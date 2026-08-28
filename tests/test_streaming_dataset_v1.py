from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import claimci.analysis.streaming_dataset as streaming_dataset
from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    ArtifactCandidate,
    ArtifactKind,
    AnalysisContractError,
    Confidence,
    FieldProvenance,
    GitCommitSha,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    ScanReason,
    ScanState,
    Sha256Digest,
    StreamingIntegrityError,
    StreamingArtifactError,
    StreamingLimits,
    artifact_source_from_snapshot,
)
from claimci.analysis.streaming_dataset import (
    DatasetScanAuditContext,
    stream_dataset_audit_context,
)
from claimci.analysis.adapters import extract_registered_source
from claimci.analysis.adapters.dataset import PassiveJsonLinesDatasetAdapter
from claimci.audit import audit_research
from claimci.dataset_check import hash_sample
from claimci.models import Impact, Verdict


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("6" * 40)


def _source(root: Path, relative: str):
    target = root.joinpath(*relative.split("/"))
    content = target.read_bytes()
    candidate = ArtifactCandidate(
        path=RepositoryPath(relative),
        kind=ArtifactKind.DATASET,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming dataset fixture",
        relevant_claim_ids=("claim-dataset",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "streaming dataset fixture",
            RepositoryPath(relative),
            "fixture:streaming-dataset",
        ),
    )
    return artifact_source_from_snapshot(
        REPOSITORY,
        HEAD,
        root,
        candidate,
    )


def _sources(root: Path):
    return {
        "baseline_train": _source(root, "base/train.jsonl"),
        "baseline_eval": _source(root, "base/eval.jsonl"),
        "candidate_train": _source(root, "candidate/train.jsonl"),
        "candidate_eval": _source(root, "candidate/eval.jsonl"),
    }


def test_passive_and_streaming_dataset_evidence_ids_have_exact_occurrence_parity(
    tmp_path: Path,
) -> None:
    content = b'{"prompt":"a"}\n{"prompt":"b"}\n'
    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "train.jsonl"
    target.write_bytes(content)
    source = _source(root, "train.jsonl")
    passive = passive_artifact(
        source.candidate,
        content,
        repository=source.repository,
        snapshot_role=source.snapshot_role,
        commit=source.head_sha,
    )
    adapter = PassiveJsonLinesDatasetAdapter()
    match = adapter.probe(passive)
    streamed = extract_registered_source(source)

    assert match is not None
    assert streamed is not None
    assert streamed.evidence is not None
    passive_evidence = adapter.extract(passive, match)
    assert passive_evidence.evidence_id.startswith("evidence-v2-")
    assert streamed.evidence.evidence_id == passive_evidence.evidence_id


def _write_rows(path: Path, rows: list[object], *, blank_lines: bool = False) -> None:
    rendered = []
    for row in rows:
        if blank_lines:
            rendered.append("\n")
        rendered.append(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    path.write_text("".join(rendered), encoding="utf-8")


def _rule(result, rule_id: str):
    return next((item for item in result.findings if item.rule_id == rule_id), None)


def test_complete_large_pair_may_emit_no_leakage(
    tmp_path: Path,
    study_factory,
) -> None:
    manifest = study_factory()
    root = manifest.parent
    _write_rows(root / "base" / "train.jsonl", [{"id": f"bt-{i}"} for i in range(2000)])
    _write_rows(root / "base" / "eval.jsonl", [{"id": f"eval-{i}"} for i in range(1000)])
    _write_rows(root / "candidate" / "train.jsonl", [{"id": f"ct-{i}"} for i in range(2000)])
    _write_rows(root / "candidate" / "eval.jsonl", [{"id": f"eval-{i}"} for i in range(1000)])
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(TypeError, match="verified streaming materialization"):
        DatasetScanAuditContext()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        **_sources(root),
    ) as context:
        result = audit_research(manifest, dataset_scan_context=context)
        assert context.evaluation_identity("baseline").record_count == 1000

    assert len([item for item in result.findings if item.rule_id == "DATASET.NO_LEAKAGE"]) == 2
    assert _rule(result, "DATASET.SCAN_INCOMPLETE") is None
    assert tuple(scratch.iterdir()) == ()
    with pytest.raises(AnalysisContractError, match="no longer active"):
        context.scan("baseline", "eval")


def test_integrity_verified_partial_without_overlap_cannot_emit_no_leakage(
    tmp_path: Path,
    study_factory,
) -> None:
    manifest = study_factory()
    root = manifest.parent
    rows = [{"id": f"row-{index}"} for index in range(20)]
    _write_rows(root / "base" / "train.jsonl", rows)
    _write_rows(root / "base" / "eval.jsonl", [{"id": f"eval-{i}"} for i in range(20)])
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        limits=StreamingLimits(max_semantic_records=10),
        **_sources(root),
    ) as context:
        result = audit_research(manifest, dataset_scan_context=context)
        assert all(item.completeness.integrity_verified for item in context.scans)
        assert context.evaluation_identity("baseline") is None

    assert not any(
        item.rule_id == "DATASET.NO_LEAKAGE"
        and item.evidence["experiment"] == "baseline"
        for item in result.findings
    )
    incomplete = _rule(result, "DATASET.SCAN_INCOMPLETE")
    assert incomplete is not None
    assert incomplete.impact is Impact.INSUFFICIENT
    assert tuple(scratch.iterdir()) == ()


def test_integrity_verified_partial_exact_overlap_remains_invalidating(
    tmp_path: Path,
    study_factory,
) -> None:
    shared = {"id": "confirmed-shared"}
    manifest = study_factory()
    root = manifest.parent
    _write_rows(
        root / "base" / "train.jsonl",
        [shared, *({"id": f"train-{i}"} for i in range(20))],
    )
    _write_rows(
        root / "base" / "eval.jsonl",
        [shared, *({"id": f"eval-{i}"} for i in range(20))],
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        limits=StreamingLimits(max_semantic_records=10),
        **_sources(root),
    ) as context:
        result = audit_research(manifest, dataset_scan_context=context)

    leakage = _rule(result, "DATASET.EXACT_LEAKAGE")
    assert leakage is not None
    assert leakage.impact is Impact.INVALIDATES
    assert _rule(result, "DATASET.SCAN_INCOMPLETE") is not None
    assert result.verdict is Verdict.NOT_SUPPORTED
    assert tuple(scratch.iterdir()) == ()


def test_blank_lines_remain_compatible_and_preview_is_bounded_to_128(
    tmp_path: Path,
    study_factory,
) -> None:
    overlaps = [{"id": f"shared-{index}"} for index in range(200)]
    manifest = study_factory()
    root = manifest.parent
    _write_rows(root / "base" / "train.jsonl", overlaps, blank_lines=True)
    _write_rows(root / "base" / "eval.jsonl", overlaps, blank_lines=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        **_sources(root),
    ) as context:
        result = audit_research(manifest, dataset_scan_context=context)

    leakage = _rule(result, "DATASET.EXACT_LEAKAGE")
    assert leakage is not None
    assert leakage.evidence["overlap_count"] == 200
    assert len(leakage.evidence["hashes"]) == 128
    assert leakage.evidence["hash_preview_truncated"] is True


@pytest.mark.parametrize(
    "malformed",
    [
        b'{"id":1,"id":2}\n',
        b'{"id":1}\nnot-json\n',
    ],
)
def test_malformed_dataset_discards_its_provisional_rows_and_cleans_up(
    tmp_path: Path,
    study_factory,
    malformed: bytes,
) -> None:
    manifest = study_factory()
    root = manifest.parent
    (root / "base" / "train.jsonl").write_bytes(malformed)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        **_sources(root),
    ) as context:
        failed = context.scan("baseline", "train")
        assert failed.completeness.state is ScanState.FAILED
        assert failed.indexed_records == 0
        result = audit_research(manifest, dataset_scan_context=context)

    assert not any(
        item.rule_id == "DATASET.NO_LEAKAGE"
        and item.evidence["experiment"] == "baseline"
        for item in result.findings
    )
    assert _rule(result, "DATASET.EXACT_LEAKAGE") is None
    assert tuple(scratch.iterdir()) == ()


def test_scratch_exhaustion_is_incomplete_and_always_removed(
    tmp_path: Path,
    study_factory,
) -> None:
    manifest = study_factory()
    root = manifest.parent
    many = [{"id": f"unique-{index:06d}"} for index in range(5000)]
    _write_rows(root / "base" / "train.jsonl", many)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        limits=StreamingLimits(max_scratch_bytes=32 * 1024),
        **_sources(root),
    ) as context:
        assert any(
            item.completeness.reason is ScanReason.SCRATCH_BUDGET
            for item in context.scans
        )
        result = audit_research(manifest, dataset_scan_context=context)

    assert _rule(result, "DATASET.SCAN_INCOMPLETE") is not None
    assert tuple(scratch.iterdir()) == ()


def test_integrity_mismatch_yields_no_context_and_removes_spill(
    tmp_path: Path,
    study_factory,
) -> None:
    manifest = study_factory()
    root = manifest.parent
    sources = _sources(root)
    target = root / "base" / "train.jsonl"
    original = target.read_bytes()
    target.write_bytes(b"x" * len(original))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(StreamingIntegrityError):
        with stream_dataset_audit_context(
            scratch_root=scratch,
            **sources,
        ):
            pytest.fail("integrity-mismatched context must not be issued")

    assert tuple(scratch.iterdir()) == ()


def test_spill_schema_is_fixed_upserts_are_parameterized_and_permissions_restrictive(
    tmp_path: Path,
    study_factory,
) -> None:
    assert "CREATE TABLE dataset_hash_counts" in streaming_dataset._SCHEMA_SQL
    assert "CHECK (experiment IN ('baseline','candidate'))" in streaming_dataset._SCHEMA_SQL
    assert "VALUES (?, ?, ?, ?)" in streaming_dataset._UPSERT_SQL
    manifest = study_factory()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        **_sources(manifest.parent),
    ):
        files = tuple(scratch.iterdir())
        assert len(files) == 1
        if os.name != "nt":
            assert stat.S_IMODE(files[0].stat().st_mode) == 0o600

    assert tuple(scratch.iterdir()) == ()


def test_scratch_budget_too_small_for_fixed_schema_fails_typed_and_cleans(
    tmp_path: Path,
    study_factory,
) -> None:
    manifest = study_factory()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(StreamingArtifactError, match="scratch budget"):
        with stream_dataset_audit_context(
            scratch_root=scratch,
            limits=StreamingLimits(max_scratch_bytes=4096),
            **_sources(manifest.parent),
        ):
            pytest.fail("undersized scratch must not issue an Audit context")

    assert tuple(scratch.iterdir()) == ()


def test_spill_cleanup_runs_when_audit_raises(
    tmp_path: Path,
    study_factory,
    monkeypatch,
) -> None:
    manifest = study_factory()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(
        "claimci.audit.determine_verdict",
        lambda _findings: (_ for _ in ()).throw(RuntimeError("forced Audit error")),
    )

    with pytest.raises(RuntimeError, match="forced Audit error"):
        with stream_dataset_audit_context(
            scratch_root=scratch,
            **_sources(manifest.parent),
        ) as context:
            audit_research(manifest, dataset_scan_context=context)

    assert tuple(scratch.iterdir()) == ()


def test_complete_multiset_identity_matches_legacy_measurement_commitment(
    tmp_path: Path,
    study_factory,
) -> None:
    rows = [{"id": "a"}, {"id": "duplicate"}, {"id": "duplicate"}]
    manifest = study_factory(baseline_eval=rows, candidate_eval=rows)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with stream_dataset_audit_context(
        scratch_root=scratch,
        **_sources(manifest.parent),
    ) as context:
        identity = context.evaluation_identity("baseline")
        assert identity is not None

    hashes = [hash_sample(row) for row in rows]
    expected_rows = [
        {"sha256": digest, "count": hashes.count(digest)}
        for digest in sorted(set(hashes))
    ]
    expected = hashlib.sha256(
        json.dumps(
            expected_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert identity.record_count == 3
    assert identity.canonical_sha256 == expected
