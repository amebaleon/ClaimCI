from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from claimci.analysis import (
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    DatasetScanAuditContext,
    ExperimentRole,
    FieldProvenance,
    GitCommitSha,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    ScanPurpose,
    ScanReason,
    ScanState,
    Sha256Digest,
    StreamingIntegrityError,
    StreamingLimits,
    artifact_source_from_snapshot,
    extract_metric_candidate_scan,
    stream_dataset_audit_context,
)
from claimci.analysis.adapters import scan_delimited_schema, scan_jsonl_schema
from claimci.analysis.artifact_source import ArtifactSource
from claimci.dataset_check import check_streaming_dataset_context


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("7" * 40)
STREAMING_MODULES = (
    "claimci/analysis/artifact_source.py",
    "claimci/analysis/adapters/streaming.py",
    "claimci/analysis/streaming_dataset.py",
)


def _candidate(
    content: bytes,
    path: str,
    *,
    kind: ArtifactKind = ArtifactKind.RESULTS,
) -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming security fixture",
        relevant_claim_ids=("claim-streaming-security",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "streaming security fixture",
            RepositoryPath(path),
            "fixture:streaming-security",
        ),
    )


def _source(
    root: Path,
    path: str,
    content: bytes,
    *,
    kind: ArtifactKind = ArtifactKind.RESULTS,
) -> ArtifactSource:
    target = root.joinpath(*path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return artifact_source_from_snapshot(
        REPOSITORY,
        HEAD,
        root,
        _candidate(content, path, kind=kind),
    )


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are unavailable: {error}")


def _dataset_sources(root: Path, rows: dict[tuple[str, str], list[str]]):
    sources = {}
    for (experiment, split), values in rows.items():
        content = b"".join(
            (json.dumps({"value": value}, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            for value in values
        )
        path = f"{experiment}/{split}.jsonl"
        sources[(experiment, split)] = _source(
            root,
            path,
            content,
            kind=ArtifactKind.DATASET,
        )
    return sources


def test_factory_rejects_a_symlink_root_before_normalizing_it(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    content = b'{"acc":0.9}\n'
    target = actual / "results.jsonl"
    target.write_bytes(content)
    linked = tmp_path / "linked"
    _symlink_or_skip(linked, actual, directory=True)

    with pytest.raises(StreamingIntegrityError, match="root is not trusted"):
        artifact_source_from_snapshot(
            REPOSITORY,
            HEAD,
            linked,
            _candidate(content, "results.jsonl"),
        )


def test_reopen_rejects_same_bytes_file_substitution(tmp_path: Path) -> None:
    content = b'{"acc":0.9}\n'
    source = _source(tmp_path, "results.jsonl", content)
    target = tmp_path / "results.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(content)
    replacement.replace(target)

    with pytest.raises(StreamingIntegrityError, match="source identity changed"):
        source.open_scan(ScanPurpose.SCHEMA)


@pytest.mark.parametrize("link_kind", ["parent", "final"])
def test_source_rejects_symlinks_below_the_trusted_root(
    tmp_path: Path,
    link_kind: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    content = b'{"acc":0.9}\n'
    (actual / "results.jsonl").write_bytes(content)
    if link_kind == "parent":
        _symlink_or_skip(root / "linked", actual, directory=True)
        path = "linked/results.jsonl"
    else:
        (root / "results").mkdir()
        _symlink_or_skip(
            root / "results" / "linked.jsonl",
            actual / "results.jsonl",
            directory=False,
        )
        path = "results/linked.jsonl"

    with pytest.raises(StreamingIntegrityError, match="symbolic link"):
        artifact_source_from_snapshot(
            REPOSITORY,
            HEAD,
            root,
            _candidate(content, path),
        )


def test_truncated_descriptor_read_never_yields_verified_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b'{"acc":0.9}\n' * 100
    source = _source(tmp_path, "results.jsonl", content)
    original_read = os.read
    calls = 0

    def truncated_read(descriptor: int, maximum: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls > 1:
            return b""
        return original_read(descriptor, min(maximum, 32))

    monkeypatch.setattr(os, "read", truncated_read)
    with source.open_scan(ScanPurpose.SCHEMA) as scan:
        with pytest.raises(StreamingIntegrityError, match="size mismatch"):
            scan.finish(records_scanned=0, semantic_complete=False)


def test_descriptor_mutation_during_scan_fails_closed(tmp_path: Path) -> None:
    content = b'{"acc":0.9}\n' * 100
    source = _source(tmp_path, "results.jsonl", content)
    target = tmp_path / "results.jsonl"

    with source.open_scan(ScanPurpose.SCHEMA) as scan:
        assert scan.read_chunk(32)
        try:
            target.write_bytes(b'{"acc":0.8}\n' * 100)
        except PermissionError as error:
            pytest.skip(f"open-file mutation is unavailable: {error}")
        with pytest.raises(StreamingIntegrityError):
            scan.finish(records_scanned=1, semantic_complete=False)


def test_trusted_streaming_contexts_are_factory_only() -> None:
    with pytest.raises(TypeError, match="trusted snapshot factory"):
        ArtifactSource()
    with pytest.raises(TypeError, match="verified streaming materialization"):
        DatasetScanAuditContext()
    with pytest.raises(ValueError):
        RepositoryPath("../outside.jsonl")


def test_candidate_explosion_and_json_graph_limits_expose_no_candidates(
    tmp_path: Path,
) -> None:
    many = "{" + ",".join(f'\"metric_{index}\":{index}' for index in range(80)) + "}\n"
    source = _source(tmp_path, "many.jsonl", many.encode("utf-8"))
    exploded = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
        limits=StreamingLimits(max_metric_candidates=4),
    )

    assert exploded.candidates == ()
    assert exploded.match is None
    assert exploded.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert exploded.completeness.reason is ScanReason.CANDIDATE_LIMIT

    deep_content = b'{"a":{"b":{"c":1}}}\n'
    deep = _source(tmp_path, "deep.jsonl", deep_content)
    depth = scan_jsonl_schema(deep, limits=StreamingLimits(max_depth=2))
    assert depth.match is None
    assert depth.completeness.reason is ScanReason.DEPTH_LIMIT


def test_quoted_csv_record_abuse_is_bounded_and_non_authoritative(
    tmp_path: Path,
) -> None:
    content = b'run,acc\n1,"a\nb\nc\nd\ne\nf\ng\nh\ni\nj"\n'
    source = _source(tmp_path, "results.csv", content)
    outcome = scan_delimited_schema(
        source,
        limits=StreamingLimits(max_logical_record_bytes=16),
    )

    assert outcome.match is None
    assert outcome.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert outcome.completeness.reason is ScanReason.LOGICAL_RECORD_LIMIT


def test_sql_metacharacters_remain_data_and_spill_is_always_removed(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    injected = "x'); DROP TABLE dataset_hash_counts; --"
    sources = _dataset_sources(
        checkout,
        {
            ("baseline", "train"): [injected],
            ("baseline", "eval"): [injected],
            ("candidate", "train"): [injected],
            ("candidate", "eval"): [injected],
        },
    )

    with stream_dataset_audit_context(
        baseline_train=sources[("baseline", "train")],
        baseline_eval=sources[("baseline", "eval")],
        candidate_train=sources[("candidate", "train")],
        candidate_eval=sources[("candidate", "eval")],
        scratch_root=scratch,
    ) as context:
        assert context.overlap_fact("baseline").overlap_count == 1
        assert context.overlap_fact("candidate").overlap_count == 1
        assert context.evaluation_alignment_fact() is not None

    assert list(scratch.iterdir()) == []


def test_partial_dataset_scan_never_authorizes_universal_negative_claims(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sources = _dataset_sources(
        checkout,
        {
            ("baseline", "train"): ["bt-1", "bt-2"],
            ("baseline", "eval"): ["be-1", "bt-2"],
            ("candidate", "train"): ["ct-1", "ct-2"],
            ("candidate", "eval"): ["ce-1", "ct-2"],
        },
    )

    with stream_dataset_audit_context(
        baseline_train=sources[("baseline", "train")],
        baseline_eval=sources[("baseline", "eval")],
        candidate_train=sources[("candidate", "train")],
        candidate_eval=sources[("candidate", "eval")],
        scratch_root=scratch,
        limits=StreamingLimits(max_semantic_records=1),
    ) as context:
        _overlaps, findings = check_streaming_dataset_context(context)

    rules = {finding.rule_id for finding in findings}
    assert "DATASET.SCAN_INCOMPLETE" in rules
    assert "DATASET.NO_LEAKAGE" not in rules
    assert "DATASET.EVALUATION_ALIGNED" not in rules
    assert "DATASET.EVALUATION_MISMATCH" not in rules


def test_streaming_modules_reject_whole_artifact_apis_and_raw_row_lists() -> None:
    forbidden_attributes = {"read_bytes", "read_text", "splitlines"}
    forbidden_accumulators = {
        "chunks",
        "contents",
        "raw_records",
        "raw_rows",
        "records",
        "rows",
    }
    violations: list[str] = []
    root = Path(__file__).parents[1]
    for relative in STREAMING_MODULES:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attribute = node.func.attr
            if attribute in forbidden_attributes:
                violations.append(f"{relative}:{node.lineno}:{attribute}")
            if attribute == "read" and not node.args and not node.keywords:
                violations.append(f"{relative}:{node.lineno}:unbounded read")
            if attribute == "decode":
                receiver = node.func.value
                if not isinstance(receiver, ast.Name) or receiver.id not in {"raw", "encoded"}:
                    violations.append(f"{relative}:{node.lineno}:whole-source decode")
            if attribute == "append" and isinstance(node.func.value, ast.Name):
                if node.func.value.id in forbidden_accumulators:
                    violations.append(
                        f"{relative}:{node.lineno}:raw record accumulation"
                    )

    assert violations == []


def test_generated_source_over_eight_mib_has_bounded_peak_buffer(
    tmp_path: Path,
) -> None:
    line = b'{"acc":0.9,"padding":"' + (b"x" * 900) + b'"}\n'
    content = line * ((8 * 1024 * 1024 // len(line)) + 2)
    assert len(content) > 8 * 1024 * 1024
    source = _source(tmp_path, "large.jsonl", content)
    limits = StreamingLimits(
        max_integrity_bytes=len(content),
        max_semantic_bytes=len(content),
        max_logical_record_bytes=1024,
    )

    outcome = scan_jsonl_schema(source, limits=limits)

    assert outcome.completeness.complete is True
    assert outcome.completeness.integrity_verified is True
    assert outcome.completeness.peak_buffer_bytes <= 1024 + (64 * 1024)
    assert outcome.completeness.integrity_bytes == len(content)
