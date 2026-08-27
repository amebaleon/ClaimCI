"""Evidence-v2 identity material and trusted occurrence regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    ArtifactCandidate,
    ArtifactKind,
    ArtifactSnapshotRole,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    artifact_source_from_snapshot,
)
from claimci.analysis.adapters.core import _evidence_id
from claimci.analysis.adapters.streaming import (
    scan_jsonl_observations,
    scan_jsonl_schema,
)


REPOSITORY = RepositoryIdentity("claimci-tests", "identity-v2")
HEAD = GitCommitSha("a" * 40)


def _expected_v2(material: dict[str, object]) -> str:
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "evidence-v2-" + hashlib.sha256(encoded).hexdigest()


def _candidate(
    content: bytes,
    *,
    path: str = "results/metrics.json",
    kind: ArtifactKind = ArtifactKind.RESULTS,
    claim_ids: tuple[str, ...] = ("claim-original",),
    detail: str = "original provenance detail",
) -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.9),
        discovery_reason="identity fixture",
        relevant_claim_ids=claim_ids,
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            detail,
            RepositoryPath(path),
            "identity-provenance-original",
        ),
    )


def _mapping(
    candidate: ArtifactCandidate,
    *,
    target: str,
    expression: str,
    detail: str = "selector provenance detail",
) -> FieldMapping:
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        detail,
        candidate.path,
        "selector-provenance-original",
    )
    return FieldMapping(
        target_field=target,
        selector=EvidenceSelector(
            SelectorKind.JSON_POINTER,
            expression,
            provenance,
        ),
        provenance=provenance,
    )


def _artifact(
    content: bytes,
    *,
    path: str = "results/metrics.json",
    kind: ArtifactKind = ArtifactKind.RESULTS,
    repository: RepositoryIdentity = REPOSITORY,
    snapshot_role: ArtifactSnapshotRole = ArtifactSnapshotRole.HEAD,
    commit: GitCommitSha = HEAD,
    claim_ids: tuple[str, ...] = ("claim-original",),
    detail: str = "original provenance detail",
) -> PassiveArtifact:
    candidate = _candidate(
        content,
        path=path,
        kind=kind,
        claim_ids=claim_ids,
        detail=detail,
    )
    return passive_artifact(
        candidate,
        content,
        repository=repository,
        snapshot_role=snapshot_role,
        commit=commit,
    )


def _base_fixture() -> tuple[PassiveArtifact, tuple[FieldMapping, ...]]:
    content = b'{"metrics":{"accuracy":0.9},"run":"one"}'
    artifact = _artifact(content)
    # Deliberately pass mappings in the reverse order; canonical material must
    # sort by field_mapping_identity rather than trust tuple order.
    mappings = (
        _mapping(
            artifact.candidate,
            target="run_id",
            expression="/run",
        ),
        _mapping(
            artifact.candidate,
            target="metric_value",
            expression="/metrics/accuracy",
        ),
    )
    return artifact, mappings


def _v2_material(
    artifact: PassiveArtifact,
    *,
    adapter_id: str = "claimci-json-v1",
    version: str = "1",
) -> dict[str, object]:
    # Keep this dictionary literal and independent from the production helper.
    # Its selector entries are intentionally written in canonical target order.
    return {
        "schema_version": 2,
        "repository": {"owner": "claimci-tests", "name": "identity-v2"},
        "snapshot": {"role": "head", "commit": "a" * 40},
        "artifact": {
            "path": "results/metrics.json",
            "kind": "results",
            "sha256": str(artifact.candidate.sha256),
            "size": artifact.candidate.size,
        },
        "adapter": {"id": adapter_id, "version": version},
        "selector": [
            {
                "target": "metric_value",
                "kind": "json_pointer",
                "expression": "/metrics/accuracy",
            },
            {
                "target": "run_id",
                "kind": "json_pointer",
                "expression": "/run",
            },
        ],
    }


def test_canonical_selector_material_is_sorted_and_provenance_free() -> None:
    artifact, mappings = _base_fixture()
    import claimci.analysis.evidence_identity as identity

    assert identity.canonical_selector_material(mappings) == (
        {
            "target": "metric_value",
            "kind": "json_pointer",
            "expression": "/metrics/accuracy",
        },
        {
            "target": "run_id",
            "kind": "json_pointer",
            "expression": "/run",
        },
    )
    assert "selector-provenance-original" not in repr(
        identity.canonical_selector_material(mappings)
    )
    assert artifact.candidate.relevant_claim_ids == ("claim-original",)


def test_v2_identity_uses_exact_literal_material_and_is_stable() -> None:
    artifact, mappings = _base_fixture()
    expected = _expected_v2(_v2_material(artifact))

    first = _evidence_id("claimci-json-v1", "1", artifact, mappings)
    second = _evidence_id("claimci-json-v1", "1", artifact, mappings)

    assert first == expected
    assert second == expected
    assert len(first) == 76
    assert first.startswith("evidence-v2-")


@pytest.mark.parametrize(
    ("label", "kwargs"),
    (
        (
            "repository",
            {"repository": RepositoryIdentity("other-owner", "identity-v2")},
        ),
        ("snapshot role", {"snapshot_role": ArtifactSnapshotRole.BASE}),
        ("commit", {"commit": GitCommitSha("b" * 40)}),
        ("path", {"path": "results/other.json"}),
        ("kind", {"kind": ArtifactKind.BENCHMARK}),
        (
            "full sha and size",
            {"content": b'{"metrics":{"accuracy":0.91},"run":"one"}'},
        ),
    ),
)
def test_v2_identity_separates_physical_occurrence_material(
    label: str,
    kwargs: dict[str, object],
) -> None:
    del label
    base, mappings = _base_fixture()
    changed_content = kwargs.pop("content", base.content)
    changed = _artifact(changed_content, **kwargs)  # type: ignore[arg-type]

    assert _evidence_id("claimci-json-v1", "1", changed, mappings) != (
        _evidence_id("claimci-json-v1", "1", base, mappings)
    )


@pytest.mark.parametrize(
    ("adapter_id", "version"),
    (
        ("claimci-jsonl-v1", "1"),
        ("claimci-json-v1", "2"),
    ),
)
def test_v2_identity_separates_trusted_adapter_id_and_version(
    adapter_id: str,
    version: str,
) -> None:
    artifact, mappings = _base_fixture()
    assert _evidence_id(adapter_id, version, artifact, mappings) != _evidence_id(
        "claimci-json-v1", "1", artifact, mappings
    )


def test_v2_identity_separates_canonical_selector_identity() -> None:
    artifact, mappings = _base_fixture()
    changed = (
        mappings[0],
        _mapping(
            artifact.candidate,
            target="metric_value",
            expression="/metrics/loss",
        ),
    )
    assert _evidence_id("claimci-json-v1", "1", artifact, changed) != _evidence_id(
        "claimci-json-v1", "1", artifact, mappings
    )


def test_v2_identity_excludes_claim_provenance_and_scientific_role() -> None:
    content = b'{"metrics":{"accuracy":0.9},"run":"one"}'
    original = _artifact(content)
    changed_metadata = _artifact(
        content,
        claim_ids=("claim-rewritten",),
        detail="provider supplied provenance detail",
    )
    original_mapping = _mapping(
        original.candidate,
        target="metric_value",
        expression="/metrics/accuracy",
    )
    changed_mapping = _mapping(
        changed_metadata.candidate,
        target="metric_value",
        expression="/metrics/accuracy",
        detail="provider supplied selector detail",
    )

    expected = _evidence_id("claimci-json-v1", "1", original, (original_mapping,))
    assert _evidence_id("claimci-json-v1", "1", changed_metadata, (changed_mapping,)) == expected

    class RoleEnvelope:
        def __init__(self, role: ExperimentRole) -> None:
            self.occurrence = original.occurrence
            self.candidate = original.candidate
            self.experiment_role = role

    for role in ExperimentRole:
        # Scientific role is deliberately an extra observation-context field;
        # it is not an input to the physical identity helper.
        assert (
            _evidence_id(
                "claimci-json-v1",
                "1",
                RoleEnvelope(role),  # type: ignore[arg-type]
                (original_mapping,),
            )
            == expected
        )


def test_v2_identity_requires_a_concrete_factory_issued_occurrence() -> None:
    artifact, mappings = _base_fixture()

    class LegacyEnvelope:
        candidate = artifact.candidate

    with pytest.raises(TypeError, match="occurrence|Occurrence"):
        _evidence_id("claimci-json-v1", "1", LegacyEnvelope(), mappings)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="semantic version"):
        _evidence_id("claimci-json-v1", "", artifact, mappings)


def test_v2_identity_rejects_candidate_not_bound_to_occurrence() -> None:
    occurrence_artifact = _artifact(
        b'{"metrics":{"accuracy":0.9},"run":"one"}',
        path="results/occurrence-a.json",
    )
    unbound_candidate = _candidate(
        b'{"metrics":{"accuracy":0.8},"run":"two"}',
        path="results/candidate-b.json",
    )

    class MismatchedEnvelope:
        occurrence = occurrence_artifact.occurrence
        candidate = unbound_candidate

    expected_if_unbound_candidate_were_hashed = _expected_v2(
        {
            "schema_version": 2,
            "repository": {"owner": "claimci-tests", "name": "identity-v2"},
            "snapshot": {"role": "head", "commit": "a" * 40},
            "artifact": {
                "path": "results/candidate-b.json",
                "kind": "results",
                "sha256": str(unbound_candidate.sha256),
                "size": unbound_candidate.size,
            },
            "adapter": {"id": "claimci-json-v1", "version": "1"},
            "selector": [],
        }
    )

    try:
        observed = _evidence_id(
            "claimci-json-v1",
            "1",
            MismatchedEnvelope(),  # type: ignore[arg-type]
            (),
        )
    except TypeError as error:
        assert "candidate" in str(error)
        return

    assert observed == expected_if_unbound_candidate_were_hashed
    pytest.fail(
        "v2 identity accepted an unbound candidate; expected a TypeError "
        f"(observed {observed})"
    )


def test_streaming_jsonl_uses_fixed_version_and_real_canonical_mappings(
    tmp_path: Path,
) -> None:
    content = b'{"accuracy":0.90,"run":"one"}\n{"accuracy":0.80,"run":"two"}\n'
    target = tmp_path / "results" / "runs.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    candidate = _candidate(content, path="results/runs.jsonl")
    source = artifact_source_from_snapshot(REPOSITORY, HEAD, tmp_path, candidate)

    schema = scan_jsonl_schema(source)
    assert schema.match is not None
    extraction = scan_jsonl_observations(source, schema.match)
    assert extraction.evidence is not None
    evidence = extraction.evidence

    expected = _expected_v2(
        {
            "schema_version": 2,
            "repository": {"owner": "claimci-tests", "name": "identity-v2"},
            "snapshot": {"role": "head", "commit": "a" * 40},
            "artifact": {
                "path": "results/runs.jsonl",
                "kind": "results",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            },
            "adapter": {"id": "claimci-jsonl-v1", "version": "1"},
            "selector": [
                {
                    "target": "metric_value",
                    "kind": "json_pointer",
                    "expression": "/accuracy",
                },
                {
                    "target": "run_id",
                    "kind": "json_pointer",
                    "expression": "/run",
                },
            ],
        }
    )
    assert evidence.evidence_id == expected
    assert len(evidence.evidence_id) == 76
