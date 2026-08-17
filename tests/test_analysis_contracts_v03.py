"""Behavioral contracts for the ClaimCI v0.3 shared analysis model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest

from claimci.analysis import (
    AdapterMatch,
    AnalysisAuthority,
    AnalysisState,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingTrust,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)


RAW_RESULTS = b'{"runs":[{"seed":1,"accuracy":0.9}]}\n'
RAW_DIGEST = hashlib.sha256(RAW_RESULTS).hexdigest()


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail="issued repository index matched the artifact",
        source_path=RepositoryPath("results/candidate.json"),
        source_id="repository-index",
    )


def _mapping(target: str = "metric_value") -> FieldMapping:
    provenance = _provenance(ProvenanceKind.ADAPTER_EXTRACTION)
    return FieldMapping(
        target_field=target,
        selector=EvidenceSelector(
            kind=SelectorKind.JSON_POINTER,
            expression="/runs/0/accuracy",
            provenance=provenance,
        ),
        provenance=provenance,
    )


def _candidate(**overrides: object) -> ArtifactCandidate:
    values: dict[str, object] = {
        "path": RepositoryPath("results/candidate.json"),
        "kind": ArtifactKind.RESULTS,
        "sha256": Sha256Digest(RAW_DIGEST),
        "size": len(RAW_RESULTS),
        "confidence": Confidence(0.9),
        "discovery_reason": "results filename and payload shape matched",
        "relevant_claim_ids": ("claim-1",),
        "provenance": _provenance(),
    }
    values.update(overrides)
    return ArtifactCandidate(**values)  # type: ignore[arg-type]


def test_confidence_is_finite_bounded_and_immutable() -> None:
    confidence = Confidence(0.75)

    assert confidence.value == 0.75
    assert float(confidence) == 0.75
    with pytest.raises(FrozenInstanceError):
        confidence.value = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", None, math.nan, math.inf, -math.inf, -0.01, 1.01],
)
def test_confidence_rejects_boolean_non_numeric_nonfinite_and_out_of_range_values(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Confidence(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "results/candidate.json",
        "README.md",
        ".claimci/review.yaml",
        "research outputs/metrics.json",
    ],
)
def test_repository_path_accepts_canonical_portable_file_paths(value: str) -> None:
    path = RepositoryPath(value)

    assert isinstance(path, str)
    assert path == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        ".",
        "./result.json",
        "../result.json",
        "data/../result.json",
        "/result.json",
        "C:/result.json",
        "C:result.json",
        "//server/share/result.json",
        r"data\result.json",
        "data//result.json",
        "data/./result.json",
        "data/result.json/",
        " data/result.json",
        "data/result.json ",
        "data/\x00result.json",
    ],
)
def test_repository_path_rejects_nonportable_unconfined_or_noncanonical_values(
    value: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RepositoryPath(value)


def test_repository_path_rejects_non_text_values() -> None:
    with pytest.raises(TypeError):
        RepositoryPath(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "digest",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "", 123],
)
def test_sha256_digest_rejects_wrong_length_case_alphabet_and_type(
    digest: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Sha256Digest(digest)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sha",
    ["a" * 39, "a" * 41, "a" * 63, "a" * 65, "A" * 40, "g" * 40, "0" * 40],
)
def test_git_commit_sha_rejects_malformed_or_zero_values(sha: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        GitCommitSha(sha)


def test_public_enum_wire_values_are_stable() -> None:
    assert {item.name: item.value for item in ArtifactKind} == {
        "RESULTS": "results",
        "CONFIG": "config",
        "DATASET": "dataset",
        "BENCHMARK": "benchmark",
        "DOCUMENT": "document",
        "SOURCE": "source",
        "TEST": "test",
        "MANIFEST": "manifest",
    }
    assert {item.name: item.value for item in ExperimentRole} == {
        "BASELINE": "baseline",
        "CANDIDATE": "candidate",
        "REFERENCE": "reference",
        "UNSPECIFIED": "unspecified",
    }
    assert {item.name: item.value for item in ProvenanceKind} == {
        "DETERMINISTIC_DISCOVERY": "deterministic_discovery",
        "ADAPTER_EXTRACTION": "adapter_extraction",
        "MANIFEST_HINT": "manifest_hint",
        "PROVIDER_PROPOSAL": "provider_proposal",
        "USER_APPROVED": "user_approved",
    }
    assert {item.name: item.value for item in MappingTrust} == {
        "INFERRED": "inferred",
        "MANIFEST_HINT": "manifest_hint",
        "USER_APPROVED": "user_approved",
    }
    assert {item.name: item.value for item in SelectorKind} == {
        "JSON_POINTER": "json_pointer",
        "DOTTED_PATH": "dotted_path",
        "COLUMN": "column",
    }
    assert {item.name: item.value for item in AnalysisState} == {
        "COMPLETE": "complete",
        "MAPPING_NEEDED": "mapping_needed",
        "PARTIAL": "partial",
        "UNAVAILABLE": "unavailable",
    }
    assert {item.name: item.value for item in AnalysisAuthority} == {
        "DETERMINISTIC": "deterministic",
        "ADVISORY": "advisory",
    }


def test_field_provenance_is_validated_and_frozen() -> None:
    provenance = _provenance(ProvenanceKind.PROVIDER_PROPOSAL)

    assert provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL
    assert provenance.source_path == "results/candidate.json"
    with pytest.raises(FrozenInstanceError):
        provenance.detail = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "provider_proposal"},
        {"detail": ""},
        {"detail": "x" * 4097},
        {"source_path": "../outside.json"},
        {"source_id": ""},
    ],
)
def test_field_provenance_rejects_invalid_fields(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "kind": ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "detail": "matched by trusted repository index",
        "source_path": RepositoryPath("results/candidate.json"),
        "source_id": "repository-index",
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        FieldProvenance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kind,expression",
    [
        (SelectorKind.JSON_POINTER, "/runs/0/accuracy"),
        (SelectorKind.DOTTED_PATH, "runs.0.accuracy"),
        (SelectorKind.COLUMN, "validation accuracy"),
    ],
)
def test_evidence_selector_accepts_bounded_declarative_syntax(
    kind: SelectorKind,
    expression: str,
) -> None:
    selector = EvidenceSelector(kind, expression, _provenance())

    assert selector.expression == expression


@pytest.mark.parametrize(
    "kind,expression",
    [
        (SelectorKind.JSON_POINTER, "runs/0/accuracy"),
        (SelectorKind.JSON_POINTER, "/runs\naccuracy"),
        (SelectorKind.DOTTED_PATH, "runs..accuracy"),
        (SelectorKind.DOTTED_PATH, ".runs"),
        (SelectorKind.COLUMN, ""),
        (SelectorKind.COLUMN, "accuracy\nvalue"),
    ],
)
def test_evidence_selector_rejects_malformed_or_active_syntax(
    kind: SelectorKind,
    expression: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        EvidenceSelector(kind, expression, _provenance())


def test_field_mapping_rejects_invalid_target_fields() -> None:
    selector = EvidenceSelector(
        SelectorKind.COLUMN,
        "accuracy",
        _provenance(),
    )

    for target in ("", "metric value", "__import__('os')", "x" * 129):
        with pytest.raises((TypeError, ValueError)):
            FieldMapping(target, selector, _provenance())


def test_artifact_candidate_preserves_typed_metadata_and_is_frozen() -> None:
    candidate = _candidate()

    assert candidate.path == "results/candidate.json"
    assert candidate.kind is ArtifactKind.RESULTS
    assert candidate.sha256 == RAW_DIGEST
    assert candidate.size == len(RAW_RESULTS)
    assert candidate.relevant_claim_ids == ("claim-1",)
    with pytest.raises(FrozenInstanceError):
        candidate.size = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "../outside.json"},
        {"kind": "results"},
        {"sha256": "A" * 64},
        {"size": -1},
        {"size": True},
        {"confidence": 0.9},
        {"discovery_reason": ""},
        {"relevant_claim_ids": ["claim-1"]},
        {"relevant_claim_ids": ("claim-1", "claim-1")},
        {"relevant_claim_ids": ("",)},
        {"provenance": "repository-index"},
    ],
)
def test_artifact_candidate_rejects_invalid_contract_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _candidate(**changes)


def test_passive_artifact_verifies_bytes_without_execution_surface() -> None:
    artifact = PassiveArtifact(_candidate(), RAW_RESULTS)

    assert artifact.content is RAW_RESULTS
    assert not hasattr(artifact, "execute")
    assert not hasattr(artifact, "repository_root")
    with pytest.raises(FrozenInstanceError):
        artifact.content = b"changed"  # type: ignore[misc]


def test_passive_artifact_rejects_wrong_size_digest_and_mutable_content() -> None:
    with pytest.raises((TypeError, ValueError), match="size"):
        PassiveArtifact(_candidate(size=1), RAW_RESULTS)
    with pytest.raises((TypeError, ValueError), match="sha256|digest"):
        PassiveArtifact(_candidate(sha256=Sha256Digest("b" * 64)), RAW_RESULTS)
    with pytest.raises((TypeError, ValueError)):
        PassiveArtifact(_candidate(), bytearray(RAW_RESULTS))  # type: ignore[arg-type]


def test_adapter_match_carries_validated_mappings_and_match_evidence() -> None:
    match = AdapterMatch(
        adapter_id="json-runs-v1",
        path=RepositoryPath("results/candidate.json"),
        confidence=Confidence(0.95),
        mappings=(_mapping(),),
        match_evidence=(_provenance(),),
    )

    assert match.adapter_id == "json-runs-v1"
    assert match.mappings[0].target_field == "metric_value"
    with pytest.raises(FrozenInstanceError):
        match.adapter_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"adapter_id": ""},
        {"adapter_id": "package.module:Adapter"},
        {"path": "../outside.json"},
        {"confidence": 0.9},
        {"mappings": [_mapping()]},
        {"mappings": (_mapping(), _mapping())},
        {"match_evidence": [_provenance()]},
        {"match_evidence": ("matched",)},
    ],
)
def test_adapter_match_rejects_invalid_or_executable_contract_fields(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "adapter_id": "json-runs-v1",
        "path": RepositoryPath("results/candidate.json"),
        "confidence": Confidence(0.95),
        "mappings": (_mapping(),),
        "match_evidence": (_provenance(),),
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        AdapterMatch(**values)  # type: ignore[arg-type]
