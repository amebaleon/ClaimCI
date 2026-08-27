"""Generic bounded JSON and per-record JSONL adapter behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ExperimentRole,
    FieldProvenance,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.adapters.core import (
    MAX_LOGICAL_RECORD_BYTES,
    MAX_RECORDS,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _mapping,
)
from claimci.analysis.adapters.structured import JsonAdapter, JsonLinesAdapter


FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def _passive(
    content: bytes,
    *,
    path: str,
    kind: ArtifactKind = ArtifactKind.RESULTS,
) -> PassiveArtifact:
    return passive_artifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=kind,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.9),
            discovery_reason="structured adapter fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "fixture bytes",
                RepositoryPath(path),
                "fixture",
            ),
        ),
        content,
    )


def _fixture(name: str, *, kind: ArtifactKind = ArtifactKind.RESULTS) -> PassiveArtifact:
    return _passive(
        (FIXTURES / name).read_bytes(),
        path=f"artifacts/{name}",
        kind=kind,
    )


def _targets(match: AdapterMatch) -> dict[str, str]:
    return {
        mapping.target_field: mapping.selector.expression
        for mapping in match.mappings
    }


def test_nested_json_probe_and_extract_preserve_field_provenance() -> None:
    artifact = _fixture("nested_metric.json")
    adapter = JsonAdapter()

    match = adapter.probe(artifact)
    assert match is not None
    assert match.adapter_id == "claimci-json-v1"
    assert _targets(match) == {
        "metric_value": "/metrics/accuracy",
        "run_id": "/run_id",
        "seed": "/seed",
    }

    evidence = adapter.extract(artifact, match)
    assert len(evidence.observations) == 1
    observation = evidence.observations[0]
    assert observation.metric_name == "accuracy"
    assert observation.metric_value == pytest.approx(0.91)
    assert observation.run_id == "candidate-7"
    assert observation.seed == 41
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED
    assert observation.provenance.source_path == artifact.candidate.path
    assert artifact.candidate.sha256 in observation.provenance.detail
    assert "/metrics/accuracy" in observation.provenance.detail


def test_json_preserves_one_explicit_metric_name_field_when_available() -> None:
    artifact = _passive(
        b'{"metric_name":"f1_score","value":0.81}',
        path="artifacts/named.json",
    )
    adapter = JsonAdapter()
    match = adapter.probe(artifact)
    assert match is not None
    assert _targets(match) == {
        "metric_name": "/metric_name",
        "metric_value": "/value",
    }
    observation = adapter.extract(artifact, match).observations[0]
    assert observation.metric_name == "f1_score"
    assert observation.metric_value == pytest.approx(0.81)


def test_ambiguous_json_requires_external_metric_mapping_without_guessing() -> None:
    artifact = _fixture("ambiguous_metrics.json")
    adapter = JsonAdapter()
    ambiguous = adapter.probe(artifact)

    assert ambiguous is not None
    assert "metric_value" not in _targets(ambiguous)
    assert float(ambiguous.confidence) < 0.8
    with pytest.raises(AdapterSelectorError, match="metric_value|mapping|ambiguous"):
        adapter.extract(artifact, ambiguous)

    metric = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/accuracy",
        inferred=False,
    )
    explicit = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(metric,),
        match_evidence=(metric.provenance,),
    )
    observation = adapter.extract(artifact, explicit).observations[0]
    assert observation.metric_name == "accuracy"
    assert observation.metric_value == pytest.approx(0.91)
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED


def test_json_config_uses_stable_structural_mappings_and_no_role_inference() -> None:
    artifact = _passive(
        b'{"model":{"name":"tiny"},"training":{"batch_size":16}}',
        path="configs/model.json",
        kind=ArtifactKind.CONFIG,
    )
    adapter = JsonAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    selectors = tuple(mapping.selector.expression for mapping in match.mappings)
    assert selectors == ("/model/name", "/training/batch_size")
    observation = adapter.extract(artifact, match).observations[0]
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED
    assert tuple((item.key, item.value) for item in observation.config_values) == (
        ("model.name", "tiny"),
        ("training.batch_size", 16),
    )


def test_jsonl_pointer_is_resolved_inside_each_logical_record() -> None:
    artifact = _fixture("seed_runs.jsonl")
    adapter = JsonLinesAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    assert _targets(match) == {
        "metric_value": "/metrics/accuracy",
        "run_id": "/run_id",
        "seed": "/seed",
    }
    evidence = adapter.extract(artifact, match)
    assert [item.metric_value for item in evidence.observations] == [0.88, 0.90, 0.92]
    assert [item.run_id for item in evidence.observations] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert [item.seed for item in evidence.observations] == [11, 12, 13]
    assert all(
        item.experiment_role is ExperimentRole.UNSPECIFIED
        for item in evidence.observations
    )

    global_pointer = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/0/metrics/accuracy",
        inferred=False,
    )
    invalid = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(global_pointer,),
        match_evidence=(global_pointer.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="missing|pointer"):
        adapter.extract(artifact, invalid)


@pytest.mark.parametrize(
    "content, message",
    [
        (b'\n{"accuracy":0.9}', "blank"),
        (b'[1,2,3]', "object"),
        (b'{"accuracy":0.9,"accuracy":0.8}', "duplicate"),
        (b'{"accuracy":NaN}', "finite"),
        (b'{broken}', "malformed"),
    ],
)
def test_jsonl_rejects_malformed_or_ambiguous_records(
    content: bytes, message: str
) -> None:
    artifact = _passive(content, path="runs/data.jsonl")
    with pytest.raises(AdapterParseError, match=message):
        JsonLinesAdapter().probe(artifact)


def test_jsonl_record_and_record_size_limits_are_whole_artifact_failures() -> None:
    too_many = b"\n".join(b'{"accuracy":0.9}' for _ in range(MAX_RECORDS + 1))
    with pytest.raises(AdapterLimitError, match="records"):
        JsonLinesAdapter().probe(_passive(too_many, path="runs/many.jsonl"))

    too_large = b'{"text":"' + (b"x" * MAX_LOGICAL_RECORD_BYTES) + b'"}'
    with pytest.raises(AdapterLimitError, match="logical record|1 MiB"):
        JsonLinesAdapter().probe(_passive(too_large, path="runs/large.jsonl"))


def test_json_and_jsonl_reject_wrong_kind_and_do_not_infer_roles_from_paths() -> None:
    role_named = _passive(
        b'{"accuracy":0.9}',
        path="baseline/candidate_results.json",
    )
    observation = JsonAdapter().extract(
        role_named,
        JsonAdapter().probe(role_named),  # type: ignore[arg-type]
    ).observations[0]
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED

    unsupported = _passive(
        b'{"accuracy":0.9}',
        path="data/results.json",
        kind=ArtifactKind.DATASET,
    )
    assert JsonAdapter().probe(unsupported) is None
    assert JsonLinesAdapter().probe(unsupported) is None


def test_structured_extract_rejects_wrong_adapter_path_selector_and_sha() -> None:
    artifact = _fixture("nested_metric.json")
    adapter = JsonAdapter()
    match = adapter.probe(artifact)
    assert match is not None

    wrong_id = AdapterMatch(
        adapter_id="claimci-jsonl-v1",
        path=match.path,
        confidence=match.confidence,
        mappings=match.mappings,
        match_evidence=match.match_evidence,
    )
    with pytest.raises(AdapterSelectorError, match="adapter"):
        adapter.extract(artifact, wrong_id)

    column = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field="metric_value",
        selector_kind=SelectorKind.COLUMN,
        selector="accuracy",
        inferred=False,
    )
    wrong_selector = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(column,),
        match_evidence=(column.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="selector"):
        adapter.extract(artifact, wrong_selector)

    object.__setattr__(artifact, "content", artifact.content + b" ")
    with pytest.raises(Exception, match="integrity|sha256|size"):
        adapter.extract(artifact, match)


def test_json_probe_rejects_duplicate_keys_and_nonfinite_values() -> None:
    for content in (
        b'{"outer":{"x":1,"x":2}}',
        b'{"accuracy":Infinity}',
    ):
        with pytest.raises(AdapterParseError):
            JsonAdapter().probe(_passive(content, path="results/data.json"))


def test_identical_json_results_at_distinct_paths_have_distinct_v2_ids() -> None:
    content = b'{"accuracy":0.9}'
    adapter = JsonAdapter()
    baseline = _passive(content, path="baseline-results.json")
    candidate = _passive(content, path="candidate-results.json")
    baseline_match = adapter.probe(baseline)
    candidate_match = adapter.probe(candidate)
    reread = _passive(content, path="baseline-results.json")
    reread_match = adapter.probe(reread)

    assert baseline_match is not None
    assert candidate_match is not None
    assert reread_match is not None
    baseline_evidence = adapter.extract(baseline, baseline_match)
    candidate_evidence = adapter.extract(candidate, candidate_match)
    reread_evidence = adapter.extract(reread, reread_match)

    assert baseline_evidence.evidence_id.startswith("evidence-v2-")
    assert len(baseline_evidence.evidence_id) == 76
    assert baseline_evidence.evidence_id != candidate_evidence.evidence_id
    assert baseline_evidence.evidence_id == reread_evidence.evidence_id
