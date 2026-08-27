"""Native ClaimCI compatibility adapters remain evidence-only."""

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
    RepoMapping,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.adapters.core import (
    AdapterParseError,
    AdapterSelectorError,
    _mapping,
)
from claimci.analysis.adapters.native import (
    NativeConfigAdapter,
    NativeManifestAdapter,
    NativeResultsAdapter,
)


ROOT = Path(__file__).parents[1]


def _passive(
    content: bytes,
    *,
    path: str,
    kind: ArtifactKind,
) -> PassiveArtifact:
    return passive_artifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=kind,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.99),
            discovery_reason="native ClaimCI fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "native fixture bytes",
                RepositoryPath(path),
                "fixture",
            ),
        ),
        content,
    )


def test_native_manifest_preserves_explicit_roles_as_manifest_hints_only() -> None:
    source = ROOT / "examples" / "day2_demo" / "research.yaml"
    artifact = _passive(
        source.read_bytes(),
        path="examples/day2_demo/research.yaml",
        kind=ArtifactKind.MANIFEST,
    )
    adapter = NativeManifestAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    assert match.adapter_id == "claimci-native-manifest-v1"
    assert all(
        item.provenance.kind is ProvenanceKind.MANIFEST_HINT
        for item in match.mappings
    )
    evidence = adapter.extract(artifact, match)
    assert len(evidence.observations) == 3
    claim, baseline, candidate = evidence.observations
    assert claim.experiment_role is ExperimentRole.UNSPECIFIED
    assert baseline.experiment_role is ExperimentRole.BASELINE
    assert candidate.experiment_role is ExperimentRole.CANDIDATE
    assert all(
        item.provenance.kind is ProvenanceKind.MANIFEST_HINT
        for item in evidence.observations
    )

    assert tuple((item.key, item.value) for item in claim.config_values) == (
        ("claim.metric", "accuracy"),
        ("claim.minimum_improvement", 0.05),
    )
    assert tuple((item.key, item.value) for item in baseline.config_values) == (
        ("config", "examples/day2_demo/baseline-config.yaml"),
        ("results", "examples/day2_demo/baseline-results.json"),
    )
    assert tuple(item.path for item in baseline.dataset_references) == (
        RepositoryPath("examples/day2_demo/baseline-train.jsonl"),
        RepositoryPath("examples/day2_demo/baseline-eval.jsonl"),
    )
    assert tuple(item.path for item in candidate.dataset_references) == (
        RepositoryPath("examples/day2_demo/candidate-train.jsonl"),
        RepositoryPath("examples/day2_demo/candidate-eval.jsonl"),
    )
    assert all(
        item.provenance.kind is ProvenanceKind.MANIFEST_HINT
        for observation in evidence.observations
        for item in (*observation.config_values, *observation.dataset_references)
    )
    assert not isinstance(evidence, RepoMapping)
    for forbidden in (
        "authoritative_verdict",
        "deterministic",
        "approved_by",
        "audit_plan",
    ):
        assert not hasattr(evidence, forbidden)


def test_native_manifest_resolves_parent_segments_but_rejects_repository_escape() -> None:
    nested = b"""claim:\n  metric: accuracy\n  minimum_improvement: 0.1\nbaseline:\n  config: ../../configs/base.yaml\n  results: ../../results/base.json\n  train_dataset: ../../data/base-train.jsonl\n  eval_dataset: ../../data/base-eval.jsonl\ncandidate:\n  config: ../../configs/candidate.yaml\n  results: ../../results/candidate.json\n  train_dataset: ../../data/candidate-train.jsonl\n  eval_dataset: ../../data/candidate-eval.jsonl\n"""
    artifact = _passive(
        nested,
        path="experiments/manifests/research.yaml",
        kind=ArtifactKind.MANIFEST,
    )
    match = NativeManifestAdapter().probe(artifact)
    assert match is not None
    observations = NativeManifestAdapter().extract(artifact, match).observations
    assert observations[1].config_values[0].value == "configs/base.yaml"

    escaping = nested.replace(b"../../configs/base.yaml", b"../../../escape.yaml")
    with pytest.raises(AdapterParseError, match="escape|repository|path"):
        NativeManifestAdapter().probe(
            _passive(
                escaping,
                path="experiments/manifests/research.yaml",
                kind=ArtifactKind.MANIFEST,
            )
        )


@pytest.mark.parametrize(
    "declared",
    ["C:/outside.yaml", "/outside.yaml", "..\\outside.yaml", "data//file.yaml"],
)
def test_native_manifest_rejects_nonportable_declared_paths(declared: str) -> None:
    content = f"""claim:\n  metric: accuracy\n  minimum_improvement: 0.1\nbaseline:\n  config: {declared!r}\n  results: base.json\n  train_dataset: train.jsonl\n  eval_dataset: eval.jsonl\ncandidate:\n  config: candidate.yaml\n  results: candidate.json\n  train_dataset: candidate-train.jsonl\n  eval_dataset: candidate-eval.jsonl\n""".encode()
    with pytest.raises(AdapterParseError, match="portable|path|repository"):
        NativeManifestAdapter().probe(
            _passive(content, path="research.yaml", kind=ArtifactKind.MANIFEST)
        )


def test_native_results_emit_repeated_runs_and_summary_without_role_or_verdict() -> None:
    source = ROOT / "examples" / "shared" / "base" / "results.json"
    artifact = _passive(
        source.read_bytes(),
        path="examples/shared/base/results.json",
        kind=ArtifactKind.RESULTS,
    )
    adapter = NativeResultsAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    targets = {
        item.target_field: item.selector.expression for item in match.mappings
    }
    assert targets == {"metric_value": "/accuracy", "seed": "/seed"}
    evidence = adapter.extract(artifact, match)
    metric_observations = [item for item in evidence.observations if item.metric_name]
    assert [item.metric_value for item in metric_observations] == [0.58, 0.60, 0.62]
    assert [item.seed for item in metric_observations] == [1, 2, 3]
    assert all(
        item.experiment_role is ExperimentRole.UNSPECIFIED
        for item in evidence.observations
    )
    assert all(
        f"record[{index}]/accuracy" in item.provenance.detail
        for index, item in enumerate(metric_observations)
    )
    summary = [item for item in evidence.observations if item.config_values]
    assert len(summary) == 1
    assert tuple((item.key, item.value) for item in summary[0].config_values) == (
        ("summary.accuracy", 0.60),
    )


def test_native_results_with_multiple_metrics_require_external_mapping() -> None:
    content = b'{"runs":[{"seed":1,"accuracy":0.9,"loss":0.2},{"seed":2,"accuracy":0.8,"loss":0.3}]}'
    artifact = _passive(content, path="results.json", kind=ArtifactKind.RESULTS)
    adapter = NativeResultsAdapter()
    ambiguous = adapter.probe(artifact)
    assert ambiguous is not None
    assert "metric_value" not in {
        item.target_field for item in ambiguous.mappings
    }
    with pytest.raises(AdapterSelectorError, match="metric_value|ambiguous|mapping"):
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
    assert [
        item.metric_value
        for item in adapter.extract(artifact, explicit).observations
        if item.metric_name
    ] == [0.9, 0.8]


def test_native_config_reports_raw_compute_inputs_without_comparison() -> None:
    source = ROOT / "examples" / "shared" / "base" / "config.yaml"
    artifact = _passive(
        source.read_bytes(),
        path="examples/shared/base/config.yaml",
        kind=ArtifactKind.CONFIG,
    )
    adapter = NativeConfigAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    observation = adapter.extract(artifact, match).observations[0]
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED
    assert tuple(item.name for item in observation.compute_evidence) == (
        "batch_size",
        "epochs",
        "training_steps",
    )
    assert tuple(item.value for item in observation.compute_evidence) == (8.0, 4.0, 100.0)
    assert not any("ratio" in item.name or "verdict" in item.name for item in observation.compute_evidence)
    assert ("model", "tiny-baseline") in tuple(
        (item.key, item.value) for item in observation.config_values
    )


def test_native_adapters_return_none_for_non_native_shapes() -> None:
    generic_results = _passive(
        b'{"accuracy":0.9}', path="results.json", kind=ArtifactKind.RESULTS
    )
    generic_config = _passive(
        b"model: tiny\n", path="config.yaml", kind=ArtifactKind.CONFIG
    )
    wrong_manifest_name = _passive(
        b"claim: {}\n", path="manifest.yaml", kind=ArtifactKind.MANIFEST
    )
    assert NativeResultsAdapter().probe(generic_results) is None
    assert NativeConfigAdapter().probe(generic_config) is None
    assert NativeManifestAdapter().probe(wrong_manifest_name) is None


def test_identical_native_configs_at_distinct_paths_have_distinct_v2_ids() -> None:
    content = b"model: identical\nbatch_size: 8\nepochs: 4\ntraining_steps: 100\n"
    adapter = NativeConfigAdapter()
    baseline = _passive(content, path="baseline-config.yaml", kind=ArtifactKind.CONFIG)
    candidate = _passive(content, path="candidate-config.yaml", kind=ArtifactKind.CONFIG)
    baseline_match = adapter.probe(baseline)
    candidate_match = adapter.probe(candidate)
    reread = _passive(content, path="baseline-config.yaml", kind=ArtifactKind.CONFIG)
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
