"""Real Discovery/Adapter/Planner/Audit integration for ClaimCI v0.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from claimci.analysis import (
    AdapterMatch,
    AnalysisState,
    ArtifactKind,
    Confidence,
    DatasetReference,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MaterializationLimits,
    MappingTrust,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    PlanningState,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    RepoMapping,
    RuntimeExecutionContext,
    SelectorKind,
    execute_ephemeral_audit,
    plan_ephemeral_audit,
    planning_request_from_discovery,
    run_unified_analysis,
)
from claimci.analysis.adapters import get_adapter
from claimci.analysis.discovery import discover_repository
from claimci.models import Verdict
from claimci.review import ProviderUsage, ReviewConfig
from claimci.review.provider import ProviderResponse, StructuredRequest


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI-Demo")
HEAD_SHA = GitCommitSha("d" * 40)


class _SynthesisProvider:
    def __init__(self) -> None:
        self.calls: list[StructuredRequest] = []

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        raise AssertionError("unified integration must not extract claims")

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            json.dumps(
                {
                    "summary": "The deterministic findings remain authoritative.",
                    "interpretations": ["The supplied comparison is not fair."],
                    "missing_evidence": [],
                    "confidence": 0.8,
                }
            ),
            "fake",
            "fake-model",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def _write_repository(root: Path) -> None:
    files = {
        "README.md": b"Accuracy improved by at least 0.05.\n",
        "evidence/accuracy_baseline_results.csv": (
            b"run_id,seed,accuracy\nbase-1,1,0.5\nbase-2,2,0.6\nbase-3,3,0.7\n"
        ),
        "evidence/accuracy_candidate_results.csv": (
            b"run_id,seed,accuracy\ncandidate-1,11,0.9\n"
        ),
        "evidence/accuracy_baseline_config.yaml": (
            b"training_steps: 100\n"
            b"batch_size: 8\n"
            b"epochs: 2\n"
            b"learning_rate: 0.001\n"
            b"model:\n  name: demo\n  version: v1\n"
            b"dataset:\n  identifier: train\n  version: v1\n"
            b"evaluation:\n  dataset_identifier: eval\n  dataset_version: v1\n  split: test\n"
        ),
        "evidence/accuracy_candidate_config.yaml": (
            b"training_steps: 300\n"
            b"batch_size: 8\n"
            b"epochs: 2\n"
            b"learning_rate: 0.001\n"
            b"model:\n  name: demo\n  version: v1\n"
            b"dataset:\n  identifier: train\n  version: v1\n"
            b"evaluation:\n  dataset_identifier: eval\n  dataset_version: v1\n  split: test\n"
        ),
        "evidence/accuracy_baseline_train_data.jsonl": b'{"id":"baseline-train"}\n',
        "evidence/accuracy_baseline_eval_data.jsonl": b'{"id":"baseline-eval"}\n',
        "evidence/accuracy_candidate_train_data.jsonl": (
            b'{"id":"shared"}\n{"id":"candidate-train"}\n'
        ),
        "evidence/accuracy_candidate_eval_data.jsonl": b'{"id":"shared"}\n',
    }
    for relative, content in files.items():
        destination = root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _write_second_claim_repository(root: Path) -> None:
    readme = root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "F1 improved by at least 0.03.\n",
        encoding="utf-8",
    )
    files = {
        "evidence/f1_baseline_results.csv": (
            b"run_id,seed,f1\nbase-1,1,0.4\nbase-2,2,0.5\nbase-3,3,0.6\n"
        ),
        "evidence/f1_candidate_results.csv": (
            b"run_id,seed,f1\ncandidate-1,11,0.7\n"
        ),
        "evidence/f1_baseline_config.yaml": (
            b"training_steps: 100\nbatch_size: 8\nepochs: 2\n"
            b"learning_rate: 0.001\nmodel:\n  name: demo\n  version: v1\n"
            b"dataset:\n  identifier: f1-train\n  version: v1\n"
            b"evaluation:\n  dataset_identifier: f1-eval\n"
            b"  dataset_version: v1\n  split: test\n"
        ),
        "evidence/f1_candidate_config.yaml": (
            b"training_steps: 100\nbatch_size: 8\nepochs: 2\n"
            b"learning_rate: 0.001\nmodel:\n  name: demo\n  version: v1\n"
            b"dataset:\n  identifier: f1-train\n  version: v1\n"
            b"evaluation:\n  dataset_identifier: f1-eval\n"
            b"  dataset_version: v1\n  split: test\n"
        ),
        "evidence/f1_baseline_train_data.jsonl": b'{"id":"f1-base-train"}\n',
        "evidence/f1_baseline_eval_data.jsonl": b'{"id":"f1-base-eval"}\n',
        "evidence/f1_candidate_train_data.jsonl": b'{"id":"f1-candidate-train"}\n',
        "evidence/f1_candidate_eval_data.jsonl": b'{"id":"f1-candidate-eval"}\n',
    }
    for relative, content in files.items():
        destination = root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _write_manifest(root: Path) -> None:
    (root / "research.yaml").write_text(
        """claim:
  metric: accuracy
  minimum_improvement: 0.05
baseline:
  config: evidence/accuracy_baseline_config.yaml
  results: evidence/accuracy_baseline_results.csv
  train_dataset: evidence/accuracy_baseline_train_data.jsonl
  eval_dataset: evidence/accuracy_baseline_eval_data.jsonl
candidate:
  config: evidence/accuracy_candidate_config.yaml
  results: evidence/accuracy_candidate_results.csv
  train_dataset: evidence/accuracy_candidate_train_data.jsonl
  eval_dataset: evidence/accuracy_candidate_eval_data.jsonl
""",
        encoding="utf-8",
    )


def _dataset_evidence(artifact, content: bytes) -> NormalizedEvidence:
    path = str(artifact.path)
    split = "train" if "train" in path else "eval"
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        "bounded passive JSONL dataset identity test stub",
        artifact.path,
        "test-passive-dataset-v1",
    )
    mapping = FieldMapping(
        "dataset.path",
        EvidenceSelector(
            SelectorKind.DOTTED_PATH,
            "dataset.path",
            provenance,
        ),
        provenance,
    )
    PassiveArtifact(artifact, content)
    return NormalizedEvidence(
        "evidence-dataset-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16],
        artifact,
        AdapterMatch(
            "test-passive-dataset-v1",
            artifact.path,
            Confidence(0.99),
            (mapping,),
            (provenance,),
        ),
        (
            NormalizedObservation(
                provenance=provenance,
                experiment_role=ExperimentRole.UNSPECIFIED,
                dataset_references=(
                    DatasetReference(artifact.path, split, provenance),
                ),
            ),
        ),
    )


def _extract_evidence(root: Path, discovery) -> tuple[NormalizedEvidence, ...]:
    extracted: list[NormalizedEvidence] = []
    for artifact in discovery.artifacts:
        content = (root / Path(str(artifact.path))).read_bytes()
        passive = PassiveArtifact(artifact, content)
        if artifact.kind is ArtifactKind.RESULTS:
            adapter = get_adapter("claimci-csv-v1")
        elif artifact.kind is ArtifactKind.CONFIG:
            adapter = get_adapter("claimci-yaml-config-v1")
        elif artifact.kind is ArtifactKind.DATASET:
            extracted.append(_dataset_evidence(artifact, content))
            continue
        else:
            continue
        match = adapter.probe(passive)
        assert match is not None
        evidence = adapter.extract(passive, match)
        assert {
            item.experiment_role for item in evidence.observations
        } == {ExperimentRole.UNSPECIFIED}
        extracted.append(evidence)
    return tuple(extracted)


def _provider_mapping_payload(
    evidence: tuple[NormalizedEvidence, ...],
) -> dict[str, object]:
    bindings: list[dict[str, object]] = []
    for item in evidence:
        path = str(item.artifact.path)
        role = "baseline" if "baseline" in path else "candidate"
        bindings.append(
            {
                "path": path,
                "kind": item.artifact.kind.value,
                "role": role,
                "adapter_id": item.adapter_match.adapter_id,
                "mappings": [
                    {
                        "target_field": mapping.target_field,
                        "selector": {
                            "kind": mapping.selector.kind.value,
                            "expression": mapping.selector.expression,
                        },
                    }
                    for mapping in item.adapter_match.mappings
                ],
            }
        )
    return {"mappings": [{"confidence": 1.0, "bindings": bindings}]}


def test_real_discovery_adapters_planner_materializer_audit_and_unified_result(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    scratch = tmp_path / "scratch"
    base.mkdir()
    head.mkdir()
    scratch.mkdir()
    (base / "README.md").write_text("No result declared.\n", encoding="utf-8")
    _write_repository(head)

    discovery = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
        base_root=base,
    )
    claim_id = discovery.claims[0].reference.claim_id
    evidence = _extract_evidence(head, discovery)

    request = planning_request_from_discovery(
        discovery,
        claim_id=claim_id,
        normalized_evidence=evidence,
    )
    planning = plan_ephemeral_audit(request)

    assert planning.state is PlanningState.READY
    assert planning.plan is not None
    assert planning.plan.head_sha == HEAD_SHA
    assert {
        item.observations[0].experiment_role
        for item in (*planning.plan.baseline_evidence, *planning.plan.candidate_evidence)
    } == {ExperimentRole.UNSPECIFIED}
    assert {
        binding.role for binding in planning.plan.selected_mapping.bindings
    } == {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}

    runtime = RuntimeExecutionContext(
        repository=REPOSITORY,
        checkout_root=head,
        head_sha=HEAD_SHA,
        scratch_root=scratch,
        limits=MaterializationLimits(
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
        ),
    )
    audit_result = execute_ephemeral_audit(planning.plan, runtime)
    assert audit_result.verdict is Verdict.NOT_SUPPORTED
    assert list(scratch.iterdir()) == []

    provider = _SynthesisProvider()
    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.research_interpretation is not None
    assert len(provider.calls) == 1
    assert list(scratch.iterdir()) == []


def test_real_multi_claim_discovery_scopes_artifacts_evidence_and_mapping(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    _write_repository(head)
    _write_second_claim_repository(head)

    discovery = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
    )
    claims_by_metric = {claim.metric: claim for claim in discovery.claims}
    assert set(claims_by_metric) == {"accuracy", "f1"}
    accuracy_id = claims_by_metric["accuracy"].reference.claim_id
    f1_id = claims_by_metric["f1"].reference.claim_id
    for artifact in discovery.artifacts:
        path = str(artifact.path)
        if "accuracy_" in path:
            assert artifact.relevant_claim_ids == (accuracy_id,)
        elif "f1_" in path:
            assert artifact.relevant_claim_ids == (f1_id,)

    evidence = _extract_evidence(head, discovery)
    request = planning_request_from_discovery(
        discovery,
        claim_id=accuracy_id,
        normalized_evidence=evidence,
    )
    planning = plan_ephemeral_audit(request)

    assert all(
        "accuracy_" in str(item.path) or item.kind is ArtifactKind.MANIFEST
        for item in request.artifacts
    )
    assert all("accuracy_" in str(item.artifact.path) for item in request.normalized_evidence)
    assert all(
        "accuracy_" in str(binding.path)
        for mapping in request.mapping_candidates
        for binding in mapping.bindings
    )
    assert request.upstream_mapping_question is None
    assert planning.state is PlanningState.READY
    assert planning.plan is not None
    assert all(
        "accuracy_" in str(binding.path)
        for binding in planning.plan.selected_mapping.bindings
    )


def test_real_manifest_hint_does_not_override_equivalent_valid_inference(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    _write_repository(head)
    _write_manifest(head)

    discovery = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
    )
    evidence = _extract_evidence(head, discovery)
    claim_id = discovery.claims[0].reference.claim_id
    request = planning_request_from_discovery(
        discovery,
        claim_id=claim_id,
        normalized_evidence=evidence,
    )
    planning = plan_ephemeral_audit(request)

    assert {item.trust for item in request.mapping_candidates} == {
        MappingTrust.INFERRED,
        MappingTrust.MANIFEST_HINT,
    }
    assert planning.state is PlanningState.READY
    assert planning.plan is not None
    assert type(planning.plan.selected_mapping) is not RepoMapping
    assert planning.plan.selected_mapping.trust is MappingTrust.INFERRED
    assert (
        planning.plan.selected_mapping.provenance.kind
        is ProvenanceKind.DETERMINISTIC_DISCOVERY
    )


def test_real_provider_proposal_requires_approval_then_precedes_inference(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    _write_repository(head)
    initial = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
    )
    initial_evidence = _extract_evidence(head, initial)
    payload = _provider_mapping_payload(initial_evidence)

    proposed = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
        provider_mapping_payload=payload,
    )
    proposed_evidence = _extract_evidence(head, proposed)
    claim_id = proposed.claims[0].reference.claim_id
    proposed_request = planning_request_from_discovery(
        proposed,
        claim_id=claim_id,
        normalized_evidence=proposed_evidence,
    )
    unapproved_planning = plan_ephemeral_audit(proposed_request)
    assert unapproved_planning.state is PlanningState.READY
    assert unapproved_planning.plan is not None
    assert (
        unapproved_planning.plan.mapping_provenance[0].kind
        is ProvenanceKind.DETERMINISTIC_DISCOVERY
    )
    provider_candidate = next(
        candidate
        for candidate in proposed.mapping_candidates
        if candidate.provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        provider_candidate,
        approved_by="repository-owner",
    )

    rediscovered = discover_repository(
        head,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=17,
        approved_mapping=approved,
        provider_mapping_payload=payload,
    )
    rediscovered_evidence = _extract_evidence(head, rediscovered)
    approved_request = planning_request_from_discovery(
        rediscovered,
        claim_id=rediscovered.claims[0].reference.claim_id,
        normalized_evidence=rediscovered_evidence,
    )
    planning = plan_ephemeral_audit(approved_request)

    assert planning.state is PlanningState.READY
    assert planning.plan is not None
    assert type(planning.plan.selected_mapping) is RepoMapping
    assert planning.plan.selected_mapping.approved_by == "repository-owner"
