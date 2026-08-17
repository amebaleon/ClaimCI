"""Behavioral contracts for the ClaimCI v0.3 shared analysis model."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from claimci.analysis import (
    Adapter,
    AdapterMatch,
    AdvisoryResearchInterpretation,
    AnalysisAuthority,
    AnalysisState,
    ArtifactCandidate,
    ArtifactBinding,
    ArtifactKind,
    ClaimReference,
    ComputeEvidence,
    Confidence,
    ConfigValue,
    DatasetReference,
    DeterministicAuditOutcome,
    EvidenceSelector,
    EphemeralAuditPlan,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingChoice,
    MappingQuestion,
    MappingTrust,
    MissingEvidence,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    UnifiedAnalysisResult,
    to_jsonable,
)
from claimci.models import AuditResult, Finding, Impact, Severity, Verdict


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
        "data/line\nbreak.json",
        "data/foo:bar.json",
        "data/CON",
        "data/aux.txt",
        "data/CONIN$",
        "data/conout$.log",
        "data/COM¹.txt",
        "data/LPT³",
        "data/trailing.",
        "data/trailing /result.json",
        "data/.. /result.json",
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
        (SelectorKind.JSON_POINTER, "/metrics/~2accuracy"),
        (SelectorKind.JSON_POINTER, "/metrics/~"),
        (SelectorKind.JSON_POINTER, "/runs\naccuracy"),
        (SelectorKind.DOTTED_PATH, "runs..accuracy"),
        (SelectorKind.DOTTED_PATH, ".runs"),
        (SelectorKind.COLUMN, ""),
        (SelectorKind.COLUMN, "accuracy\nvalue"),
        (SelectorKind.COLUMN, "SELECT * FROM results"),
        (SelectorKind.COLUMN, "accuracy; command"),
        (SelectorKind.COLUMN, "__import__('os')"),
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
        {"relevant_claim_ids": ("claim\n1",)},
        {"relevant_claim_ids": ("claim\x7f1",)},
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


def _adapter_match(path: str = "results/candidate.json") -> AdapterMatch:
    return AdapterMatch(
        adapter_id="json-runs-v1",
        path=RepositoryPath(path),
        confidence=Confidence(0.95),
        mappings=(_mapping(),),
        match_evidence=(_provenance(),),
    )


def _observation(
    role: ExperimentRole = ExperimentRole.CANDIDATE,
) -> NormalizedObservation:
    provenance = _provenance(ProvenanceKind.ADAPTER_EXTRACTION)
    return NormalizedObservation(
        provenance=provenance,
        metric_name="accuracy",
        metric_value=0.9,
        run_id="candidate-run-1",
        seed=7,
        experiment_role=role,
        config_values=(ConfigValue("batch_size", 32, provenance),),
        dataset_references=(
            DatasetReference(
                RepositoryPath("data/candidate-eval.jsonl"),
                "eval",
                provenance,
            ),
        ),
        compute_evidence=(
            ComputeEvidence("training_steps", 1_000, "steps", provenance),
        ),
    )


def _normalized_evidence(
    *,
    role: ExperimentRole = ExperimentRole.CANDIDATE,
    evidence_id: str = "evidence-candidate-results",
) -> NormalizedEvidence:
    candidate = _candidate()
    return NormalizedEvidence(
        evidence_id=evidence_id,
        artifact=candidate,
        adapter_match=_adapter_match(),
        observations=(_observation(role),),
    )


def _binding(
    path: str,
    kind: ArtifactKind,
    role: ExperimentRole,
    *,
    provenance: FieldProvenance | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        path=RepositoryPath(path),
        kind=kind,
        role=role,
        adapter_id="json-runs-v1" if kind is ArtifactKind.RESULTS else None,
        mappings=(_mapping(),) if kind is ArtifactKind.RESULTS else (),
        provenance=provenance or _provenance(),
    )


def _mapping_candidate(
    *,
    trust: MappingTrust = MappingTrust.INFERRED,
    provenance: FieldProvenance | None = None,
    bindings: tuple[ArtifactBinding, ...] | None = None,
) -> MappingCandidate:
    return MappingCandidate(
        mapping_id="mapping-1",
        bindings=bindings
        or (
            _binding("base/results.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
            _binding(
                "candidate/results.json",
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
            ),
        ),
        confidence=Confidence(0.8),
        trust=trust,
        provenance=provenance or _provenance(),
    )


def test_normalized_observation_represents_optional_typed_evidence() -> None:
    observation = _observation()

    assert observation.metric_name == "accuracy"
    assert observation.metric_value == 0.9
    assert observation.run_id == "candidate-run-1"
    assert observation.seed == 7
    assert observation.experiment_role is ExperimentRole.CANDIDATE
    assert observation.config_values[0].value == 32
    assert observation.dataset_references[0].path == "data/candidate-eval.jsonl"
    assert observation.compute_evidence[0].value == 1_000
    with pytest.raises(FrozenInstanceError):
        observation.metric_value = 1.0  # type: ignore[misc]


def test_normalized_observation_does_not_force_every_evidence_category() -> None:
    observation = NormalizedObservation(
        provenance=_provenance(),
        config_values=(ConfigValue("model", "resnet", _provenance()),),
    )

    assert observation.metric_name is None
    assert observation.dataset_references == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"metric_name": "accuracy", "metric_value": None},
        {"metric_name": None, "metric_value": 0.9},
        {"metric_name": "accuracy", "metric_value": math.nan},
        {"seed": True},
        {"experiment_role": "candidate"},
        {"config_values": []},
        {"dataset_references": []},
        {"compute_evidence": []},
        {"provenance": "adapter"},
    ],
)
def test_normalized_observation_rejects_inconsistent_or_mutable_fields(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "provenance": _provenance(),
        "metric_name": "accuracy",
        "metric_value": 0.9,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        NormalizedObservation(**values)  # type: ignore[arg-type]


def test_normalized_observation_rejects_empty_content() -> None:
    with pytest.raises((TypeError, ValueError), match="evidence|observation"):
        NormalizedObservation(provenance=_provenance())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConfigValue("metric", math.inf, _provenance()),
        lambda: ConfigValue("", 1, _provenance()),
        lambda: DatasetReference("../outside.jsonl", "eval", _provenance()),
        lambda: DatasetReference("data/eval.jsonl", "", _provenance()),
        lambda: ComputeEvidence("steps", True, "steps", _provenance()),
        lambda: ComputeEvidence("steps", math.nan, "steps", _provenance()),
    ],
)
def test_normalized_subvalues_reject_malformed_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_normalized_evidence_ties_observations_to_artifact_and_adapter() -> None:
    evidence = _normalized_evidence()

    assert evidence.artifact.path == evidence.adapter_match.path
    assert evidence.observations[0].experiment_role is ExperimentRole.CANDIDATE


def test_normalized_evidence_rejects_mismatched_path_or_empty_observations() -> None:
    with pytest.raises((TypeError, ValueError), match="path"):
        NormalizedEvidence(
            evidence_id="evidence-1",
            artifact=_candidate(),
            adapter_match=_adapter_match("results/other.json"),
            observations=(_observation(),),
        )
    with pytest.raises((TypeError, ValueError), match="observation"):
        NormalizedEvidence(
            evidence_id="evidence-1",
            artifact=_candidate(),
            adapter_match=_adapter_match(),
            observations=(),
        )


def test_trivial_adapter_operates_only_on_passive_bytes() -> None:
    class FakeAdapter:
        def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
            assert isinstance(artifact.content, bytes)
            return _adapter_match()

        def extract(
            self,
            artifact: PassiveArtifact,
            match: AdapterMatch,
        ) -> NormalizedEvidence:
            return NormalizedEvidence(
                evidence_id="fake-evidence",
                artifact=artifact.candidate,
                adapter_match=match,
                observations=(_observation(),),
            )

    fake: Adapter = FakeAdapter()
    passive = PassiveArtifact(_candidate(), RAW_RESULTS)
    match = fake.probe(passive)
    assert match is not None
    assert fake.extract(passive, match).evidence_id == "fake-evidence"


def test_mapping_candidate_rejects_conflicting_experiment_roles_for_same_path() -> None:
    bindings = (
        _binding("results/shared.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _binding("results/shared.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )

    with pytest.raises((TypeError, ValueError), match="conflict|baseline|candidate"):
        _mapping_candidate(bindings=bindings)


def test_mapping_candidate_rejects_approved_trust_and_inconsistent_manifest_hint() -> None:
    with pytest.raises((TypeError, ValueError)):
        _mapping_candidate(trust=MappingTrust.USER_APPROVED)
    with pytest.raises((TypeError, ValueError)):
        _mapping_candidate(trust=MappingTrust.MANIFEST_HINT)

    manifest = _mapping_candidate(
        trust=MappingTrust.MANIFEST_HINT,
        provenance=FieldProvenance(
            ProvenanceKind.MANIFEST_HINT,
            "explicit research.yaml mapping",
            RepositoryPath("research.yaml"),
            "research.yaml",
        ),
    )
    assert manifest.trust is MappingTrust.MANIFEST_HINT


@pytest.mark.parametrize("trust", ["inferred", "manifest_hint"])
def test_mapping_candidate_rejects_raw_string_trust_values(trust: str) -> None:
    with pytest.raises((TypeError, ValueError), match="trust|MappingTrust"):
        _mapping_candidate(trust=trust)  # type: ignore[arg-type]


def test_mapping_candidate_is_deeply_immutable() -> None:
    mapping = _mapping_candidate()

    with pytest.raises(FrozenInstanceError):
        mapping.bindings = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        _mapping_candidate(bindings=list(mapping.bindings))  # type: ignore[arg-type]


def _choice(choice_id: str, path: str) -> MappingChoice:
    return MappingChoice(
        choice_id=choice_id,
        label=f"Use {path}",
        bindings=(
            _binding(path, ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
        ),
    )


def test_mapping_question_has_bounded_unique_candidate_choices() -> None:
    question = MappingQuestion(
        question_id="candidate-results",
        prompt="Which file contains the candidate results?",
        choices=(
            _choice("results-a", "results/a.json"),
            _choice("results-b", "results/b.json"),
        ),
        relevant_claim_id="claim-1",
    )

    assert len(question.choices) == 2
    assert question.choices[0].bindings[0].role is ExperimentRole.CANDIDATE


@pytest.mark.parametrize("count", [0, 1, 9])
def test_mapping_question_rejects_choice_counts_outside_two_through_eight(
    count: int,
) -> None:
    choices = tuple(
        _choice(f"choice-{index}", f"results/{index}.json") for index in range(count)
    )
    with pytest.raises((TypeError, ValueError), match="two|eight|choice"):
        MappingQuestion("question", "Which result?", choices)


def test_mapping_question_rejects_duplicate_choice_ids_and_mutable_choices() -> None:
    duplicate = (
        _choice("same", "results/a.json"),
        _choice("same", "results/b.json"),
    )
    with pytest.raises((TypeError, ValueError), match="unique|duplicate"):
        MappingQuestion("question", "Which result?", duplicate)
    with pytest.raises(TypeError):
        MappingQuestion("question", "Which result?", list(duplicate))  # type: ignore[arg-type]


def test_repo_mapping_requires_explicit_approval_factory() -> None:
    repository = RepositoryIdentity(owner="amebaleon", name="ClaimCI-Demo")
    candidate = _mapping_candidate()

    with pytest.raises(TypeError):
        RepoMapping()
    with pytest.raises(TypeError):
        RepoMapping(  # type: ignore[call-arg]
            repository=repository,
            bindings=candidate.bindings,
        )

    approved = RepoMapping.approve(
        repository=repository,
        candidate=candidate,
        approved_by="owner:amebaleon",
    )
    assert approved.trust is MappingTrust.USER_APPROVED
    assert approved.source_trust is MappingTrust.INFERRED
    assert approved.bindings == candidate.bindings
    assert approved.approval_provenance.kind is ProvenanceKind.USER_APPROVED


def test_repo_mapping_approval_rejects_subclassed_trust_inputs() -> None:
    candidate = _mapping_candidate()

    class MappingCandidateSubclass(MappingCandidate):
        pass

    class RepositoryIdentitySubclass(RepositoryIdentity):
        pass

    subclassed_candidate = MappingCandidateSubclass(
        candidate.mapping_id,
        candidate.bindings,
        candidate.confidence,
        candidate.trust,
        candidate.provenance,
    )
    repository = RepositoryIdentity("amebaleon", "ClaimCI-Demo")
    with pytest.raises(TypeError, match="MappingCandidate"):
        RepoMapping.approve(repository, subclassed_candidate, approved_by="owner")
    with pytest.raises(TypeError, match="RepositoryIdentity"):
        RepoMapping.approve(
            RepositoryIdentitySubclass("amebaleon", "ClaimCI-Demo"),
            candidate,
            approved_by="owner",
        )


def test_provider_mapping_can_only_become_approved_through_explicit_transition() -> None:
    proposed = _mapping_candidate(
        provenance=FieldProvenance(
            ProvenanceKind.PROVIDER_PROPOSAL,
            "provider proposed a candidate results path",
            RepositoryPath("candidate/results.json"),
            "provider-call-1",
        )
    )
    assert proposed.trust is MappingTrust.INFERRED

    approved = RepoMapping.approve(
        RepositoryIdentity("amebaleon", "ClaimCI-Demo"),
        proposed,
        approved_by="owner:amebaleon",
    )
    assert approved.trust is MappingTrust.USER_APPROVED
    assert approved.source_trust is MappingTrust.INFERRED


@pytest.mark.parametrize(
    "owner,name",
    [("", "repo"), ("owner/name", "repo"), ("owner", ""), ("owner", "bad/name")],
)
def test_repository_identity_rejects_invalid_owner_or_name(owner: str, name: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        RepositoryIdentity(owner, name)


def test_ephemeral_audit_plan_carries_required_identity_and_evidence() -> None:
    baseline = _normalized_evidence(
        role=ExperimentRole.BASELINE,
        evidence_id="evidence-baseline-results",
    )
    candidate = _normalized_evidence()
    plan = EphemeralAuditPlan(
        plan_id="plan-1",
        repository=RepositoryIdentity("amebaleon", "ClaimCI-Demo"),
        pr_number=1,
        head_sha=GitCommitSha("a" * 40),
        claim=ClaimReference(
            claim_id="claim-1",
            text="Candidate improves accuracy.",
            source_path=RepositoryPath("CLAIM.md"),
            confidence=Confidence(0.9),
            provenance=_provenance(),
        ),
        baseline_evidence=(baseline,),
        candidate_evidence=(candidate,),
        mapping_provenance=(_provenance(),),
        missing_evidence=(
            MissingEvidence(
                kind=ArtifactKind.DATASET,
                role=ExperimentRole.CANDIDATE,
                description="candidate training data is not identified",
                claim_id="claim-1",
            ),
        ),
        confidence=Confidence(0.75),
    )

    assert plan.repository.full_name == "amebaleon/ClaimCI-Demo"
    assert plan.pr_number == 1
    assert plan.ephemeral is True
    assert not hasattr(plan, "committed_path")
    assert not hasattr(plan, "write")


def test_ephemeral_plan_rejects_role_conflicts_and_invalid_pr_number() -> None:
    shared: dict[str, object] = {
        "plan_id": "plan-1",
        "repository": RepositoryIdentity("amebaleon", "ClaimCI-Demo"),
        "pr_number": 1,
        "head_sha": GitCommitSha("a" * 40),
        "claim": ClaimReference(
            "claim-1",
            "Candidate improves accuracy.",
            None,
            Confidence(0.9),
            _provenance(),
        ),
        "baseline_evidence": (
            _normalized_evidence(role=ExperimentRole.CANDIDATE),
        ),
        "candidate_evidence": (_normalized_evidence(),),
        "mapping_provenance": (_provenance(),),
        "missing_evidence": (),
        "confidence": Confidence(0.75),
    }
    with pytest.raises((TypeError, ValueError), match="baseline|role"):
        EphemeralAuditPlan(**shared)  # type: ignore[arg-type]

    shared["baseline_evidence"] = (
        _normalized_evidence(
            role=ExperimentRole.BASELINE,
            evidence_id="evidence-baseline",
        ),
    )
    shared["pr_number"] = 0
    with pytest.raises((TypeError, ValueError), match="PR|pr_number|positive"):
        EphemeralAuditPlan(**shared)  # type: ignore[arg-type]


def _audit_result(
    verdict: Verdict = Verdict.NOT_SUPPORTED,
    evidence: dict[str, object] | None = None,
) -> AuditResult:
    finding = Finding(
        rule_id="CONFIG.COMPUTE_MISMATCH",
        severity=Severity.CRITICAL,
        title="Candidate compute proxy is too large",
        explanation="The candidate proxy is three times the baseline proxy.",
        evidence=evidence or {"ratio": 3.0, "nested": [{"path": "config.yaml"}]},
        impact=Impact.INVALIDATES,
    )
    return AuditResult(
        verdict=verdict,
        findings=(finding,),
        metric="accuracy",
        minimum_improvement=0.05,
    )


def _advisory() -> AdvisoryResearchInterpretation:
    return AdvisoryResearchInterpretation(
        summary="Provider says SUPPORTED, but this remains advisory.",
        interpretations=("NOT_SUPPORTED", "SUPPORTED"),
        missing_evidence=("A second candidate run is missing.",),
        confidence=Confidence(1.0),
    )


def _mapping_question() -> MappingQuestion:
    return MappingQuestion(
        question_id="candidate-results",
        prompt="Which file contains the candidate results?",
        choices=(
            _choice("results-a", "results/a.json"),
            _choice("results-b", "results/b.json"),
        ),
        relevant_claim_id="claim-1",
    )


def test_only_actual_audit_result_can_create_deterministic_authority() -> None:
    with pytest.raises(TypeError):
        DeterministicAuditOutcome()
    with pytest.raises(TypeError):
        DeterministicAuditOutcome.from_audit_result(  # type: ignore[arg-type]
            {"verdict": "SUPPORTED"}
        )
    with pytest.raises(TypeError):
        DeterministicAuditOutcome.from_audit_result(_advisory())  # type: ignore[arg-type]

    class AuditResultSubclass(AuditResult):
        pass

    with pytest.raises(TypeError, match="actual AuditResult"):
        DeterministicAuditOutcome.from_audit_result(
            AuditResultSubclass(Verdict.SUPPORTED, ())
        )

    class FakeVerdict:
        value = "SUPPORTED"

    with pytest.raises(TypeError, match="Verdict"):
        DeterministicAuditOutcome.from_audit_result(
            AuditResult(FakeVerdict(), ())  # type: ignore[arg-type]
        )

    outcome = DeterministicAuditOutcome.from_audit_result(_audit_result())

    assert outcome.verdict is Verdict.NOT_SUPPORTED
    assert outcome.authority is AnalysisAuthority.DETERMINISTIC
    assert outcome.payload["verdict"] == "NOT_SUPPORTED"


def test_factory_only_trust_types_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="subclass|final|factory"):

        class ForgedOutcome(DeterministicAuditOutcome):
            pass

    with pytest.raises(TypeError, match="subclass|final|factory"):

        class ForgedRepoMapping(RepoMapping):
            pass

    with pytest.raises(TypeError, match="subclass|final|authority"):

        class ForgedAdvisory(AdvisoryResearchInterpretation):
            verdict = Verdict.SUPPORTED

    with pytest.raises(TypeError, match="subclass|final|authority"):

        class ForgedUnifiedResult(UnifiedAnalysisResult):
            @property
            def authoritative_verdict(self) -> Verdict:
                return Verdict.SUPPORTED


def test_deterministic_snapshot_is_deeply_immutable_and_detached() -> None:
    evidence: dict[str, object] = {
        "ratio": 3.0,
        "nested": [{"path": "config.yaml"}],
    }
    outcome = DeterministicAuditOutcome.from_audit_result(
        _audit_result(evidence=evidence)
    )
    evidence["ratio"] = 1.0
    evidence["nested"] = []

    finding = outcome.payload["findings"][0]  # type: ignore[index]
    assert finding["evidence"]["ratio"] == 3.0  # type: ignore[index]
    assert finding["evidence"]["nested"][0]["path"] == "config.yaml"  # type: ignore[index]
    with pytest.raises(TypeError):
        outcome.payload["verdict"] = "SUPPORTED"  # type: ignore[index]
    with pytest.raises(TypeError):
        finding["evidence"]["ratio"] = 1.0  # type: ignore[index]


def test_advisory_interpretation_has_fixed_nonblocking_authority_and_no_verdict() -> None:
    advisory = _advisory()

    assert advisory.authority is AnalysisAuthority.ADVISORY
    assert not hasattr(advisory, "verdict")
    assert not hasattr(advisory, "severity")
    assert not hasattr(advisory, "impact")
    with pytest.raises(TypeError):
        AdvisoryResearchInterpretation(  # type: ignore[call-arg]
            summary="advisory",
            interpretations=(),
            missing_evidence=(),
            confidence=Confidence(0.5),
            verdict=Verdict.SUPPORTED,
        )


def test_advisory_text_cannot_override_authoritative_verdict() -> None:
    unified = UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=DeterministicAuditOutcome.from_audit_result(_audit_result()),
        research_interpretation=_advisory(),
    )

    assert unified.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert unified.research_interpretation is not None
    assert "SUPPORTED" in unified.research_interpretation.summary


def test_unified_result_complete_requires_deterministic_authority() -> None:
    with pytest.raises((TypeError, ValueError), match="COMPLETE|complete|deterministic"):
        UnifiedAnalysisResult(
            state=AnalysisState.COMPLETE,
            research_interpretation=_advisory(),
        )

    result = UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=DeterministicAuditOutcome.from_audit_result(_audit_result()),
    )
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED


def test_unified_result_mapping_needed_requires_question_and_no_authority() -> None:
    with pytest.raises((TypeError, ValueError), match="question|mapping"):
        UnifiedAnalysisResult(state=AnalysisState.MAPPING_NEEDED)
    with pytest.raises((TypeError, ValueError), match="deterministic|mapping"):
        UnifiedAnalysisResult(
            state=AnalysisState.MAPPING_NEEDED,
            deterministic=DeterministicAuditOutcome.from_audit_result(_audit_result()),
            mapping_question=_mapping_question(),
        )

    result = UnifiedAnalysisResult(
        state=AnalysisState.MAPPING_NEEDED,
        mapping_question=_mapping_question(),
    )
    assert result.authoritative_verdict is None


def test_unified_result_unavailable_requires_reason_and_no_results() -> None:
    with pytest.raises((TypeError, ValueError), match="reason|unavailable"):
        UnifiedAnalysisResult(state=AnalysisState.UNAVAILABLE)
    with pytest.raises((TypeError, ValueError), match="unavailable|result"):
        UnifiedAnalysisResult(
            state=AnalysisState.UNAVAILABLE,
            research_interpretation=_advisory(),
            unavailable_reason="adapter unavailable",
        )

    result = UnifiedAnalysisResult(
        state=AnalysisState.UNAVAILABLE,
        unavailable_reason="no eligible evidence artifacts were found",
    )
    assert result.authoritative_verdict is None


def test_unified_result_partial_requires_explicit_available_state() -> None:
    with pytest.raises((TypeError, ValueError), match="partial|available"):
        UnifiedAnalysisResult(state=AnalysisState.PARTIAL)

    result = UnifiedAnalysisResult(
        state=AnalysisState.PARTIAL,
        research_interpretation=_advisory(),
        unavailable_reason="deterministic adapter mapping is incomplete",
    )
    assert result.authoritative_verdict is None


def test_json_serialization_is_detached_finite_and_preserves_ephemeral_marker() -> None:
    baseline = _normalized_evidence(
        role=ExperimentRole.BASELINE,
        evidence_id="evidence-baseline-results",
    )
    plan = EphemeralAuditPlan(
        plan_id="plan-serialization",
        repository=RepositoryIdentity("amebaleon", "ClaimCI-Demo"),
        pr_number=None,
        head_sha=GitCommitSha("a" * 40),
        claim=ClaimReference(
            "claim-1",
            "Candidate improves accuracy.",
            None,
            Confidence(0.9),
            _provenance(),
        ),
        baseline_evidence=(baseline,),
        candidate_evidence=(_normalized_evidence(),),
        mapping_provenance=(_provenance(),),
        missing_evidence=(),
        confidence=Confidence(0.75),
    )

    serialized = to_jsonable(plan)
    encoded = json.dumps(serialized, allow_nan=False, sort_keys=True)

    assert serialized["ephemeral"] is True  # type: ignore[index]
    assert serialized["repository"]["owner"] == "amebaleon"  # type: ignore[index]
    assert serialized["confidence"] == 0.75  # type: ignore[index]
    assert '"head_sha": "aaaaaaaa' in encoded
    serialized["repository"]["owner"] = "mutated"  # type: ignore[index]
    assert plan.repository.owner == "amebaleon"

    serialized_path = to_jsonable(RepositoryPath("results/candidate.json"))
    assert type(serialized_path) is str
    assert serialized_path == "results/candidate.json"


def test_json_serialization_rejects_nonfinite_and_unsupported_values() -> None:
    with pytest.raises((TypeError, ValueError)):
        to_jsonable(math.nan)
    with pytest.raises((TypeError, ValueError)):
        to_jsonable(object())
    with pytest.raises((TypeError, ValueError), match="integer|JSON"):
        to_jsonable(10**5_000)


def test_legacy_manifest_is_an_optional_high_confidence_mapping_hint() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = (root / "research.yaml").read_bytes()
    manifest_provenance = FieldProvenance(
        kind=ProvenanceKind.MANIFEST_HINT,
        detail="legacy research.yaml explicitly maps audit inputs",
        source_path=RepositoryPath("research.yaml"),
        source_id="research.yaml",
    )
    manifest = ArtifactCandidate(
        path=RepositoryPath("research.yaml"),
        kind=ArtifactKind.MANIFEST,
        sha256=Sha256Digest(hashlib.sha256(raw).hexdigest()),
        size=len(raw),
        confidence=Confidence(0.99),
        discovery_reason="canonical legacy manifest filename",
        relevant_claim_ids=("claim-legacy",),
        provenance=manifest_provenance,
    )
    bindings = (
        _binding(
            "examples/shared/base/config.yaml",
            ArtifactKind.CONFIG,
            ExperimentRole.BASELINE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/base/results.json",
            ArtifactKind.RESULTS,
            ExperimentRole.BASELINE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/base/train.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.BASELINE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/base/eval.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.BASELINE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/candidate/config.yaml",
            ArtifactKind.CONFIG,
            ExperimentRole.CANDIDATE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/candidate/results.json",
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/candidate/train.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.CANDIDATE,
            provenance=manifest_provenance,
        ),
        _binding(
            "examples/shared/candidate/eval.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.CANDIDATE,
            provenance=manifest_provenance,
        ),
    )
    mapping = MappingCandidate(
        mapping_id="legacy-research-yaml",
        bindings=bindings,
        confidence=Confidence(0.99),
        trust=MappingTrust.MANIFEST_HINT,
        provenance=manifest_provenance,
    )

    assert manifest.kind is ArtifactKind.MANIFEST
    assert float(manifest.confidence) >= 0.9
    assert mapping.trust is MappingTrust.MANIFEST_HINT
    assert len(mapping.bindings) == 8
    assert {binding.role for binding in mapping.bindings} == {
        ExperimentRole.BASELINE,
        ExperimentRole.CANDIDATE,
    }
    assert "manifest" not in EphemeralAuditPlan.__dataclass_fields__


def test_analysis_contracts_import_no_provider_or_execution_surface() -> None:
    module_path = Path(inspect.getfile(RepositoryPath)).resolve()
    package_root = module_path.parent
    forbidden_modules = {
        "subprocess",
        "openai",
        "importlib",
        "claimci.review.orchestrator",
        "claimci.review.openai_provider",
    }
    forbidden_builtins = {"eval", "exec", "compile", "__import__"}
    forbidden_attributes = {"system", "popen"}

    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({alias.name for alias in node.names} & forbidden_modules)
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_builtins
                elif isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in forbidden_attributes
