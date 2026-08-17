"""Behavioral contracts for the ClaimCI v0.3 shared analysis model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest

from claimci.analysis import (
    Adapter,
    AdapterMatch,
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
