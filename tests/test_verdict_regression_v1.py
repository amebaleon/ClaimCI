from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from claimci.analysis import (
    AdvisoryResearchInterpretation,
    AnalysisState,
    ArtifactBinding,
    ArtifactKind,
    Confidence,
    ConfigValue,
    DeterministicAuditOutcome,
    ExecutionClassification,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    HistoricalAnalysisOutcome,
    MappingChoice,
    MappingQuestion,
    OutcomeComparability,
    ProvenanceKind,
    RegressionClassification,
    ReplayRecipe,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    UnifiedAnalysisResult,
    VerdictRegressionReport,
    VerdictTransitionKind,
    classify_verdict_regression,
    derive_ephemeral_plan_id,
    execute_ephemeral_audit_with_trace,
)
from claimci.analysis.regression import _raw_verdict_transition
from claimci.models import AuditResult, Verdict
from tests.test_ephemeral_materialize_v03 import _plan_fixture


def _audit_result(verdict: Verdict = Verdict.SUPPORTED) -> AuditResult:
    return AuditResult(
        verdict=verdict,
        findings=(),
        manifest_path=Path("<ephemeral>/research.yaml"),
        metric="accuracy",
        minimum_improvement=0.05,
    )


def _deterministic(verdict: Verdict = Verdict.SUPPORTED) -> DeterministicAuditOutcome:
    return DeterministicAuditOutcome.from_audit_result(_audit_result(verdict))


def _advisory() -> AdvisoryResearchInterpretation:
    return AdvisoryResearchInterpretation(
        summary="Advisory context only.",
        interpretations=(),
        missing_evidence=(),
        confidence=Confidence(0.8),
    )


def _mapping_question() -> MappingQuestion:
    provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "bounded mapping fixture",
        RepositoryPath("results/value.json"),
    )
    mapping = FieldMapping(
        "metric_value",
        EvidenceSelector(SelectorKind.JSON_POINTER, "/accuracy", provenance),
        provenance,
    )
    baseline = ArtifactBinding(
        RepositoryPath("results/baseline.json"),
        ArtifactKind.RESULTS,
        ExperimentRole.BASELINE,
        "claimci-structured-results-v1",
        (mapping,),
        provenance,
    )
    candidate = dataclasses.replace(
        baseline,
        path=RepositoryPath("results/candidate.json"),
        role=ExperimentRole.CANDIDATE,
    )
    return MappingQuestion(
        "question-1",
        "Which validated result mapping should ClaimCI use?",
        (
            MappingChoice("choice-1", "Use the first mapping", (baseline,)),
            MappingChoice("choice-2", "Use the second mapping", (candidate,)),
        ),
        "claim-1",
    )


def _prepared_execution(
    root: Path,
    *,
    candidate_values: tuple[float, ...] = (0.85, 0.9, 0.95),
    candidate_seeds: tuple[int | None, ...] = (11, 12, 13),
    baseline_values: tuple[float, ...] = (0.5, 0.6, 0.7),
    minimum_improvement: float = 0.05,
    candidate_training_steps: int = 100,
    candidate_evaluation_version: str = "v1",
    candidate_result_suffix: bytes = b"",
    result_adapter_id: str | None = None,
    engine_source_revision: str | None = None,
) -> object:
    root.mkdir(parents=True, exist_ok=True)
    plan, runtime, checkout, _scratch = _plan_fixture(
        root,
        candidate_train_content=b'{"id":"candidate-train"}\n',
    )

    def update_evidence(
        values: tuple[object, ...],
        *,
        candidate: bool,
    ) -> tuple[object, ...]:
        updated: list[object] = []
        for evidence in values:
            if evidence.artifact.kind is ArtifactKind.RESULTS:
                observed_values = candidate_values if candidate else baseline_values
                observed_seeds = (
                    candidate_seeds
                    if candidate
                    else tuple(range(1, len(observed_values) + 1))
                )
                observation = evidence.observations[0]
                observations = tuple(
                    dataclasses.replace(
                        observation,
                        metric_value=value,
                        seed=seed,
                        run_id=f"run-{index}",
                    )
                    for index, (value, seed) in enumerate(
                        zip(observed_values, observed_seeds, strict=True),
                        start=1,
                    )
                )
                content = (
                    "accuracy,seed\n"
                    + "".join(
                        f"{value},{'' if seed is None else seed}\n"
                        for value, seed in zip(
                            observed_values,
                            observed_seeds,
                            strict=True,
                        )
                    )
                ).encode("utf-8")
                if candidate:
                    content += candidate_result_suffix
                destination = checkout / str(evidence.artifact.path)
                destination.write_bytes(content)
                adapter_match = evidence.adapter_match
                if result_adapter_id is not None:
                    adapter_match = dataclasses.replace(
                        adapter_match,
                        adapter_id=result_adapter_id,
                    )
                evidence = dataclasses.replace(
                    evidence,
                    artifact=dataclasses.replace(
                        evidence.artifact,
                        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
                        size=len(content),
                    ),
                    adapter_match=adapter_match,
                    observations=observations,
                )
            elif evidence.artifact.kind is ArtifactKind.CONFIG:
                observation = evidence.observations[0]
                config_values = [
                    item
                    for item in observation.config_values
                    if not item.key.startswith("model.")
                ]
                if candidate:
                    config_values = [
                        dataclasses.replace(item, value=candidate_training_steps)
                        if item.key == "training_steps"
                        else dataclasses.replace(
                            item,
                            value=candidate_evaluation_version,
                        )
                        if item.key == "evaluation.dataset_version"
                        else item
                        for item in config_values
                    ]
                config_values.append(
                    ConfigValue("model", "demo", observation.provenance)
                )
                evidence = dataclasses.replace(
                    evidence,
                    observations=(
                        dataclasses.replace(
                            observation,
                            config_values=tuple(
                                sorted(config_values, key=lambda item: item.key)
                            ),
                        ),
                    ),
                )
            updated.append(evidence)
        return tuple(updated)

    baseline = update_evidence(plan.baseline_evidence, candidate=False)
    candidate = update_evidence(plan.candidate_evidence, candidate=True)
    selected_mapping = plan.selected_mapping
    if result_adapter_id is not None:
        assert selected_mapping is not None
        selected_mapping = dataclasses.replace(
            selected_mapping,
            bindings=tuple(
                dataclasses.replace(binding, adapter_id=result_adapter_id)
                if binding.kind is ArtifactKind.RESULTS
                else binding
                for binding in selected_mapping.bindings
            ),
        )
    audit_claim = dataclasses.replace(
        plan.audit_claim,
        minimum_absolute_improvement=minimum_improvement,
    )
    plan = dataclasses.replace(
        plan,
        baseline_evidence=baseline,
        candidate_evidence=candidate,
        audit_claim=audit_claim,
        selected_mapping=selected_mapping,
    )
    plan = dataclasses.replace(plan, plan_id=derive_ephemeral_plan_id(plan))
    if engine_source_revision is not None:
        runtime = dataclasses.replace(
            runtime,
            engine_source_revision=engine_source_revision,
        )
    return execute_ephemeral_audit_with_trace(plan, runtime)


def _outcome_from_execution(
    execution: object,
    *,
    state: AnalysisState = AnalysisState.COMPLETE,
    advisory_summary: str = "Advisory context only.",
) -> HistoricalAnalysisOutcome:
    audit_result = execution.audit_result
    replay_recipe = execution.replay_recipe
    input_snapshot = execution.input_snapshot
    assert type(replay_recipe) is ReplayRecipe
    result = UnifiedAnalysisResult(
        state=state,
        deterministic=DeterministicAuditOutcome.from_audit_result(audit_result),
        research_interpretation=(
            AdvisoryResearchInterpretation(
                summary=advisory_summary,
                interpretations=(),
                missing_evidence=(),
                confidence=Confidence(0.8),
            )
            if state is AnalysisState.COMPLETE
            else None
        ),
        unavailable_reason=(
            "advisory Research Review unavailable"
            if state is AnalysisState.PARTIAL
            else None
        ),
        verification_input_snapshot=input_snapshot,
        replay_recipe=replay_recipe,
    )
    return HistoricalAnalysisOutcome.from_unified_result(result)


def test_verdict_regression_enums_expose_only_the_frozen_v1_vocabulary() -> None:
    assert {item.value for item in ExecutionClassification} == {
        "deterministic_complete",
        "pre_audit_evidence_partial",
        "mapping_needed",
        "post_audit_advisory_partial",
        "operational_unavailable",
        "provider_unavailable",
        "runner_snapshot_unavailable",
    }
    assert {item.value for item in OutcomeComparability} == {
        "exact",
        "captured_scope",
        "claim_changed",
        "measurement_changed",
        "baseline_changed",
        "engine_changed",
        "indeterminate",
        "legacy_unavailable",
    }
    assert {
        "unchanged",
        "supported_to_not_supported",
        "supported_to_insufficient",
        "supported_to_no_verdict",
        "not_supported_to_supported",
        "insufficient_to_supported",
        "completeness_regression",
        "completeness_recovery",
        "raw_transition_not_comparable",
    } <= {item.value for item in VerdictTransitionKind}
    assert {item.value for item in RegressionClassification} == {
        "regression",
        "evidence_regression",
        "coverage_regression",
        "resolution",
        "unchanged",
        "not_comparable",
    }


@pytest.mark.parametrize(
    ("result", "unavailable_classification", "expected"),
    (
        (
            UnifiedAnalysisResult(
                state=AnalysisState.COMPLETE,
                deterministic=_deterministic(),
                research_interpretation=_advisory(),
            ),
            None,
            ExecutionClassification.DETERMINISTIC_COMPLETE,
        ),
        (
            UnifiedAnalysisResult(
                state=AnalysisState.PARTIAL,
                unavailable_reason="required passive evidence is missing",
            ),
            None,
            ExecutionClassification.PRE_AUDIT_EVIDENCE_PARTIAL,
        ),
        (
            UnifiedAnalysisResult(
                state=AnalysisState.MAPPING_NEEDED,
                mapping_question=_mapping_question(),
            ),
            None,
            ExecutionClassification.MAPPING_NEEDED,
        ),
        (
            UnifiedAnalysisResult(
                state=AnalysisState.PARTIAL,
                deterministic=_deterministic(),
                unavailable_reason="advisory review unavailable",
            ),
            None,
            ExecutionClassification.POST_AUDIT_ADVISORY_PARTIAL,
        ),
        *tuple(
            (
                UnifiedAnalysisResult(
                    state=AnalysisState.UNAVAILABLE,
                    unavailable_reason="fixed operational reason",
                ),
                unavailable,
                unavailable,
            )
            for unavailable in (
                ExecutionClassification.OPERATIONAL_UNAVAILABLE,
                ExecutionClassification.PROVIDER_UNAVAILABLE,
                ExecutionClassification.RUNNER_SNAPSHOT_UNAVAILABLE,
            )
        ),
    ),
)
def test_historical_outcome_derives_the_exact_execution_classification(
    result: UnifiedAnalysisResult,
    unavailable_classification: ExecutionClassification | None,
    expected: ExecutionClassification,
) -> None:
    outcome = HistoricalAnalysisOutcome.from_unified_result(
        result,
        unavailable_classification=unavailable_classification,
    )

    assert outcome.execution is expected
    assert outcome.verdict is (
        None if result.deterministic is None else result.deterministic.verdict
    )


def test_unavailable_classification_is_typed_and_cannot_relabel_other_states() -> None:
    unavailable = UnifiedAnalysisResult(
        state=AnalysisState.UNAVAILABLE,
        unavailable_reason="do not parse this text",
    )
    complete = UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=_deterministic(),
        research_interpretation=_advisory(),
    )

    with pytest.raises(ValueError, match="unavailable classification"):
        HistoricalAnalysisOutcome.from_unified_result(unavailable)
    with pytest.raises(ValueError, match="unavailable classification"):
        HistoricalAnalysisOutcome.from_unified_result(
            unavailable,
            unavailable_classification=ExecutionClassification.MAPPING_NEEDED,
        )
    with pytest.raises(ValueError, match="only valid"):
        HistoricalAnalysisOutcome.from_unified_result(
            complete,
            unavailable_classification=ExecutionClassification.PROVIDER_UNAVAILABLE,
        )


def test_historical_outcome_is_exact_type_factory_only_immutable_and_final() -> None:
    result = UnifiedAnalysisResult(
        state=AnalysisState.PARTIAL,
        deterministic=_deterministic(),
        unavailable_reason="advisory review unavailable",
    )
    outcome = HistoricalAnalysisOutcome.from_unified_result(result)

    with pytest.raises(TypeError):
        HistoricalAnalysisOutcome()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="UnifiedAnalysisResult"):
        HistoricalAnalysisOutcome.from_unified_result(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="final"):
        type("ForgedHistoricalOutcome", (HistoricalAnalysisOutcome,), {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.verdict = Verdict.NOT_SUPPORTED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    (
        (None, None, VerdictTransitionKind.UNCHANGED),
        (None, Verdict.SUPPORTED, VerdictTransitionKind.NO_VERDICT_TO_SUPPORTED),
        (
            None,
            Verdict.NOT_SUPPORTED,
            VerdictTransitionKind.NO_VERDICT_TO_NOT_SUPPORTED,
        ),
        (
            None,
            Verdict.INSUFFICIENT_EVIDENCE,
            VerdictTransitionKind.NO_VERDICT_TO_INSUFFICIENT,
        ),
        (
            Verdict.SUPPORTED,
            None,
            VerdictTransitionKind.SUPPORTED_TO_NO_VERDICT,
        ),
        (
            Verdict.NOT_SUPPORTED,
            None,
            VerdictTransitionKind.NOT_SUPPORTED_TO_NO_VERDICT,
        ),
        (
            Verdict.INSUFFICIENT_EVIDENCE,
            None,
            VerdictTransitionKind.INSUFFICIENT_TO_NO_VERDICT,
        ),
        (Verdict.SUPPORTED, Verdict.SUPPORTED, VerdictTransitionKind.UNCHANGED),
        (
            Verdict.SUPPORTED,
            Verdict.NOT_SUPPORTED,
            VerdictTransitionKind.SUPPORTED_TO_NOT_SUPPORTED,
        ),
        (
            Verdict.SUPPORTED,
            Verdict.INSUFFICIENT_EVIDENCE,
            VerdictTransitionKind.SUPPORTED_TO_INSUFFICIENT,
        ),
        (
            Verdict.NOT_SUPPORTED,
            Verdict.SUPPORTED,
            VerdictTransitionKind.NOT_SUPPORTED_TO_SUPPORTED,
        ),
        (
            Verdict.NOT_SUPPORTED,
            Verdict.NOT_SUPPORTED,
            VerdictTransitionKind.UNCHANGED,
        ),
        (
            Verdict.NOT_SUPPORTED,
            Verdict.INSUFFICIENT_EVIDENCE,
            VerdictTransitionKind.NOT_SUPPORTED_TO_INSUFFICIENT,
        ),
        (
            Verdict.INSUFFICIENT_EVIDENCE,
            Verdict.SUPPORTED,
            VerdictTransitionKind.INSUFFICIENT_TO_SUPPORTED,
        ),
        (
            Verdict.INSUFFICIENT_EVIDENCE,
            Verdict.NOT_SUPPORTED,
            VerdictTransitionKind.INSUFFICIENT_TO_NOT_SUPPORTED,
        ),
        (
            Verdict.INSUFFICIENT_EVIDENCE,
            Verdict.INSUFFICIENT_EVIDENCE,
            VerdictTransitionKind.UNCHANGED,
        ),
    ),
)
def test_raw_verdict_transition_covers_the_complete_matrix(
    previous: Verdict | None,
    current: Verdict | None,
    expected: VerdictTransitionKind,
) -> None:
    assert _raw_verdict_transition(previous, current) is expected


def test_supported_to_not_supported_with_real_audits_is_a_regression(
    tmp_path: Path,
) -> None:
    supported_root = tmp_path / "supported"
    not_supported_root = tmp_path / "not-supported"
    supported_root.mkdir()
    not_supported_root.mkdir()
    previous_execution = _prepared_execution(supported_root)
    current_execution = _prepared_execution(
        not_supported_root,
        candidate_values=(0.3, 0.4, 0.5),
    )
    assert previous_execution.audit_result.verdict is Verdict.SUPPORTED
    assert current_execution.audit_result.verdict is Verdict.NOT_SUPPORTED

    report = classify_verdict_regression(
        _outcome_from_execution(previous_execution),
        _outcome_from_execution(current_execution),
    )

    assert type(report) is VerdictRegressionReport
    assert report.comparability is OutcomeComparability.CAPTURED_SCOPE
    assert report.transition_kind is VerdictTransitionKind.SUPPORTED_TO_NOT_SUPPORTED
    assert report.classification is RegressionClassification.REGRESSION
    assert report.previous_audit_sha256 is not None
    assert report.current_audit_sha256 is not None


def test_supported_to_insufficient_with_real_audits_is_an_evidence_regression(
    tmp_path: Path,
) -> None:
    supported_root = tmp_path / "supported"
    insufficient_root = tmp_path / "insufficient"
    supported_root.mkdir()
    insufficient_root.mkdir()
    previous_execution = _prepared_execution(supported_root)
    current_execution = _prepared_execution(
        insufficient_root,
        candidate_seeds=(11, None, 13),
    )
    assert previous_execution.audit_result.verdict is Verdict.SUPPORTED
    assert current_execution.audit_result.verdict is Verdict.INSUFFICIENT_EVIDENCE

    report = classify_verdict_regression(
        _outcome_from_execution(previous_execution),
        _outcome_from_execution(current_execution),
    )

    assert report.comparability is OutcomeComparability.CAPTURED_SCOPE
    assert report.transition_kind is VerdictTransitionKind.SUPPORTED_TO_INSUFFICIENT
    assert report.classification is RegressionClassification.EVIDENCE_REGRESSION


def test_not_supported_to_supported_with_compatible_frames_is_a_resolution(
    tmp_path: Path,
) -> None:
    not_supported_root = tmp_path / "not-supported"
    supported_root = tmp_path / "supported"
    not_supported_root.mkdir()
    supported_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(
            _prepared_execution(
                not_supported_root,
                candidate_values=(0.3, 0.4, 0.5),
            )
        ),
        _outcome_from_execution(_prepared_execution(supported_root)),
    )

    assert report.comparability is OutcomeComparability.CAPTURED_SCOPE
    assert report.transition_kind is VerdictTransitionKind.NOT_SUPPORTED_TO_SUPPORTED
    assert report.classification is RegressionClassification.RESOLUTION


def test_same_full_capture_is_exact_even_when_engine_build_provenance_changes(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(
            _prepared_execution(first_root, engine_source_revision="a" * 40)
        ),
        _outcome_from_execution(
            _prepared_execution(second_root, engine_source_revision="b" * 40)
        ),
    )

    assert report.comparability is OutcomeComparability.EXACT
    assert report.transition_kind is VerdictTransitionKind.UNCHANGED
    assert report.classification is RegressionClassification.UNCHANGED


def test_source_only_measurement_snapshot_change_remains_comparable(
    tmp_path: Path,
) -> None:
    first_execution = _prepared_execution(tmp_path / "first")
    second_execution = _prepared_execution(
        tmp_path / "second",
        candidate_result_suffix=b"\n",
    )
    first_snapshot = first_execution.input_snapshot
    second_snapshot = second_execution.input_snapshot
    assert first_snapshot.candidate_measurement is not None
    assert second_snapshot.candidate_measurement is not None
    assert (
        first_snapshot.candidate_measurement.semantic_protocol_id
        == second_snapshot.candidate_measurement.semantic_protocol_id
    )
    assert (
        first_snapshot.candidate_measurement.source_snapshot_id
        != second_snapshot.candidate_measurement.source_snapshot_id
    )

    report = classify_verdict_regression(
        _outcome_from_execution(first_execution),
        _outcome_from_execution(second_execution),
    )

    assert report.comparability is OutcomeComparability.EXACT
    assert report.transition_kind is VerdictTransitionKind.UNCHANGED
    assert report.classification is RegressionClassification.UNCHANGED


def test_claim_audit_spec_change_makes_the_raw_transition_not_comparable(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(first_root)),
        _outcome_from_execution(
            _prepared_execution(second_root, minimum_improvement=0.4)
        ),
    )

    assert report.comparability is OutcomeComparability.CLAIM_CHANGED
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_claim_change_precedes_other_incompatible_frame_changes(
    tmp_path: Path,
) -> None:
    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(tmp_path / "first")),
        _outcome_from_execution(
            _prepared_execution(
                tmp_path / "second",
                minimum_improvement=0.4,
                candidate_evaluation_version="v2",
                result_adapter_id="fixture.results-v2",
            )
        ),
    )

    assert report.comparability is OutcomeComparability.CLAIM_CHANGED
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_measurement_semantic_change_makes_the_raw_transition_not_comparable(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(first_root)),
        _outcome_from_execution(
            _prepared_execution(second_root, candidate_evaluation_version="v2")
        ),
    )

    assert report.comparability is OutcomeComparability.MEASUREMENT_CHANGED
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_baseline_audit_semantics_change_is_not_a_verdict_regression(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(first_root)),
        _outcome_from_execution(
            _prepared_execution(second_root, baseline_values=(0.2, 0.3, 0.4))
        ),
    )

    assert report.comparability is OutcomeComparability.BASELINE_CHANGED
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_audit_semantics_compatibility_change_is_engine_changed(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(first_root)),
        _outcome_from_execution(
            _prepared_execution(second_root, result_adapter_id="fixture.results-v2")
        ),
    )

    assert report.comparability is OutcomeComparability.ENGINE_CHANGED
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_unattributed_frame_change_is_indeterminate(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    report = classify_verdict_regression(
        _outcome_from_execution(_prepared_execution(first_root)),
        _outcome_from_execution(
            _prepared_execution(second_root, candidate_training_steps=110)
        ),
    )

    assert report.comparability is OutcomeComparability.INDETERMINATE
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_legacy_deterministic_results_without_replay_are_not_comparable() -> None:
    previous = HistoricalAnalysisOutcome.from_unified_result(
        UnifiedAnalysisResult(
            state=AnalysisState.COMPLETE,
            deterministic=_deterministic(Verdict.SUPPORTED),
            research_interpretation=_advisory(),
        )
    )
    current = HistoricalAnalysisOutcome.from_unified_result(
        UnifiedAnalysisResult(
            state=AnalysisState.COMPLETE,
            deterministic=_deterministic(Verdict.NOT_SUPPORTED),
            research_interpretation=_advisory(),
        )
    )

    report = classify_verdict_regression(previous, current)

    assert report.comparability is OutcomeComparability.LEGACY_UNAVAILABLE
    assert report.transition_kind is VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_pre_audit_partial_after_real_audit_is_coverage_not_verdict_regression(
    tmp_path: Path,
) -> None:
    previous = _outcome_from_execution(_prepared_execution(tmp_path / "previous"))
    current = HistoricalAnalysisOutcome.from_unified_result(
        UnifiedAnalysisResult(
            state=AnalysisState.PARTIAL,
            unavailable_reason="required baseline results disappeared",
        )
    )

    report = classify_verdict_regression(previous, current)

    assert report.comparability is OutcomeComparability.INDETERMINATE
    assert report.transition_kind is VerdictTransitionKind.SUPPORTED_TO_NO_VERDICT
    assert report.classification is RegressionClassification.COVERAGE_REGRESSION
    assert report.current_verdict is None


def test_mapping_needed_after_real_audit_is_workflow_attention_only(
    tmp_path: Path,
) -> None:
    previous = _outcome_from_execution(_prepared_execution(tmp_path / "previous"))
    current = HistoricalAnalysisOutcome.from_unified_result(
        UnifiedAnalysisResult(
            state=AnalysisState.MAPPING_NEEDED,
            mapping_question=_mapping_question(),
        )
    )

    report = classify_verdict_regression(previous, current)

    assert report.transition_kind is VerdictTransitionKind.SUPPORTED_TO_NO_VERDICT
    assert report.classification is RegressionClassification.NOT_COMPARABLE


@pytest.mark.parametrize(
    "execution",
    (
        ExecutionClassification.OPERATIONAL_UNAVAILABLE,
        ExecutionClassification.PROVIDER_UNAVAILABLE,
        ExecutionClassification.RUNNER_SNAPSHOT_UNAVAILABLE,
    ),
)
def test_infrastructure_unavailable_is_never_scientific_or_evidence_regression(
    tmp_path: Path,
    execution: ExecutionClassification,
) -> None:
    previous = _outcome_from_execution(_prepared_execution(tmp_path / execution.value))
    current = HistoricalAnalysisOutcome.from_unified_result(
        UnifiedAnalysisResult(
            state=AnalysisState.UNAVAILABLE,
            unavailable_reason="typed infrastructure failure",
        ),
        unavailable_classification=execution,
    )

    report = classify_verdict_regression(previous, current)

    assert report.transition_kind is VerdictTransitionKind.SUPPORTED_TO_NO_VERDICT
    assert report.classification is RegressionClassification.NOT_COMPARABLE


def test_post_audit_advisory_partial_is_only_a_completeness_regression(
    tmp_path: Path,
) -> None:
    execution = _prepared_execution(tmp_path)
    previous = _outcome_from_execution(execution)
    current = _outcome_from_execution(execution, state=AnalysisState.PARTIAL)

    report = classify_verdict_regression(previous, current)

    assert report.comparability is OutcomeComparability.EXACT
    assert report.transition_kind is VerdictTransitionKind.COMPLETENESS_REGRESSION
    assert report.classification is RegressionClassification.UNCHANGED
    assert report.previous_verdict is report.current_verdict is Verdict.SUPPORTED


def test_post_audit_completeness_recovery_is_contextual_resolution(
    tmp_path: Path,
) -> None:
    execution = _prepared_execution(tmp_path)
    previous = _outcome_from_execution(execution, state=AnalysisState.PARTIAL)
    current = _outcome_from_execution(execution)

    report = classify_verdict_regression(previous, current)

    assert report.transition_kind is VerdictTransitionKind.COMPLETENESS_RECOVERY
    assert report.classification is RegressionClassification.RESOLUTION


def test_advisory_text_cannot_change_a_verdict_regression_classification(
    tmp_path: Path,
) -> None:
    execution = _prepared_execution(tmp_path)
    previous = _outcome_from_execution(
        execution,
        advisory_summary="NOT_SUPPORTED and REGRESSION",
    )
    current = _outcome_from_execution(
        execution,
        advisory_summary="SUPPORTED and RESOLUTION",
    )

    report = classify_verdict_regression(previous, current)

    assert report.transition_kind is VerdictTransitionKind.UNCHANGED
    assert report.classification is RegressionClassification.UNCHANGED


def test_regression_report_is_contextual_immutable_final_and_has_no_authority_fields(
    tmp_path: Path,
) -> None:
    execution = _prepared_execution(tmp_path)
    outcome = _outcome_from_execution(execution)
    before_audit = execution.audit_result
    before_recipe = outcome.replay_recipe

    report = classify_verdict_regression(outcome, outcome)

    assert execution.audit_result == before_audit
    assert outcome.replay_recipe == before_recipe
    assert not hasattr(report, "findings")
    assert not hasattr(report, "impact")
    assert not hasattr(report, "authoritative_verdict")
    with pytest.raises(TypeError):
        VerdictRegressionReport()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="final"):
        type("ForgedRegressionReport", (VerdictRegressionReport,), {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.classification = RegressionClassification.REGRESSION  # type: ignore[misc]
