from __future__ import annotations

import dataclasses
import json
from dataclasses import FrozenInstanceError
import pytest
import yaml

from claimci.audit import audit_research
from claimci.measurement import (
    ClaimCIVerificationReduction,
    MeasurementAuditContext,
    MeasurementComponentKind,
    MeasurementComponentScope,
    MeasurementContractError,
    MeasurementCoverage,
    MeasurementDriftFinding,
    MeasurementDriftReport,
    MeasurementDriftState,
    MeasurementProtocolComponent,
    MeasurementProtocolIdentity,
    MeasurementProtocolPair,
    MeasurementSemanticField,
    MeasurementSourceBinding,
    MeasurementSourceSelector,
    MeasurementSourceSnapshot,
    compare_measurement_protocols,
    measurement_report_to_jsonable,
)
from claimci.models import Verdict
from claimci.report import render_json


def _binding(
    *,
    path: str = "configs/baseline.yaml",
    digest: str = "a" * 64,
    role: str = "baseline",
    kind: str = "config",
    split: str | None = None,
) -> MeasurementSourceBinding:
    return MeasurementSourceBinding(
        path=path,
        kind=kind,
        sha256=digest,
        size=128,
        adapter_id="claimci-yaml-config-v1",
        adapter_version="v1",
        selectors=(
            MeasurementSourceSelector(
                target_field="config.evaluation.metric",
                kind="dotted_path",
                expression="evaluation.metric",
            ),
        ),
        role=role,
        split=split,
    )


def _snapshot(
    binding: MeasurementSourceBinding,
    *,
    head_sha: str = "b" * 40,
) -> MeasurementSourceSnapshot:
    return MeasurementSourceSnapshot.from_bindings(
        repository_id="amebaleon/ClaimCI-Demo",
        head_sha=head_sha,
        bindings=(binding,),
    )


def _component(
    binding: MeasurementSourceBinding,
    *,
    kind: MeasurementComponentKind = MeasurementComponentKind.METRIC,
    coverage: MeasurementCoverage = MeasurementCoverage.RECOVERED,
    fields: tuple[MeasurementSemanticField, ...] = (
        MeasurementSemanticField("name", "accuracy"),
    ),
) -> MeasurementProtocolComponent:
    schema_ids = {
        MeasurementComponentKind.EVALUATOR_IMPLEMENTATION: (
            "claimci.measurement.evaluator.v1"
        ),
        MeasurementComponentKind.METRIC: "claimci.measurement.metric.v1",
    }
    return MeasurementProtocolComponent(
        kind=kind,
        scope=MeasurementComponentScope.MEASUREMENT_PROCEDURE,
        coverage=coverage,
        schema_id=schema_ids[kind],
        semantic_fields=fields,
        source_binding_ids=(binding.binding_id,),
    )


def _identity(
    binding: MeasurementSourceBinding,
    component: MeasurementProtocolComponent,
) -> MeasurementProtocolIdentity:
    return MeasurementProtocolIdentity.from_components(
        policy_id="claimci.measurement.pilot.v1",
        source_snapshot=_snapshot(binding),
        components=(component,),
    )


def test_measurement_contract_enums_are_exact_and_claim_reduction_is_separate() -> None:
    assert {item.value for item in MeasurementComponentKind} == {
        "evaluator_implementation",
        "metric",
        "evaluation_dataset",
        "evaluation_config",
        "prompt_template",
        "decoding_parameters",
        "test_selection",
        "retry_aggregation",
        "normalization_postprocessing",
        "benchmark_version",
    }
    assert {item.value for item in MeasurementComponentScope} == {
        "measurement_procedure",
        "system_under_test",
        "unknown",
    }
    assert {item.value for item in MeasurementCoverage} == {
        "recovered",
        "byte_only",
        "missing",
        "not_applicable",
    }
    assert {item.value for item in MeasurementDriftState} == {
        "verified",
        "warning",
        "insufficient",
        "invalidates",
        "not_assessed",
    }
    assert (
        ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1.value
        == "arithmetic_mean_v1"
    )
    assert "direction" not in {item.name for item in dataclasses.fields(MeasurementProtocolIdentity)}
    assert "minimum_improvement" not in {
        item.name for item in dataclasses.fields(MeasurementProtocolIdentity)
    }


def test_protocol_contracts_are_immutable_and_ids_are_factory_derived() -> None:
    binding = _binding()
    component = _component(binding)
    snapshot = _snapshot(binding)
    identity = _identity(binding, component)

    with pytest.raises(FrozenInstanceError):
        component.coverage = MeasurementCoverage.MISSING  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.bindings = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="factory"):
        MeasurementSourceSnapshot()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="factory"):
        MeasurementProtocolIdentity()  # type: ignore[call-arg]
    assert identity.semantic_protocol_id.startswith("measurement-semantic-")
    assert identity.source_snapshot_id == snapshot.source_snapshot_id


def test_component_scope_and_schema_are_fixed_by_claimci_policy() -> None:
    binding = _binding()

    with pytest.raises(MeasurementContractError, match="scope"):
        MeasurementProtocolComponent(
            kind=MeasurementComponentKind.METRIC,
            scope=MeasurementComponentScope.SYSTEM_UNDER_TEST,
            coverage=MeasurementCoverage.RECOVERED,
            schema_id="claimci.measurement.metric.v1",
            semantic_fields=(MeasurementSemanticField("name", "accuracy"),),
            source_binding_ids=(binding.binding_id,),
        )
    with pytest.raises(MeasurementContractError, match="schema"):
        MeasurementProtocolComponent(
            kind=MeasurementComponentKind.METRIC,
            scope=MeasurementComponentScope.MEASUREMENT_PROCEDURE,
            coverage=MeasurementCoverage.RECOVERED,
            schema_id="provider.metric.v999",
            semantic_fields=(MeasurementSemanticField("name", "accuracy"),),
            source_binding_ids=(binding.binding_id,),
        )


def test_source_path_head_and_hash_never_feed_semantic_protocol_identity() -> None:
    first_binding = _binding(path="configs/first.yaml", digest="a" * 64)
    second_binding = _binding(path="other/second.yaml", digest="c" * 64)
    first = MeasurementProtocolIdentity.from_components(
        policy_id="claimci.measurement.pilot.v1",
        source_snapshot=_snapshot(first_binding, head_sha="b" * 40),
        components=(_component(first_binding),),
    )
    second = MeasurementProtocolIdentity.from_components(
        policy_id="claimci.measurement.pilot.v1",
        source_snapshot=_snapshot(second_binding, head_sha="d" * 40),
        components=(_component(second_binding),),
    )

    assert first.semantic_protocol_id == second.semantic_protocol_id
    assert first.source_snapshot_id != second.source_snapshot_id


def test_measurement_source_and_semantic_fields_are_bounded_and_confined() -> None:
    with pytest.raises(MeasurementContractError, match="repository-relative"):
        _binding(path="../outside.yaml")
    with pytest.raises(MeasurementContractError, match="SHA-256"):
        _binding(digest="not-a-hash")
    with pytest.raises(MeasurementContractError, match="repository-relative"):
        _binding(path="configs/CON.yaml")
    with pytest.raises(MeasurementContractError, match="size"):
        dataclasses.replace(_binding(), size=2**63)
    with pytest.raises(MeasurementContractError, match="bounded"):
        MeasurementSourceSelector(
            target_field="config.evaluation.metric",
            kind="dotted_path",
            expression="x" * 1_025,
        )
    with pytest.raises(MeasurementContractError, match="bounded"):
        MeasurementSemanticField("name", "x" * 4_097)


def test_semantic_and_source_snapshot_ids_are_independently_stable() -> None:
    first_binding = _binding(digest="a" * 64)
    second_binding = _binding(digest="c" * 64)
    first = _identity(first_binding, _component(first_binding))
    second = _identity(second_binding, _component(second_binding))

    assert first.semantic_protocol_id == second.semantic_protocol_id
    assert first.source_snapshot_id != second.source_snapshot_id

    reordered_fields = (
        MeasurementSemanticField("version", "v1"),
        MeasurementSemanticField("name", "accuracy"),
    )
    canonical = _identity(
        first_binding,
        _component(first_binding, fields=tuple(reversed(reordered_fields))),
    )
    reordered = _identity(
        first_binding,
        _component(first_binding, fields=reordered_fields),
    )
    assert canonical.semantic_protocol_id == reordered.semantic_protocol_id


def test_source_bindings_and_protocol_components_have_canonical_order() -> None:
    config = _binding(path="configs/base.yaml", digest="a" * 64)
    results = _binding(
        path="results/base.json",
        digest="c" * 64,
        kind="results",
    )
    first_snapshot = MeasurementSourceSnapshot.from_bindings(
        repository_id="amebaleon/ClaimCI-Demo",
        head_sha="b" * 40,
        bindings=(config, results),
    )
    second_snapshot = MeasurementSourceSnapshot.from_bindings(
        repository_id="amebaleon/ClaimCI-Demo",
        head_sha="b" * 40,
        bindings=(results, config),
    )
    metric = _component(config)
    evaluator = _component(
        results,
        kind=MeasurementComponentKind.EVALUATOR_IMPLEMENTATION,
        fields=(MeasurementSemanticField("version", "v1"),),
    )
    first = MeasurementProtocolIdentity.from_components(
        policy_id="claimci.measurement.pilot.v1",
        source_snapshot=first_snapshot,
        components=(metric, evaluator),
    )
    second = MeasurementProtocolIdentity.from_components(
        policy_id="claimci.measurement.pilot.v1",
        source_snapshot=second_snapshot,
        components=(evaluator, metric),
    )

    assert first_snapshot == second_snapshot
    assert first.semantic_protocol_id == second.semantic_protocol_id
    assert first.components == second.components


def test_source_only_byte_drift_is_warning_and_never_invalidating() -> None:
    baseline_binding = _binding(path="evaluator/base.py", digest="a" * 64)
    candidate_binding = _binding(
        path="evaluator/candidate.py",
        digest="c" * 64,
        role="candidate",
    )
    baseline_component = _component(
        baseline_binding,
        kind=MeasurementComponentKind.EVALUATOR_IMPLEMENTATION,
        coverage=MeasurementCoverage.BYTE_ONLY,
        fields=(),
    )
    candidate_component = _component(
        candidate_binding,
        kind=MeasurementComponentKind.EVALUATOR_IMPLEMENTATION,
        coverage=MeasurementCoverage.BYTE_ONLY,
        fields=(),
    )
    pair = MeasurementProtocolPair(
        baseline=_identity(baseline_binding, baseline_component),
        candidate=_identity(candidate_binding, candidate_component),
    )

    report = compare_measurement_protocols(pair, native_rule_ids=())

    assert report.state is MeasurementDriftState.WARNING
    assert report.findings[0].state is MeasurementDriftState.WARNING
    assert report.findings[0].native_rule_ids == ()
    assert report.claimci_verification_reduction is (
        ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1
    )
    assert "verdict" not in measurement_report_to_jsonable(report)
    assert "impact" not in json.dumps(measurement_report_to_jsonable(report))


def test_provider_shaped_objects_cannot_inject_protocol_or_drift_authority() -> None:
    provider_payload = {
        "semantic_protocol_id": "measurement-semantic-forged",
        "scope": "measurement_procedure",
        "state": "verified",
        "verdict": "SUPPORTED",
    }

    with pytest.raises(TypeError):
        MeasurementProtocolPair(  # type: ignore[arg-type]
            baseline=provider_payload,
            candidate=provider_payload,
        )
    with pytest.raises(TypeError):
        compare_measurement_protocols(  # type: ignore[arg-type]
            provider_payload,
            native_rule_ids=(),
        )
    with pytest.raises(TypeError, match="comparator"):
        MeasurementDriftFinding()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="comparator"):
        MeasurementDriftReport()  # type: ignore[call-arg]


def _audit_context() -> MeasurementAuditContext:
    baseline = (
        _binding(path="base/results.json", kind="results"),
        _binding(path="base/config.yaml"),
        _binding(
            path="base/train.jsonl",
            kind="dataset",
            split="train",
            digest="c" * 64,
        ),
        _binding(
            path="base/eval.jsonl",
            kind="dataset",
            split="eval",
            digest="d" * 64,
        ),
    )
    candidate = (
        _binding(
            path="candidate/results.json",
            kind="results",
            role="candidate",
            digest="e" * 64,
        ),
        _binding(
            path="candidate/config.yaml",
            role="candidate",
            digest="f" * 64,
        ),
        _binding(
            path="candidate/train.jsonl",
            kind="dataset",
            split="train",
            role="candidate",
            digest="1" * 64,
        ),
        _binding(
            path="candidate/eval.jsonl",
            kind="dataset",
            split="eval",
            role="candidate",
            digest="2" * 64,
        ),
    )
    return MeasurementAuditContext(
        baseline_source=MeasurementSourceSnapshot.from_bindings(
            repository_id="amebaleon/ClaimCI-Demo",
            head_sha="b" * 40,
            bindings=baseline,
        ),
        candidate_source=MeasurementSourceSnapshot.from_bindings(
            repository_id="amebaleon/ClaimCI-Demo",
            head_sha="b" * 40,
            bindings=candidate,
        ),
        policy_id="claimci.measurement.pilot.v1",
    )


def _drift_by_kind(result: object) -> dict[MeasurementComponentKind, object]:
    report = result.measurement_drift  # type: ignore[attr-defined]
    assert report is not None
    return {item.component_kind: item for item in report.findings}


def test_native_audit_recovers_matching_protocol_without_new_rule_ids(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_config_updates={
            "evaluation": {
                "metric": "accuracy",
                "metric_definition": "exact_match_v1",
            }
        },
        candidate_config_updates={
            "evaluation": {
                "metric": "accuracy",
                "metric_definition": "exact_match_v1",
            }
        },
    )

    result = audit_research(manifest, measurement_context=_audit_context())

    assert result.verdict is Verdict.SUPPORTED
    assert result.measurement_drift is not None
    assert result.measurement_drift.state is MeasurementDriftState.VERIFIED
    by_kind = _drift_by_kind(result)
    assert by_kind[MeasurementComponentKind.METRIC].state is MeasurementDriftState.VERIFIED
    assert (
        by_kind[MeasurementComponentKind.EVALUATION_DATASET].state
        is MeasurementDriftState.VERIFIED
    )
    assert all(
        by_kind[kind].state is MeasurementDriftState.NOT_ASSESSED
        for kind in {
            MeasurementComponentKind.EVALUATOR_IMPLEMENTATION,
            MeasurementComponentKind.PROMPT_TEMPLATE,
            MeasurementComponentKind.DECODING_PARAMETERS,
            MeasurementComponentKind.TEST_SELECTION,
            MeasurementComponentKind.RETRY_AGGREGATION,
            MeasurementComponentKind.NORMALIZATION_POSTPROCESSING,
            MeasurementComponentKind.BENCHMARK_VERSION,
        }
    )
    assert not any(item.rule_id.startswith("MEASUREMENT.") for item in result.findings)


def test_recovered_metric_config_change_references_existing_native_invalidation(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_config_updates={
            "evaluation": {
                "metric": "accuracy",
                "metric_definition": "micro_accuracy_v1",
            }
        },
        candidate_config_updates={
            "evaluation": {
                "metric": "accuracy",
                "metric_definition": "macro_accuracy_v1",
            }
        },
    )

    result = audit_research(manifest, measurement_context=_audit_context())

    native = [item for item in result.findings if item.rule_id == "CONFIG.EVALUATION_MISMATCH"]
    metric = _drift_by_kind(result)[MeasurementComponentKind.METRIC]
    assert result.verdict is Verdict.NOT_SUPPORTED
    assert len(native) == 1
    assert metric.state is MeasurementDriftState.INVALIDATES
    assert metric.native_rule_ids == ("CONFIG.EVALUATION_MISMATCH",)
    assert not any(item.rule_id.startswith("MEASUREMENT.") for item in result.findings)


def test_recovered_evaluation_population_change_references_existing_dataset_rule(
    study_factory,
) -> None:
    manifest = study_factory(
        candidate_eval=[{"id": "different", "text": "different evaluation row"}],
    )

    result = audit_research(manifest, measurement_context=_audit_context())

    native = [item for item in result.findings if item.rule_id == "DATASET.EVALUATION_MISMATCH"]
    dataset = _drift_by_kind(result)[MeasurementComponentKind.EVALUATION_DATASET]
    assert result.verdict is Verdict.NOT_SUPPORTED
    assert len(native) == 1
    assert dataset.state is MeasurementDriftState.INVALIDATES
    assert dataset.native_rule_ids == ("DATASET.EVALUATION_MISMATCH",)


def test_claim_direction_and_threshold_do_not_change_measurement_semantic_identity(
    study_factory,
) -> None:
    manifest = study_factory()
    first = audit_research(manifest, measurement_context=_audit_context())
    payload = yaml.safe_load(manifest.read_text("utf-8"))
    payload["claim"]["direction"] = "lower"
    payload["claim"]["minimum_improvement"] = 0.2
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    second = audit_research(manifest, measurement_context=_audit_context())

    assert first.measurement_drift is not None
    assert second.measurement_drift is not None
    assert (
        first.measurement_drift.baseline_semantic_protocol_id
        == second.measurement_drift.baseline_semantic_protocol_id
    )
    assert (
        first.measurement_drift.candidate_semantic_protocol_id
        == second.measurement_drift.candidate_semantic_protocol_id
    )


def test_measurement_json_is_bounded_and_omits_raw_dataset_rows(study_factory) -> None:
    private_text = "private-evaluation-row-that-must-not-be-copied"
    manifest = study_factory(
        baseline_eval=[{"id": "shared", "text": private_text}],
        candidate_eval=[{"id": "shared", "text": private_text}],
    )

    result = audit_research(manifest, measurement_context=_audit_context())
    payload = json.loads(render_json(result))

    assert payload["measurement_drift"]["state"] == "verified"
    assert private_text not in json.dumps(payload["measurement_drift"])
    assert "verdict" not in payload["measurement_drift"]


def test_absent_measurement_context_preserves_legacy_json_bytes(study_factory) -> None:
    manifest = study_factory()

    result = audit_research(manifest)
    payload = json.loads(render_json(result))

    assert result.measurement_drift is None
    assert "measurement_drift" not in payload
