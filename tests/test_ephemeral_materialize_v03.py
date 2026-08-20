"""Confined compatibility materialization for zero-configuration audits."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from claimci.analysis import (
    AdapterMatch,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    AuditClaimSpec,
    BoundedValueRepresentation,
    ClaimReference,
    Confidence,
    ConfigValue,
    DatasetSplit,
    EvidenceTraceBundle,
    TraceAuthorityClass,
    TraceCompleteness,
    TraceRecordKind,
    EphemeralAuditPlan,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    MaterializationLimits,
    MaterializationPartial,
    MaterializationUnavailable,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    RuntimeExecutionContext,
    SelectorKind,
    Sha256Digest,
    execute_ephemeral_audit,
    execute_ephemeral_audit_with_trace,
    claim_evidence_policy,
    claim_semantic_projection,
    compile_audit_claim,
    derive_ephemeral_plan_id,
    recover_scientific_claim,
    to_jsonable,
)
from claimci.analysis.adapters import extract_registered_artifact
from claimci.models import AuditResult, Direction, Verdict
from claimci.measurement import MeasurementDriftState
from claimci.report import render_json


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI-Demo")
HEAD_SHA = GitCommitSha("a" * 40)
CLAIM_ID = "claim-1"


def _provenance(path: str, source_id: str) -> FieldProvenance:
    return FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        "validated fake adapter extraction",
        RepositoryPath(path),
        source_id,
    )


def _write_artifact(
    checkout: Path,
    path: str,
    kind: ArtifactKind,
    content: bytes,
) -> ArtifactCandidate:
    destination = checkout / Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="deterministic fixture",
        relevant_claim_ids=(CLAIM_ID,),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "fixture artifact",
            RepositoryPath(path),
            f"artifact:{path}",
        ),
    )


def _normalized_results(
    artifact: ArtifactCandidate,
    role: ExperimentRole,
    values: tuple[tuple[float, int | str | None], ...],
) -> NormalizedEvidence:
    provenance = _provenance(str(artifact.path), f"adapter:{artifact.path}")
    mapping = FieldMapping(
        "metric_value",
        EvidenceSelector(SelectorKind.COLUMN, "accuracy", provenance),
        provenance,
    )
    match = AdapterMatch(
        "fixture.results",
        artifact.path,
        Confidence(0.98),
        (mapping,),
        (provenance,),
    )
    return NormalizedEvidence(
        f"evidence-{str(artifact.path).replace('/', '-')}",
        artifact,
        match,
        tuple(
            NormalizedObservation(
                provenance=provenance,
                metric_name="accuracy",
                metric_value=value,
                run_id=f"run-{index}",
                seed=seed,
                experiment_role=role,
            )
            for index, (value, seed) in enumerate(values, start=1)
        ),
    )


def _normalized_config(
    artifact: ArtifactCandidate,
    role: ExperimentRole,
    values: dict[str, object],
) -> NormalizedEvidence:
    provenance = _provenance(str(artifact.path), f"adapter:{artifact.path}")
    mappings = tuple(
        FieldMapping(
            f"config.{key}",
            EvidenceSelector(SelectorKind.DOTTED_PATH, key, provenance),
            provenance,
        )
        for key in sorted(values)
    )
    return NormalizedEvidence(
        f"evidence-{str(artifact.path).replace('/', '-')}",
        artifact,
        AdapterMatch(
            "fixture.config",
            artifact.path,
            Confidence(0.98),
            mappings,
            (provenance,),
        ),
        (
            NormalizedObservation(
                provenance=provenance,
                experiment_role=role,
                config_values=tuple(
                    ConfigValue(key, value, provenance)
                    for key, value in sorted(values.items())
                ),
            ),
        ),
    )


def _normalized_dataset(
    artifact: ArtifactCandidate,
    content: bytes,
) -> NormalizedEvidence:
    evidence = extract_registered_artifact(PassiveArtifact(artifact, content))
    assert evidence is not None
    assert evidence.adapter_match.adapter_id == "claimci-jsonl-dataset-v1"
    return evidence


def _binding(
    evidence: NormalizedEvidence,
    role: ExperimentRole,
    *,
    dataset_split: DatasetSplit | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        evidence.artifact.path,
        evidence.artifact.kind,
        role,
        evidence.adapter_match.adapter_id,
        evidence.adapter_match.mappings,
        evidence.adapter_match.match_evidence[0],
        dataset_split,
    )


def _plan_fixture(
    tmp_path: Path,
    *,
    candidate_seed: int | str | None = 11,
    partial_config: bool = False,
    candidate_train_content: bytes = b'{"id":"shared"}\n{"id":"candidate-train"}\n',
) -> tuple[EphemeralAuditPlan, RuntimeExecutionContext, Path, Path]:
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    checkout.mkdir()
    scratch.mkdir()
    baseline_results = _normalized_results(
        _write_artifact(
            checkout,
            "results/baseline.csv",
            ArtifactKind.RESULTS,
            b"accuracy,seed\n0.5,1\n0.6,2\n0.7,3\n",
        ),
        ExperimentRole.BASELINE,
        ((0.5, 1), (0.6, 2), (0.7, 3)),
    )
    candidate_results = _normalized_results(
        _write_artifact(
            checkout,
            "results/candidate.json",
            ArtifactKind.RESULTS,
            b'{"accuracy":0.9,"seed":11}\n',
        ),
        ExperimentRole.CANDIDATE,
        ((0.9, candidate_seed),),
    )
    base_config_values: dict[str, object] = {
        "training_steps": 100,
        "batch_size": 8,
    }
    candidate_config_values: dict[str, object] = {
        "training_steps": 300,
        "batch_size": 8,
    }
    if not partial_config:
        common = {
            "epochs": 2,
            "learning_rate": 0.001,
            "model.name": "demo",
            "model.version": "v1",
            "dataset.identifier": "train",
            "dataset.version": "v1",
            "evaluation.dataset_identifier": "eval",
            "evaluation.dataset_version": "v1",
            "evaluation.split": "test",
        }
        base_config_values.update(common)
        candidate_config_values.update(common)
    baseline_config = _normalized_config(
        _write_artifact(
            checkout,
            "configs/baseline.yaml",
            ArtifactKind.CONFIG,
            b"training_steps: 100\n",
        ),
        ExperimentRole.BASELINE,
        base_config_values,
    )
    candidate_config = _normalized_config(
        _write_artifact(
            checkout,
            "configs/candidate.json",
            ArtifactKind.CONFIG,
            b'{"training_steps":300}\n',
        ),
        ExperimentRole.CANDIDATE,
        candidate_config_values,
    )
    baseline_train_content = b'{"id":"baseline-train"}\n'
    baseline_eval_content = b'{"id":"shared"}\n'
    candidate_eval_content = b'{"id":"shared"}\n'
    datasets = (
        _normalized_dataset(
            _write_artifact(
                checkout,
                "data/baseline-train.jsonl",
                ArtifactKind.DATASET,
                baseline_train_content,
            ),
            baseline_train_content,
        ),
        _normalized_dataset(
            _write_artifact(
                checkout,
                "data/baseline-eval.jsonl",
                ArtifactKind.DATASET,
                baseline_eval_content,
            ),
            baseline_eval_content,
        ),
        _normalized_dataset(
            _write_artifact(
                checkout,
                "data/candidate-train.jsonl",
                ArtifactKind.DATASET,
                candidate_train_content,
            ),
            candidate_train_content,
        ),
        _normalized_dataset(
            _write_artifact(
                checkout,
                "data/candidate-eval.jsonl",
                ArtifactKind.DATASET,
                candidate_eval_content,
            ),
            candidate_eval_content,
        ),
    )
    baseline = (baseline_results, baseline_config, *datasets[:2])
    candidate = (candidate_results, candidate_config, *datasets[2:])
    bindings = (
        _binding(baseline_results, ExperimentRole.BASELINE),
        _binding(baseline_config, ExperimentRole.BASELINE),
        _binding(
            datasets[0],
            ExperimentRole.BASELINE,
            dataset_split=DatasetSplit.TRAIN,
        ),
        _binding(
            datasets[1],
            ExperimentRole.BASELINE,
            dataset_split=DatasetSplit.EVAL,
        ),
        _binding(candidate_results, ExperimentRole.CANDIDATE),
        _binding(candidate_config, ExperimentRole.CANDIDATE),
        _binding(
            datasets[2],
            ExperimentRole.CANDIDATE,
            dataset_split=DatasetSplit.TRAIN,
        ),
        _binding(
            datasets[3],
            ExperimentRole.CANDIDATE,
            dataset_split=DatasetSplit.EVAL,
        ),
    )
    mapping = MappingCandidate(
        "mapping-fixture",
        bindings,
        Confidence(0.98),
        MappingTrust.INFERRED,
        FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "deterministic fixture mapping",
            source_id="mapping-fixture",
        ),
    )
    claim_provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "validated claim fixture",
        RepositoryPath("CLAIM.md"),
        "claim-source",
    )
    audit_claim = AuditClaimSpec(
        CLAIM_ID,
        "accuracy",
        Direction.HIGHER,
        0.05,
        claim_provenance,
        claim_provenance,
        claim_provenance,
    )
    plan = EphemeralAuditPlan(
        plan_id="plan-" + "1" * 24,
        repository=REPOSITORY,
        pr_number=7,
        head_sha=HEAD_SHA,
        claim=ClaimReference(
            CLAIM_ID,
            "Accuracy improved by at least 0.05.",
            RepositoryPath("CLAIM.md"),
            Confidence(0.98),
            claim_provenance,
        ),
        baseline_evidence=baseline,
        candidate_evidence=candidate,
        mapping_provenance=(mapping.provenance,),
        missing_evidence=(),
        confidence=Confidence(0.98),
        audit_claim=audit_claim,
        selected_mapping=mapping,
    )
    plan = dataclasses.replace(plan, plan_id=derive_ephemeral_plan_id(plan))
    runtime = RuntimeExecutionContext(
        repository=REPOSITORY,
        checkout_root=checkout,
        head_sha=HEAD_SHA,
        scratch_root=scratch,
        limits=MaterializationLimits(
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
        ),
    )
    return plan, runtime, checkout, scratch


def test_traced_execution_preserves_the_exact_audit_result_and_rendering(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)

    traced = execute_ephemeral_audit_with_trace(plan, runtime)
    compatibility = execute_ephemeral_audit(plan, runtime)

    assert type(traced.audit_result) is AuditResult
    assert traced.audit_result == compatibility
    assert render_json(traced.audit_result) == render_json(compatibility)
    assert traced.trace.completeness is TraceCompleteness.COMPLETE
    assert traced.trace.head_sha == plan.head_sha
    assert traced.trace.deterministic_authority.verdict is compatibility.verdict
    assert not tuple(scratch.iterdir())


def test_trace_retains_held_out_semantics_without_changing_audit_or_plan_identity(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    reference = dataclasses.replace(
        plan.claim,
        text=(
            "On the held-out test set, accuracy improved by at least 0.05."
        ),
    )
    scientific_claim = recover_scientific_claim(reference)
    assert scientific_claim is not None
    audit_claim = compile_audit_claim(scientific_claim)
    constrained = dataclasses.replace(
        plan,
        claim=reference,
        audit_claim=audit_claim,
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )
    constrained = dataclasses.replace(
        constrained,
        plan_id=derive_ephemeral_plan_id(constrained),
    )

    assert audit_claim == plan.audit_claim
    assert constrained.plan_id == plan.plan_id

    trace = execute_ephemeral_audit_with_trace(constrained, runtime).trace
    claim_entries = tuple(
        entry
        for entry in trace.entries
        if entry.record_kind is TraceRecordKind.NATURAL_LANGUAGE_CLAIM
    )
    assert len(claim_entries) == 1
    assert claim_entries[0].authority is TraceAuthorityClass.NON_AUTHORITATIVE_INPUT
    assert claim_entries[0].normalized_value == BoundedValueRepresentation.from_value(
        claim_semantic_projection(scientific_claim)
    )
    assert trace.deterministic_authority.verdict is Verdict.NOT_SUPPORTED


def _with_scientific_claim(plan: EphemeralAuditPlan) -> EphemeralAuditPlan:
    reference = dataclasses.replace(
        plan.claim,
        text="Accuracy improved from 0.60 to 0.90 by at least 0.05.",
    )
    scientific_claim = recover_scientific_claim(reference)
    assert scientific_claim is not None
    changed = dataclasses.replace(
        plan,
        claim=reference,
        audit_claim=compile_audit_claim(scientific_claim),
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )
    return dataclasses.replace(changed, plan_id=derive_ephemeral_plan_id(changed))


def test_scientific_exact_head_execution_attaches_bounded_measurement_report(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    plan = _with_scientific_claim(plan)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    report = execution.audit_result.measurement_drift
    assert report is not None
    assert report.state in {
        MeasurementDriftState.VERIFIED,
        MeasurementDriftState.INVALIDATES,
    }
    assert report.baseline_semantic_protocol_id.startswith("measurement-semantic-")
    assert report.baseline_source_snapshot_id.startswith("measurement-source-")
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    assert not any(
        finding.rule_id.startswith("MEASUREMENT.")
        for finding in execution.audit_result.findings
    )


def test_measurement_trace_failure_never_changes_native_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    plan = _with_scientific_claim(plan)

    def fail_trace(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private trace construction failure")

    monkeypatch.setattr(materialize, "_build_evidence_trace", fail_trace)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.NOT_SUPPORTED
    assert execution.audit_result.measurement_drift is not None
    assert execution.trace.completeness is TraceCompleteness.UNAVAILABLE
    assert execution.trace.deterministic_authority.verdict is Verdict.NOT_SUPPORTED


def test_comment_only_config_change_changes_source_not_semantic_identity(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_plan, first_runtime, _checkout, _scratch = _plan_fixture(first_root)
    second_plan, second_runtime, second_checkout, _scratch = _plan_fixture(second_root)
    first_plan = _with_scientific_claim(first_plan)
    second_plan = _with_scientific_claim(second_plan)
    changed_path = second_checkout / "configs" / "baseline.yaml"
    changed_content = changed_path.read_bytes() + b"# comment-only change\n"
    changed_path.write_bytes(changed_content)
    changed_baseline = tuple(
        dataclasses.replace(
            evidence,
            artifact=dataclasses.replace(
                evidence.artifact,
                sha256=Sha256Digest(hashlib.sha256(changed_content).hexdigest()),
                size=len(changed_content),
            ),
        )
        if evidence.artifact.path == RepositoryPath("configs/baseline.yaml")
        else evidence
        for evidence in second_plan.baseline_evidence
    )
    second_plan = dataclasses.replace(second_plan, baseline_evidence=changed_baseline)
    second_plan = dataclasses.replace(
        second_plan,
        plan_id=derive_ephemeral_plan_id(second_plan),
    )

    first = execute_ephemeral_audit(first_plan, first_runtime)
    second = execute_ephemeral_audit(second_plan, second_runtime)

    assert first.measurement_drift is not None
    assert second.measurement_drift is not None
    assert (
        first.measurement_drift.baseline_semantic_protocol_id
        == second.measurement_drift.baseline_semantic_protocol_id
    )
    assert (
        first.measurement_drift.baseline_source_snapshot_id
        != second.measurement_drift.baseline_source_snapshot_id
    )
    assert first.verdict is second.verdict


def test_trace_identity_does_not_depend_on_ephemeral_scratch_paths(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_root.mkdir()
    second_root.mkdir()
    first_plan, first_runtime, _checkout, _scratch = _plan_fixture(first_root)
    second_plan, second_runtime, _checkout, _scratch = _plan_fixture(second_root)

    first = execute_ephemeral_audit_with_trace(first_plan, first_runtime)
    second = execute_ephemeral_audit_with_trace(second_plan, second_runtime)

    assert first.audit_result.verdict is second.audit_result.verdict
    assert first.trace == second.trace


def test_complete_trace_links_exact_passive_sources_to_actual_audit_rules(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)
    trace = execution.trace
    passive = tuple(
        item
        for item in trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
    )

    assert len(passive) == 8
    baseline = next(
        item
        for item in passive
        if item.artifact_path == RepositoryPath("results/baseline.csv")
    )
    planned = next(
        item
        for item in plan.baseline_evidence
        if item.artifact.path == baseline.artifact_path
    )
    assert baseline.artifact_sha256 == planned.artifact.sha256
    assert baseline.artifact_size == planned.artifact.size
    assert baseline.adapter_id == "fixture.results"
    assert baseline.adapter_version is None
    assert baseline.role is ExperimentRole.BASELINE
    assert tuple(item.expression for item in baseline.selectors) == ("accuracy",)
    assert baseline.source_value is not None
    assert baseline.source_value.direct_values == (0.5, 1, 0.6, 2, 0.7, 3)
    assert baseline.normalized_value == baseline.source_value
    assert "RESULT.RECOMPUTED" in baseline.consumer_rule_ids
    assert "SEED.IMBALANCE" in baseline.consumer_rule_ids

    dataset = next(
        item
        for item in passive
        if item.artifact_path == RepositoryPath("data/candidate-train.jsonl")
    )
    assert dataset.adapter_id == "claimci-jsonl-dataset-v1"
    assert dataset.adapter_version == "v1"
    assert dataset.dataset_split is DatasetSplit.TRAIN
    assert dataset.source_value is not None
    assert dataset.source_value.direct_values == ()

    consumed = {
        rule_id
        for item in trace.entries
        for rule_id in item.consumer_rule_ids
    }
    actual = {item.rule_id for item in execution.audit_result.findings}
    assert consumed == actual


def test_trace_does_not_persist_repository_prose_or_string_config_values(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)
    payload = json.dumps(to_jsonable(execution.trace), sort_keys=True)

    assert plan.claim.text not in payload
    assert '"demo"' not in payload
    assert '"model.name": "demo"' not in payload
    assert '"v1"' in payload  # adapter version metadata is safe identity.
    assert "0.9" in payload


def test_trace_failure_never_changes_or_discards_the_audit_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    expected = execute_ephemeral_audit(plan, runtime)

    def fail_trace(*_args: object, **_kwargs: object) -> EvidenceTraceBundle:
        raise RuntimeError("trace-only failure")

    monkeypatch.setattr(
        "claimci.analysis.materialize._build_evidence_trace",
        fail_trace,
    )
    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result == expected
    assert execution.trace.completeness is TraceCompleteness.UNAVAILABLE
    assert execution.trace.deterministic_authority.verdict is expected.verdict


def test_provider_claim_hint_remains_non_authoritative_in_complete_trace(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    provider = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "validated semantic proposal",
        RepositoryPath("CLAIM.md"),
        "semantic-call-1",
    )
    assert plan.audit_claim is not None
    changed = dataclasses.replace(
        plan,
        claim=dataclasses.replace(plan.claim, provenance=provider),
        audit_claim=dataclasses.replace(
            plan.audit_claim,
            metric_provenance=provider,
            direction_provenance=provider,
            threshold_provenance=provider,
        ),
    )
    changed = dataclasses.replace(changed, plan_id=derive_ephemeral_plan_id(changed))

    trace = execute_ephemeral_audit_with_trace(changed, runtime).trace
    proposals = tuple(
        item
        for item in trace.entries
        if item.record_kind is TraceRecordKind.LLM_SEMANTIC_PROPOSAL
    )

    assert len(proposals) == 1
    assert proposals[0].authority is TraceAuthorityClass.NON_AUTHORITATIVE_INPUT
    assert all(
        item.provenance_kind is ProvenanceKind.ADAPTER_EXTRACTION
        for item in trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
    )
    assert trace.deterministic_authority.authority.value == "deterministic"


def test_provider_confirmation_is_traced_without_retyping_the_claim_or_evidence(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    provider = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "validated semantic preflight confirmation",
        source_id="semantic-call-1",
    )
    changed = dataclasses.replace(plan, semantic_proposal_provenance=provider)

    trace = execute_ephemeral_audit_with_trace(changed, runtime).trace
    proposals = tuple(
        item
        for item in trace.entries
        if item.record_kind is TraceRecordKind.LLM_SEMANTIC_PROPOSAL
    )

    assert len(proposals) == 1
    assert proposals[0].provenance_kind is ProvenanceKind.PROVIDER_PROPOSAL
    assert proposals[0].authority is TraceAuthorityClass.NON_AUTHORITATIVE_INPUT
    assert all(
        item.provenance_kind is ProvenanceKind.ADAPTER_EXTRACTION
        for item in trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
    )
    assert trace.deterministic_authority.authority.value == "deterministic"


def test_repo_mapping_approval_is_binding_trust_not_verdict_authority(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    assert type(plan.selected_mapping) is MappingCandidate
    approved = RepoMapping.approve(
        plan.repository,
        plan.selected_mapping,
        approved_by="pilot-user-1",
    )
    changed = dataclasses.replace(
        plan,
        selected_mapping=approved,
        mapping_provenance=(approved.approval_provenance,),
    )
    changed = dataclasses.replace(changed, plan_id=derive_ephemeral_plan_id(changed))

    trace = execute_ephemeral_audit_with_trace(changed, runtime).trace
    approvals = tuple(
        item
        for item in trace.entries
        if item.record_kind is TraceRecordKind.APPROVED_MAPPING
    )

    assert len(approvals) == 1
    assert approvals[0].authority is TraceAuthorityClass.EXPLICIT_MAPPING_APPROVAL
    assert approvals[0].consumer_rule_ids == ()
    assert trace.deterministic_authority.authority.value == "deterministic"


def test_runtime_context_rejects_overlapping_or_symlinked_roots(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    with pytest.raises((TypeError, ValueError), match="overlap|scratch|checkout"):
        RuntimeExecutionContext(
            REPOSITORY,
            checkout,
            HEAD_SHA,
            checkout / "scratch",
        )

    if hasattr(os, "symlink"):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable on this platform")
        with pytest.raises((TypeError, ValueError), match="symlink|root"):
            RuntimeExecutionContext(
                REPOSITORY,
                link,
                HEAD_SHA,
                tmp_path / "scratch",
            )


def test_runtime_context_rejects_subclassed_security_limits(tmp_path: Path) -> None:
    class LimitsSubclass(MaterializationLimits):
        pass

    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    checkout.mkdir()
    scratch.mkdir()

    with pytest.raises(TypeError, match="limits|MaterializationLimits"):
        RuntimeExecutionContext(
            REPOSITORY,
            checkout,
            HEAD_SHA,
            scratch,
            LimitsSubclass(),
        )


def test_runtime_scratch_root_substitution_is_unavailable(tmp_path: Path) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    original = tmp_path / "original-scratch"
    scratch.rename(original)
    scratch.mkdir()
    marker = scratch / "replacement.txt"
    marker.write_text("do not touch", encoding="utf-8")

    with pytest.raises(MaterializationUnavailable, match="scratch|root|identity"):
        execute_ephemeral_audit(plan, runtime)

    assert marker.read_text("utf-8") == "do not touch"
    assert not (scratch / plan.plan_id).exists()


def test_each_customer_artifact_is_opened_once_then_materialized_from_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    opened: list[Path] = []
    original = materialize._open_customer_artifact

    def recording_open(path: Path) -> int:
        opened.append(path)
        return original(path)

    monkeypatch.setattr(materialize, "_open_customer_artifact", recording_open)

    result = execute_ephemeral_audit(plan, runtime)

    assert type(result) is AuditResult
    assert result.verdict is Verdict.NOT_SUPPORTED
    assert len(opened) == len(set(opened)) == 8
    assert not (scratch / plan.plan_id).exists()


def test_changed_artifact_hash_fails_unavailable_and_cleans_plan_tree(
    tmp_path: Path,
) -> None:
    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    (checkout / "results" / "candidate.json").write_bytes(b"changed\n")

    with pytest.raises(MaterializationUnavailable, match="hash|size|snapshot"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_symlink_substitution_is_unavailable_without_following_target(
    tmp_path: Path,
) -> None:
    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    source = checkout / "results" / "candidate.json"
    target = checkout / "results" / "target.json"
    target.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(MaterializationUnavailable, match="symlink|snapshot|regular"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_open_time_redirection_is_rejected_by_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A link-swap-equivalent open cannot substitute another regular file."""

    import claimci.analysis.materialize as materialize

    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    victim = checkout / "results" / "candidate.json"
    outside = tmp_path / "outside-candidate.json"
    outside.write_bytes(victim.read_bytes())
    original = materialize._open_customer_artifact

    def redirected_open(path: Path) -> int:
        return original(outside if path == victim else path)

    monkeypatch.setattr(materialize, "_open_customer_artifact", redirected_open)

    with pytest.raises(MaterializationUnavailable, match="identity|snapshot"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_low_confidence_manifest_hint_cannot_bypass_runtime_mapping_gate(
    tmp_path: Path,
) -> None:
    """Materialization independently enforces the planner's hosted trust floor."""

    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert type(plan.selected_mapping) is MappingCandidate
    manifest_provenance = FieldProvenance(
        ProvenanceKind.MANIFEST_HINT,
        "unapproved low-confidence manifest hint",
        RepositoryPath("research.yaml"),
        "manifest-hint-hostile",
    )
    low_confidence = dataclasses.replace(
        plan.selected_mapping,
        confidence=Confidence(0.01),
        trust=MappingTrust.MANIFEST_HINT,
        provenance=manifest_provenance,
    )
    hostile = dataclasses.replace(
        plan,
        selected_mapping=low_confidence,
        mapping_provenance=(manifest_provenance,),
    )
    hostile = dataclasses.replace(
        hostile,
        plan_id=derive_ephemeral_plan_id(hostile),
    )

    with pytest.raises(MaterializationUnavailable, match="confidence|trust|mapping"):
        execute_ephemeral_audit(hostile, runtime)

    assert not (scratch / hostile.plan_id).exists()


def test_native_conversion_contains_only_normalized_runs_config_and_passive_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize
    from claimci.audit import audit_research as real_audit_research

    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    observed: dict[str, object] = {}

    def inspect_then_audit(path: Path, *, artifact_root: Path) -> AuditResult:
        observed["baseline_results"] = json.loads(
            (artifact_root / "baseline" / "results.json").read_text("utf-8")
        )
        observed["candidate_results"] = json.loads(
            (artifact_root / "candidate" / "results.json").read_text("utf-8")
        )
        observed["baseline_config"] = (
            artifact_root / "baseline" / "config.yaml"
        ).read_text("utf-8")
        observed["datasets"] = {
            f"{role}/{split}": (
                artifact_root / role / f"{split}.jsonl"
            ).read_bytes()
            for role in ("baseline", "candidate")
            for split in ("train", "eval")
        }
        observed["manifest"] = path.read_text("utf-8")
        return real_audit_research(path, artifact_root=artifact_root)

    monkeypatch.setattr(materialize, "audit_research", inspect_then_audit)

    result = execute_ephemeral_audit(plan, runtime)

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert observed["baseline_results"] == {
        "runs": [
            {"accuracy": 0.5, "seed": 1},
            {"accuracy": 0.6, "seed": 2},
            {"accuracy": 0.7, "seed": 3},
        ]
    }
    assert observed["candidate_results"] == {
        "runs": [{"accuracy": 0.9, "seed": 11}]
    }
    assert "summary" not in observed["manifest"]
    assert "model:" in observed["baseline_config"]
    assert "  name: demo" in observed["baseline_config"]
    assert observed["datasets"] == {
        f"{role}/{split}": (
            checkout / "data" / f"{role}-{split}.jsonl"
        ).read_bytes()
        for role in ("baseline", "candidate")
        for split in ("train", "eval")
    }
    assert not (scratch / plan.plan_id).exists()


def test_one_passive_dataset_can_fill_both_splits_and_is_captured_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    baseline_dataset_bindings = tuple(
        item
        for item in plan.selected_mapping.bindings
        if item.kind is ArtifactKind.DATASET
        and item.role is ExperimentRole.BASELINE
    )
    train_binding = next(
        item
        for item in baseline_dataset_bindings
        if item.dataset_split is DatasetSplit.TRAIN
    )
    eval_binding = next(
        item
        for item in baseline_dataset_bindings
        if item.dataset_split is DatasetSplit.EVAL
    )
    train_evidence = next(
        item
        for item in plan.baseline_evidence
        if item.artifact.path == train_binding.path
    )
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            item
            for item in plan.selected_mapping.bindings
            if item is not eval_binding
        )
        + (dataclasses.replace(train_binding, dataset_split=DatasetSplit.EVAL),),
    )
    changed_plan = dataclasses.replace(
        plan,
        baseline_evidence=tuple(
            item
            for item in plan.baseline_evidence
            if item.artifact.path != eval_binding.path
        ),
        selected_mapping=changed_mapping,
        mapping_provenance=(changed_mapping.provenance,),
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )
    original_capture = materialize._capture_artifact
    captures: list[RepositoryPath] = []

    def count_capture(*args: object, **kwargs: object) -> PassiveArtifact:
        candidate = args[0]
        assert isinstance(candidate, ArtifactCandidate)
        captures.append(candidate.path)
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(materialize, "_capture_artifact", count_capture)

    result = execute_ephemeral_audit(changed_plan, runtime)

    assert captures.count(train_evidence.artifact.path) == 1
    assert any(
        item.rule_id == "DATASET.EXACT_LEAKAGE"
        and item.evidence["experiment"] == "baseline"
        for item in result.findings
    )
    assert not (scratch / changed_plan.plan_id).exists()


def test_dataset_normalized_split_injection_fails_fresh_adapter_revalidation(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    split_by_path = {
        item.path: item.dataset_split
        for item in plan.selected_mapping.bindings
        if item.kind is ArtifactKind.DATASET
    }

    def inject_split(item: NormalizedEvidence) -> NormalizedEvidence:
        if item.artifact.kind is not ArtifactKind.DATASET:
            return item
        split = split_by_path[item.artifact.path]
        assert split is not None
        observation = item.observations[0]
        reference = observation.dataset_references[0]
        return dataclasses.replace(
            item,
            observations=(
                dataclasses.replace(
                    observation,
                    dataset_references=(
                        dataclasses.replace(reference, split=split.value),
                    ),
                ),
            ),
        )

    changed_plan = dataclasses.replace(
        plan,
        baseline_evidence=tuple(inject_split(item) for item in plan.baseline_evidence),
        candidate_evidence=tuple(inject_split(item) for item in plan.candidate_evidence),
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationUnavailable, match="adapter|identity|normalized"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / changed_plan.plan_id).exists()


def test_dataset_adapter_is_revalidated_once_per_unique_captured_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.adapters as adapters

    plan, runtime, _checkout, _scratch = _plan_fixture(tmp_path)
    original = adapters.get_adapter
    selected: list[str] = []

    def record(adapter_id: str) -> object:
        selected.append(adapter_id)
        return original(adapter_id)

    monkeypatch.setattr(adapters, "get_adapter", record)

    execute_ephemeral_audit(plan, runtime)

    assert selected == ["claimci-jsonl-dataset-v1"] * 4


def test_dataset_adapter_identity_drift_fails_closed_before_audit(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    dataset = next(
        item
        for item in plan.candidate_evidence
        if item.artifact.kind is ArtifactKind.DATASET
    )
    changed_dataset = dataclasses.replace(
        dataset,
        adapter_match=dataclasses.replace(
            dataset.adapter_match,
            adapter_id="fixture.dataset",
        ),
    )
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            dataclasses.replace(item, adapter_id="fixture.dataset")
            if item.path == dataset.artifact.path
            else item
            for item in plan.selected_mapping.bindings
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        candidate_evidence=tuple(
            changed_dataset if item is dataset else item
            for item in plan.candidate_evidence
        ),
        selected_mapping=changed_mapping,
        mapping_provenance=(changed_mapping.provenance,),
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationUnavailable, match="adapter"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / changed_plan.plan_id).exists()


def test_dataset_selector_injection_fails_closed_before_audit(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    dataset = next(
        item
        for item in plan.baseline_evidence
        if item.artifact.kind is ArtifactKind.DATASET
    )
    provenance = dataset.adapter_match.match_evidence[0]
    injected = FieldMapping(
        "dataset.path",
        EvidenceSelector(
            SelectorKind.DOTTED_PATH,
            "dataset.path",
            provenance,
        ),
        provenance,
    )
    changed_dataset = dataclasses.replace(
        dataset,
        adapter_match=dataclasses.replace(
            dataset.adapter_match,
            mappings=(injected,),
        ),
    )
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            dataclasses.replace(item, mappings=(injected,))
            if item.path == dataset.artifact.path
            else item
            for item in plan.selected_mapping.bindings
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        baseline_evidence=tuple(
            changed_dataset if item is dataset else item
            for item in plan.baseline_evidence
        ),
        selected_mapping=changed_mapping,
        mapping_provenance=(changed_mapping.provenance,),
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationUnavailable, match="selector"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / changed_plan.plan_id).exists()


def test_duplicate_dataset_split_binding_fails_closed_before_audit(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    candidate_eval = next(
        item
        for item in plan.selected_mapping.bindings
        if item.kind is ArtifactKind.DATASET
        and item.role is ExperimentRole.CANDIDATE
        and item.dataset_split is DatasetSplit.EVAL
    )
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            dataclasses.replace(item, dataset_split=DatasetSplit.TRAIN)
            if item is candidate_eval
            else item
            for item in plan.selected_mapping.bindings
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        selected_mapping=changed_mapping,
        mapping_provenance=(changed_mapping.provenance,),
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationUnavailable, match="duplicate split|train and one eval"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / changed_plan.plan_id).exists()


def test_malformed_jsonl_is_left_for_the_existing_deterministic_audit(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(
        tmp_path,
        candidate_train_content=b"not JSON\n",
    )

    result = execute_ephemeral_audit(plan, runtime)

    assert any(item.rule_id == "DATASET.INVALID" for item in result.findings)


def test_genuinely_absent_seed_remains_absent(tmp_path: Path) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, _scratch = _plan_fixture(
        tmp_path,
        candidate_seed=None,
    )
    original = materialize.audit_research
    observed: dict[str, object] = {}

    def inspect_then_audit(path: Path, *, artifact_root: Path) -> AuditResult:
        observed.update(
            json.loads(
                (artifact_root / "candidate" / "results.json").read_text("utf-8")
            )
        )
        return original(path, artifact_root=artifact_root)

    from pytest import MonkeyPatch

    patcher = MonkeyPatch()
    patcher.setattr(materialize, "audit_research", inspect_then_audit)
    try:
        execute_ephemeral_audit(plan, runtime)
    finally:
        patcher.undo()

    assert observed == {"runs": [{"accuracy": 0.9}]}


def test_present_non_integer_seed_is_partial_and_not_coerced_to_missing(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(
        tmp_path,
        candidate_seed="seed-eleven",
    )

    with pytest.raises(MaterializationPartial, match="seed|represent"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_partial_config_reaches_existing_audit_missing_fields_policy(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, _scratch = _plan_fixture(
        tmp_path,
        partial_config=True,
    )

    result = execute_ephemeral_audit(plan, runtime)

    missing = tuple(
        item for item in result.findings if item.rule_id == "CONFIG.MISSING_FIELDS"
    )
    assert len(missing) == 1


def test_duplicate_config_key_across_observations_is_partial(tmp_path: Path) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    config = next(
        item
        for item in plan.candidate_evidence
        if item.artifact.kind is ArtifactKind.CONFIG
    )
    provenance = config.observations[0].provenance
    duplicate_observation = NormalizedObservation(
        provenance=provenance,
        experiment_role=ExperimentRole.CANDIDATE,
        config_values=(ConfigValue("training_steps", 999, provenance),),
    )
    changed_config = dataclasses.replace(
        config,
        observations=(*config.observations, duplicate_observation),
    )
    changed_candidate = tuple(
        changed_config if item is config else item for item in plan.candidate_evidence
    )
    changed_plan = dataclasses.replace(plan, candidate_evidence=changed_candidate)

    with pytest.raises(MaterializationPartial, match="duplicate"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_missing_whole_artifact_category_is_partial_without_placeholder(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    config = next(
        item
        for item in plan.candidate_evidence
        if item.artifact.kind is ArtifactKind.CONFIG
    )
    assert isinstance(plan.selected_mapping, MappingCandidate)
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            item
            for item in plan.selected_mapping.bindings
            if not (
                item.path == config.artifact.path
                and item.kind is ArtifactKind.CONFIG
            )
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        candidate_evidence=tuple(
            item for item in plan.candidate_evidence if item is not config
        ),
        selected_mapping=changed_mapping,
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationPartial, match="missing|category"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_existing_plan_tree_collision_is_unavailable_and_not_deleted(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    existing = scratch / plan.plan_id
    existing.mkdir()
    marker = existing / "caller-owned.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(MaterializationUnavailable, match="exclusive|exists|collision"):
        execute_ephemeral_audit(plan, runtime)

    assert marker.read_text("utf-8") == "keep"


@pytest.mark.parametrize("failure", ["write", "audit"])
def test_controlled_internal_failures_clean_only_created_plan_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    sibling = scratch / "caller-owned"
    sibling.mkdir()
    marker = sibling / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    if failure == "write":
        original = materialize._exclusive_text
        calls = 0

        def fail_second_write(path: Path, content: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected write failure")
            original(path, content)

        monkeypatch.setattr(materialize, "_exclusive_text", fail_second_write)
    else:
        monkeypatch.setattr(
            materialize,
            "audit_research",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected audit failure")
            ),
        )

    with pytest.raises(MaterializationUnavailable, match="failed closed"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()
    assert marker.read_text("utf-8") == "keep"
    assert scratch.exists()


def test_stale_runtime_head_is_unavailable_before_materialization(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    stale = dataclasses.replace(runtime, head_sha=GitCommitSha("b" * 40))

    with pytest.raises(MaterializationUnavailable, match="head|snapshot"):
        execute_ephemeral_audit(plan, stale)

    assert not (scratch / plan.plan_id).exists()


def test_provider_originated_normalized_observation_is_unavailable(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    results = next(
        item
        for item in plan.candidate_evidence
        if item.artifact.kind is ArtifactKind.RESULTS
    )
    provider = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "provider supplied an untrusted result value",
        results.artifact.path,
        "provider-call-1",
    )
    changed_results = dataclasses.replace(
        results,
        observations=(
            dataclasses.replace(results.observations[0], provenance=provider),
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        candidate_evidence=tuple(
            changed_results if item is results else item
            for item in plan.candidate_evidence
        ),
    )

    with pytest.raises(MaterializationUnavailable, match="provider|adapter-validated"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_non_jsonl_dataset_representation_is_partial(tmp_path: Path) -> None:
    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    train_binding = next(
        item
        for item in plan.selected_mapping.bindings
        if item.kind is ArtifactKind.DATASET
        and item.role is ExperimentRole.CANDIDATE
        and item.dataset_split is DatasetSplit.TRAIN
    )
    dataset = next(
        item
        for item in plan.candidate_evidence
        if item.artifact.kind is ArtifactKind.DATASET
        and item.artifact.path == train_binding.path
    )
    old_source = checkout / Path(str(dataset.artifact.path))
    new_path = RepositoryPath("data/candidate-train.csv")
    new_source = checkout / Path(str(new_path))
    new_source.write_bytes(old_source.read_bytes())
    old_source.unlink()
    changed_artifact = dataclasses.replace(dataset.artifact, path=new_path)
    changed_reference = dataclasses.replace(
        dataset.observations[0].dataset_references[0],
        path=new_path,
    )
    changed_observation = dataclasses.replace(
        dataset.observations[0],
        dataset_references=(changed_reference,),
    )
    changed_match = dataclasses.replace(dataset.adapter_match, path=new_path)
    changed_dataset = dataclasses.replace(
        dataset,
        artifact=changed_artifact,
        adapter_match=changed_match,
        observations=(changed_observation,),
    )
    changed_mapping = dataclasses.replace(
        plan.selected_mapping,
        bindings=tuple(
            dataclasses.replace(item, path=new_path)
            if item.path == dataset.artifact.path
            else item
            for item in plan.selected_mapping.bindings
        ),
    )
    changed_plan = dataclasses.replace(
        plan,
        candidate_evidence=tuple(
            changed_dataset if item is dataset else item
            for item in plan.candidate_evidence
        ),
        selected_mapping=changed_mapping,
    )
    changed_plan = dataclasses.replace(
        changed_plan,
        plan_id=derive_ephemeral_plan_id(changed_plan),
    )

    with pytest.raises(MaterializationPartial, match="JSONL|dataset"):
        execute_ephemeral_audit(changed_plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_individual_native_files_use_exclusive_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    original = materialize._exclusive_text
    injected = False

    def collide_once(path: Path, content: str) -> None:
        nonlocal injected
        if not injected:
            injected = True
            path.write_text("caller collision", encoding="utf-8")
        original(path, content)

    monkeypatch.setattr(materialize, "_exclusive_text", collide_once)

    with pytest.raises(MaterializationUnavailable, match="failed closed"):
        execute_ephemeral_audit(plan, runtime)

    assert not (scratch / plan.plan_id).exists()


def test_materializer_has_no_customer_execution_or_network_surface() -> None:
    source = Path("claimci/analysis/materialize.py").read_text(encoding="utf-8")

    for forbidden in (
        "subprocess",
        "importlib",
        "requests",
        "urllib",
        "socket",
        "eval(",
        "exec(",
    ):
        assert forbidden not in source


def test_executor_rejects_canonical_looking_but_tampered_plan_id(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    tampered = dataclasses.replace(plan, plan_id="plan-" + "f" * 24)

    with pytest.raises(MaterializationUnavailable, match="identity|plan"):
        execute_ephemeral_audit(tampered, runtime)

    assert not (scratch / tampered.plan_id).exists()


def test_executor_rejects_mapping_provenance_drift(tmp_path: Path) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)
    drifted = dataclasses.replace(
        plan,
        mapping_provenance=(
            FieldProvenance(
                ProvenanceKind.PROVIDER_PROPOSAL,
                "forged mapping provenance",
                source_id="provider-call-9",
            ),
        ),
    )

    with pytest.raises(MaterializationUnavailable, match="mapping|provenance"):
        execute_ephemeral_audit(drifted, runtime)

    assert not (scratch / plan.plan_id).exists()
