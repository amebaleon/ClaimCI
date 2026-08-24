"""Exact-head materialization for streamed evidence and dataset authority."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from claimci.analysis import (
    AdapterMatch,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    AuditClaimSpec,
    ClaimReference,
    Confidence,
    ConfigValue,
    DatasetSplit,
    EphemeralAuditPlan,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    MaterializationLimits,
    MaterializationUnavailable,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    RuntimeExecutionContext,
    ScanPurpose,
    ScanState,
    SelectorKind,
    Sha256Digest,
    StreamingLimits,
    TraceRecordKind,
    approve_metric_binding,
    artifact_source_from_snapshot,
    derive_ephemeral_plan_id,
    execute_ephemeral_audit_with_trace,
    extract_bound_metric_evidence,
    extract_metric_candidate_scan,
    integrate_metric_binding,
    resolve_metric_binding,
)
from claimci.analysis.adapters import (
    extract_registered_artifact,
    extract_registered_source,
)
from claimci.models import AuditResult, Direction, Impact, Verdict


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("7" * 40)
CLAIM_ID = "claim-streaming-materialization"


def _write_candidate(
    checkout: Path,
    relative: str,
    kind: ArtifactKind,
    content: bytes,
) -> ArtifactCandidate:
    path = checkout / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactCandidate(
        path=RepositoryPath(relative),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming materialization fixture",
        relevant_claim_ids=(CLAIM_ID,),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "exact fixture artifact",
            RepositoryPath(relative),
            f"artifact:{relative}",
        ),
    )


def _config_evidence(
    artifact: ArtifactCandidate,
    role: ExperimentRole,
) -> NormalizedEvidence:
    values: dict[str, object] = {
        "batch_size": 8,
        "dataset.identifier": "train",
        "dataset.version": "v1",
        "epochs": 2,
        "evaluation.dataset_identifier": "eval",
        "evaluation.dataset_version": "v1",
        "evaluation.split": "test",
        "learning_rate": 0.001,
        "model": "demo-v1",
        "training_steps": 100,
    }
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        "fixed config fixture extraction",
        artifact.path,
        f"adapter:{artifact.path}",
    )
    mappings = tuple(
        FieldMapping(
            f"config.{key}",
            EvidenceSelector(SelectorKind.DOTTED_PATH, key, provenance),
            provenance,
        )
        for key in sorted(values)
    )
    return NormalizedEvidence(
        evidence_id=f"evidence-{str(artifact.path).replace('/', '-')}",
        artifact=artifact,
        adapter_match=AdapterMatch(
            "fixture.config",
            artifact.path,
            Confidence(0.99),
            mappings,
            (provenance,),
        ),
        observations=(
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


def _dataset_evidence(
    checkout: Path,
    artifact: ArtifactCandidate,
    content: bytes,
    *,
    streamed: bool,
    limits: StreamingLimits,
) -> NormalizedEvidence:
    if streamed:
        outcome = extract_registered_source(
            artifact_source_from_snapshot(REPOSITORY, HEAD, checkout, artifact),
            limits=limits,
        )
        assert outcome is not None and outcome.evidence is not None
        return outcome.evidence
    evidence = extract_registered_artifact(PassiveArtifact(artifact, content))
    assert evidence is not None
    return evidence


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for record in records
    )


def _plan_fixture(
    tmp_path: Path,
    *,
    stream_datasets: bool = False,
    dataset_overlap_in_prefix: bool = False,
    semantic_records: int = 100_000,
) -> tuple[EphemeralAuditPlan, RuntimeExecutionContext, Path, Path]:
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    checkout.mkdir()
    scratch.mkdir()
    stream_limits = StreamingLimits(
        max_integrity_bytes=4 * 1024 * 1024,
        max_semantic_bytes=4 * 1024 * 1024,
        max_semantic_records=semantic_records,
        max_scratch_bytes=4 * 1024 * 1024,
    )

    result_pairs: list[tuple[ArtifactCandidate, object]] = []
    for role, value in (
        (ExperimentRole.BASELINE, 0.50),
        (ExperimentRole.CANDIDATE, 0.75),
    ):
        content = _jsonl(
            [
                {
                    "acc": value + offset,
                    "padding": "r" * 900,
                    "run": index,
                    "seed": index,
                }
                for index, offset in (
                    ((1, 0.0),)
                    if semantic_records == 1
                    else ((1, 0.0), (2, 0.01), (3, -0.01))
                )
            ]
        )
        artifact = _write_candidate(
            checkout,
            f"results/{role.value}.jsonl",
            ArtifactKind.RESULTS,
            content,
        )
        source = artifact_source_from_snapshot(REPOSITORY, HEAD, checkout, artifact)
        result_pairs.append((artifact, source))

    baseline_candidates = extract_metric_candidate_scan(
        result_pairs[0][1],
        role=ExperimentRole.BASELINE,
        limits=stream_limits,
    )
    candidate_candidates = extract_metric_candidate_scan(
        result_pairs[1][1],
        role=ExperimentRole.CANDIDATE,
        limits=stream_limits,
    )
    pending = resolve_metric_binding(
        (*baseline_candidates.candidates, *candidate_candidates.candidates),
        repository=REPOSITORY,
        head_sha=HEAD,
        canonical_metric="accuracy",
    )
    assert pending.question is not None
    assert len(pending.question.proposals) == 1
    metric_binding = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )
    baseline_result = extract_bound_metric_evidence(
        result_pairs[0][1],
        metric_binding,
        role=ExperimentRole.BASELINE,
        limits=stream_limits,
    )
    candidate_result = extract_bound_metric_evidence(
        result_pairs[1][1],
        metric_binding,
        role=ExperimentRole.CANDIDATE,
        limits=stream_limits,
    )

    configs = tuple(
        _config_evidence(
            _write_candidate(
                checkout,
                f"configs/{role.value}.json",
                ArtifactKind.CONFIG,
                b'{"training_steps":100}\n',
            ),
            role,
        )
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
    )

    pad = "d" * 500 if stream_datasets else ""
    shared = {"id": "shared", "padding": pad}
    dataset_records = {
        (ExperimentRole.BASELINE, DatasetSplit.TRAIN): (
            [shared, {"id": "baseline-train-tail", "padding": pad}]
            if dataset_overlap_in_prefix
            else [
                {"id": "baseline-train", "padding": pad},
                {"id": "baseline-train-tail", "padding": pad},
            ]
        ),
        (ExperimentRole.BASELINE, DatasetSplit.EVAL): [
            shared,
            {"id": "baseline-eval-tail", "padding": pad},
        ],
        (ExperimentRole.CANDIDATE, DatasetSplit.TRAIN): [
            {"id": "candidate-train", "padding": pad},
            {"id": "candidate-train-tail", "padding": pad},
        ],
        (ExperimentRole.CANDIDATE, DatasetSplit.EVAL): [
            shared,
            {"id": "baseline-eval-tail", "padding": pad},
        ],
    }
    datasets: dict[tuple[ExperimentRole, DatasetSplit], NormalizedEvidence] = {}
    dataset_bindings: list[ArtifactBinding] = []
    for (role, split), records in dataset_records.items():
        content = _jsonl(records)
        artifact = _write_candidate(
            checkout,
            f"data/{role.value}-{split.value}.jsonl",
            ArtifactKind.DATASET,
            content,
        )
        evidence = _dataset_evidence(
            checkout,
            artifact,
            content,
            streamed=stream_datasets,
            limits=stream_limits,
        )
        datasets[(role, split)] = evidence
        dataset_bindings.append(
            ArtifactBinding(
                artifact.path,
                ArtifactKind.DATASET,
                role,
                evidence.adapter_match.adapter_id,
                evidence.adapter_match.mappings,
                evidence.adapter_match.match_evidence[0],
                split,
            )
        )

    mapping_provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "exact streaming fixture mapping",
        source_id="mapping:streaming",
    )
    mapping = MappingCandidate(
        mapping_id="mapping-streaming",
        bindings=(
            ArtifactBinding(
                result_pairs[0][0].path,
                ArtifactKind.RESULTS,
                ExperimentRole.BASELINE,
                baseline_result.adapter_match.adapter_id,
                tuple(
                    item
                    for item in baseline_result.adapter_match.mappings
                    if item.target_field != "metric_value"
                ),
                baseline_result.adapter_match.match_evidence[0],
            ),
            ArtifactBinding(
                result_pairs[1][0].path,
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
                candidate_result.adapter_match.adapter_id,
                tuple(
                    item
                    for item in candidate_result.adapter_match.mappings
                    if item.target_field != "metric_value"
                ),
                candidate_result.adapter_match.match_evidence[0],
            ),
            ArtifactBinding(
                configs[0].artifact.path,
                ArtifactKind.CONFIG,
                ExperimentRole.BASELINE,
                configs[0].adapter_match.adapter_id,
                configs[0].adapter_match.mappings,
                configs[0].adapter_match.match_evidence[0],
            ),
            ArtifactBinding(
                configs[1].artifact.path,
                ArtifactKind.CONFIG,
                ExperimentRole.CANDIDATE,
                configs[1].adapter_match.adapter_id,
                configs[1].adapter_match.mappings,
                configs[1].adapter_match.match_evidence[0],
            ),
            *dataset_bindings,
        ),
        confidence=Confidence(0.99),
        trust=MappingTrust.INFERRED,
        provenance=mapping_provenance,
    )
    selected_mapping = integrate_metric_binding(mapping, metric_binding)
    provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "fixed streaming claim",
        RepositoryPath("CLAIM.md"),
        "claim:streaming",
    )
    audit_claim = AuditClaimSpec(
        CLAIM_ID,
        "accuracy",
        Direction.HIGHER,
        0.05,
        provenance,
        provenance,
        provenance,
    )
    plan = EphemeralAuditPlan(
        plan_id="plan-" + "0" * 24,
        repository=REPOSITORY,
        pr_number=17,
        head_sha=HEAD,
        claim=ClaimReference(
            CLAIM_ID,
            "Accuracy improved by at least 0.05.",
            RepositoryPath("CLAIM.md"),
            Confidence(0.99),
            provenance,
        ),
        baseline_evidence=(
            baseline_result,
            configs[0],
            datasets[(ExperimentRole.BASELINE, DatasetSplit.TRAIN)],
            datasets[(ExperimentRole.BASELINE, DatasetSplit.EVAL)],
        ),
        candidate_evidence=(
            candidate_result,
            configs[1],
            datasets[(ExperimentRole.CANDIDATE, DatasetSplit.TRAIN)],
            datasets[(ExperimentRole.CANDIDATE, DatasetSplit.EVAL)],
        ),
        mapping_provenance=(selected_mapping.provenance,),
        missing_evidence=(),
        confidence=Confidence(0.99),
        audit_claim=audit_claim,
        selected_mapping=selected_mapping,
        metric_binding=metric_binding,
    )
    plan = dataclasses.replace(plan, plan_id=derive_ephemeral_plan_id(plan))
    runtime = RuntimeExecutionContext(
        repository=REPOSITORY,
        checkout_root=checkout,
        head_sha=HEAD,
        scratch_root=scratch,
        limits=MaterializationLimits(
            max_file_bytes=256,
            max_total_bytes=2 * 1024 * 1024,
            max_stream_file_bytes=4 * 1024 * 1024,
            max_stream_total_bytes=16 * 1024 * 1024,
            streaming_limits=stream_limits,
        ),
    )
    return plan, runtime, checkout, scratch


def _finding(result: AuditResult, rule_id: str):
    return next((item for item in result.findings if item.rule_id == rule_id), None)


def test_large_atomic_acc_pair_streams_through_exact_binding_and_real_audit(
    tmp_path: Path,
) -> None:
    plan, runtime, _checkout, scratch = _plan_fixture(tmp_path)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.SUPPORTED
    assert _finding(execution.audit_result, "RESULT.METRIC_IDENTITY_VERIFIED")
    assert len(execution.scan_completeness) == 2
    assert all(
        report.state is ScanState.COMPLETE and report.integrity_verified
        for report in execution.scan_completeness
    )
    streamed_results = tuple(
        item
        for item in execution.trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
        and item.artifact_kind is ArtifactKind.RESULTS
    )
    assert len(streamed_results) == 2
    assert all(item.scan_completeness is not None for item in streamed_results)
    assert all(
        item.source_value is not None
        and item.source_value.byte_count == item.artifact_size
        for item in streamed_results
    )
    assert not tuple(scratch.iterdir())


@pytest.mark.parametrize(
    "role",
    [ExperimentRole.BASELINE, ExperimentRole.CANDIDATE],
)
def test_changed_stream_endpoint_invalidates_the_entire_metric_pair(
    tmp_path: Path,
    role: ExperimentRole,
) -> None:
    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    path = checkout / "results" / f"{role.value}.jsonl"
    changed = path.read_bytes().replace(b'"acc":0.', b'"acc":1.', 1)
    assert len(changed) == path.stat().st_size
    path.write_bytes(changed)

    with pytest.raises(MaterializationUnavailable, match="entire metric pair"):
        execute_ephemeral_audit_with_trace(plan, runtime)

    assert not tuple(scratch.iterdir())


@pytest.mark.parametrize("overlap", [False, True])
def test_partial_dataset_scan_preserves_only_existential_authority_and_no_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap: bool,
) -> None:
    import claimci.analysis.materialize as materialize
    from claimci.audit import audit_research as real_audit_research

    plan, runtime, _checkout, scratch = _plan_fixture(
        tmp_path,
        stream_datasets=True,
        dataset_overlap_in_prefix=overlap,
        semantic_records=1,
    )
    observed: dict[str, bool] = {}

    def inspect_then_audit(path: Path, **kwargs: object) -> AuditResult:
        root = Path(kwargs["artifact_root"])
        observed["dataset_copy_absent"] = all(
            not (root / role / f"{split}.jsonl").exists()
            for role in ("baseline", "candidate")
            for split in ("train", "eval")
        )
        return real_audit_research(path, **kwargs)

    monkeypatch.setattr(materialize, "audit_research", inspect_then_audit)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert observed == {"dataset_copy_absent": True}
    assert _finding(execution.audit_result, "DATASET.SCAN_INCOMPLETE")
    leakage = _finding(execution.audit_result, "DATASET.EXACT_LEAKAGE")
    assert (leakage is not None) is overlap
    assert _finding(execution.audit_result, "DATASET.NO_LEAKAGE") is None
    if overlap:
        assert leakage is not None and leakage.impact is Impact.INVALIDATES
        assert execution.audit_result.verdict is Verdict.NOT_SUPPORTED
    else:
        assert execution.audit_result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    dataset_reports = tuple(
        report
        for report in execution.scan_completeness
        if report.purpose is ScanPurpose.DATASET_RECORDS
    )
    assert len(dataset_reports) == 4
    assert all(
        report.state is ScanState.INCOMPLETE_BUDGET
        and report.integrity_verified
        for report in dataset_reports
    )
    dataset_trace = tuple(
        item
        for item in execution.trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
        and item.artifact_kind is ArtifactKind.DATASET
    )
    assert len(dataset_trace) == 4
    assert all(
        item.scan_completeness is not None
        and item.scan_completeness.state is ScanState.INCOMPLETE_BUDGET
        and item.source_value is not None
        and item.source_value.byte_count == item.artifact_size
        for item in dataset_trace
    )
    assert not tuple(scratch.iterdir())
