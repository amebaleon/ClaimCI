"""Release regressions for Metric Binding over streamed shared tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    MetricBindingState,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    ScanState,
    Sha256Digest,
    TablePredicate,
    TableScalarType,
    TableSelector,
    approve_metric_binding,
    artifact_source_from_snapshot,
    extract_metric_candidate_scan,
    integrate_metric_binding,
    resolve_metric_binding,
)


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("9" * 40)


def _source(tmp_path: Path, suffix: str):
    delimiter = "," if suffix == "csv" else "\t"
    content = (
        delimiter.join(("role", "status", "enabled", "version", "seed", "acc"))
        + "\n"
        + delimiter.join(("baseline", "ok", "true", "1", "1", "0.60"))
        + "\n"
        + delimiter.join(("candidate", "ok", "false", "2", "2", "0.70"))
        + "\n"
    ).encode()
    relative = RepositoryPath(f"results/shared.{suffix}")
    target = tmp_path / str(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "combined streaming security fixture",
        relative,
        f"fixture:metric-streaming:{suffix}",
    )
    candidate = ArtifactCandidate(
        path=relative,
        kind=ArtifactKind.RESULTS,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="combined metric streaming security fixture",
        relevant_claim_ids=("claim-accuracy",),
        provenance=provenance,
    )
    return artifact_source_from_snapshot(REPOSITORY, HEAD, tmp_path, candidate)


def _provenance(source, label: str) -> FieldProvenance:
    return FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        f"{label}; sha256={source.candidate.sha256}",
        source.candidate.path,
        f"fixture:{label}",
    )


def _table_mapping(
    source,
    *,
    target_field: str,
    column: str,
    predicates: tuple[TablePredicate, ...],
) -> FieldMapping:
    provenance = _provenance(source, f"table:{target_field}:{column}")
    return FieldMapping(
        target_field,
        TableSelector(column, predicates, 1, provenance),
        provenance,
    )


def _adapter_match(
    source,
    *,
    row_role: str,
    extra_predicates: tuple[TablePredicate, ...] = (),
) -> AdapterMatch:
    suffix = str(source.candidate.path).rsplit(".", 1)[1]
    provenance = _provenance(source, f"metric:{suffix}:{row_role}")
    selector = TableSelector(
        "acc",
        (
            TablePredicate("role", TableScalarType.STRING, row_role),
            *extra_predicates,
        ),
        1,
        provenance,
    )
    return AdapterMatch(
        f"claimci-{suffix}-v1",
        source.candidate.path,
        Confidence(0.99),
        (FieldMapping("metric_value", selector, provenance),),
        (provenance,),
    )


def _metric_candidate(source, role: ExperimentRole, row_role: str):
    outcome = extract_metric_candidate_scan(
        source,
        role=role,
        adapter_match=_adapter_match(source, row_role=row_role),
    )
    assert outcome.completeness.state is ScanState.COMPLETE
    assert len(outcome.candidates) == 1
    return outcome.candidates[0]


def _approved_acc_binding(source, *, candidate_row: str):
    baseline = _metric_candidate(source, ExperimentRole.BASELINE, "baseline")
    candidate = _metric_candidate(
        source,
        ExperimentRole.CANDIDATE,
        candidate_row,
    )
    resolution = resolve_metric_binding(
        (baseline, candidate),
        repository=REPOSITORY,
        head_sha=HEAD,
        canonical_metric="accuracy",
        relevant_claim_id="claim-accuracy",
    )
    assert resolution.state is MetricBindingState.APPROVAL_REQUIRED
    assert resolution.question is not None
    assert len(resolution.question.proposals) == 1
    assert resolution.question.proposals[0].label == "[acc]"
    binding = approve_metric_binding(
        resolution.question,
        proposal_id=resolution.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )
    return baseline, candidate, binding


def _mapping_candidate(
    source,
    baseline_mappings: tuple[FieldMapping, ...],
    candidate_mappings: tuple[FieldMapping, ...],
) -> MappingCandidate:
    provenance = _provenance(source, "mapping")
    suffix = str(source.candidate.path).rsplit(".", 1)[1]
    return MappingCandidate(
        mapping_id="mapping-metric-streaming-security",
        bindings=(
            ArtifactBinding(
                source.candidate.path,
                source.candidate.kind,
                ExperimentRole.BASELINE,
                f"claimci-{suffix}-v1",
                baseline_mappings,
                provenance,
            ),
            ArtifactBinding(
                source.candidate.path,
                source.candidate.kind,
                ExperimentRole.CANDIDATE,
                f"claimci-{suffix}-v1",
                candidate_mappings,
                provenance,
            ),
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=provenance,
    )


def _role_predicate(value: str) -> tuple[TablePredicate, ...]:
    return (TablePredicate("role", TableScalarType.STRING, value),)


def _safe_seed_mapping(source) -> MappingCandidate:
    return _mapping_candidate(
        source,
        (
            _table_mapping(
                source,
                target_field="seed",
                column="seed",
                predicates=_role_predicate("baseline"),
            ),
        ),
        (
            _table_mapping(
                source,
                target_field="seed",
                column="seed",
                predicates=_role_predicate("candidate"),
            ),
        ),
    )


def test_same_path_overlapping_table_selectors_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path, "csv")
    overlapping = _role_predicate("baseline")

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _mapping_candidate(
            source,
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=overlapping,
                ),
            ),
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=overlapping,
                ),
            ),
        )


@pytest.mark.parametrize("suffix", ["csv", "tsv"])
def test_streaming_metric_binding_cannot_reintroduce_cross_role_overlap(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = _source(tmp_path, suffix)
    _, _, binding = _approved_acc_binding(source, candidate_row="baseline")

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        integrate_metric_binding(_safe_seed_mapping(source), binding)


@pytest.mark.parametrize(
    ("baseline_predicate", "candidate_predicate"),
    [
        (
            TablePredicate("enabled", TableScalarType.BOOLEAN, True),
            TablePredicate("enabled", TableScalarType.STRING, "true"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1.0"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1e0"),
        ),
    ],
)
def test_typed_lexical_aliases_remain_overlapping(
    tmp_path: Path,
    baseline_predicate: TablePredicate,
    candidate_predicate: TablePredicate,
) -> None:
    source = _source(tmp_path, "csv")

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _mapping_candidate(
            source,
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=(baseline_predicate,),
                ),
            ),
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=(candidate_predicate,),
                ),
            ),
        )


def test_subset_superset_predicates_remain_overlapping(tmp_path: Path) -> None:
    source = _source(tmp_path, "csv")

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _mapping_candidate(
            source,
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=_role_predicate("baseline"),
                ),
            ),
            (
                _table_mapping(
                    source,
                    target_field="metric_value",
                    column="acc",
                    predicates=(
                        *_role_predicate("baseline"),
                        TablePredicate("status", TableScalarType.STRING, "ok"),
                    ),
                ),
            ),
        )


def test_disjoint_role_predicates_stream_and_bind_successfully(tmp_path: Path) -> None:
    source = _source(tmp_path, "csv")
    baseline, candidate, binding = _approved_acc_binding(
        source,
        candidate_row="candidate",
    )

    integrated = integrate_metric_binding(_safe_seed_mapping(source), binding)

    assert baseline.candidate_id != candidate.candidate_id
    assert len(integrated.bindings) == 2
    assert {
        (
            item.role,
            next(
                mapping.selector.predicates[0].value
                for mapping in item.mappings
                if mapping.target_field == "metric_value"
            ),
        )
        for item in integrated.bindings
    } == {
        (ExperimentRole.BASELINE, "baseline"),
        (ExperimentRole.CANDIDATE, "candidate"),
    }


def test_atomic_acc_pair_keeps_two_exact_streaming_endpoints(tmp_path: Path) -> None:
    source = _source(tmp_path, "tsv")
    baseline, candidate, binding = _approved_acc_binding(
        source,
        candidate_row="candidate",
    )

    assert binding.baseline_candidate == baseline
    assert binding.candidate_candidate == candidate
    assert binding.baseline_candidate.candidate_id != binding.candidate_candidate.candidate_id
    assert binding.baseline_candidate.experiment_role is ExperimentRole.BASELINE
    assert binding.candidate_candidate.experiment_role is ExperimentRole.CANDIDATE
    assert binding.baseline_candidate.artifact_path == binding.candidate_candidate.artifact_path
    assert binding.baseline_candidate.artifact_sha256 == binding.candidate_candidate.artifact_sha256
    assert binding.baseline_candidate.selector != binding.candidate_candidate.selector
