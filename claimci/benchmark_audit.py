"""Native deterministic checks for the Benchmark Measurement evidence profile."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Mapping

from .analysis.contracts import AuditClaimSpec, NormalizedEvidence
from .analysis.profiles import (
    _BENCHMARK_PROFILE_FIELDS,
    _canonical_profile_scalar,
    BenchmarkResultForm,
    BenchmarkVariant,
    EvidenceProfileId,
    EvidenceProfileSelection,
)
from .audit import determine_verdict
from .models import AuditResult, ClaimCIError, Finding, Impact, Severity
from .measurement import (
    ClaimCIVerificationReduction,
    MeasurementAuditContext,
    compare_measurement_protocols,
    recover_benchmark_measurement_protocol_pair,
)
from .result_check import LoadedResults, check_reported_aggregates, check_results


def _canonical_scalar(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _config_projection(
    evidence: tuple[NormalizedEvidence, ...],
) -> dict[str, object]:
    values: dict[str, object] = {}
    encoded: dict[str, str] = {}
    for item in evidence:
        for observation in item.observations:
            for config in observation.config_values:
                if config.key not in _BENCHMARK_PROFILE_FIELDS:
                    continue
                try:
                    recovered = _canonical_profile_scalar(config.key, config.value)
                except Exception as error:
                    raise ClaimCIError(
                        f"benchmark component {config.key!r} is invalid"
                    ) from error
                canonical = _canonical_scalar(recovered)
                previous = encoded.get(config.key)
                if previous is not None and previous != canonical:
                    raise ClaimCIError(
                        f"benchmark component {config.key!r} has conflicting values"
                    )
                values[config.key] = recovered
                encoded[config.key] = canonical
    return values


def _component(
    values: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    return {
        key[len(prefix) + 1 :]: value
        for key, value in values.items()
        if key.startswith(prefix + ".")
    }


def _mismatch_finding(
    *,
    rule_id: str,
    title: str,
    component: str,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=Severity.CRITICAL,
        title=title,
        explanation=(
            f"The represented baseline and candidate {component} components differ, "
            "so their metric values are not directly comparable."
        ),
        evidence={
            "component": component,
            "baseline": dict(sorted(baseline.items())),
            "candidate": dict(sorted(candidate.items())),
        },
        impact=Impact.INVALIDATES,
    )


def _procedure_component(
    prefix: str,
    values: Mapping[str, object],
    selection: EvidenceProfileSelection,
) -> dict[str, object]:
    component = _component(values, prefix)
    excluded: set[str] = set()
    if prefix == "benchmark.evaluator":
        excluded.update({"unit_value", "quantity"})
    if (
        prefix == "benchmark.environment"
        and selection.benchmark_variant is BenchmarkVariant.COST
    ):
        excluded.update({"provider", "sku"})
    return {key: value for key, value in component.items() if key not in excluded}


def _effective_component(
    prefix: str,
    shared_values: Mapping[str, object],
    role_values: Mapping[str, object],
    selection: EvidenceProfileSelection,
) -> tuple[dict[str, object], dict[str, object]]:
    shared = _procedure_component(prefix, shared_values, selection)
    role = _procedure_component(prefix, role_values, selection)
    conflicts = {
        key: role[key]
        for key in shared.keys() & role.keys()
        if shared[key] != role[key]
    }
    effective = dict(shared)
    effective.update(role)
    return effective, conflicts


def _procedure_insufficient(component: str) -> Finding:
    return Finding(
        rule_id="BENCHMARK.PROCEDURE_INSUFFICIENT",
        severity=Severity.WARNING,
        title="Benchmark measurement procedure is incomplete",
        explanation=(
            f"Required {component} evidence could not be independently recovered "
            "from the passive benchmark inputs."
        ),
        evidence={"component": component},
        impact=Impact.INSUFFICIENT,
    )


def _raw_results(
    evidence: tuple[NormalizedEvidence, ...],
    metric: str,
    *,
    role: str,
) -> tuple[LoadedResults | None, tuple[Finding, ...]]:
    values: list[float] = []
    seeds: list[int | None] = []
    mismatches: set[str] = set()
    paths: list[str] = []
    for item in evidence:
        paths.append(str(item.artifact.path))
        for observation in item.observations:
            if observation.metric_name is None:
                continue
            if observation.metric_name != metric:
                mismatches.add(observation.metric_name)
                continue
            assert observation.metric_value is not None
            values.append(observation.metric_value)
            seeds.append(
                observation.seed
                if isinstance(observation.seed, int)
                and not isinstance(observation.seed, bool)
                else None
            )
    findings: list[Finding] = []
    if mismatches:
        findings.append(
            Finding(
                rule_id="BENCHMARK.METRIC_MISMATCH",
                severity=Severity.CRITICAL,
                title=f"{role.title()} measured metric does not match the claim",
                explanation=(
                    f"The claim asserts {metric!r}, while the selected {role} "
                    "evidence represents a different metric identity."
                ),
                evidence={
                    "role": role,
                    "claim_metric": metric,
                    "represented_metrics": sorted(mismatches),
                },
                impact=Impact.INVALIDATES,
            )
        )
    if not values:
        findings.append(_procedure_insufficient(f"{role} result values"))
        return None, tuple(findings)
    return (
        LoadedResults(
            path=Path(paths[0] if paths else f"<{role}-benchmark>"),
            values=tuple(values),
            seeds=tuple(seeds),
            claimed_summary=None,
        ),
        tuple(findings),
    )


def _required_value(values: Mapping[str, object], key: str) -> object:
    if key not in values:
        raise ClaimCIError(f"required benchmark component {key!r} is missing")
    return values[key]


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 2**63 - 1
    ):
        raise ClaimCIError(f"{label} must be a positive integer")
    return value


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ClaimCIError(f"{label} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ClaimCIError(f"{label} must be a finite decimal") from exc
    if not number.is_finite():
        raise ClaimCIError(f"{label} must be a finite decimal")
    return number


def _single_metric_value(
    evidence: tuple[NormalizedEvidence, ...],
    metric: str,
    *,
    role: str,
) -> tuple[float | None, tuple[Finding, ...]]:
    loaded, findings = _raw_results(evidence, metric, role=role)
    if loaded is None:
        return None, findings
    if len(loaded.values) != 1:
        return (
            None,
            (*findings, _procedure_insufficient(f"{role} aggregate scalar")),
        )
    return loaded.values[0], findings


def _represented_metric_identity(
    evidence: tuple[NormalizedEvidence, ...],
    *,
    role: str,
) -> str:
    represented = {
        observation.metric_name
        for item in evidence
        for observation in item.observations
        if observation.metric_name is not None
    }
    if len(represented) != 1:
        raise ClaimCIError(
            f"{role} benchmark evidence has no single measured metric identity"
        )
    return next(iter(represented))


def _append_result_check(
    findings: list[Finding],
    checked: object,
) -> tuple[object, object, object, object]:
    from .result_check import ResultCheck

    if type(checked) is not ResultCheck:
        raise TypeError("Benchmark result comparison is invalid")
    findings.extend(checked.findings)
    return (
        checked.baseline_summary,
        checked.candidate_summary,
        checked.absolute_improvement,
        checked.relative_improvement,
    )


def audit_benchmark(
    *,
    audit_claim: AuditClaimSpec,
    selection: EvidenceProfileSelection,
    baseline_evidence: tuple[NormalizedEvidence, ...],
    candidate_evidence: tuple[NormalizedEvidence, ...],
    reference_evidence: tuple[NormalizedEvidence, ...] = (),
    measurement_context: MeasurementAuditContext | None = None,
) -> AuditResult:
    """Audit one freshly materialized Benchmark profile through AuditResult."""

    if type(audit_claim) is not AuditClaimSpec:
        raise TypeError("Benchmark Audit requires AuditClaimSpec")
    if type(selection) is not EvidenceProfileSelection or (
        selection.profile_id is not EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    ):
        raise TypeError("Benchmark Audit requires its deterministic profile selection")
    if not selection.complete:
        raise ClaimCIError("Benchmark Audit requires a complete evidence profile")
    for label, values in (
        ("baseline", baseline_evidence),
        ("candidate", candidate_evidence),
        ("reference", reference_evidence),
    ):
        if not isinstance(values, tuple) or not all(
            type(item) is NormalizedEvidence for item in values
        ):
            raise TypeError(f"Benchmark {label} evidence is invalid")
    if audit_claim.minimum_absolute_improvement is None:
        raise ClaimCIError("Benchmark Audit requires an absolute improvement threshold")
    if measurement_context is not None and type(
        measurement_context
    ) is not MeasurementAuditContext:
        raise TypeError("Benchmark measurement context is invalid")

    findings: list[Finding] = []
    baseline_config = _config_projection(baseline_evidence)
    candidate_config = _config_projection(candidate_evidence)
    reference_config = _config_projection(reference_evidence)
    baseline_metric_identities = {
        observation.metric_name
        for item in baseline_evidence
        for observation in item.observations
        if observation.metric_name is not None
    }
    candidate_metric_identities = {
        observation.metric_name
        for item in candidate_evidence
        for observation in item.observations
        if observation.metric_name is not None
    }
    comparisons = (
        (
            "benchmark.workload",
            "BENCHMARK.WORKLOAD_MISMATCH",
            "Benchmark workload changed",
        ),
        (
            "benchmark.measurement_config",
            "BENCHMARK.CONFIG_MISMATCH",
            "Benchmark measurement configuration changed",
        ),
        (
            "benchmark.run_protocol",
            "BENCHMARK.CONFIG_MISMATCH",
            "Benchmark run protocol changed",
        ),
        (
            "benchmark.environment",
            "BENCHMARK.CONFIG_MISMATCH",
            "Benchmark environment changed",
        ),
        (
            "benchmark.evaluator",
            "BENCHMARK.CONFIG_MISMATCH",
            "Benchmark evaluator changed",
        ),
        (
            "measurement.metric_identity",
            "BENCHMARK.METRIC_MISMATCH",
            "Measured metric identity changed",
        ),
    )
    protocol_mismatch = not (
        len(baseline_metric_identities) == 1
        and baseline_metric_identities == candidate_metric_identities
    )
    for prefix, rule_id, title in comparisons:
        baseline, baseline_reference_conflicts = _effective_component(
            prefix, reference_config, baseline_config, selection
        )
        candidate, candidate_reference_conflicts = _effective_component(
            prefix, reference_config, candidate_config, selection
        )
        reference = _procedure_component(prefix, reference_config, selection)
        for role, conflicts in (
            ("baseline", baseline_reference_conflicts),
            ("candidate", candidate_reference_conflicts),
        ):
            if conflicts:
                protocol_mismatch = True
                findings.append(
                    _mismatch_finding(
                        rule_id=rule_id,
                        title=title,
                        component=f"{prefix}.{role}_reference",
                        baseline={key: reference[key] for key in conflicts},
                        candidate=conflicts,
                    )
                )
        if baseline and candidate and baseline != candidate:
            protocol_mismatch = True
            findings.append(
                _mismatch_finding(
                    rule_id=rule_id,
                    title=title,
                    component=prefix,
                    baseline=baseline,
                    candidate=candidate,
                )
            )
    if not protocol_mismatch:
        findings.append(
            Finding(
                rule_id="BENCHMARK.PROTOCOL_VERIFIED",
                severity=Severity.VERIFIED,
                title="Benchmark measurement protocol is comparable",
                explanation=(
                    "The independently recovered baseline, candidate, and shared "
                    "reference measurement components are compatible under the "
                    "selected fixed Benchmark profile policy."
                ),
                evidence={
                    "profile_id": selection.profile_id.value,
                    "variant": selection.benchmark_variant.value
                    if selection.benchmark_variant is not None
                    else None,
                    "activation_policy_id": selection.activation_policy_id,
                },
                impact=Impact.NONE,
            )
        )

    baseline_metrics = None
    candidate_metrics = None
    absolute_improvement = None
    relative_improvement = None
    if selection.result_form is BenchmarkResultForm.RAW_RUN_SERIES:
        baseline, baseline_findings = _raw_results(
            baseline_evidence,
            audit_claim.metric,
            role="baseline",
        )
        candidate, candidate_findings = _raw_results(
            candidate_evidence,
            audit_claim.metric,
            role="candidate",
        )
        findings.extend(baseline_findings)
        findings.extend(candidate_findings)
        if baseline is not None and candidate is not None:
            checked = check_results(
                baseline,
                candidate,
                audit_claim.metric,
                audit_claim.minimum_absolute_improvement,
                audit_claim.direction,
            )
            (
                baseline_metrics,
                candidate_metrics,
                absolute_improvement,
                relative_improvement,
            ) = _append_result_check(findings, checked)
    elif selection.result_form is BenchmarkResultForm.REPORTED_AGGREGATE:
        baseline_value, baseline_findings = _single_metric_value(
            baseline_evidence, audit_claim.metric, role="baseline"
        )
        candidate_value, candidate_findings = _single_metric_value(
            candidate_evidence, audit_claim.metric, role="candidate"
        )
        findings.extend(baseline_findings)
        findings.extend(candidate_findings)
        try:
            baseline_count = _positive_integer(
                _required_value(
                    baseline_config, "benchmark.run_protocol.sample_count"
                ),
                label="baseline aggregate sample count",
            )
            candidate_count = _positive_integer(
                _required_value(
                    candidate_config, "benchmark.run_protocol.sample_count"
                ),
                label="candidate aggregate sample count",
            )
            statistic = _required_value(
                baseline_config, "benchmark.run_protocol.statistic"
            )
            aggregation = _required_value(
                baseline_config, "benchmark.run_protocol.aggregation"
            )
            if not isinstance(statistic, str) or not isinstance(aggregation, str):
                raise ClaimCIError("aggregate identities must be exact text")
            if baseline_value is None or candidate_value is None:
                raise ClaimCIError("aggregate scalar is unavailable")
            checked = check_reported_aggregates(
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                baseline_sample_count=baseline_count,
                candidate_sample_count=candidate_count,
                statistic=statistic,
                aggregation=aggregation,
                metric=audit_claim.metric,
                minimum_improvement=audit_claim.minimum_absolute_improvement,
                direction=audit_claim.direction,
            )
        except ClaimCIError:
            findings.append(_procedure_insufficient("reported aggregate metadata"))
        else:
            (
                baseline_metrics,
                candidate_metrics,
                absolute_improvement,
                relative_improvement,
            ) = _append_result_check(findings, checked)
    elif selection.result_form is BenchmarkResultForm.DETERMINISTIC_DERIVATION:
        derived: dict[str, float] = {}
        for role, config, role_evidence in (
            ("baseline", baseline_config, baseline_evidence),
            ("candidate", candidate_config, candidate_evidence),
        ):
            try:
                family = _required_value(config, "benchmark.evaluator.formula_family")
                if family != "multiply_v1":
                    raise ClaimCIError("deterministic formula family is unsupported")
                unit = _decimal(
                    _required_value(config, "benchmark.evaluator.unit_value"),
                    label=f"{role} unit value",
                )
                quantity = _decimal(
                    _required_value(config, "benchmark.evaluator.quantity"),
                    label=f"{role} quantity",
                )
                with localcontext() as context:
                    context.prec = 40
                    recomputed_decimal = unit * quantity
                recomputed = float(recomputed_decimal)
                if not math.isfinite(recomputed):
                    raise ClaimCIError("deterministic derivation is not finite")
                represented, represented_findings = _single_metric_value(
                    role_evidence,
                    audit_claim.metric,
                    role=role,
                )
                findings.extend(represented_findings)
                if represented is None:
                    raise ClaimCIError("represented derived value is unavailable")
                if Decimal(str(represented)) != recomputed_decimal:
                    findings.append(
                        Finding(
                            rule_id="BENCHMARK.DERIVATION_MISMATCH",
                            severity=Severity.CRITICAL,
                            title=f"{role.title()} deterministic derivation does not match",
                            explanation=(
                                "ClaimCI recomputed the fixed multiply_v1 formula from "
                                "exact passive inputs and obtained a different value."
                            ),
                            evidence={
                                "role": role,
                                "formula_family": "multiply_v1",
                                "represented": represented,
                                "recomputed": recomputed,
                            },
                            impact=Impact.INVALIDATES,
                        )
                    )
                derived[role] = recomputed
            except ClaimCIError:
                findings.append(_procedure_insufficient(f"{role} deterministic derivation"))
        if set(derived) == {"baseline", "candidate"}:
            checked = check_reported_aggregates(
                baseline_value=derived["baseline"],
                candidate_value=derived["candidate"],
                baseline_sample_count=1,
                candidate_sample_count=1,
                statistic="deterministic_derivation",
                aggregation="multiply_v1",
                metric=audit_claim.metric,
                minimum_improvement=audit_claim.minimum_absolute_improvement,
                direction=audit_claim.direction,
            )
            (
                baseline_metrics,
                candidate_metrics,
                absolute_improvement,
                relative_improvement,
            ) = _append_result_check(findings, checked)

    measurement_drift = None
    if measurement_context is not None:
        baseline_protocol: dict[str, Mapping[str, object]] = {}
        candidate_protocol: dict[str, Mapping[str, object]] = {}
        for prefix, _rule_id, _title in comparisons:
            baseline, _baseline_conflicts = _effective_component(
                prefix, reference_config, baseline_config, selection
            )
            candidate, _candidate_conflicts = _effective_component(
                prefix, reference_config, candidate_config, selection
            )
            if baseline:
                baseline_protocol[prefix] = baseline
            if candidate:
                candidate_protocol[prefix] = candidate
        protocols = recover_benchmark_measurement_protocol_pair(
            measurement_context,
            baseline_metric=_represented_metric_identity(
                baseline_evidence,
                role="baseline",
            ),
            candidate_metric=_represented_metric_identity(
                candidate_evidence,
                role="candidate",
            ),
            baseline_components=baseline_protocol,
            candidate_components=candidate_protocol,
        )
        measurement_drift = compare_measurement_protocols(
            protocols,
            native_rule_ids=tuple(item.rule_id for item in findings),
            claimci_verification_reduction=(
                ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1
                if selection.result_form is BenchmarkResultForm.RAW_RUN_SERIES
                else ClaimCIVerificationReduction.NOT_APPLICABLE
            ),
        )
    return AuditResult(
        verdict=determine_verdict(findings),
        findings=tuple(findings),
        manifest_path=Path("."),
        metric=audit_claim.metric,
        minimum_improvement=audit_claim.minimum_absolute_improvement,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        direction=audit_claim.direction,
        measurement_drift=measurement_drift,
    )


__all__ = ["audit_benchmark"]
