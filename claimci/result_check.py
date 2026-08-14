"""Load raw experiment runs and recompute claim metrics."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Any, Sequence

from .models import (
    ClaimCIError,
    Direction,
    Finding,
    Impact,
    MetricSummary,
    Severity,
    coerce_direction,
)
from .parsing import unique_json_object, validate_json_graph


@dataclass(frozen=True)
class LoadedResults:
    path: Path
    values: tuple[float, ...]
    seeds: tuple[int | None, ...]
    claimed_summary: float | None


@dataclass(frozen=True)
class ResultCheck:
    baseline_summary: MetricSummary
    candidate_summary: MetricSummary
    absolute_improvement: float
    relative_improvement: float | None
    findings: tuple[Finding, ...]


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON value {token}")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _decimal_number(value: object, *, label: str) -> Decimal:
    """Convert a finite numeric value to its deterministic decimal spelling."""

    number = _finite_number(value)
    if number is None:
        raise ClaimCIError(f"{label} must be finite")
    try:
        decimal_number = Decimal(str(number))
    except (DecimalException, ValueError, TypeError) as exc:
        raise ClaimCIError(f"{label} must be finite") from exc
    if not decimal_number.is_finite():
        raise ClaimCIError(f"{label} must be finite")
    return decimal_number


def _finite_float(value: Decimal, *, label: str) -> float:
    """Convert a Decimal result to a finite public float or reject it."""

    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ClaimCIError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ClaimCIError(f"{label} must be finite")
    return number


def load_results(path: Path, metric: str) -> LoadedResults:
    try:
        path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ClaimCIError(f"invalid results file path {path!r}") from exc
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ClaimCIError(f"results file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ClaimCIError(f"could not decode results file {path}: {exc}") from exc
    except ValueError as exc:
        raise ClaimCIError(f"could not read results file {path}: {exc}") from exc
    except OSError as exc:
        raise ClaimCIError(f"could not read results file {path}: {exc}") from exc

    try:
        payload = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=unique_json_object,
        )
        validate_json_graph(payload, label=f"results JSON {path}")
    except (json.JSONDecodeError, ValueError, OverflowError, TypeError, RecursionError) as exc:
        raise ClaimCIError(f"invalid JSON in results file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClaimCIError(f"results file {path} must contain a JSON object")

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ClaimCIError(f"results file {path} must contain a non-empty runs list")

    values: list[float] = []
    seeds: list[int | None] = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ClaimCIError(f"results file {path} run {index} must be an object")
        if metric not in run:
            raise ClaimCIError(
                f"results file {path} run {index} is missing metric {metric!r}"
            )
        value = _finite_number(run[metric])
        if value is None:
            raise ClaimCIError(
                f"results file {path} run {index} metric {metric!r} must be finite"
            )
        values.append(value)
        seed = run.get("seed")
        seeds.append(seed if isinstance(seed, int) and not isinstance(seed, bool) else None)

    claimed_summary: float | None = None
    summary = payload.get("summary")
    if summary is not None:
        if not isinstance(summary, dict):
            raise ClaimCIError(f"results file {path} summary must be an object")
        claimed_value = summary.get(metric, summary.get("mean"))
        if claimed_value is not None:
            claimed_summary = _finite_number(claimed_value)
            if claimed_summary is None:
                raise ClaimCIError(
                    f"results file {path} claimed summary for {metric!r} must be finite"
                )

    # Validate derived statistics while the artifact error can still be tied to
    # this result path.  ``check_results`` recomputes the summaries for its
    # comparison, but malformed numeric ranges must never escape as a raw
    # OverflowError from the parser.
    summarize_runs(values)

    try:
        resolved_path = path.resolve()
    except (OSError, ValueError) as exc:
        raise ClaimCIError(f"could not resolve results file {path}: {exc}") from exc

    return LoadedResults(
        path=resolved_path,
        values=tuple(values),
        seeds=tuple(seeds),
        claimed_summary=claimed_summary,
    )


def summarize_runs(values: Sequence[float]) -> MetricSummary:
    if not values:
        raise ClaimCIError("cannot summarize an empty run list")

    normalized: list[float] = []
    for index, value in enumerate(values, start=1):
        number = _finite_number(value)
        if number is None:
            raise ClaimCIError(f"run {index} metric must be finite")
        normalized.append(number)

    try:
        statistics_mean = statistics.fmean(normalized)
    except (OverflowError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ClaimCIError(f"could not compute finite summary statistics: {exc}") from exc
    if not math.isfinite(statistics_mean):
        raise ClaimCIError("could not compute finite summary statistics: mean is not finite")

    # ``statistics.fmean`` is useful for detecting unsupported overflow, but
    # its binary floating-point summation can leave a representational tail
    # (for example, 0.7000000000000001 for three decimal inputs).  Recompute
    # the public mean from each value's decimal spelling so threshold and
    # claimed-summary comparisons remain deterministic at the metric's scale.
    try:
        decimal_values = [Decimal(str(number)) for number in normalized]
        with localcontext() as context:
            context.prec = max(
                28,
                max(len(value.as_tuple().digits) for value in decimal_values)
                + len(str(len(decimal_values)))
                + 4,
            )
            decimal_mean = sum(decimal_values, Decimal(0)) / Decimal(len(decimal_values))
        mean = _finite_float(decimal_mean, label="summary mean")
    except (DecimalException, OverflowError, ValueError, TypeError) as exc:
        if isinstance(exc, ClaimCIError):
            raise
        raise ClaimCIError(f"could not compute finite summary statistics: {exc}") from exc

    if len(normalized) >= 2:
        try:
            standard_deviation = statistics.stdev(normalized)
        except (OverflowError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise ClaimCIError(
                f"could not compute finite summary statistics: {exc}"
            ) from exc
        if not math.isfinite(standard_deviation):
            raise ClaimCIError(
                "could not compute finite summary statistics: standard deviation is not finite"
            )
    else:
        standard_deviation = None

    return MetricSummary(
        count=len(normalized),
        mean=mean,
        standard_deviation=standard_deviation,
    )


def _summary_evidence(summary: MetricSummary) -> dict[str, float | int | None]:
    return asdict(summary)


def _summary_mismatch_finding(
    experiment: str,
    loaded: LoadedResults,
    summary: MetricSummary,
) -> Finding | None:
    claimed = loaded.claimed_summary
    if claimed is None:
        return None
    claimed_decimal = _decimal_number(claimed, label="claimed summary")
    recomputed_decimal = _decimal_number(summary.mean, label="recomputed mean")
    if claimed_decimal == recomputed_decimal:
        return None
    return Finding(
        rule_id="RESULT.SUMMARY_MISMATCH",
        severity=Severity.CRITICAL,
        title=f"{experiment.title()} claimed summary does not match raw runs",
        explanation=(
            f"The {experiment} summary reports {claimed:.6g}, but the raw-run mean "
            f"recomputes to {summary.mean:.6g}."
        ),
        evidence={
            "experiment": experiment,
            "claimed": claimed,
            "recomputed": summary.mean,
        },
        impact=Impact.INVALIDATES,
    )


def _seed_findings(experiment: str, loaded: LoadedResults) -> list[Finding]:
    findings: list[Finding] = []
    missing_indices = [index for index, seed in enumerate(loaded.seeds, start=1) if seed is None]
    if missing_indices:
        findings.append(
            Finding(
                rule_id="SEED.MISSING_METADATA",
                severity=Severity.WARNING,
                title=f"{experiment.title()} has missing seed metadata",
                explanation="Every raw run needs an integer seed to establish independent evidence.",
                evidence={"experiment": experiment, "run_indices": missing_indices},
                impact=Impact.INSUFFICIENT,
            )
        )

    present = [seed for seed in loaded.seeds if seed is not None]
    duplicate_counts = Counter(present)
    duplicates = sorted(seed for seed, count in duplicate_counts.items() if count > 1)
    if duplicates:
        findings.append(
            Finding(
                rule_id="SEED.DUPLICATE",
                severity=Severity.WARNING,
                title=f"{experiment.title()} repeats seed metadata",
                explanation="Repeated seeds do not constitute independent runs.",
                evidence={"experiment": experiment, "duplicate_seeds": duplicates},
                impact=Impact.INSUFFICIENT,
            )
        )

    if len(loaded.values) == 1:
        findings.append(
            Finding(
                rule_id="SEED.SINGLE_RUN",
                severity=Severity.WARNING,
                title=f"{experiment.title()} has only one run",
                explanation="A single run cannot provide a sample standard deviation.",
                evidence={"experiment": experiment, "run_count": 1},
                impact=Impact.INSUFFICIENT,
            )
        )
    return findings


def check_results(
    baseline: LoadedResults,
    candidate: LoadedResults,
    metric: str,
    minimum_improvement: float,
    direction: Direction | str = Direction.HIGHER,
) -> ResultCheck:
    normalized_direction = coerce_direction(direction)
    baseline_summary = summarize_runs(baseline.values)
    candidate_summary = summarize_runs(candidate.values)
    baseline_decimal = _decimal_number(baseline_summary.mean, label="baseline mean")
    candidate_decimal = _decimal_number(candidate_summary.mean, label="candidate mean")
    threshold_number = _finite_number(minimum_improvement)
    if threshold_number is None or threshold_number < 0:
        raise ClaimCIError(
            "minimum improvement must be a finite non-negative number"
        )
    threshold_decimal = _decimal_number(
        threshold_number, label="minimum improvement"
    )

    try:
        if normalized_direction is Direction.HIGHER:
            absolute_decimal = candidate_decimal - baseline_decimal
        else:
            absolute_decimal = baseline_decimal - candidate_decimal
    except (DecimalException, ValueError, TypeError) as exc:
        raise ClaimCIError("absolute improvement must be finite") from exc
    absolute_improvement = _finite_float(
        absolute_decimal, label="absolute improvement"
    )

    if baseline_decimal == 0:
        relative_improvement = None
    else:
        try:
            relative_decimal = absolute_decimal / abs(baseline_decimal)
            relative_improvement = float(relative_decimal)
        except (DecimalException, OverflowError, ValueError, TypeError, ZeroDivisionError):
            relative_improvement = None
        else:
            if not math.isfinite(relative_improvement):
                relative_improvement = None

    findings: list[Finding] = [
        Finding(
            rule_id="RESULT.RECOMPUTED",
            severity=Severity.VERIFIED,
            title=f"Raw {metric} recomputed successfully",
            explanation="ClaimCI calculated both experiment summaries from raw run values.",
            evidence={
                "baseline": _summary_evidence(baseline_summary),
                "candidate": _summary_evidence(candidate_summary),
            },
            impact=Impact.NONE,
        )
    ]

    for experiment, loaded, summary in (
        ("baseline", baseline, baseline_summary),
        ("candidate", candidate, candidate_summary),
    ):
        mismatch = _summary_mismatch_finding(experiment, loaded, summary)
        if mismatch is not None:
            findings.append(mismatch)
        findings.extend(_seed_findings(experiment, loaded))

    baseline_count = len(baseline.values)
    candidate_count = len(candidate.values)
    smaller = min(baseline_count, candidate_count)
    larger = max(baseline_count, candidate_count)
    if smaller and larger - smaller >= 2 and larger / smaller >= 3.0:
        findings.append(
            Finding(
                rule_id="SEED.IMBALANCE",
                severity=Severity.WARNING,
                title="Baseline and candidate evidence counts are severely imbalanced",
                explanation=(
                    f"Baseline has {baseline_count} runs while candidate has "
                    f"{candidate_count}."
                ),
                evidence={
                    "baseline_count": baseline_count,
                    "candidate_count": candidate_count,
                    "ratio": larger / smaller,
                },
                impact=Impact.NONE,
            )
        )

    has_seed_problem = any(finding.rule_id.startswith("SEED.") for finding in findings)
    if not has_seed_problem:
        findings.append(
            Finding(
                rule_id="SEED.EVIDENCE_VERIFIED",
                severity=Severity.VERIFIED,
                title="Seed evidence is complete",
                explanation="Both experiments contain multiple runs with unique integer seeds.",
                evidence={
                    "baseline_count": baseline_count,
                    "candidate_count": candidate_count,
                },
                impact=Impact.NONE,
            )
        )

    claim_evidence: dict[str, Any] = {
        "minimum_improvement": threshold_number,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "direction": normalized_direction.value,
    }
    directional_prefix = (
        f"Candidate minus baseline {metric} is {absolute_improvement:.6g}, "
        if normalized_direction is Direction.HIGHER
        else f"Baseline minus candidate {metric} is {absolute_improvement:.6g}, "
    )
    if absolute_decimal >= threshold_decimal:
        findings.append(
            Finding(
                rule_id="RESULT.CLAIM_SUPPORTED",
                severity=Severity.VERIFIED,
                title="Recomputed improvement meets the claim threshold",
                explanation=(
                    f"{directional_prefix}meeting the required "
                    f"{minimum_improvement:.6g}."
                ),
                evidence=claim_evidence,
                impact=Impact.NONE,
            )
        )
    else:
        findings.append(
            Finding(
                rule_id="RESULT.CLAIM_NOT_SUPPORTED",
                severity=Severity.CRITICAL,
                title="Recomputed improvement is below the claim threshold",
                explanation=(
                    f"{directional_prefix}below the required "
                    f"{minimum_improvement:.6g}."
                ),
                evidence=claim_evidence,
                impact=Impact.INVALIDATES,
            )
        )

    return ResultCheck(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        findings=tuple(findings),
    )
