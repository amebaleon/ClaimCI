"""Contract tests for raw result, seed, and improvement checks.

The fixtures in this module use hand-derived values so that a statistic or
threshold implementation cannot make its own expectation pass.  Findings are
asserted by their public rule id and policy impact because those values are
part of the report contract.
"""

import json
from pathlib import Path

import pytest

from claimci.audit import ClaimCIError
from claimci.models import Impact, Severity
from claimci.result_check import check_results, load_results, summarize_runs


METRIC = "accuracy"

RULE_RECOMPUTED = "RESULT.RECOMPUTED"
RULE_SUMMARY_MISMATCH = "RESULT.SUMMARY_MISMATCH"
RULE_CLAIM_SUPPORTED = "RESULT.CLAIM_SUPPORTED"
RULE_CLAIM_NOT_SUPPORTED = "RESULT.CLAIM_NOT_SUPPORTED"
RULE_MISSING_SEED = "SEED.MISSING_METADATA"
RULE_DUPLICATE_SEED = "SEED.DUPLICATE"
RULE_SINGLE_RUN = "SEED.SINGLE_RUN"
RULE_SEED_IMBALANCE = "SEED.IMBALANCE"
RULE_SEED_VERIFIED = "SEED.EVIDENCE_VERIFIED"


def _write_results(
    root: Path,
    name: str,
    values: list[float],
    *,
    seeds: list[int | None] | None = None,
    summary: dict[str, object] | None = None,
) -> Path:
    """Write a small raw-results artifact while preserving omitted seeds."""

    if seeds is None:
        seeds = list(range(1, len(values) + 1))
    assert len(seeds) == len(values)
    runs: list[dict[str, object]] = []
    for value, seed in zip(values, seeds):
        run: dict[str, object] = {METRIC: value}
        if seed is not None:
            run["seed"] = seed
        runs.append(run)
    payload: dict[str, object] = {"runs": runs}
    if summary is not None:
        payload["summary"] = summary
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _finding(result: object, rule_id: str) -> object:
    findings = getattr(result, "findings")
    return next(item for item in findings if item.rule_id == rule_id)


def test_summarize_runs_reports_literal_count_mean_and_sample_stdev() -> None:
    summary = summarize_runs([0.8, 0.9, 1.0])

    assert summary.count == 3
    assert summary.mean == pytest.approx(0.9)
    # Hand-derived sample stdev for [0.8, 0.9, 1.0], not population stdev.
    assert summary.standard_deviation == pytest.approx(0.1, rel=0.0, abs=1e-12)


def test_check_results_accepts_exact_absolute_threshold_boundary(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.70, 0.70]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.75, 0.75]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.absolute_improvement == pytest.approx(0.05)
    recomputed = _finding(result, RULE_RECOMPUTED)
    assert recomputed.severity is Severity.VERIFIED
    assert recomputed.impact is Impact.NONE
    assert recomputed.evidence["baseline"] == {
        "count": 2,
        "mean": 0.70,
        "standard_deviation": 0.0,
    }
    assert recomputed.evidence["candidate"] == {
        "count": 2,
        "mean": 0.75,
        "standard_deviation": 0.0,
    }
    supported = _finding(result, RULE_CLAIM_SUPPORTED)
    assert supported.severity is Severity.VERIFIED
    assert supported.impact is Impact.NONE
    assert supported.evidence["minimum_improvement"] == pytest.approx(0.05)
    assert supported.evidence["absolute_improvement"] == pytest.approx(0.05)
    assert supported.evidence["relative_improvement"] == pytest.approx(
        0.07142857142857142
    )
    assert RULE_CLAIM_NOT_SUPPORTED not in {
        finding.rule_id for finding in result.findings
    }


def test_absolute_threshold_does_not_use_a_passing_relative_percentage(
    tmp_path: Path,
) -> None:
    # +0.02 is below the absolute five-point threshold even though it is a
    # literal +10% relative improvement.
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.20, 0.20]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.22, 0.22]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.absolute_improvement == pytest.approx(0.02)
    assert result.relative_improvement == pytest.approx(0.10)
    finding = _finding(result, RULE_CLAIM_NOT_SUPPORTED)
    assert finding.severity is Severity.CRITICAL
    assert finding.impact is Impact.INVALIDATES
    assert finding.evidence["minimum_improvement"] == pytest.approx(0.05)
    assert finding.evidence["absolute_improvement"] == pytest.approx(0.02)
    assert finding.evidence["relative_improvement"] == pytest.approx(0.10)


def test_candidate_decline_is_a_signed_below_threshold_improvement(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.70, 0.75]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.absolute_improvement == pytest.approx(-0.075)
    assert result.relative_improvement == pytest.approx(-0.09375)
    finding = _finding(result, RULE_CLAIM_NOT_SUPPORTED)
    assert finding.rule_id == RULE_CLAIM_NOT_SUPPORTED
    assert finding.severity is Severity.CRITICAL
    assert finding.impact is Impact.INVALIDATES
    assert finding.evidence["minimum_improvement"] == pytest.approx(0.05)
    assert finding.evidence["absolute_improvement"] == pytest.approx(-0.075)
    assert finding.evidence["relative_improvement"] == pytest.approx(-0.09375)


def test_zero_baseline_reports_no_relative_improvement_and_keeps_absolute_rule(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.0, 0.0]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.05, 0.05]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.baseline_summary.mean == 0.0
    assert result.candidate_summary.mean == 0.05
    assert result.absolute_improvement == 0.05
    assert result.relative_improvement is None
    supported = _finding(result, RULE_CLAIM_SUPPORTED)
    assert supported.severity is Severity.VERIFIED
    assert supported.impact is Impact.NONE
    assert supported.evidence["minimum_improvement"] == pytest.approx(0.05)
    assert supported.evidence["absolute_improvement"] == pytest.approx(0.05)
    assert supported.evidence["relative_improvement"] is None
    assert RULE_CLAIM_NOT_SUPPORTED not in {
        finding.rule_id for finding in result.findings
    }


def test_claimed_summary_mismatch_is_critical_and_invalidating(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC
    )
    candidate = load_results(
        _write_results(
            tmp_path,
            "candidate.json",
            [0.80, 0.80],
            summary={"mean": 0.55},
        ),
        METRIC,
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    finding = _finding(result, RULE_SUMMARY_MISMATCH)
    assert finding.severity is Severity.CRITICAL
    assert finding.impact is Impact.INVALIDATES
    assert finding.evidence["experiment"] == "candidate"
    assert finding.evidence["claimed"] == pytest.approx(0.55)
    assert finding.evidence["recomputed"] == pytest.approx(0.80)


@pytest.mark.parametrize(
    ("raw_text", "expected_fragment"),
    [
        ('{"runs":', "JSON"),
        ('{"summary": {"mean": 0.8}}', "runs"),
        ('{"runs": [{"seed": 1}]}', METRIC),
        ('{"runs": [{"seed": 1, "accuracy": NaN}]}', "finite"),
        ('{"runs": [{"seed": 1, "accuracy": Infinity}]}', "finite"),
    ],
)
def test_load_results_rejects_malformed_missing_or_nonfinite_raw_metric(
    tmp_path: Path,
    raw_text: str,
    expected_fragment: str,
) -> None:
    path = tmp_path / "bad-results.json"
    path.write_text(raw_text, encoding="utf-8")

    with pytest.raises(ClaimCIError, match=expected_fragment):
        load_results(path, METRIC)


def test_load_results_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ClaimCIError, match="results"):
        load_results(tmp_path / "missing.json", METRIC)


def test_missing_seed_metadata_is_a_warning_with_insufficient_impact(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC
    )
    candidate = load_results(
        _write_results(
            tmp_path,
            "candidate.json",
            [0.85, 0.86],
            seeds=[11, None],
        ),
        METRIC,
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    finding = _finding(result, RULE_MISSING_SEED)
    assert finding.severity is Severity.WARNING
    assert finding.impact is Impact.INSUFFICIENT
    assert finding.evidence["experiment"] == "candidate"


def test_duplicate_seed_metadata_is_a_warning_with_insufficient_impact(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC
    )
    candidate = load_results(
        _write_results(
            tmp_path,
            "candidate.json",
            [0.85, 0.86],
            seeds=[11, 11],
        ),
        METRIC,
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    finding = _finding(result, RULE_DUPLICATE_SEED)
    assert finding.severity is Severity.WARNING
    assert finding.impact is Impact.INSUFFICIENT
    assert finding.evidence["experiment"] == "candidate"


def test_one_run_experiment_is_insufficient_even_when_raw_mean_is_available(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.90, 0.91]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.baseline_summary.count == 1
    assert result.baseline_summary.mean == 0.80
    assert result.baseline_summary.standard_deviation is None
    finding = _finding(result, RULE_SINGLE_RUN)
    assert finding.severity is Severity.WARNING
    assert finding.impact is Impact.INSUFFICIENT
    assert finding.evidence["experiment"] == "baseline"


def test_five_to_one_seed_count_imbalance_is_warning_without_invalidating_claim(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80] * 5), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.85]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    finding = _finding(result, RULE_SEED_IMBALANCE)
    assert finding.severity is Severity.WARNING
    assert finding.impact is Impact.NONE
    assert finding.evidence["baseline_count"] == 5
    assert finding.evidence["candidate_count"] == 1
    assert finding.evidence["ratio"] == pytest.approx(5.0)


def test_equal_means_with_positive_tiny_threshold_are_not_supported(
    tmp_path: Path,
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.80, 0.80]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 5e-13)

    assert result.absolute_improvement == 0.0
    finding = _finding(result, RULE_CLAIM_NOT_SUPPORTED)
    assert finding.impact is Impact.INVALIDATES


def test_exact_decimal_threshold_boundary_remains_supported(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.10, 0.10]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [0.15, 0.15]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.absolute_improvement == pytest.approx(0.05)
    assert _finding(result, RULE_CLAIM_SUPPORTED).impact is Impact.NONE
    assert RULE_CLAIM_NOT_SUPPORTED not in {
        finding.rule_id for finding in result.findings
    }


def test_tiny_claimed_summary_difference_is_a_mismatch(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [0.0, 0.0]), METRIC
    )
    candidate = load_results(
        _write_results(
            tmp_path,
            "candidate.json",
            [0.0, 0.0],
            summary={"mean": 5e-13},
        ),
        METRIC,
    )

    result = check_results(baseline, candidate, METRIC, 0.0)

    finding = _finding(result, RULE_SUMMARY_MISMATCH)
    assert finding.impact is Impact.INVALIDATES
    assert finding.evidence["claimed"] == 5e-13
    assert finding.evidence["recomputed"] == 0.0


def test_load_results_wraps_decode_errors_as_claimci_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-encoding.json"
    path.write_bytes(b"{\xff")

    with pytest.raises(ClaimCIError, match="read|decode|results"):
        load_results(path, METRIC)


def test_load_results_wraps_nul_path_value_errors_as_claimci_errors() -> None:
    with pytest.raises(ClaimCIError, match="results"):
        load_results(Path("\0"), METRIC)


def test_load_results_rejects_huge_integer_conversion(tmp_path: Path) -> None:
    path = tmp_path / "huge-integer.json"
    path.write_text(
        json.dumps({"runs": [{"seed": 1, METRIC: 10**400}]}),
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="finite|statistics|results"):
        load_results(path, METRIC)


def test_summarize_runs_rejects_statistics_overflow() -> None:
    with pytest.raises(ClaimCIError, match="statistics|summary|finite"):
        summarize_runs([1.7e308, -1.7e308])


def test_absolute_improvement_overflow_is_a_claimci_error(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [1.7e308]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [-1.7e308]), METRIC
    )

    with pytest.raises(ClaimCIError, match="improvement|finite"):
        check_results(baseline, candidate, METRIC, 0.0)


def test_relative_improvement_overflow_is_reported_as_none(tmp_path: Path) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline.json", [1e-320]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate.json", [1.0]), METRIC
    )

    result = check_results(baseline, candidate, METRIC, 0.0)

    assert result.relative_improvement is None
    assert result.absolute_improvement == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), True, False])
def test_direct_check_results_rejects_invalid_claim_thresholds(
    tmp_path: Path, threshold: object
) -> None:
    baseline = load_results(
        _write_results(tmp_path, "baseline-threshold.json", [1.0, 1.0]), METRIC
    )
    candidate = load_results(
        _write_results(tmp_path, "candidate-threshold.json", [0.0, 0.0]), METRIC
    )

    with pytest.raises(ClaimCIError, match="finite non-negative"):
        check_results(baseline, candidate, METRIC, threshold)  # type: ignore[arg-type]
