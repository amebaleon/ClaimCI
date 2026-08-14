"""Day 2 metric-direction contracts.

These tests intentionally exercise the public direction value, rather than a
private parser or arithmetic helper.  A lower-is-better claim must be the
same deterministic audit as a higher-is-better claim with the sign reversed;
omitting ``direction`` must remain byte-for-byte compatible with Day 1 JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from claimci.audit import audit_research, load_research_spec
from claimci.models import ClaimCIError, Direction, Verdict
from claimci.report import render_human, render_json
from claimci.result_check import check_results, load_results


METRIC = "accuracy"


def _write_results(
    root: Path,
    name: str,
    values: list[float],
    *,
    seeds: list[int | None] | None = None,
) -> Path:
    """Write a small raw-results artifact with deterministic seed metadata."""

    if seeds is None:
        seeds = list(range(1, len(values) + 1))
    assert len(seeds) == len(values)
    runs: list[dict[str, object]] = []
    for value, seed in zip(values, seeds):
        run: dict[str, object] = {METRIC: value}
        if seed is not None:
            run["seed"] = seed
        runs.append(run)
    path = root / name
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return path


def _finding(result: object, rule_id: str) -> object:
    return next(item for item in getattr(result, "findings") if item.rule_id == rule_id)


def _set_manifest_direction(manifest: Path, value: object) -> None:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["claim"]["direction"] = value
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_direction_enum_exposes_strict_lowercase_wire_values() -> None:
    assert Direction.HIGHER.value == "higher"
    assert Direction.LOWER.value == "lower"
    assert Direction("higher") is Direction.HIGHER
    assert Direction("lower") is Direction.LOWER
    with pytest.raises(ValueError):
        Direction("HIGHER")
    with pytest.raises(ValueError):
        Direction("sideways")


def test_manifest_omitted_direction_defaults_to_higher(study_factory) -> None:
    spec = load_research_spec(study_factory())

    assert spec.direction is Direction.HIGHER


def test_manifest_direction_strips_surrounding_whitespace(study_factory) -> None:
    manifest = study_factory()
    _set_manifest_direction(manifest, "  lower  ")

    spec = load_research_spec(manifest)

    assert spec.direction is Direction.LOWER


@pytest.mark.parametrize(
    "raw_direction",
    ["HIGHER", "Lower", "sideways", "", None, True, 1, {"value": "lower"}],
)
def test_manifest_rejects_every_noncanonical_direction_value(
    study_factory,
    raw_direction: object,
) -> None:
    manifest = study_factory()
    _set_manifest_direction(manifest, raw_direction)

    with pytest.raises(ClaimCIError, match="direction"):
        load_research_spec(manifest)


def test_check_results_four_argument_api_remains_higher_by_default(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.70, 0.70]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.75, 0.75]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05)

    assert result.absolute_improvement == pytest.approx(0.05)
    assert result.relative_improvement == pytest.approx(0.05 / 0.70)
    supported = _finding(result, "RESULT.CLAIM_SUPPORTED")
    assert supported.evidence["direction"] == "higher"


def test_check_results_explicit_higher_direction_preserves_boundary(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.70, 0.70]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.75, 0.75]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05, Direction.HIGHER)

    assert result.absolute_improvement == pytest.approx(0.05)
    assert _finding(result, "RESULT.CLAIM_SUPPORTED").impact.value == "NONE"


def test_check_results_lower_direction_reverses_absolute_and_relative_improvement(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.70, 0.70]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05, Direction.LOWER)

    assert result.absolute_improvement == pytest.approx(0.10)
    assert result.relative_improvement == pytest.approx(0.125)
    supported = _finding(result, "RESULT.CLAIM_SUPPORTED")
    assert supported.evidence["direction"] == "lower"
    assert supported.evidence["absolute_improvement"] == pytest.approx(0.10)


def test_check_results_lower_direction_accepts_exact_threshold_boundary(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.75, 0.75]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05, Direction.LOWER)

    assert result.absolute_improvement == pytest.approx(0.05)
    assert _finding(result, "RESULT.CLAIM_SUPPORTED").impact.value == "NONE"


def test_check_results_lower_direction_decline_is_negative_and_invalidating(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.80, 0.80]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.90, 0.90]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05, Direction.LOWER)

    assert result.absolute_improvement == pytest.approx(-0.10)
    assert result.relative_improvement == pytest.approx(-0.125)
    failed = _finding(result, "RESULT.CLAIM_NOT_SUPPORTED")
    assert failed.evidence["direction"] == "lower"
    assert failed.impact.value == "INVALIDATES"


def test_check_results_lower_direction_zero_baseline_has_no_relative_value(
    tmp_path: Path,
) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.0, 0.0]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [-0.05, -0.05]), METRIC)

    result = check_results(baseline, candidate, METRIC, 0.05, Direction.LOWER)

    assert result.absolute_improvement == pytest.approx(0.05)
    assert result.relative_improvement is None
    assert _finding(result, "RESULT.CLAIM_SUPPORTED").impact.value == "NONE"


def test_check_results_rejects_unknown_direction_value(tmp_path: Path) -> None:
    baseline = load_results(_write_results(tmp_path, "baseline.json", [0.70, 0.70]), METRIC)
    candidate = load_results(_write_results(tmp_path, "candidate.json", [0.75, 0.75]), METRIC)

    with pytest.raises(ClaimCIError, match="direction"):
        check_results(baseline, candidate, METRIC, 0.05, "sideways")


def test_audit_defaults_direction_to_higher_for_day1_manifests(study_factory) -> None:
    result = audit_research(study_factory())

    assert result.verdict is Verdict.SUPPORTED
    assert result.direction is Direction.HIGHER
    assert result.absolute_improvement == pytest.approx(0.10)


def test_audit_propagates_lower_direction_and_computes_favorable_delta(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, METRIC: 0.78},
            {"seed": 2, METRIC: 0.80},
            {"seed": 3, METRIC: 0.82},
        ],
        candidate_runs=[
            {"seed": 1, METRIC: 0.68},
            {"seed": 2, METRIC: 0.70},
            {"seed": 3, METRIC: 0.72},
        ],
    )
    _set_manifest_direction(manifest, "lower")

    result = audit_research(manifest)

    assert result.verdict is Verdict.SUPPORTED
    assert result.direction is Direction.LOWER
    assert result.absolute_improvement == pytest.approx(0.10)
    assert result.relative_improvement == pytest.approx(0.125)
    supported = _finding(result, "RESULT.CLAIM_SUPPORTED")
    assert supported.evidence["direction"] == "lower"


def test_audit_lower_direction_decline_is_not_supported(study_factory) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, METRIC: 0.78},
            {"seed": 2, METRIC: 0.80},
            {"seed": 3, METRIC: 0.82},
        ],
        candidate_runs=[
            {"seed": 1, METRIC: 0.88},
            {"seed": 2, METRIC: 0.90},
            {"seed": 3, METRIC: 0.92},
        ],
    )
    _set_manifest_direction(manifest, "lower")

    result = audit_research(manifest)

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert result.absolute_improvement == pytest.approx(-0.10)
    failed = _finding(result, "RESULT.CLAIM_NOT_SUPPORTED")
    assert failed.evidence["direction"] == "lower"


def test_human_lower_report_explicitly_explains_direction(study_factory) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, "loss": 0.78},
            {"seed": 2, "loss": 0.80},
            {"seed": 3, "loss": 0.82},
        ],
        candidate_runs=[
            {"seed": 1, "loss": 0.68},
            {"seed": 2, "loss": 0.70},
            {"seed": 3, "loss": 0.72},
        ],
    )
    # The fixture writes accuracy by default; replace both result artifacts
    # with the declared loss metric before loading the lower-direction claim.
    for relative, values in (
        ("base/results.json", [0.78, 0.80, 0.82]),
        ("candidate/results.json", [0.68, 0.70, 0.72]),
    ):
        path = manifest.parent / relative
        path.write_text(
            json.dumps(
                {"runs": [{"seed": i, "loss": value} for i, value in enumerate(values, 1)]}
            ),
            encoding="utf-8",
        )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["claim"]["metric"] = "loss"
    payload["claim"]["direction"] = "lower"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = render_human(audit_research(manifest))

    assert "candidate lowers loss by >=" in report
    assert "lower is better" in report.lower()
    assert "direction: lower" in report.lower()
    assert "Absolute improvement: 0.100000 metric units" in report


def test_default_higher_json_keeps_day1_claim_shape(study_factory) -> None:
    payload = json.loads(render_json(audit_research(study_factory())))

    assert payload["claim"] == {"metric": METRIC, "minimum_improvement": 0.05}


def test_lower_json_claim_records_direction_and_directional_metrics(study_factory) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, METRIC: 0.78},
            {"seed": 2, METRIC: 0.80},
            {"seed": 3, METRIC: 0.82},
        ],
        candidate_runs=[
            {"seed": 1, METRIC: 0.68},
            {"seed": 2, METRIC: 0.70},
            {"seed": 3, METRIC: 0.72},
        ],
    )
    _set_manifest_direction(manifest, "lower")

    payload = json.loads(render_json(audit_research(manifest)))

    assert payload["claim"] == {
        "metric": METRIC,
        "minimum_improvement": 0.05,
        "direction": "lower",
    }
    assert payload["metrics"]["absolute_improvement"] == pytest.approx(0.10)
    claim_finding = next(
        finding
        for finding in payload["findings"]
        if finding["rule_id"] == "RESULT.CLAIM_SUPPORTED"
    )
    assert claim_finding["evidence"]["direction"] == "lower"
