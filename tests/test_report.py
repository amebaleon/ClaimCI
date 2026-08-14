from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from claimci.audit import audit_research
from claimci.models import AuditResult, Finding, Impact, MetricSummary, Severity, Verdict
from claimci.report import render_human, render_json


def test_human_report_separates_absolute_and_relative_improvement(study_factory) -> None:
    result = audit_research(study_factory())

    report = render_human(result)

    assert report.startswith("ClaimCI Audit\n")
    assert "Claim:" in report
    assert "candidate improves accuracy by >= 5.00 percentage points" in report
    assert "Verdict:\nSUPPORTED" in report
    assert "Absolute improvement: 0.100000 (10.00 percentage points)" in report
    assert "Relative improvement: 16.67%" in report
    assert "VERIFIED" in report
    assert "INFO" in report


def test_json_report_is_structured_and_deterministic(study_factory) -> None:
    result = audit_research(study_factory())

    first = render_json(result)
    second = render_json(result)
    payload = json.loads(first)

    assert first == second
    assert payload["verdict"] == "SUPPORTED"
    assert payload["claim"] == {"metric": "accuracy", "minimum_improvement": 0.05}
    assert payload["metrics"]["baseline"]["count"] == 3
    assert payload["metrics"]["absolute_improvement"] == pytest.approx(0.10)
    assert all("rule_id" in finding and "evidence" in finding for finding in payload["findings"])


class _UnknownScalar:
    def __str__(self) -> str:
        return "opaque"


def test_reports_normalize_non_json_evidence_without_nonfinite_tokens() -> None:
    finding = Finding(
        rule_id="TEST.REPORT_NORMALIZATION",
        severity=Severity.WARNING,
        title="Evidence needs normalization",
        explanation="Evidence includes values outside the JSON-native scalar set.",
        evidence={
            1: {
                "nan": math.nan,
                "positive_inf": math.inf,
                "negative_inf": -math.inf,
                "path": Path("artifacts/results.json"),
                "tuple": ("nested", 3),
                7: _UnknownScalar(),
            },
            "mixed": {2: ("tuple", math.nan)},
        },
        impact=Impact.NONE,
    )
    result = AuditResult(verdict=Verdict.SUPPORTED, findings=(finding,))

    human = render_human(result)
    first = render_json(result)
    second = render_json(result)
    payload = json.loads(first)
    evidence = payload["findings"][0]["evidence"]

    assert first == second
    assert "Evidence:" in human
    assert "NaN" in human and "Infinity" in human
    assert evidence["1"] == {
        "7": "opaque",
        "nan": "NaN",
        "negative_inf": "-Infinity",
        "path": str(Path("artifacts/results.json")),
        "positive_inf": "Infinity",
        "tuple": ["nested", 3],
    }
    assert evidence["mixed"]["2"] == ["tuple", "NaN"]
    assert "NaN" in first and "Infinity" in first
    assert ": NaN" not in first and ": Infinity" not in first


def test_human_report_uses_metric_units_for_non_accuracy_metrics() -> None:
    result = AuditResult(
        verdict=Verdict.SUPPORTED,
        findings=(),
        metric="loss",
        minimum_improvement=0.1,
        baseline_metrics=MetricSummary(count=2, mean=1.0, standard_deviation=0.1),
        candidate_metrics=MetricSummary(count=2, mean=0.8, standard_deviation=0.1),
        absolute_improvement=0.2,
        relative_improvement=0.2,
    )

    report = render_human(result)

    assert "candidate improves loss by >= 0.100000 metric units" in report
    assert "Absolute improvement: 0.200000 metric units" in report
    assert "percentage points" not in report


def test_json_report_normalizes_yaml_sets_in_stable_sorted_order() -> None:
    finding = Finding(
        rule_id="TEST.SET_EVIDENCE",
        severity=Severity.INFO,
        title="Set-valued YAML evidence",
        explanation="Safe YAML can construct sets, so reports must order them.",
        evidence={"values": {"delta", "alpha", "beta"}},
        impact=Impact.NONE,
    )
    result = AuditResult(verdict=Verdict.SUPPORTED, findings=(finding,))

    first = render_json(result)
    second = render_json(result)
    evidence = json.loads(first)["findings"][0]["evidence"]

    assert first == second
    assert evidence["values"] == ["alpha", "beta", "delta"]
