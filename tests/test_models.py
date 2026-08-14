"""Behavioral contract tests for ClaimCI's immutable model layer.

These tests intentionally exercise the public model values rather than any
implementation details.  The rule and enum values are serialized by the CLI,
so changing them would make reports and downstream consumers incompatible.
"""

from dataclasses import FrozenInstanceError

import pytest

from claimci.models import (
    AuditResult,
    Finding,
    Impact,
    MetricSummary,
    OverlapSummary,
    Severity,
    Verdict,
)


def test_severity_impact_and_verdict_values_are_stable() -> None:
    assert {member.name: member.value for member in Severity} == {
        "CRITICAL": "CRITICAL",
        "WARNING": "WARNING",
        "VERIFIED": "VERIFIED",
        "INFO": "INFO",
    }
    assert {member.name: member.value for member in Impact} == {
        "NONE": "NONE",
        "INVALIDATES": "INVALIDATES",
        "INSUFFICIENT": "INSUFFICIENT",
    }
    assert {member.name: member.value for member in Verdict} == {
        "SUPPORTED": "SUPPORTED",
        "NOT_SUPPORTED": "NOT_SUPPORTED",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    }


def test_finding_is_frozen_and_preserves_stable_rule_and_policy_fields() -> None:
    finding = Finding(
        rule_id="CONFIG.COMPUTE_MISMATCH",
        severity=Severity.CRITICAL,
        title="Candidate compute is too large",
        explanation="The candidate uses at least 1.5x the baseline compute proxy.",
        evidence={"baseline_proxy": 100.0, "candidate_proxy": 300.0, "ratio": 3.0},
        impact=Impact.INVALIDATES,
    )

    assert finding.rule_id == "CONFIG.COMPUTE_MISMATCH"
    assert finding.severity is Severity.CRITICAL
    assert finding.impact is Impact.INVALIDATES
    assert finding.evidence["ratio"] == 3.0

    with pytest.raises(FrozenInstanceError):
        finding.title = "mutated"  # type: ignore[misc]


def test_metric_summary_is_frozen_and_exposes_sample_statistics() -> None:
    summary = MetricSummary(count=3, mean=0.8, standard_deviation=0.1)

    assert summary.count == 3
    assert summary.mean == pytest.approx(0.8)
    assert summary.standard_deviation == pytest.approx(0.1)
    with pytest.raises(FrozenInstanceError):
        summary.mean = 0.0  # type: ignore[misc]


def test_overlap_summary_is_frozen_and_reports_duplicate_aware_rate() -> None:
    summary = OverlapSummary(
        experiment="baseline",
        train_count=4,
        eval_count=5,
        overlap_count=2,
        overlap_rate=0.4,
    )

    assert summary.experiment == "baseline"
    assert summary.train_count == 4
    assert summary.eval_count == 5
    assert summary.overlap_count == 2
    assert summary.overlap_rate == pytest.approx(0.4)
    with pytest.raises(FrozenInstanceError):
        summary.overlap_count = 0  # type: ignore[misc]


def test_audit_result_is_frozen_and_keeps_verdict_and_findings() -> None:
    finding = Finding(
        rule_id="CONFIG.FAIRNESS_VERIFIED",
        severity=Severity.VERIFIED,
        title="Configs match",
        explanation="All supported fairness fields match.",
        evidence={},
        impact=Impact.NONE,
    )
    result = AuditResult(verdict=Verdict.SUPPORTED, findings=(finding,))

    assert result.verdict is Verdict.SUPPORTED
    assert result.findings == (finding,)
    with pytest.raises(FrozenInstanceError):
        result.verdict = Verdict.NOT_SUPPORTED  # type: ignore[misc]
