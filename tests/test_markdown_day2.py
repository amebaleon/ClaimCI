"""Day 2 Markdown and CLI contract tests.

These tests deliberately treat the JSON report as the machine-readable source of
truth.  A Markdown renderer may choose a different layout, but it must preserve
the same verdict, claim, metric summaries, improvement values, and stable finding
IDs without allowing artifact-controlled text to change the table structure.
"""

from __future__ import annotations

import json
from html import unescape
from pathlib import Path

import pytest

from claimci.cli import main
from claimci.models import (
    AuditResult,
    Direction,
    Finding,
    Impact,
    MetricSummary,
    Severity,
    Verdict,
)
from claimci.report import render_json, render_markdown


def _result(
    *,
    metric: str = "loss",
    verdict: Verdict = Verdict.SUPPORTED,
    direction: Direction = Direction.HIGHER,
) -> AuditResult:
    if verdict is Verdict.NOT_SUPPORTED:
        finding = Finding(
            rule_id="DATASET.EXACT_LEAKAGE",
            severity=Severity.CRITICAL,
            title="Evaluation leakage detected",
            explanation="An evaluation record appears in training data.",
            evidence={"overlap_count": 1},
            impact=Impact.INVALIDATES,
        )
    elif verdict is Verdict.INSUFFICIENT_EVIDENCE:
        finding = Finding(
            rule_id="SEED.SINGLE_RUN",
            severity=Severity.WARNING,
            title="Only one run is available",
            explanation="More independent seeds are required.",
            evidence={"experiment": "candidate", "count": 1},
            impact=Impact.INSUFFICIENT,
        )
    else:
        finding = Finding(
            rule_id="RESULT.CLAIM_SUPPORTED",
            severity=Severity.VERIFIED,
            title="Improvement meets threshold",
            explanation="The recomputed directional improvement meets the claim.",
            evidence={
                "minimum_improvement": 0.1,
                "absolute_improvement": 0.123456,
                "relative_improvement": 0.152415,
            },
            impact=Impact.NONE,
        )
    return AuditResult(
        verdict=verdict,
        findings=(finding,),
        manifest_path=Path("research.yaml"),
        metric=metric,
        minimum_improvement=0.1,
        direction=direction,
        baseline_metrics=MetricSummary(
            count=3,
            mean=0.810000,
            standard_deviation=0.012345,
        ),
        candidate_metrics=MetricSummary(
            count=3,
            mean=0.686544 if direction is Direction.LOWER else 0.933456,
            standard_deviation=0.006789,
        ),
        absolute_improvement=0.123456,
        relative_improvement=0.152415,
    )


def test_markdown_is_deterministic_and_preserves_json_values() -> None:
    result = _result()

    first = render_markdown(result)
    second = render_markdown(result)
    machine = json.loads(render_json(result))

    assert first == second
    assert first.startswith("## ClaimCI Audit\n")
    assert machine["verdict"] in first
    assert machine["claim"]["metric"] in first
    assert f"{machine['claim']['minimum_improvement']:.6f}" in first
    assert f"{machine['metrics']['baseline']['mean']:.6f}" in first
    assert f"{machine['metrics']['candidate']['mean']:.6f}" in first
    assert f"{machine['metrics']['absolute_improvement']:.6f}" in first
    assert f"{machine['metrics']['relative_improvement']:.6f}" in first
    assert "RESULT.CLAIM_SUPPORTED" in first


def test_markdown_reports_lower_direction_without_recomputing_values() -> None:
    result = _result(metric="loss", direction=Direction.LOWER)
    # A lower-direction result is represented by the same signed values that
    # the auditor computed; Markdown must not silently turn them into higher-is-
    # better wording or derive a new difference from the two means.
    markdown = render_markdown(result)

    assert "loss" in markdown
    assert "0.123456" in markdown
    assert "0.152415" in markdown or "15.24%" in markdown
    # The lower wording is required by the public contract even when the
    # dataclass is constructed directly (without a manifest round-trip).
    assert "lower" in markdown.casefold() or "less" in markdown.casefold()


@pytest.mark.parametrize("verdict", list(Verdict))
def test_markdown_always_exposes_verdict_and_all_finding_ids(verdict: Verdict) -> None:
    finding = Finding(
        rule_id="CONFIG.SUSPICIOUS|ROW",
        severity=Severity.CRITICAL,
        title="Artifact-controlled | title",
        explanation="line one\nline two",
        evidence={"payload": "value | <script>alert(1)</script>"},
        impact=Impact.INVALIDATES,
    )
    result = _result(verdict=verdict)
    result = AuditResult(**{**result.__dict__, "findings": (finding,)})

    markdown = render_markdown(result)

    assert verdict.value in markdown
    assert finding.rule_id in markdown


def test_markdown_escapes_artifact_controlled_table_and_evidence_text() -> None:
    result = _result(metric="accuracy | <script>alert(1)</script>\nnext")
    finding = Finding(
        rule_id="TEST.MARKDOWN_ESCAPE",
        severity=Severity.WARNING,
        title="bad | title",
        explanation="first line\n| second line",
        evidence={
            "unsafe": "x | y\n<script>alert('x')</script>",
            "backticks": "```\n| injected row",
        },
        impact=Impact.INSUFFICIENT,
    )
    result = AuditResult(**{**result.__dict__, "findings": (finding,)})

    markdown = render_markdown(result)

    # Literal control characters must not be able to create a new table row or
    # executable HTML.  Either Markdown backslash escaping or HTML escaping is
    # acceptable; the raw unsafe forms are not.
    assert "accuracy | <script>" not in markdown
    assert "<script>alert(1)</script>" not in markdown
    assert "x | y\n" not in markdown
    assert "<script>alert('x')</script>" not in markdown
    assert "| injected row" not in markdown
    assert "\\|" in markdown or "&#124;" in markdown or "&vert;" in markdown
    assert "&lt;script&gt;" in markdown or "\\<script>" in markdown


def test_markdown_evidence_renders_as_readable_json_not_entity_text() -> None:
    finding = Finding(
        rule_id="CONFIG.COMPUTE_MISMATCH",
        severity=Severity.CRITICAL,
        title="Compute mismatch",
        explanation="Candidate compute is too large.",
        evidence={
            "baseline_basis": "training_steps*batch_size",
            "markup": "__bold__ ~~gone~~",
            "ratio": 3.0,
        },
        impact=Impact.INVALIDATES,
    )
    result = AuditResult(**{**_result().__dict__, "findings": (finding,)})

    markdown = render_markdown(result)
    evidence_line = next(
        line for line in markdown.splitlines() if line.startswith("  - Evidence:")
    )
    encoded = evidence_line.split("<code>", 1)[1].rsplit("</code>", 1)[0]

    assert unescape(encoded) == (
        '{"baseline_basis":"training_steps*batch_size",'
        '"markup":"__bold__ ~~gone~~","ratio":3.0}'
    )
    assert "&#95;&#95;bold&#95;&#95;" in encoded
    assert "&#126;&#126;gone&#126;&#126;" in encoded
    assert "`{&quot;" not in evidence_line


def test_common_metric_name_renders_exactly_inside_markdown_table() -> None:
    markdown = render_markdown(_result(metric="f1_score:mean"))
    metric_row = next(line for line in markdown.splitlines() if line.startswith("| Metric"))
    encoded = metric_row.split("<code>", 1)[1].split("</code>", 1)[0]

    assert unescape(encoded) == "f1_score:mean"
    assert "`f1&#95;score" not in metric_row


def test_cli_markdown_output_and_exit_code(study_factory, capsys) -> None:
    manifest = study_factory(candidate_config_updates={"training_steps": 300})

    exit_code = main(["audit", str(manifest), "--markdown"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    assert captured.out.startswith("## ClaimCI Audit\n")
    assert "NOT_SUPPORTED" in captured.out


def test_cli_markdown_preserves_insufficient_evidence_exit(study_factory, capsys) -> None:
    manifest = study_factory(write_candidate_results=False)

    exit_code = main(["audit", str(manifest), "--markdown"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == ""
    assert "INSUFFICIENT_EVIDENCE" in captured.out
    assert "RESULT.MISSING" in captured.out


def test_cli_rejects_json_and_markdown_together(study_factory, capsys) -> None:
    manifest = study_factory()

    with pytest.raises(SystemExit) as raised:
        main(["audit", str(manifest), "--json", "--markdown"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "mutually exclusive" in captured.err
