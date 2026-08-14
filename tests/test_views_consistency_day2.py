"""End-to-end consistency across every public Day 2 report surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimci.audit import audit_research
from claimci.github import build_check_payload
from claimci.models import Verdict
from claimci.report import render_json, render_markdown


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    "valid",
    "compute_mismatch",
    "seed_mismatch",
    "data_leak",
    "fake_or_unsupported_result",
    "different_eval_dataset",
    "missing_result",
    "multiple_failures",
    "day2_demo",
)
CONCLUSIONS = {
    Verdict.SUPPORTED: "success",
    Verdict.NOT_SUPPORTED: "failure",
    Verdict.INSUFFICIENT_EVIDENCE: "action_required",
}


@pytest.mark.parametrize("fixture", FIXTURES)
def test_json_markdown_and_github_check_are_one_audit_view(fixture: str) -> None:
    result = audit_research(ROOT / "examples" / fixture / "research.yaml")
    machine = json.loads(render_json(result))
    markdown = render_markdown(result)
    check = build_check_payload(machine, markdown, "c" * 40)

    assert machine["verdict"] == result.verdict.value
    assert check["conclusion"] == CONCLUSIONS[result.verdict]
    assert check["output"]["summary"] == markdown
    assert result.verdict.value in markdown

    machine_rules = [finding["rule_id"] for finding in machine["findings"]]
    result_rules = [finding.rule_id for finding in result.findings]
    assert machine_rules == result_rules
    for rule_id in machine_rules:
        assert rule_id in markdown

    json.dumps(check, allow_nan=False)
