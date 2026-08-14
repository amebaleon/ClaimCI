"""GitHub Check Run adapter and workflow contract tests for Day 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from claimci.github import build_check_payload, verdict_to_conclusion
from claimci.models import Verdict
from claimci.report import render_json
from tests.test_markdown_day2 import _result


TEST_SHA = "a" * 40


@pytest.mark.parametrize(
    ("verdict", "conclusion"),
    [
        (Verdict.SUPPORTED, "success"),
        ("SUPPORTED", "success"),
        (Verdict.NOT_SUPPORTED, "failure"),
        ("NOT_SUPPORTED", "failure"),
        (Verdict.INSUFFICIENT_EVIDENCE, "action_required"),
        ("INSUFFICIENT_EVIDENCE", "action_required"),
    ],
)
def test_verdict_mapping_is_explicit_and_nonpassing_is_distinct(
    verdict: Verdict | str, conclusion: str
) -> None:
    assert verdict_to_conclusion(verdict) == conclusion


@pytest.mark.parametrize("unknown", [None, "", "UNKNOWN", object(), 3])
def test_unknown_verdicts_map_to_failure(unknown: object) -> None:
    assert verdict_to_conclusion(unknown) == "failure"


def _json_report(verdict: Verdict = Verdict.SUPPORTED) -> dict[str, Any]:
    return json.loads(render_json(_result(verdict=verdict)))


def test_check_payload_uses_head_sha_and_clean_summary() -> None:
    markdown = "## ClaimCI Audit\n\nSUPPORTED\n"
    payload = build_check_payload(
        _json_report(), markdown, TEST_SHA, details_url="https://ci.example/run/7"
    )

    assert payload["name"] == "ClaimCI Audit"
    assert payload["head_sha"] == TEST_SHA
    assert payload["status"] == "completed"
    assert payload["conclusion"] == "success"
    assert payload["output"]["title"]
    assert payload["output"]["summary"] == markdown
    assert payload["details_url"] == "https://ci.example/run/7"
    # The returned object must be accepted by strict JSON encoders.
    json.dumps(payload, allow_nan=False)


def test_check_payload_accepts_audit_result_without_a_render_round_trip() -> None:
    result = _result(verdict=Verdict.INSUFFICIENT_EVIDENCE)

    payload = build_check_payload(result, "## Audit\n", TEST_SHA)

    assert payload["conclusion"] == "action_required"


@pytest.mark.parametrize("verdict", list(Verdict))
def test_check_payload_conclusion_follows_report_verdict(verdict: Verdict) -> None:
    payload = build_check_payload(_json_report(verdict), "summary", TEST_SHA)

    expected = {
        Verdict.SUPPORTED: "success",
        Verdict.NOT_SUPPORTED: "failure",
        Verdict.INSUFFICIENT_EVIDENCE: "action_required",
    }[verdict]
    assert payload["conclusion"] == expected


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"verdict": "UNKNOWN"},
        {"verdict": None},
        {"verdict": "SUPPORTED", "metrics": {"bad": float("nan")}},
        [],
        None,
    ],
)
def test_malformed_or_unknown_report_produces_failure_payload(report: object) -> None:
    payload = build_check_payload(report, "summary", TEST_SHA)

    assert payload["conclusion"] == "failure"
    assert payload["status"] == "completed"
    assert isinstance(payload["output"]["summary"], str)
    json.dumps(payload, allow_nan=False)


def test_missing_head_sha_produces_failure_payload() -> None:
    payload = build_check_payload(_json_report(), "summary", "")

    assert payload["conclusion"] == "failure"
    assert "head" in payload["output"]["summary"].casefold()


@pytest.mark.parametrize("head_sha", ["deadbeef", "g" * 40, "a" * 39, "a" * 65])
def test_malformed_head_sha_produces_failure_payload(head_sha: str) -> None:
    payload = build_check_payload(_json_report(), "summary", head_sha)

    assert payload["conclusion"] == "failure"
    assert "head" in payload["output"]["summary"].casefold()


def test_structurally_minimal_forged_supported_report_fails_closed() -> None:
    forged = {
        "verdict": "SUPPORTED",
        "claim": {},
        "metrics": {},
        "findings": [],
    }

    payload = build_check_payload(forged, "summary", TEST_SHA)

    assert payload["conclusion"] == "failure"


def test_nonfinite_in_memory_audit_result_cannot_become_success() -> None:
    result = _result()
    malformed = result.__class__(
        **{
            **result.__dict__,
            "baseline_metrics": result.baseline_metrics.__class__(
                count=3,
                mean=float("nan"),
                standard_deviation=0.1,
            ),
        }
    )

    payload = build_check_payload(malformed, "summary", TEST_SHA)

    assert payload["conclusion"] == "failure"


def test_check_summary_is_deterministically_bounded() -> None:
    oversized = "x" * 100_000

    first = build_check_payload(_json_report(), oversized, TEST_SHA)
    second = build_check_payload(_json_report(), oversized, TEST_SHA)

    summary = first["output"]["summary"]
    assert first == second
    assert len(summary) <= 65_535
    assert "truncat" in summary.casefold()
    json.dumps(first, allow_nan=False)


def _workflow_text() -> str:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "claimci.yml"
    return workflow.read_text(encoding="utf-8")


def _workflow_mapping() -> dict[str, Any]:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "claimci.yml"
    parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_workflow_uses_trusted_pr_target_context() -> None:
    text = _workflow_text()

    assert "pull_request_target:" in text
    assert "workflow_dispatch" not in text


def test_workflow_has_minimal_read_and_check_write_permissions() -> None:
    parsed = _workflow_mapping()
    permissions = parsed.get("permissions")
    if permissions is None:
        permissions = parsed.get(True)
    assert permissions == {"contents": "read", "checks": "write"}


def test_workflow_checks_out_pr_head_and_writes_job_summary() -> None:
    text = _workflow_text()

    assert "actions/checkout@" in text
    assert "github.event.pull_request.head.sha" in text
    assert "actions/setup-python@" in text
    assert "pip install" in text
    assert "research.yaml" in text
    assert "--markdown" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text


def test_workflow_publishes_check_via_input_file_for_every_pr() -> None:
    text = _workflow_text()

    # Report content must travel through --input rather than shell -f/-d
    # interpolation, which would permit metric/evidence text to alter a request.
    assert "gh api" in text
    assert "--input" in text
    assert "check-runs" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" not in text
    # The explicit check must be anchored to the PR head, not a merge commit.
    assert "--head-sha" in text
    assert "Fork pull-request Check limitation" not in text


def test_workflow_installs_trusted_base_auditor_and_treats_pr_as_data() -> None:
    text = _workflow_text()

    assert "github.event.pull_request.base.sha" in text
    assert "path: claimci-trusted" in text
    assert "path: pull-request" in text
    assert "allow-unsafe-pr-checkout: true" in text
    assert "python -m pip install ./claimci-trusted" in text
    assert "run: python -m pip install .\n" not in text
    assert '--artifact-root pull-request' in text


def test_workflow_preserves_valid_nonzero_audit_verdicts() -> None:
    text = _workflow_text()

    # ClaimCI deliberately returns 1 and 2 for valid NOT_SUPPORTED and
    # INSUFFICIENT_EVIDENCE reports.  Only disagreement between the two
    # deterministic render passes is an adapter failure.
    assert 'if [ "$json_status" -ne "$markdown_status" ]; then' in text
    assert 'if [ "$markdown_status" -ne 0 ]; then' not in text


def test_workflow_has_no_duplicate_scientific_verdict_enforcement_step() -> None:
    parsed = _workflow_mapping()
    steps = parsed["jobs"]["claimci"]["steps"]
    names = [step.get("name") for step in steps]

    # The explicit ClaimCI Audit Check Run is authoritative.  Once the
    # adapter and GitHub publication succeed, a valid scientific verdict must
    # not independently turn the Actions job red.
    assert "Enforce ClaimCI verdict" not in names


def test_workflow_always_publishes_the_check_after_adapter_failure() -> None:
    parsed = _workflow_mapping()
    steps = parsed["jobs"]["claimci"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish ClaimCI Check")

    # Invalid adapter input returns nonzero but still writes a failure payload;
    # always() ensures that payload is published before the job remains failed.
    assert publish.get("if") == "always()"
