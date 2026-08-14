"""Adversarial Day 2 tests for the GitHub adapter and report boundary.

These tests deliberately stay local: they exercise the pure Check Run payload
builder, its file-oriented CLI, and the static workflow contract without
calling GitHub.  The cases here cover failure boundaries that are easy to
accidentally turn into an optimistic green check (or an uncaught exception).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from claimci.github import MAX_SUMMARY_LENGTH, build_check_payload, main
from claimci.models import Finding, Impact, Severity, Verdict
from claimci.report import render_json, render_markdown
from tests.test_markdown_day2 import _result


TEST_SHA = "b" * 40


def _report(verdict: Verdict = Verdict.SUPPORTED) -> dict[str, Any]:
    """Return a renderer-shaped report for adapter tests."""

    return json.loads(render_json(_result(verdict=verdict)))


def _write_inputs(tmp_path: Path, *, verdict: Verdict = Verdict.SUPPORTED) -> tuple[Path, Path]:
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "summary.md"
    report_path.write_text(json.dumps(_report(verdict)), encoding="utf-8")
    markdown_path.write_text("## ClaimCI Audit\n\nsummary\n", encoding="utf-8")
    return report_path, markdown_path


def _run_adapter(
    report: Path | str,
    markdown: Path | str,
    output: Path | str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any] | None]:
    code = main(
        [
            "--report",
            str(report),
            "--markdown",
            str(markdown),
            "--head-sha",
            TEST_SHA,
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    if output != "-" and isinstance(output, Path) and output.is_file():
        return code, json.loads(output.read_text(encoding="utf-8"))
    # Keep stderr assertions useful to callers while avoiding a second helper
    # API just for the stdout path.
    assert captured.err == "" or code == 2
    return code, None


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8", "nul"])
def test_unreadable_report_is_a_published_failure_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """Bad report paths must not become a skipped or green Check Run."""

    _, markdown = _write_inputs(tmp_path)
    output = tmp_path / f"{kind}-payload.json"
    if kind == "missing":
        report: Path | str = tmp_path / "does-not-exist.json"
    elif kind == "directory":
        report_dir = tmp_path / "report-dir"
        report_dir.mkdir()
        report = report_dir
    elif kind == "invalid_utf8":
        invalid = tmp_path / "invalid.json"
        invalid.write_bytes(b"{\xff\n")
        report = invalid
    else:
        report = "\x00claimci-report.json"

    code, payload = _run_adapter(report, markdown, output, capsys)

    # A writable output is enough to publish an explicit integration failure;
    # the adapter itself must not hide the Check Run behind exit status 2.
    assert code == 0
    assert payload is not None
    assert payload["conclusion"] == "failure"
    assert payload["status"] == "completed"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8", "nul"])
def test_unreadable_markdown_is_a_published_failure_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """A missing or malformed summary must fail closed, never publish report text."""

    report, _ = _write_inputs(tmp_path)
    output = tmp_path / f"{kind}-payload.json"
    if kind == "missing":
        markdown: Path | str = tmp_path / "does-not-exist.md"
    elif kind == "directory":
        markdown_dir = tmp_path / "summary-dir"
        markdown_dir.mkdir()
        markdown = markdown_dir
    elif kind == "invalid_utf8":
        invalid = tmp_path / "invalid.md"
        invalid.write_bytes(b"summary\xff")
        markdown = invalid
    else:
        markdown = "\x00claimci-summary.md"

    code, payload = _run_adapter(report, markdown, output, capsys)

    assert code == 0
    assert payload is not None
    assert payload["conclusion"] == "failure"
    assert "trustworthy" in payload["output"]["summary"].casefold()
    json.dumps(payload, allow_nan=False)


def test_unwritable_or_nul_output_path_returns_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A payload that cannot be written is an adapter/integration error."""

    report, markdown = _write_inputs(tmp_path)
    missing_parent = tmp_path / "not-created" / "payload.json"
    code = main(
        [
            "--report",
            str(report),
            "--markdown",
            str(markdown),
            "--head-sha",
            TEST_SHA,
            "--output",
            str(missing_parent),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "adapter error" in captured.err.casefold()

    code = main(
        [
            "--report",
            str(report),
            "--markdown",
            str(markdown),
            "--head-sha",
            TEST_SHA,
            "--output",
            "\x00claimci-payload.json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "adapter error" in captured.err.casefold()


@pytest.mark.parametrize("verdict", list(Verdict))
def test_adapter_uses_verdict_not_claimci_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verdict: Verdict
) -> None:
    """Exit code 2 is shared by insufficient evidence and malformed input."""

    report, markdown = _write_inputs(tmp_path, verdict=verdict)
    output = tmp_path / f"{verdict.value}.json"
    code, payload = _run_adapter(report, markdown, output, capsys)

    assert code == 0
    assert payload is not None
    expected = {
        Verdict.SUPPORTED: "success",
        Verdict.NOT_SUPPORTED: "failure",
        Verdict.INSUFFICIENT_EVIDENCE: "action_required",
    }[verdict]
    assert payload["conclusion"] == expected

    # An unknown verdict has the same shape as a valid report but is an
    # integration failure, not insufficient evidence.
    unknown = _report()
    unknown["verdict"] = "UNKNOWN"
    assert build_check_payload(unknown, "summary", TEST_SHA)["conclusion"] == "failure"


def test_non_strict_input_never_leaks_nan_or_infinity_into_payload() -> None:
    """The Checks API payload must remain standards-compliant JSON."""

    report = _report()
    report["metrics"]["baseline"] = float("nan")
    report["metrics"]["candidate"] = float("inf")

    payload = build_check_payload(report, "summary", TEST_SHA)

    assert payload["conclusion"] == "failure"
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert json.loads(encoded) == payload


def test_summary_exact_limit_is_preserved_and_next_character_truncates() -> None:
    """Boundary behavior is exact and truncation is deterministic."""

    report = _report()
    exact = "x" * MAX_SUMMARY_LENGTH
    exact_payload = build_check_payload(report, exact, TEST_SHA)
    assert exact_payload["output"]["summary"] == exact
    assert len(exact_payload["output"]["summary"]) == MAX_SUMMARY_LENGTH

    oversized = exact + "y"
    first = build_check_payload(report, oversized, TEST_SHA)
    second = build_check_payload(report, oversized, TEST_SHA)
    assert first == second
    summary = first["output"]["summary"]
    assert len(summary) == MAX_SUMMARY_LENGTH
    assert summary.startswith(exact[:100])
    assert "summary truncated" in summary.casefold()


def test_payload_is_deterministic_and_strictly_serializable() -> None:
    """Repeated builds produce byte-identical JSON after canonical encoding."""

    report = _report(Verdict.NOT_SUPPORTED)
    markdown = "## ClaimCI Audit\n\n| evidence | stable |\n"
    first = build_check_payload(report, markdown, TEST_SHA, "https://ci.example/run/1")
    second = build_check_payload(report, markdown, TEST_SHA, "https://ci.example/run/1")
    assert first == second
    encoded_first = json.dumps(first, ensure_ascii=True, sort_keys=True, allow_nan=False)
    encoded_second = json.dumps(second, ensure_ascii=True, sort_keys=True, allow_nan=False)
    assert encoded_first == encoded_second


def test_markdown_escapes_backticks_links_and_controlled_rows() -> None:
    """Artifact text cannot close code spans, create links, or add rows."""

    finding = Finding(
        rule_id="TEST.MARKDOWN_ATTACK",
        severity=Severity.CRITICAL,
        title="`** [click](javascript:alert(1)) | title",
        explanation="line one\n| injected row\n```",
        evidence={
            "payload": "`\n| next row\n<script>alert(1)</script>",
            "url": "[click](javascript:alert(1))",
        },
        impact=Impact.INVALIDATES,
    )
    result = _result(metric="accuracy` | metric\n| injected", verdict=Verdict.NOT_SUPPORTED)
    result = result.__class__(**{**result.__dict__, "findings": (finding,)})

    markdown = render_markdown(result)

    assert "<script>" not in markdown
    assert "javascript:alert" not in markdown
    assert "[click](" not in markdown
    assert "| injected row" not in markdown
    assert "```" not in markdown
    # A malicious line break must be represented inline, not as a new table row.
    assert "accuracy` | metric\n" not in markdown
    assert "&#124;" in markdown or "\\|" in markdown


@pytest.mark.parametrize("verdict", list(Verdict))
def test_markdown_status_chrome_is_portable_to_windows_korean_console(
    verdict: Verdict,
) -> None:
    """Built-in status decoration must not crash a cp949 PowerShell capture."""

    markdown = render_markdown(_result(verdict=verdict))

    markdown.encode("cp949")


def _workflow() -> tuple[str, dict[str, Any]]:
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "claimci.yml"
    text = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def test_workflow_handles_ready_for_review_without_privileged_trigger() -> None:
    """Draft-to-ready PR transitions run the trusted data-only target job."""

    text, parsed = _workflow()
    on = parsed.get("on", parsed.get(True))
    assert isinstance(on, dict)
    pull_request = on.get("pull_request_target")
    assert isinstance(pull_request, dict)
    types = pull_request.get("types", [])
    assert "ready_for_review" in types
    assert "python -m pip install ./claimci-trusted" in text
    assert "--artifact-root pull-request" in text


def test_check_publication_failure_is_not_suppressed() -> None:
    """A failed Checks API call must fail the PR job visibly."""

    text, parsed = _workflow()
    steps = parsed["jobs"]["claimci"]["steps"]
    publish = next(step for step in steps if step.get("name", "").startswith("Publish ClaimCI Check"))
    assert publish.get("continue-on-error", False) is not True
    assert publish.get("if") == "always()"
    run = str(publish.get("run", ""))
    assert "gh api" in run
    assert "--input" in run
    assert "|| true" not in run
    assert "check-runs" in run
    assert "--input claimci-check.json" in run
    assert "if [ ! -s claimci-check.json ]; then" in run
    assert 'conclusion: "failure"' in run
    # The target workflow may publish only because it installs trusted base
    # code and treats the path-confined PR checkout as passive artifact data.
    assert "pull_request_target" in text
    assert "run: python -m pip install .\n" not in text
