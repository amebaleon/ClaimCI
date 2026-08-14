"""GitHub Check Run integration for ClaimCI.

The adapter deliberately has no dependency on the GitHub SDK.  It turns the
machine-readable ClaimCI report and its already-rendered Markdown summary into
the small JSON object accepted by ``POST /repos/{owner}/{repo}/check-runs``.
All validation happens before a conclusion is selected: an unknown verdict,
malformed report, missing head SHA, or unusable summary is a failure rather
than a silently optimistic check.

``python -m claimci.github`` is the workflow-facing entry point.  It writes a
payload file even when the input report is invalid, allowing the workflow to
publish a trustworthy ``failure`` Check Run instead of losing the result to a
shell/JSON parsing error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import AuditResult, Verdict
from .parsing import unique_json_object


# GitHub's Check Run ``output.summary`` limit is 65,535 characters.  Keep the
# value in one place so both the pure adapter and its CLI obey the same bound.
MAX_SUMMARY_LENGTH = 65_535
CHECK_NAME = "ClaimCI Audit"
_HEAD_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_SEVERITIES = {"CRITICAL", "WARNING", "VERIFIED", "INFO"}
_IMPACTS = {"NONE", "INVALIDATES", "INSUFFICIENT"}


def verdict_to_conclusion(verdict: object) -> str:
    """Map a ClaimCI verdict to an explicit GitHub Check conclusion.

    ``action_required`` is intentionally used for insufficient evidence.  It
    is non-passing while remaining distinguishable from a claim that was
    actively rejected.  Anything outside the public verdict enum is treated
    as an integration failure.
    """

    if isinstance(verdict, Verdict):
        normalized = verdict.value
    elif isinstance(verdict, str):
        normalized = verdict
    else:
        normalized = None
    return {
        Verdict.SUPPORTED.value: "success",
        Verdict.NOT_SUPPORTED.value: "failure",
        Verdict.INSUFFICIENT_EVIDENCE.value: "action_required",
    }.get(normalized, "failure")


def _strict_json_safe(value: object) -> bool:
    """Return whether *value* can be encoded as strict JSON.

    ``allow_nan=False`` is important here: Python's default JSON encoder emits
    NaN/Infinity tokens that are not valid JSON accepted by the Checks API.
    """

    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


def _finite_number(value: object) -> bool:
    """Return whether *value* is a finite JSON number (never a bool)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _valid_metric_summary(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    count = value.get("count")
    deviation = value.get("standard_deviation")
    return (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 1
        and _finite_number(value.get("mean"))
        and (
            deviation is None
            or (_finite_number(deviation) and float(deviation) >= 0.0)
        )
    )


def _valid_finding(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("rule_id"), str)
        and bool(value["rule_id"].strip())
        and value.get("severity") in _SEVERITIES
        and value.get("impact") in _IMPACTS
        and isinstance(value.get("title"), str)
        and isinstance(value.get("explanation"), str)
        and isinstance(value.get("evidence"), Mapping)
    )


def _valid_overlap(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    counts = [
        value.get(name)
        for name in ("train_count", "eval_count", "overlap_count")
    ]
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in counts
    ):
        return False
    return (
        isinstance(value.get("experiment"), str)
        and bool(value["experiment"].strip())
        and _finite_number(value.get("overlap_rate"))
        and 0.0 <= float(value["overlap_rate"]) <= 1.0
        and counts[2] <= counts[0]
        and counts[2] <= counts[1]
        and isinstance(value.get("overlapping_hashes", []), list)
    )


def _valid_report(report: object) -> tuple[bool, str | None, str | None]:
    """Validate and extract the report verdict.

    Reports are produced by :func:`claimci.report.render_json`; requiring its
    stable top-level sections catches truncated or unrelated JSON before it
    can be published as a successful Check Run.  Extra keys remain allowed for
    forward compatibility.
    """

    if not isinstance(report, Mapping):
        return False, None, "report is not a JSON object"
    if not _strict_json_safe(report):
        return False, None, "report is not strict JSON"

    verdict = report.get("verdict")
    if not isinstance(verdict, str) or verdict not in {
        Verdict.SUPPORTED.value,
        Verdict.NOT_SUPPORTED.value,
        Verdict.INSUFFICIENT_EVIDENCE.value,
    }:
        return False, None, "report has an unknown verdict"

    manifest = report.get("manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        return False, None, "report is missing a valid manifest"

    claim = report.get("claim")
    if not isinstance(claim, Mapping):
        return False, None, "report is missing a valid claim section"
    metric = claim.get("metric")
    threshold = claim.get("minimum_improvement")
    direction = claim.get("direction", "higher")
    if not isinstance(metric, str) or not metric.strip():
        return False, None, "report claim has an invalid metric"
    if not _finite_number(threshold) or float(threshold) < 0.0:
        return False, None, "report claim has an invalid minimum improvement"
    if direction not in {"higher", "lower"}:
        return False, None, "report claim has an invalid direction"

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, None, "report is missing a valid metrics section"
    for arm in ("baseline", "candidate"):
        summary = metrics.get(arm)
        if summary is not None and not _valid_metric_summary(summary):
            return False, None, f"report has an invalid {arm} metric summary"
    for name in ("absolute_improvement", "relative_improvement"):
        value = metrics.get(name)
        if value is not None and not _finite_number(value):
            return False, None, f"report has an invalid {name}"

    findings = report.get("findings")
    if not isinstance(findings, list) or not findings:
        return False, None, "report is missing valid findings"
    if not all(_valid_finding(finding) for finding in findings):
        return False, None, "report has an invalid finding"

    impacts = {finding["impact"] for finding in findings}
    expected_verdict = (
        Verdict.NOT_SUPPORTED.value
        if "INVALIDATES" in impacts
        else Verdict.INSUFFICIENT_EVIDENCE.value
        if "INSUFFICIENT" in impacts
        else Verdict.SUPPORTED.value
    )
    if verdict != expected_verdict:
        return False, None, "report verdict contradicts finding impacts"

    if verdict == Verdict.SUPPORTED.value:
        baseline = metrics.get("baseline")
        candidate = metrics.get("candidate")
        improvement = metrics.get("absolute_improvement")
        rules = {finding["rule_id"] for finding in findings}
        if (
            not _valid_metric_summary(baseline)
            or not _valid_metric_summary(candidate)
            or not _finite_number(improvement)
            or "RESULT.CLAIM_SUPPORTED" not in rules
        ):
            return False, None, "supported report lacks verified metric evidence"
        baseline_mean = Decimal(str(baseline["mean"]))
        candidate_mean = Decimal(str(candidate["mean"]))
        expected_improvement = (
            baseline_mean - candidate_mean
            if direction == "lower"
            else candidate_mean - baseline_mean
        )
        if Decimal(str(improvement)) != expected_improvement:
            return False, None, "supported report has inconsistent improvement arithmetic"
        if Decimal(str(improvement)) < Decimal(str(threshold)):
            return False, None, "supported report does not meet its claim threshold"

    overlaps = report.get("overlaps")
    if not isinstance(overlaps, list) or not all(
        _valid_overlap(overlap) for overlap in overlaps
    ):
        return False, None, "report has invalid overlap summaries"

    return True, verdict, None


def _bounded_summary(summary: str) -> str:
    """Bound a Markdown summary with a deterministic truncation notice."""

    if len(summary) <= MAX_SUMMARY_LENGTH:
        return summary

    notice = (
        "\n\n> **ClaimCI:** summary truncated to satisfy the GitHub Check Run "
        "output limit."
    )
    keep = max(0, MAX_SUMMARY_LENGTH - len(notice))
    return summary[:keep] + notice


def _failure_summary(reason: str) -> str:
    """Return a stable, non-artifact-controlled failure summary."""

    return (
        "## ClaimCI Audit\n\n"
        "**FAILURE - ClaimCI could not publish a trustworthy audit verdict.**\n\n"
        f"Adapter validation: {reason}.\n"
    )


def build_check_payload(
    report: object,
    markdown: object,
    head_sha: object,
    details_url: object | None = None,
) -> dict[str, Any]:
    """Build a strict JSON-safe GitHub Check Run payload.

    The function is intentionally total: malformed inputs produce a completed
    failure payload with a bounded, deterministic summary.  This lets a CI
    workflow publish an integration failure instead of skipping the Check Run
    entirely.  Valid payloads contain only values accepted by the Checks API.
    """

    if isinstance(report, AuditResult):
        # Keep this convenience path coupled to the public renderer rather
        # than maintaining a second AuditResult-to-JSON schema here.
        from .report import render_json

        try:
            report = json.loads(render_json(report))
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            report = None

    report_ok, verdict, report_reason = _valid_report(report)
    reasons: list[str] = []
    if not report_ok:
        reasons.append(report_reason or "invalid report")

    if not isinstance(head_sha, str) or not _HEAD_SHA.fullmatch(head_sha.strip()):
        reasons.append("head SHA is missing or malformed")
        normalized_head_sha = ""
    else:
        normalized_head_sha = head_sha.strip()

    if not isinstance(markdown, str) or not markdown.strip():
        reasons.append("Markdown summary is missing")
        normalized_summary = _failure_summary("Markdown summary is missing")
    else:
        # Keep valid renderer output byte-for-byte intact (including trailing
        # newlines), only bounding it when the GitHub API requires truncation.
        normalized_summary = _bounded_summary(markdown)

    normalized_details_url: str | None = None
    if details_url is not None:
        if not isinstance(details_url, str) or not details_url.strip():
            reasons.append("details URL is invalid")
        else:
            normalized_details_url = details_url.strip()

    if reasons:
        conclusion = "failure"
        summary = _bounded_summary(_failure_summary("; ".join(reasons)))
        title = "ClaimCI Audit - integration failure"
    else:
        # ``verdict`` is known to be one of the three canonical values after
        # _valid_report succeeds, but keep the fallback defensive for future
        # enum additions or a type-checking blind spot.
        conclusion = verdict_to_conclusion(verdict)
        summary = normalized_summary
        title = f"ClaimCI Audit - {verdict}"

    payload: dict[str, Any] = {
        "name": CHECK_NAME,
        "head_sha": normalized_head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": title,
            "summary": summary,
        },
    }
    if normalized_details_url is not None:
        payload["details_url"] = normalized_details_url

    # This should always hold because every field above is a primitive, but
    # retain a final guard so future edits cannot leak non-standard JSON.
    if not _strict_json_safe(payload):
        return {
            "name": CHECK_NAME,
            "head_sha": normalized_head_sha,
            "status": "completed",
            "conclusion": "failure",
            "output": {
                "title": "ClaimCI Audit - integration failure",
                "summary": _failure_summary("payload is not strict JSON"),
            },
        }
    return payload


def _read_report(path: Path) -> object | None:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def _read_markdown(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m claimci.github",
        description="Build a GitHub Check Run payload from ClaimCI audit files.",
    )
    parser.add_argument("--report", required=True, help="path to ClaimCI JSON output")
    parser.add_argument(
        "--markdown", required=True, help="path to ClaimCI Markdown output"
    )
    parser.add_argument("--head-sha", required=True, help="pull-request head commit SHA")
    parser.add_argument("--details-url", default=None, help="optional workflow run URL")
    parser.add_argument(
        "--output",
        default="-",
        help="payload destination (default: stdout; use - for stdout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _read_report(Path(args.report))
    markdown = _read_markdown(Path(args.markdown))
    payload = build_check_payload(
        report,
        markdown,
        args.head_sha,
        details_url=args.details_url,
    )
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output == "-":
        sys.stdout.write(encoded)
    else:
        try:
            Path(args.output).write_text(encoded, encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"ClaimCI GitHub adapter error: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the workflow
    raise SystemExit(main())
