"""Human-readable and machine-readable ClaimCI audit reports."""

from __future__ import annotations

import json
import math
from html import escape as html_escape
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .models import AuditResult, Direction, Finding, MetricSummary, Severity


SEVERITY_ORDER = (
    Severity.CRITICAL,
    Severity.WARNING,
    Severity.VERIFIED,
    Severity.INFO,
)

# Accuracy is the only metric with an explicit percentage-point convention in
# the Day 1 contract.  Other metrics stay in their native metric units.
PERCENTAGE_POINT_METRICS = frozenset({"accuracy"})
MAX_REPORT_NESTING_DEPTH = 256


def _stable_json_sort_key(value: Any) -> str:
    """Return a deterministic ordering key for normalized set members."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        # ``_json_safe`` normally makes this unreachable.  A repr fallback
        # keeps normalization total for unusual user-provided evidence.
        return repr(value)


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe representation of an arbitrary value."""

    def normalize(current: Any, depth: int, active: set[int]) -> Any:
        if depth > MAX_REPORT_NESTING_DEPTH:
            return "<maximum report nesting depth>"
        if current is None or isinstance(current, (str, bool, int)):
            return current
        if isinstance(current, float):
            if math.isnan(current):
                return "NaN"
            if math.isinf(current):
                return "Infinity" if current > 0 else "-Infinity"
            return current

        containers = (Mapping, list, tuple, set, frozenset)
        if isinstance(current, containers):
            identity = id(current)
            if identity in active:
                return "<recursive report value>"
            active.add(identity)
            try:
                if isinstance(current, Mapping):
                    return {
                        _json_key(key): normalize(item, depth + 1, active)
                        for key, item in current.items()
                    }
                if isinstance(current, (list, tuple)):
                    return [normalize(item, depth + 1, active) for item in current]
                normalized = [normalize(item, depth + 1, active) for item in current]
                return sorted(normalized, key=_stable_json_sort_key)
            finally:
                active.remove(identity)

        try:
            return str(current)
        except Exception:
            value_type = type(current)
            return f"<{value_type.__module__}.{value_type.__qualname__}>"

    return normalize(value, 0, set())


def _json_key(value: Any) -> str:
    normalized = _json_safe(value)
    return normalized if isinstance(normalized, str) else str(normalized)


def _uses_percentage_points(metric: str) -> bool:
    return str(metric).casefold() in PERCENTAGE_POINT_METRICS


def _threshold_text(metric: str, threshold: float) -> str:
    if _uses_percentage_points(metric):
        return f"{threshold * 100:.2f} percentage points"
    return f"{threshold:.6f} metric units"


def _summary_text(label: str, summary: MetricSummary | None) -> str:
    if summary is None:
        return f"- {label}: unavailable"
    stddev = (
        "unavailable"
        if summary.standard_deviation is None
        else f"{summary.standard_deviation:.6f}"
    )
    return (
        f"- {label}: count={summary.count}, mean={summary.mean:.6f}, "
        f"sample stddev={stddev}"
    )


def _finding_text(finding: Finding) -> str:
    evidence = json.dumps(
        _json_safe(finding.evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"- [{finding.rule_id}] {finding.title}: {finding.explanation} Evidence: {evidence}"


def render_human(result: AuditResult) -> str:
    claim_verb = "improves" if result.direction is Direction.HIGHER else "lowers"
    lines = [
        "ClaimCI Audit",
        "",
        "Claim:",
        (
            f"candidate {claim_verb} {result.metric} by >= "
            f"{_threshold_text(result.metric, result.minimum_improvement)}"
        ),
        "",
        "Verdict:",
        result.verdict.value.replace("_", " "),
        "",
        "Metrics:",
        _summary_text("Baseline", result.baseline_metrics),
        _summary_text("Candidate", result.candidate_metrics),
    ]
    # Keep the Day 1 higher-is-better wording byte-compatible.  Lower claims
    # need an explicit cue so a reader cannot mistake the signed directional
    # improvement for the raw candidate-minus-baseline difference.
    if result.direction is Direction.LOWER:
        lines.extend(("", "Direction: lower (lower is better)"))
    if result.absolute_improvement is None:
        lines.append("- Absolute improvement: unavailable")
    else:
        if _uses_percentage_points(result.metric):
            lines.append(
                f"- Absolute improvement: {result.absolute_improvement:.6f} "
                f"({result.absolute_improvement * 100:.2f} percentage points)"
            )
        else:
            lines.append(
                f"- Absolute improvement: {result.absolute_improvement:.6f} metric units"
            )
    if result.relative_improvement is None:
        lines.append("- Relative improvement: unavailable")
    else:
        lines.append(f"- Relative improvement: {result.relative_improvement:.2%}")

    for severity in SEVERITY_ORDER:
        grouped = [finding for finding in result.findings if finding.severity is severity]
        if not grouped:
            continue
        lines.extend(("", severity.value))
        lines.extend(_finding_text(finding) for finding in grouped)
    return "\n".join(lines) + "\n"


def _metric_dict(summary: MetricSummary | None) -> dict[str, Any] | None:
    return None if summary is None else asdict(summary)


def render_json(result: AuditResult) -> str:
    claim: dict[str, Any] = {
        "metric": result.metric,
        "minimum_improvement": result.minimum_improvement,
    }
    # Omit the default to preserve the Day 1 JSON shape exactly.  Explicit
    # lower-direction claims carry their policy in the machine-readable claim
    # object as well as in claim-decision finding evidence.
    if result.direction is Direction.LOWER:
        claim["direction"] = Direction.LOWER.value
    payload = {
        "manifest": str(result.manifest_path),
        "claim": claim,
        "verdict": result.verdict.value,
        "metrics": {
            "baseline": _metric_dict(result.baseline_metrics),
            "candidate": _metric_dict(result.candidate_metrics),
            "absolute_improvement": result.absolute_improvement,
            "relative_improvement": result.relative_improvement,
        },
        "overlaps": [asdict(overlap) for overlap in result.overlaps],
        "findings": [
            {
                "rule_id": finding.rule_id,
                "severity": finding.severity.value,
                "impact": finding.impact.value,
                "title": finding.title,
                "explanation": finding.explanation,
                "evidence": finding.evidence,
            }
            for finding in result.findings
        ],
    }
    return (
        json.dumps(
            _json_safe(payload),
            # Machine output stays valid and printable even on legacy Windows
            # consoles; json.loads reconstructs the original Unicode values.
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _markdown_text(value: object) -> str:
    """Escape one artifact-controlled value for GitHub-flavored Markdown.

    HTML is escaped first and table delimiters are encoded everywhere.  Line
    breaks become ``<br>`` only after the angle brackets from the input have
    been escaped, so a metric name or finding cannot add table rows or active
    HTML to a pull-request report.
    """

    text = html_escape(str(value), quote=True)
    # Encode Markdown's active delimiters as entities. This prevents a
    # metric/finding/evidence value from closing an inline-code span, creating
    # a link (including a javascript: URL), adding emphasis, or injecting a
    # table row. Colons are encoded as well so dangerous URL schemes never
    # survive as active or visually ambiguous text.
    for character, entity in (
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("(", "&#40;"),
        (")", "&#41;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        (":", "&#58;"),
    ):
        text = text.replace(character, entity)
    return text.replace("\r\n", "<br>").replace("\r", "<br>").replace(
        "\n", "<br>"
    )


def _markdown_number(value: float | int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6f}"


def _markdown_code(value: object) -> str:
    """Wrap artifact text in safe HTML code whose entities render readably.

    Unlike entities inside a Markdown backtick span, entities inside the HTML
    element are decoded by the renderer. Normal JSON therefore displays as
    normal JSON, while markup delimiters cannot escape into an active link,
    table row, inline-code span, emphasis node, or HTML tag.
    """

    text = html_escape(str(value), quote=False)
    for character, entity in (
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        ("~", "&#126;"),
        (":", "&#58;"),
    ):
        text = text.replace(character, entity)
    return f"<code>{text}</code>"


def _markdown_stddev(summary: MetricSummary | None) -> str:
    if summary is None or summary.standard_deviation is None:
        return "unavailable"
    return f"{summary.standard_deviation:.6f}"


def _verdict_presentation(result: AuditResult) -> tuple[str, str]:
    if result.verdict.value == "SUPPORTED":
        return "PASS", "The declared claim is supported by every required check."
    if result.verdict.value == "NOT_SUPPORTED":
        return "FAIL", "The declared claim is not supported by the audited evidence."
    return "ACTION REQUIRED", "More or valid evidence is required before this claim can pass."


def _markdown_finding(finding: Finding) -> list[str]:
    evidence = json.dumps(
        _json_safe(finding.evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # Rule IDs are internal stable constants and are intentionally left
    # literal inside inline code.  All artifact-derived prose and evidence is
    # escaped independently.
    return [
        f"- `{finding.rule_id}` - **{_markdown_text(finding.title)}**",
        f"  - {_markdown_text(finding.explanation)}",
        f"  - Evidence: {_markdown_code(evidence)}",
    ]


def render_markdown(result: AuditResult) -> str:
    """Render a deterministic GitHub Check/job-summary view of an audit."""

    icon, interpretation = _verdict_presentation(result)
    direction_label = (
        "higher is better"
        if result.direction is Direction.HIGHER
        else "lower is better"
    )
    baseline = result.baseline_metrics
    candidate = result.candidate_metrics
    metric = _markdown_code(result.metric)

    lines = [
        "## ClaimCI Audit",
        "",
        f"### {icon} `{result.verdict.value}`",
        "",
        interpretation,
        "",
        "### Claim",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Metric | {metric} |",
        f"| Direction | `{result.direction.value}` ({direction_label}) |",
        (
            "| Required directional improvement | "
            f"`{result.minimum_improvement:.6f}` "
            f"({_markdown_text(_threshold_text(result.metric, result.minimum_improvement))}) |"
        ),
        "",
        "### Recomputed metrics",
        "",
        "| Experiment | Runs | Mean | Sample standard deviation |",
        "|---|---:|---:|---:|",
        (
            "| Baseline | "
            f"{baseline.count if baseline is not None else 'unavailable'} | "
            f"{_markdown_number(None if baseline is None else baseline.mean)} | "
            f"{_markdown_stddev(baseline)} |"
        ),
        (
            "| Candidate | "
            f"{candidate.count if candidate is not None else 'unavailable'} | "
            f"{_markdown_number(None if candidate is None else candidate.mean)} | "
            f"{_markdown_stddev(candidate)} |"
        ),
        "",
    ]

    if result.absolute_improvement is None:
        lines.append("- **Directional improvement:** unavailable")
    elif _uses_percentage_points(result.metric):
        lines.append(
            "- **Directional improvement:** "
            f"`{result.absolute_improvement:.6f}` "
            f"({result.absolute_improvement * 100:.2f} percentage points)"
        )
    else:
        lines.append(
            f"- **Directional improvement:** `{result.absolute_improvement:.6f}` metric units"
        )
    if result.relative_improvement is None:
        lines.append("- **Relative improvement:** unavailable")
    else:
        lines.append(
            f"- **Relative improvement:** `{result.relative_improvement:.6f}` "
            f"({result.relative_improvement:.2%})"
        )

    drivers = [finding for finding in result.findings if finding.impact.value != "NONE"]
    lines.extend(("", "### Decision drivers", ""))
    if drivers:
        for finding in drivers:
            lines.extend(_markdown_finding(finding))
    else:
        lines.append("- No invalidating or insufficient-evidence findings.")

    lines.extend(("", "<details>", "<summary>All audit findings</summary>", ""))
    for severity in SEVERITY_ORDER:
        grouped = [finding for finding in result.findings if finding.severity is severity]
        if not grouped:
            continue
        lines.extend((f"#### {severity.value}", ""))
        for finding in grouped:
            lines.extend(_markdown_finding(finding))
        lines.append("")
    lines.extend(("</details>", ""))
    return "\n".join(lines)
