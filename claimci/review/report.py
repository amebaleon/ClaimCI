"""Advisory JSON and Markdown views for a :class:`ResearchReview`.

The review layer is intentionally a presentation boundary.  Values produced by
the provider are serialized as interpretations and claims; only immutable
snapshots received from the deterministic bridge are rendered under the
``DETERMINISTIC_EVIDENCE`` heading.  Nothing in this module constructs a
deterministic ClaimCI verdict or finding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from html import escape as html_escape
from pathlib import Path
from typing import Any

from .evidence import EvidenceBundle
from .models import ProviderUsage, ReviewStatus
from .orchestrator import ClaimInterpretation, ResearchReview


MAX_REPORT_DEPTH = 128


def _unicode_safe(value: str) -> str:
    """Render lone UTF-16 surrogate code points as visible escape text."""

    return "".join(
        f"\\u{ord(character):04x}"
        if 0xD800 <= ord(character) <= 0xDFFF
        else character
        for character in value
    )


def _decimal_json_number(value: Decimal | None) -> float | None:
    if value is None or not value.is_finite():
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _json_safe(value: Any, *, depth: int = 0, active: set[int] | None = None) -> Any:
    """Return a finite, deterministic JSON-compatible representation.

    Review model values are bounded and immutable, but deterministic evidence
    maps can contain arbitrary JSON-like values.  This defensive normalizer
    avoids leaking non-finite numbers, recursive objects, or implementation
    reprs into the machine-readable report.
    """

    if depth > MAX_REPORT_DEPTH:
        return "<maximum report nesting depth>"
    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _unicode_safe(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else "<non-finite number>"
    if isinstance(value, Decimal):
        return _decimal_json_number(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, depth=depth + 1, active=active)
    if isinstance(value, Path):
        return _unicode_safe(value.as_posix())
    if is_dataclass(value):
        identity = id(value)
        if identity in active:
            return "<recursive report value>"
        active.add(identity)
        try:
            return {
                field.name: _json_safe(
                    getattr(value, field.name), depth=depth + 1, active=active
                )
                for field in fields(value)
            }
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return "<recursive report value>"
        active.add(identity)
        try:
            return {
                _unicode_safe(str(key)): _json_safe(
                    item, depth=depth + 1, active=active
                )
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in active:
            return "<recursive report value>"
        active.add(identity)
        try:
            items = [
                _json_safe(item, depth=depth + 1, active=active) for item in value
            ]
            # Sets are not expected in review models, but stable ordering keeps
            # this fallback deterministic if one appears in evidence.
            if isinstance(value, (set, frozenset)):
                items.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
            return items
        finally:
            active.remove(identity)
    try:
        return _unicode_safe(str(value))
    except Exception:
        return f"<{type(value).__name__}>"


def _usage_dict(usage: ProviderUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": (
            _decimal_json_number(usage.estimated_cost_usd)
        ),
    }


def _sorted_claims(review: ResearchReview) -> list[dict[str, Any]]:
    return [
        _json_safe(claim)
        for claim in sorted(review.claims, key=lambda item: item.claim_id)
    ]


def _sorted_interpretations(review: ResearchReview) -> list[dict[str, Any]]:
    return [
        _json_safe(item)
        for item in sorted(review.interpretations, key=lambda item: item.claim_id)
    ]


def _review_payload(review: ResearchReview) -> dict[str, Any]:
    if not isinstance(review, ResearchReview):
        raise TypeError("review must be a ResearchReview")
    evidence = review.evidence
    if not isinstance(evidence, EvidenceBundle):
        raise TypeError("review evidence must be an EvidenceBundle")
    calls = sorted(
        review.provider_calls,
        key=lambda call: (call.task, call.request_id or "", call.model),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "review": {
            "status": review.status.value,
            "policy": "advisory",
            "blocking": False,
        },
        "claims": _sorted_claims(review),
        "interpretations": _sorted_interpretations(review),
        "evidence": {
            "references": [
                _json_safe(reference)
                for reference in sorted(
                    evidence.references, key=lambda item: item.evidence_id
                )
            ],
            "missing": [
                _json_safe(item)
                for item in sorted(
                    evidence.missing,
                    key=lambda item: (
                        item.claim_id,
                        item.requested_path or "",
                        item.reason,
                    ),
                )
            ],
            "total_chars": evidence.total_chars,
            "routing_incomplete": evidence.routing_incomplete,
        },
        "deterministic_audits": [
            _json_safe(item)
            for item in sorted(
                review.deterministic_audits,
                key=lambda item: item.manifest_path,
            )
        ],
        "provider": {
            "calls": [
                _json_safe(item)
                for item in calls
            ],
            "usage": _usage_dict(review.usage),
        },
        "error": (
            None
            if review.error_code is None and review.error_message is None
            else {
                "code": review.error_code,
                "message": review.error_message,
            }
        ),
    }
    return _json_safe(payload)


def render_review_json(review: ResearchReview) -> str:
    """Render one stable advisory JSON document."""

    return (
        json.dumps(
            _review_payload(review),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _markdown_text(value: object) -> str:
    """Escape repository/provider text before placing it in Markdown."""

    text = html_escape(_unicode_safe(str(value)), quote=True)
    # Encode Markdown delimiters as entities so untrusted text cannot create a
    # table row, link, emphasis span, or executable HTML.
    for character, entity in (
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("(", "&#40;"),
        (")", "&#41;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        ("~", "&#126;"),
        (":", "&#58;"),
    ):
        text = text.replace(character, entity)
    return text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")


def _deterministic_text(value: object) -> str:
    """Render deterministic metadata without trusting artifact-derived prose."""

    # Finding explanations can contain submitted artifact paths and values,
    # even though the finding itself came from trusted deterministic code.
    return _markdown_text(value)


def _source_label(claim: Any) -> str:
    source = claim.source
    path = f" path={source.path}" if source.path is not None else ""
    return (
        f"{source.source_id} ({source.kind.value}{path}, "
        f"lines {source.start_line}-{source.end_line})"
    )


def _claim_lines(review: ResearchReview) -> list[str]:
    lines: list[str] = []
    for claim in sorted(review.claims, key=lambda item: item.claim_id):
        lines.extend(
            (
                f"#### `{_markdown_text(claim.claim_id)}`",
                "",
                f"- **Claim type:** {_markdown_text(claim.claim_type.value)}",
                f"- **Subject:** {_markdown_text(claim.subject)}",
                f"- **Source:** {_markdown_text(_source_label(claim))}",
                f"- **Original text:** {_markdown_text(claim.source_text)}",
                f"- **Confidence:** {_markdown_text(claim.confidence)}",
                "",
            )
        )
    if not lines:
        lines.append("- No claims were extracted.")
    return lines


def _deterministic_lines(review: ResearchReview) -> list[str]:
    lines: list[str] = []
    audits = sorted(review.deterministic_audits, key=lambda item: item.manifest_path)
    if not audits:
        lines.append(
            "- No deterministic ClaimCI audit snapshot was available for this "
            "bounded review scope."
        )
        return lines
    for audit in audits:
        lines.extend(
            (
                f"- Manifest `{_markdown_text(audit.manifest_path)}`: "
                f"verdict **{_markdown_text(audit.verdict)}**, metric "
                f"`{_markdown_text(audit.metric)}`, direction "
                f"`{_markdown_text(audit.direction)}`.",
            )
        )
        if not audit.findings:
            lines.append("  - No deterministic findings.")
            continue
        for finding in audit.findings:
            lines.extend(
                (
                    f"  - [{_deterministic_text(finding.rule_id)}] "
                    f"**{_deterministic_text(finding.title)}**: "
                    f"{_deterministic_text(finding.explanation)}",
                    f"    - Severity={_markdown_text(finding.severity)}, "
                    f"impact={_markdown_text(finding.impact)}; evidence "
                    f"`{_markdown_text(json.dumps(_json_safe(finding.evidence), sort_keys=True, ensure_ascii=False))}`",
                )
            )
    return lines


def _interpretation_lines(review: ResearchReview) -> list[str]:
    lines: list[str] = []
    if not review.interpretations:
        lines.append("- No LLM interpretation was produced.")
        return lines
    for interpretation in sorted(review.interpretations, key=lambda item: item.claim_id):
        lines.extend(
            (
                f"- **Claim `{_markdown_text(interpretation.claim_id)}`:** "
                f"{_markdown_text(interpretation.interpretation)}",
                f"  - Citations: "
                f"{_markdown_text(', '.join(interpretation.citations) or 'none')}",
            )
        )
    return lines


def _missing_lines(review: ResearchReview) -> list[str]:
    missing = sorted(
        review.evidence.missing,
        key=lambda item: (item.claim_id, item.requested_path or "", item.reason),
    )
    lines = [
        f"- **Claim `{_markdown_text(item.claim_id)}`:** "
        f"{_markdown_text(item.description or item.reason)}"
        + (
            f" (requested `{_markdown_text(item.requested_path)}`)"
            if item.requested_path
            else ""
        )
        for item in missing
    ]
    for interpretation in sorted(review.interpretations, key=lambda item: item.claim_id):
        for note in interpretation.missing_evidence:
            lines.append(
                f"- **Claim `{_markdown_text(interpretation.claim_id)}` (LLM note):** "
                f"{_markdown_text(note)}"
            )
    return lines or ["- No missing evidence was recorded by deterministic discovery."]


def _unsupported_lines(review: ResearchReview) -> list[str]:
    rows: list[str] = []
    for interpretation in sorted(review.interpretations, key=lambda item: item.claim_id):
        for note in interpretation.unsupported_inferences:
            rows.append(
                f"- **Claim `{_markdown_text(interpretation.claim_id)}`:** "
                f"{_markdown_text(note)}"
            )
    return rows or ["- No unsupported inferences were recorded."]


def render_review_markdown(review: ResearchReview) -> str:
    """Render a safe GitHub review/job-summary view.

    Sections are deliberately ordered so deterministic evidence is visually and
    textually separate from model interpretation.  The advisory status can
    never become a blocking GitHub quality gate through this view.
    """

    if not isinstance(review, ResearchReview):
        raise TypeError("review must be a ResearchReview")
    lines = [
        "## ClaimCI Research Review (Advisory)",
        "",
        (
            "This review is advisory. Only the separate deterministic ClaimCI "
            "Audit can block a pull request."
        ),
        "",
        f"**Status:** `{_markdown_text(review.status.value)}`",
        "",
        "### Claims",
        "",
        *_claim_lines(review),
        "### DETERMINISTIC_EVIDENCE",
        "",
        *_deterministic_lines(review),
        "",
        "### LLM_INTERPRETATION",
        "",
        *_interpretation_lines(review),
        "",
        "### MISSING_EVIDENCE",
        "",
        (
            "Evidence gaps are limited to ClaimCI's bounded analyzed snapshot; "
            "they do not establish that evidence is absent from the original repository."
        ),
        "",
        *_missing_lines(review),
        "",
        "### UNSUPPORTED_INFERENCE",
        "",
        *_unsupported_lines(review),
        "",
        "### Evidence and provider usage",
        "",
        f"- Evidence references: {len(review.evidence.references)}; "
        f"bounded characters: {review.evidence.total_chars}.",
        "- Routing incomplete: "
        f"{'yes' if review.evidence.routing_incomplete else 'no'}.",
    ]
    for reference in sorted(review.evidence.references, key=lambda item: item.evidence_id):
        lines.extend(
            (
                f"- Evidence `{_markdown_text(reference.evidence_id)}` "
                f"({', '.join(_markdown_text(claim_id) for claim_id in reference.claim_ids)}): "
                f"provenance **{_markdown_text(reference.provenance.value)}**; "
                f"locality **{_markdown_text(reference.excerpt_locality.value)}**; "
                f"excerpt complete **{'yes' if reference.excerpt_complete else 'no'}**; "
                f"`{_markdown_text(reference.path)}` lines "
                f"{reference.start_line}-{reference.end_line}; "
                f"excerpt: {_markdown_text(reference.excerpt)}",
            )
        )
    lines.extend(
        (
        f"- Provider calls: {len(review.provider_calls)}; input tokens: "
        f"{_markdown_text(review.usage.input_tokens if review.usage.input_tokens is not None else 'unavailable')}; "
        f"output tokens: {_markdown_text(review.usage.output_tokens if review.usage.output_tokens is not None else 'unavailable')}; "
        f"total tokens: {_markdown_text(review.usage.total_tokens if review.usage.total_tokens is not None else 'unavailable')}; "
        f"estimated cost USD: {_markdown_text(review.usage.estimated_cost_usd if review.usage.estimated_cost_usd is not None else 'unavailable')}.",
        )
    )
    if review.error_message:
        lines.extend(("", f"**Review note:** {_markdown_text(review.error_message)}"))
    return "\n".join(lines) + "\n"


__all__ = ["render_review_json", "render_review_markdown"]
