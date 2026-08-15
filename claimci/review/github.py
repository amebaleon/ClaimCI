"""Advisory GitHub Check adapter for the ClaimCI research review.

The deterministic ``ClaimCI Audit`` Check is implemented by
``claimci.github`` and remains the only quality gate.  This module deliberately
has a separate name and always publishes a ``neutral`` Check conclusion.  It
only adapts an already-rendered review; it never parses model prose or makes a
scientific decision.

``python -m claimci.review.github`` is a small workflow-facing helper.  It
reads the review JSON/Markdown files produced by the trusted ``claimci review``
command and writes a strict Check Run payload without contacting GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from claimci.parsing import unique_json_object

from .models import ReviewError, ReviewStatus
from .orchestrator import ResearchReview


CHECK_NAME = "ClaimCI Research Review"
# GitHub enforces this field in UTF-8 bytes.  Stay below the documented
# 65,535-byte ceiling so serialization and platform edge cases retain margin.
MAX_CHECK_SUMMARY_BYTES = 60_000
_HEAD_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def _strict_json_safe(value: object) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


def _bounded_summary(summary: str) -> str:
    """Bound final renderer output without splitting a UTF-8 code point."""

    encoded = summary.encode("utf-8")
    if len(encoded) <= MAX_CHECK_SUMMARY_BYTES:
        return summary
    notice = (
        "\n\n> **ClaimCI:** advisory review summary truncated to satisfy the "
        "GitHub Check Run output limit."
    )
    notice_bytes = notice.encode("utf-8")
    keep = max(0, MAX_CHECK_SUMMARY_BYTES - len(notice_bytes))
    # The input is already rendered and escaped. Decode only the retained byte
    # prefix, dropping at most the incomplete final code point at the cut.
    prefix = encoded[:keep].decode("utf-8", errors="ignore")
    return prefix + notice


def _integration_summary(reason: str) -> str:
    # This text is fixed and intentionally excludes provider/model output,
    # paths, credentials, and exception details supplied by untrusted input.
    return (
        "## ClaimCI Research Review (Advisory)\n\n"
        "**Review unavailable; the advisory integration could not render a "
        "trustworthy review.**\n\n"
        f"Integration note: {reason}.\n\n"
        "Only the separate deterministic ClaimCI Audit can block a pull "
        "request."
    )


def _valid_review(review: object) -> bool:
    """Require the trusted immutable review result, not an arbitrary mapping."""

    if not isinstance(review, ResearchReview):
        return False
    return isinstance(review.status, ReviewStatus)


def _valid_markdown(markdown: object) -> bool:
    """Accept renderer output while rejecting placeholder/error fragments.

    The report renderer always emits the advisory heading.  An oversized
    string is still accepted so the adapter can exercise its deterministic
    truncation behavior before publication.
    """

    if not isinstance(markdown, str) or not markdown.strip():
        return False
    try:
        encoded_size = len(markdown.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    if encoded_size >= MAX_CHECK_SUMMARY_BYTES:
        return True
    return markdown.startswith("## ClaimCI Research Review (Advisory)")


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _HEAD_SHA.fullmatch(value.strip()) is not None


def _safe_details_url(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or any(ord(char) < 0x20 or ord(char) == 0x7F for char in stripped):
        return None
    return stripped


def build_review_check_payload(
    review: object,
    markdown: object,
    head_sha: object,
    *,
    details_url: object | None = None,
) -> dict[str, Any]:
    """Build a strict, always-neutral advisory Check Run payload.

    Malformed inputs produce a completed neutral payload containing a fixed
    integration/unavailable message.  This is intentionally total: a broken
    optional review can never become a passing or blocking scientific Check.
    """

    reasons: list[str] = []
    if not _valid_review(review):
        reasons.append("review result is invalid")
    if not _valid_markdown(markdown):
        reasons.append("review Markdown is missing or invalid")
    if not _valid_sha(head_sha):
        reasons.append("head SHA is missing or malformed")

    if reasons:
        summary = _integration_summary("; ".join(reasons))
    else:
        # _valid_markdown guarantees a string, but keep the cast local so
        # future type-checkers do not widen the payload to arbitrary objects.
        summary = _bounded_summary(markdown)  # type: ignore[arg-type]

    payload: dict[str, Any] = {
        "name": CHECK_NAME,
        "head_sha": head_sha.strip() if _valid_sha(head_sha) else "",
        "status": "completed",
        # Advisory review status never controls the deterministic quality gate.
        "conclusion": "neutral",
        "output": {
            "title": "ClaimCI Research Review - advisory",
            "summary": summary,
        },
    }
    normalized_details_url = _safe_details_url(details_url)
    if normalized_details_url is not None:
        payload["details_url"] = normalized_details_url

    if not _strict_json_safe(payload):
        # All fields above are primitives; retain a fixed fallback if a future
        # edit accidentally introduces a non-JSON value.
        return {
            "name": CHECK_NAME,
            "head_sha": "",
            "status": "completed",
            "conclusion": "neutral",
            "output": {
                "title": "ClaimCI Research Review - advisory",
                "summary": _integration_summary("payload is not strict JSON"),
            },
        }
    return payload


def _read_json(path: Path) -> object | None:
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


def _review_from_json(value: object) -> ResearchReview | None:
    if not isinstance(value, Mapping):
        return None
    review_section = value.get("review")
    if not isinstance(review_section, Mapping):
        return None
    status = review_section.get("status")
    try:
        return ResearchReview(status=ReviewStatus(status))
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m claimci.review.github",
        description="Build an advisory ClaimCI Research Review Check payload.",
    )
    parser.add_argument(
        "--review-json",
        default=None,
        help="path to the JSON review output (missing/invalid is safely unavailable)",
    )
    parser.add_argument("--markdown", required=True, help="path to review Markdown")
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
    review = _review_from_json(_read_json(Path(args.review_json))) if args.review_json else None
    markdown = _read_markdown(Path(args.markdown))
    payload = build_review_check_payload(
        review,
        markdown,
        args.head_sha,
        details_url=args.details_url,
    )
    encoded = json.dumps(
        payload,
        # Preserve UTF-8 instead of expanding every non-ASCII code point into
        # JSON escapes; the payload then remains inside the bounded job handoff.
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.output == "-":
        sys.stdout.write(encoded)
        return 0
    try:
        Path(args.output).write_text(encoded, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ClaimCI Research Review adapter error: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - workflow entry point
    raise SystemExit(main())
