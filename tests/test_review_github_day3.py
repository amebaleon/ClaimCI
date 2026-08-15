"""Adversarial tests for the Day 3 advisory GitHub Check adapter.

The research-review Check is deliberately a separate surface from the
deterministic ``ClaimCI Audit`` Check.  These tests use only value objects and
in-memory strings: they never call GitHub or a provider, and they do not need
an API key.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from claimci.github import build_check_payload
from claimci.models import Verdict
from claimci.report import render_json
from claimci.review.models import ReviewStatus
from claimci.review.orchestrator import ResearchReview
from tests.test_markdown_day2 import _result


GOOD_SHA = "a" * 40
GOOD_SHA_64 = "B" * 64
MAX_CHECK_SUMMARY_BYTES = 60_000


def _review(status: ReviewStatus = ReviewStatus.COMPLETE) -> ResearchReview:
    """Return the smallest trusted review result for adapter tests."""

    return ResearchReview(status=status)


def _build(
    review: object,
    markdown: object = "## ClaimCI Research Review (Advisory)\n\nNo blocking decision.",
    head_sha: object = GOOD_SHA,
    details_url: object | None = "https://ci.example/runs/7",
) -> dict[str, Any]:
    # Keep the import at call time so the rest of this file still reports
    # useful static/workflow failures while the production adapter is RED.
    from claimci.review.github import build_review_check_payload

    return build_review_check_payload(
        review,
        markdown,
        head_sha,
        details_url=details_url,
    )


@pytest.mark.parametrize(
    "status",
    [
        ReviewStatus.COMPLETE,
        ReviewStatus.PARTIAL,
        ReviewStatus.UNAVAILABLE,
        ReviewStatus.DISABLED,
    ],
)
def test_review_check_has_its_own_name_and_is_always_neutral(
    status: ReviewStatus,
) -> None:
    """Review status is advisory metadata, never a scientific conclusion."""

    payload = _build(_review(status))

    assert payload["name"] == "ClaimCI Research Review"
    assert payload["status"] == "completed"
    assert payload["head_sha"] == GOOD_SHA
    assert payload["conclusion"] == "neutral"
    assert payload["output"]["summary"].startswith(
        "## ClaimCI Research Review (Advisory)"
    )
    assert payload["details_url"] == "https://ci.example/runs/7"
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    ("verdict", "expected_conclusion"),
    [
        (Verdict.SUPPORTED, "success"),
        (Verdict.NOT_SUPPORTED, "failure"),
        (Verdict.INSUFFICIENT_EVIDENCE, "action_required"),
    ],
)
def test_review_check_never_changes_the_existing_audit_conclusion(
    verdict: Verdict,
    expected_conclusion: str,
) -> None:
    """Adding an advisory payload cannot turn the deterministic gate green/red."""

    # ``build_check_payload`` is the existing authoritative adapter.  Its
    # failure conclusion is intentionally different from the review's neutral
    # conclusion, and both payloads must remain independently named.
    deterministic_report = json.loads(render_json(_result(verdict=verdict)))
    audit = build_check_payload(deterministic_report, "audit", GOOD_SHA)
    review = _build(_review(ReviewStatus.COMPLETE))

    assert audit["name"] == "ClaimCI Audit"
    assert audit["conclusion"] == expected_conclusion
    assert review["name"] == "ClaimCI Research Review"
    assert review["conclusion"] == "neutral"


@pytest.mark.parametrize(
    ("review", "markdown", "head_sha"),
    [
        (None, "summary", GOOD_SHA),
        (object(), "summary", GOOD_SHA),
        ({"status": "COMPLETE"}, "summary", GOOD_SHA),
        (_review(), None, GOOD_SHA),
        (_review(), "", GOOD_SHA),
        (_review(), "summary", ""),
        (_review(), "summary", "not-a-commit"),
        (_review(), "summary", "g" * 40),
        (_review(), "summary", "a" * 39),
        (_review(), "summary", "a" * 65),
    ],
)
def test_malformed_adapter_inputs_are_safe_neutral_advisory_payloads(
    review: object,
    markdown: object,
    head_sha: object,
) -> None:
    """Malformed review data cannot be mistaken for a passing scientific review."""

    payload = _build(review, markdown, head_sha)
    summary = payload["output"]["summary"]

    assert payload["name"] == "ClaimCI Research Review"
    assert payload["status"] == "completed"
    assert payload["conclusion"] == "neutral"
    assert isinstance(summary, str)
    assert "advisory" in summary.casefold()
    assert "unavailable" in summary.casefold() or "integration" in summary.casefold()
    assert not any(
        marker in summary.casefold()
        for marker in ("scientific pass", "success", "action_required")
    )
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("head_sha", [GOOD_SHA, GOOD_SHA_64, GOOD_SHA.upper()])
def test_adapter_accepts_only_hex_commit_sha_shapes(head_sha: str) -> None:
    """Both Git SHA-1 and SHA-256 forms are accepted, without shell parsing."""

    payload = _build(_review(), "summary", head_sha)

    assert payload["head_sha"] == head_sha
    assert payload["conclusion"] == "neutral"


def test_summary_is_bounded_deterministically_and_remains_strict_json() -> None:
    """The post-render Check summary is bounded by UTF-8 bytes, not characters."""

    escaped_unit = "\ud55c&lt;record&gt;"
    oversized = (
        "## ClaimCI Research Review (Advisory)\n\n"
        + escaped_unit * 4_000
    )
    assert len(oversized) < 65_535
    assert len(oversized.encode("utf-8")) > 65_535
    first = _build(_review(), oversized)
    second = _build(_review(), oversized)

    assert first == second
    summary = first["output"]["summary"]
    encoded = summary.encode("utf-8")
    assert len(encoded) <= MAX_CHECK_SUMMARY_BYTES
    assert encoded.decode("utf-8") == summary
    assert summary.startswith("## ClaimCI Research Review (Advisory)")
    assert "truncat" in summary.casefold()
    json.dumps(first, ensure_ascii=False, allow_nan=False)


def test_malformed_oversized_input_uses_small_neutral_fallback() -> None:
    """Publisher-bound malformed data cannot retain an oversized summary."""

    oversized = "\ud55c" * 30_000
    payload = _build(None, oversized)
    summary = payload["output"]["summary"]

    assert payload["conclusion"] == "neutral"
    assert "unavailable" in summary.casefold()
    assert len(summary.encode("utf-8")) <= MAX_CHECK_SUMMARY_BYTES
    assert oversized not in summary


def test_cli_preserves_utf8_and_fits_the_cross_job_handoff(tmp_path: Path) -> None:
    """Escaped JSON must not re-expand a bounded Unicode summary past handoff."""

    from claimci.review.github import main

    review_json = tmp_path / "review.json"
    markdown = tmp_path / "review.md"
    output = tmp_path / "check.json"
    review_json.write_text(
        json.dumps({"review": {"status": "PARTIAL"}}), encoding="utf-8"
    )
    markdown.write_text(
        "## ClaimCI Research Review (Advisory)\n\n" + "\U0001f9ea" * 20_000,
        encoding="utf-8",
    )

    assert main(
        [
            "--review-json",
            str(review_json),
            "--markdown",
            str(markdown),
            "--head-sha",
            GOOD_SHA,
            "--output",
            str(output),
        ]
    ) == 0

    encoded_payload = output.read_bytes()
    assert len(base64.b64encode(encoded_payload)) < 100_000
    payload = json.loads(encoded_payload.decode("utf-8"))
    summary = payload["output"]["summary"]
    assert len(summary.encode("utf-8")) <= MAX_CHECK_SUMMARY_BYTES
    assert "truncat" in summary.casefold()


def test_adapter_preserves_renderer_escaping_byte_for_byte() -> None:
    """The report renderer owns escaping; the Check adapter must not undo it."""

    escaped = (
        "## ClaimCI Research Review (Advisory)\n\n"
        "&lt;script&gt;alert(1)&lt;/script&gt; &#124; escaped row\n"
        "[literal link](not-a-url)"
    )
    payload = _build(_review(), escaped)

    assert payload["output"]["summary"] == escaped
    assert "<script>" not in payload["output"]["summary"]
    assert "| escaped row" not in payload["output"]["summary"]
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("details_url", [None, "", object(), "\x00bad-url"])
def test_invalid_details_url_does_not_break_or_greenlight_the_check(
    details_url: object,
) -> None:
    payload = _build(_review(), "summary", GOOD_SHA, details_url)

    assert payload["name"] == "ClaimCI Research Review"
    assert payload["conclusion"] == "neutral"
    json.dumps(payload, allow_nan=False)
