"""Hostile regressions for the zero-config passive repository boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from claimci.analysis.contracts import RepositoryPath
from claimci.analysis.discovery.models import (
    ArtifactIssue,
    DiscoveryError,
    DiscoveryLimits,
)
from claimci.analysis.discovery.repository import (
    collect_repository_context,
    inspect_artifact,
    read_artifact_text,
)
from claimci.review.evidence import discover_evidence
from claimci.review.models import (
    ClaimDirection,
    ClaimType,
    ReviewError,
    ScientificClaim,
    SourceKind,
    SourceLocation,
)
from claimci.review.sources import collect_review_sources


def _redirect_open_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    victim: Path,
    replacement: Path,
) -> None:
    """Model an OS pathname resolving to another file at the open boundary."""

    original_path_open = Path.open
    original_os_open = os.open

    def redirected_path_open(self: Path, *args: object, **kwargs: object):
        selected = replacement if Path(self) == victim else Path(self)
        return original_path_open(selected, *args, **kwargs)

    def redirected_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        selected = replacement if Path(path) == victim else path
        return original_os_open(selected, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", redirected_path_open)
    monkeypatch.setattr(os, "open", redirected_os_open)


def _metric_claim() -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim-hostile",
        source_text="Candidate improves accuracy.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="candidate model",
        metric="accuracy",
        direction=ClaimDirection.HIGHER,
        source=SourceLocation(
            source_id="source-readme",
            kind=SourceKind.REPOSITORY_FILE,
            path="README.md",
            start_line=1,
            end_line=1,
        ),
    )


def test_claim_source_open_swap_fails_closed_without_exposing_outside_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    source = head / "README.md"
    source.write_text("SAFE CLAIM\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("OUTSIDE_SECRET_SOURCE\n", encoding="utf-8")
    _redirect_open_after_validation(
        monkeypatch,
        victim=source,
        replacement=outside,
    )

    with pytest.raises(ReviewError, match="changed|confined|identity|symlink"):
        collect_review_sources(head)


def test_evidence_open_swap_is_unavailable_without_exposing_outside_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    source = head / "candidate_results.json"
    source.write_text('{"accuracy": 0.9}\n', encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("OUTSIDE_SECRET_EVIDENCE\n", encoding="utf-8")
    _redirect_open_after_validation(
        monkeypatch,
        victim=source,
        replacement=outside,
    )

    result = discover_evidence(
        head,
        (_metric_claim(),),
        ("candidate_results.json",),
    )

    assert result.references == ()
    assert any(item.reason == "unreadable" for item in result.missing)
    assert all(
        "OUTSIDE_SECRET_EVIDENCE" not in item.description
        for item in result.missing
    )


def test_artifact_open_swap_is_rejected_instead_of_pinning_redirected_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    source = head / "candidate_results.json"
    source.write_bytes(b'{"safe": true}\n')
    outside = tmp_path / "outside.json"
    outside.write_bytes(b'{"secret": 1}\n')
    limits = DiscoveryLimits()
    context = collect_repository_context(head, limits=limits)
    _redirect_open_after_validation(
        monkeypatch,
        victim=source,
        replacement=outside,
    )

    result = inspect_artifact(
        context,
        RepositoryPath("candidate_results.json"),
        limits=limits,
    )

    assert isinstance(result, ArtifactIssue)
    assert "changed" in result.reason or "confined" in result.reason

    with pytest.raises(DiscoveryError, match="indexed"):
        read_artifact_text(
            context,
            RepositoryPath("outside.json"),
            limits=limits,
        )
