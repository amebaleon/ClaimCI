"""Detached threshold recovery remains source-bound and fail-closed."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from claimci.analysis import (
    GitCommitSha,
    RepositoryIdentity,
    to_jsonable,
)
from claimci.analysis.discovery import discover_repository
from claimci.analysis.discovery.claims import _deterministic_claims
from claimci.analysis.discovery.repository import RepositoryContext
from claimci.review import SourceBundle, SourceKind, SourceRecord


REPOSITORY = RepositoryIdentity("amebaleon", "detached-threshold-fixture")
HEAD_SHA = GitCommitSha("a" * 40)
EXACT_PRODUCTION_SMOKE_CLAIM = (
    "The candidate improves accuracy from 0.60 to 0.70 under the same "
    "configuration and evaluation dataset."
)


def _discover(tmp_path: Path, document: str):
    root = tmp_path / "repository"
    root.mkdir()
    return discover_repository(
        root,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_title="Production smoke: verify supported result",
        pr_description=document,
    )


def _discover_raw_source(tmp_path: Path, document: str):
    """Exercise deterministic discovery before Review's normal line-ending fold."""

    record = SourceRecord(
        source_id="source-line-ending-normalization",
        kind=SourceKind.PULL_REQUEST_DESCRIPTION,
        path=None,
        text=document,
        sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
    )
    context = RepositoryContext(
        head_root=tmp_path,
        base_root=None,
        source_bundle=SourceBundle(sources=(record,), total_chars=len(document)),
        repository_paths=(),
        changed_paths=(),
    )
    return _deterministic_claims(context)


@pytest.mark.parametrize(
    (
        "case",
        "document",
        "expected_threshold",
        "expected_claim_count",
        "expects_binding",
    ),
    (
        (
            "A",
            "The candidate improves accuracy from 0.60 to 0.70 by at least 0.05"
            + "\nDeclared minimum improvement: 0.05",
            0.05,
            1,
            True,
        ),
        (
            "B",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\n\tminimum improvement:\t0.05\t",
            0.05,
            1,
            True,
        ),
        (
            "C",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\nREQUIRED MINIMUM IMPROVEMENT: 0.05",
            0.05,
            1,
            True,
        ),
        (
            "D",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\nSuggested minimum improvement: 0.05",
            None,
            1,
            False,
        ),
        (
            "E",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\nDeclared minimum improvement: 0.05 because it matters",
            None,
            1,
            False,
        ),
        (
            "F",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\nDeclared minimum improvement: 0.05"
            + "\nMinimum improvement: 0.05",
            None,
            1,
            False,
        ),
        (
            "G",
            EXACT_PRODUCTION_SMOKE_CLAIM
            + "\nDeclared minimum improvement: 0.05"
            + "\ncandidate improves loss from 0.40 to 0.30",
            None,
            2,
            False,
        ),
        (
            "H",
            "The candidate improves accuracy from 0.60 to 0.70 by at least 0.04"
            + "\nDeclared minimum improvement: 0.05",
            None,
            0,
            False,
        ),
    ),
)
def test_detached_threshold_a_h_matrix_is_explicit_and_fail_closed(
    tmp_path: Path,
    case: str,
    document: str,
    expected_threshold: float | None,
    expected_claim_count: int,
    expects_binding: bool,
) -> None:
    """Break caught: accepted labels can be missed or untrusted prose can bind."""

    discovery = _discover(tmp_path, document)

    assert len(discovery.claims) == expected_claim_count, case
    if expected_claim_count == 0:
        return
    primary = discovery.claims[0]
    observed = (
        None
        if primary.minimum_improvement is None
        else primary.minimum_improvement.value
    )
    assert observed == expected_threshold, case
    assert primary.scientific_claim is not None
    assert (primary.scientific_claim._source_binding is not None) is expects_binding


def test_detached_threshold_binding_is_immutable_source_certified_and_not_public(
    tmp_path: Path,
) -> None:
    """Break caught: threshold data could lose its trusted document/span scope."""

    document = (
        EXACT_PRODUCTION_SMOKE_CLAIM
        + "\nDeclared minimum improvement: 0.05"
    )
    discovery = _discover(tmp_path, document)

    assert len(discovery.claims) == 1
    claim = discovery.claims[0]
    assert claim.scientific_claim is not None
    binding = claim.scientific_claim._source_binding
    assert binding is not None
    assert binding.document.source_id == claim.source.source_id
    assert binding.document.sha256 == hashlib.sha256(document.encode("utf-8")).hexdigest()
    assert binding.document.text[binding.primary_start : binding.primary_end] == (
        EXACT_PRODUCTION_SMOKE_CLAIM
    )
    assert binding.document.text[
        binding.declaration_start : binding.declaration_end
    ] == "Declared minimum improvement: 0.05"
    assert claim.minimum_improvement is not None
    assert claim.minimum_improvement.provenance is claim.reference.provenance
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.primary_start = 0  # type: ignore[misc]

    serialized = to_jsonable(claim.scientific_claim)
    assert "_source_binding" not in serialized
    assert "source_binding" not in serialized
    assert "document" not in serialized
    assert "declaration_start" not in serialized


def test_detached_threshold_terminal_period_is_preserved_in_private_binding(
    tmp_path: Path,
) -> None:
    """Break caught: the exact fixture punctuation becomes non-authoritative."""

    declaration = "Declared minimum improvement: 0.05."
    discovery = _discover(
        tmp_path,
        EXACT_PRODUCTION_SMOKE_CLAIM + "\n" + declaration,
    )

    assert len(discovery.claims) == 1
    claim = discovery.claims[0]
    assert claim.minimum_improvement is not None
    assert claim.minimum_improvement.value == 0.05
    assert claim.scientific_claim is not None
    primary = claim.scientific_claim.primary
    assert primary.minimum_improvement is not None
    assert primary.minimum_improvement.raw == "0.05"
    binding = claim.scientific_claim._source_binding
    assert binding is not None
    assert binding.document.text[
        binding.declaration_start : binding.declaration_end
    ] == declaration


def test_invalid_second_recognized_declaration_fails_closed_cardinality(
    tmp_path: Path,
) -> None:
    """Break caught: filtering a negative declaration masks a duplicate label."""

    discovery = _discover(
        tmp_path,
        "\n".join(
            (
                EXACT_PRODUCTION_SMOKE_CLAIM,
                "Declared minimum improvement: 0.05",
                "Minimum improvement: -0.01",
            )
        ),
    )

    assert len(discovery.claims) == 1
    claim = discovery.claims[0]
    assert claim.minimum_improvement is None
    assert claim.scientific_claim is not None
    assert claim.scientific_claim._source_binding is None


def test_existing_exact_production_smoke_claim_id_remains_unchanged(
    tmp_path: Path,
) -> None:
    """Break caught: detached recovery changes the established claim-ID algorithm."""

    discovery = _discover(tmp_path, EXACT_PRODUCTION_SMOKE_CLAIM)

    assert len(discovery.claims) == 1
    assert discovery.claims[0].reference.claim_id == "claim-79792fed7eb306e5"


@pytest.mark.parametrize("line_ending", ("\r\n", "\r"))
def test_detached_threshold_normalizes_crlf_and_lone_cr_without_changing_spans(
    tmp_path: Path,
    line_ending: str,
) -> None:
    """Break caught: CR line endings change IDs or make a valid declaration fail."""

    document = line_ending.join(
        (
            EXACT_PRODUCTION_SMOKE_CLAIM,
            "Declared minimum improvement: 0.05",
        )
    )
    baseline = _discover_raw_source(
        tmp_path,
        "\n".join(
            (
                EXACT_PRODUCTION_SMOKE_CLAIM,
                "Declared minimum improvement: 0.05",
            )
        ),
    )
    claims = _discover_raw_source(tmp_path, document)

    assert len(baseline) == 1
    assert len(claims) == 1
    claim = claims[0]
    assert claim.reference.text == EXACT_PRODUCTION_SMOKE_CLAIM
    assert claim.reference.claim_id == baseline[0].reference.claim_id
    assert claim.minimum_improvement is not None
    assert claim.minimum_improvement.value == 0.05
    assert claim.scientific_claim is not None
    binding = claim.scientific_claim._source_binding
    assert binding is not None
    assert binding.document.text[binding.primary_start : binding.primary_end] == (
        EXACT_PRODUCTION_SMOKE_CLAIM
    )
    assert binding.document.text[
        binding.declaration_start : binding.declaration_end
    ] == "Declared minimum improvement: 0.05"
