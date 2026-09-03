"""Regression coverage for exact Java evidence routing in nested monorepos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from claimci.review.evidence import (
    EvidenceKind,
    EvidenceProvenance,
    discover_evidence,
)
from claimci.review.models import (
    ClaimType,
    ProviderUsage,
    ReviewConfig,
    ReviewLimits,
    ReviewStatus,
    ScientificClaim,
    SourceKind,
    SourceLocation,
)
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest
from claimci.review.sources import collect_review_sources


SERVICE = (
    "apps/example-backend/src/main/java/com/example/domain/"
    "DatasetService.java"
)
DAO = (
    "apps/example-backend/src/main/java/com/example/domain/"
    "DatasetItemDAO.java"
)
SERVICE_TEST = (
    "apps/example-backend/src/test/java/com/example/domain/"
    "DatasetServiceEnrichmentTest.java"
)
CLAIM_TEXT = "Fully versioned datasets skip the legacy item-count query."


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _opik_shaped_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    _write(base, SERVICE, "final class DatasetService { int mode = 1; }\n")
    _write(head, SERVICE, "final class DatasetService { int mode = 2; }\n")
    _write(base, DAO, "interface DatasetItemDAO { long count(); }\n")
    _write(head, DAO, "interface DatasetItemDAO { long countDistinct(); }\n")
    _write(
        head,
        SERVICE_TEST,
        "final class DatasetServiceEnrichmentTest { void skipsCount() {} }\n",
    )
    return base, head


def _claim(*, hints: tuple[str, ...] = ()) -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim-java-routing",
        source_text=CLAIM_TEXT,
        claim_type=ClaimType.IMPLEMENTATION_CLAIM,
        subject="DatasetService item-count control flow",
        source=SourceLocation(
            source_id="source-description",
            kind=SourceKind.PULL_REQUEST_DESCRIPTION,
            path=None,
            start_line=1,
            end_line=1,
        ),
        evidence_hints=hints,
    )


def _config() -> ReviewConfig:
    return ReviewConfig(
        enabled=True,
        provider="openai",
        model="fake-model",
        limits=ReviewLimits(timeout_seconds=30.0),
    )


def _extraction_output(request: StructuredRequest, hints: tuple[str, ...]) -> str:
    source = next(
        item
        for item in request.payload["sources"]
        if item["kind"] == "pull_request_description"
    )
    return json.dumps(
        {
            "claims": [
                {
                    "source_text": CLAIM_TEXT,
                    "claim_type": "implementation_claim",
                    "subject": "DatasetService item-count control flow",
                    "metric": None,
                    "direction": "not_applicable",
                    "claimed_magnitude": None,
                    "qualifiers": [],
                    "source": {
                        "source_id": source["source_id"],
                        "start_line": 1,
                        "end_line": 1,
                    },
                    "confidence": 0.9,
                    "evidence_hints": list(hints),
                }
            ]
        }
    )


def _synthesis_output(request: StructuredRequest) -> str:
    claim_id = request.payload["claims"][0]["claim_id"]
    return json.dumps(
        {
            "interpretations": [
                {
                    "claim_id": claim_id,
                    "interpretation": "The claim remains advisory.",
                    "citations": [],
                    "missing_evidence": [],
                    "unsupported_inferences": [],
                    "confidence": 0.5,
                }
            ]
        }
    )


class _FakeProvider:
    def __init__(
        self,
        *,
        hints: tuple[str, ...] = (),
        inspect_extraction: Callable[[StructuredRequest], None] | None = None,
    ) -> None:
        self.hints = hints
        self.inspect_extraction = inspect_extraction
        self.calls: list[StructuredRequest] = []

    def _response(self, request: StructuredRequest, output: str) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            output_text=output,
            provider="fake",
            model="fake-model",
            request_id=f"fake-{len(self.calls)}",
            usage=ProviderUsage(),
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        if self.inspect_extraction is not None:
            self.inspect_extraction(request)
        return self._response(request, _extraction_output(request, self.hints))

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        return self._response(request, _synthesis_output(request))


def _run(
    base: Path,
    head: Path,
    *,
    hints: tuple[str, ...] = (),
    inspect_extraction: Callable[[StructuredRequest], None] | None = None,
):
    provider = _FakeProvider(
        hints=hints,
        inspect_extraction=inspect_extraction,
    )
    route_path = (
        "docs/review.md" if (head / "docs" / "review.md").is_file() else SERVICE
    )
    result = run_review(
        ReviewInputs(
            repository_root=head,
            base_root=base,
            pr_description=(
                f"{CLAIM_TEXT}\nBenchmark evidence: 1 run in {route_path}"
            ),
        ),
        _config(),
        provider=provider,
    )
    return result, provider


def test_changed_nested_java_source_and_test_paths_share_one_trusted_policy(
    tmp_path: Path,
) -> None:
    """Removing Java from either policy must lose a changed path or its kind."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)

    assert {SERVICE, DAO, SERVICE_TEST} <= set(sources.changed_paths)

    evidence = discover_evidence(
        head,
        [_claim()],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )
    by_path = {reference.path: reference for reference in evidence.references}
    assert by_path[SERVICE].kind is EvidenceKind.SOURCE
    assert by_path[DAO].kind is EvidenceKind.SOURCE
    assert by_path[SERVICE_TEST].kind is EvidenceKind.TEST
    assert all(
        reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
        for reference in by_path.values()
    )


@pytest.mark.parametrize("suffix", [".py", ".js", ".ts", ".rs", ".go"])
def test_existing_source_suffixes_keep_changed_routing_behavior(
    tmp_path: Path,
    suffix: str,
) -> None:
    """The shared policy must preserve every previously supported language."""

    relative = f"apps/example-backend/src/main/language/Service{suffix}"
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    _write(base, relative, "old source\n")
    _write(head, relative, "new source\n")

    sources = collect_review_sources(head, base_root=base)
    assert relative in sources.changed_paths

    evidence = discover_evidence(
        head,
        [_claim()],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )
    reference = next(item for item in evidence.references if item.path == relative)
    assert reference.kind is EvidenceKind.SOURCE
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT


def test_exact_indexed_changed_java_hint_issues_a_supporting_reference(
    tmp_path: Path,
) -> None:
    """An exact path must route, while ClaimCI still assigns its identity and kind."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    evidence = discover_evidence(
        head,
        [_claim(hints=(SERVICE,))],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    reference = next(item for item in evidence.references if item.path == SERVICE)
    assert reference.kind is EvidenceKind.SOURCE
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert evidence.routing_incomplete is False


def test_descriptive_safe_hint_is_unresolved_without_claiming_file_absence(
    tmp_path: Path,
) -> None:
    """A provider description must never be interpreted as a missing literal file."""

    head = tmp_path / "head"
    head.mkdir()
    _write(head, SERVICE_TEST, "final class DatasetServiceEnrichmentTest {}\n")
    hint = "DatasetServiceEnrichmentTest source"

    evidence = discover_evidence(
        head,
        [_claim(hints=(hint,))],
        (SERVICE_TEST,),
        changed_paths=(),
    )

    unresolved = next(item for item in evidence.missing if item.requested_path == hint)
    assert unresolved.reason == "unresolved_provider_hint"
    assert "unresolved" in unresolved.description.casefold()
    assert "not available in the analyzed snapshot" not in unresolved.description.casefold()
    assert evidence.routing_incomplete is True


def test_descriptive_hint_does_not_block_changed_java_fallback(
    tmp_path: Path,
) -> None:
    """Removing deterministic fallback must leave this claim materially unresolved."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    hint = "DatasetServiceEnrichmentTest source"
    evidence = discover_evidence(
        head,
        [_claim(hints=(hint,))],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    assert {reference.path for reference in evidence.references} >= {
        SERVICE,
        SERVICE_TEST,
    }
    assert any(item.reason == "unresolved_provider_hint" for item in evidence.missing)
    assert evidence.routing_incomplete is False


def test_materially_unresolved_provider_routing_is_advisory_partial(
    tmp_path: Path,
) -> None:
    """A valid synthesis must not hide an uncovered explicit routing failure."""

    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    _write(head, "docs/review.md", "Bounded review context.\n")
    result, provider = _run(
        base,
        head,
        hints=("DatasetServiceEnrichmentTest source",),
    )

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "EVIDENCE_ROUTING_INCOMPLETE"
    assert len(result.claims) == len(result.interpretations) == 1
    assert result.evidence.routing_incomplete is True
    assert [call.task for call in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def test_legitimate_zero_evidence_without_routing_failure_remains_complete(
    tmp_path: Path,
) -> None:
    """Zero references alone must never force the advisory result to PARTIAL."""

    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    _write(head, "docs/review.md", "Bounded review context.\n")
    result, provider = _run(base, head)

    assert result.status is ReviewStatus.COMPLETE
    assert result.error_code is None
    assert result.evidence.references == ()
    assert result.evidence.routing_incomplete is False
    assert [call.task for call in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def test_extraction_contract_requires_exact_issued_repository_paths(
    tmp_path: Path,
) -> None:
    """Deleting the path-shaped contract must let descriptive hints recur."""

    base, head = _opik_shaped_snapshot(tmp_path)

    def inspect(request: StructuredRequest) -> None:
        contract = request.payload["extraction_contract"]["evidence_hints"]
        assert "exact repository-relative paths" in contract
        assert "repository_paths" in contract
        assert "empty list" in contract

    result, _ = _run(base, head, inspect_extraction=inspect)

    assert result.status is ReviewStatus.COMPLETE
