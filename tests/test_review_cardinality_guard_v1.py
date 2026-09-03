"""End-to-end regressions for bounded Java test-cardinality synthesis guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimci.review.evidence import EvidenceKind, EvidenceProvenance
from claimci.review.models import ProviderUsage, ReviewConfig, ReviewLimits, ReviewStatus
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest


TEST_PATH = (
    "apps/example-backend/src/test/java/com/example/domain/"
    "DatasetServiceEnrichmentTest.java"
)
COUNT_CLAIM = (
    "Added DatasetServiceEnrichmentTest (7 cases) covering item-count source "
    "selection."
)
SECOND_CLAIM = "The release notes describe the test organization."
FALSE_AFFIRMATION = (
    "The complete Java test source confirms the claimed seven test cases."
)


COMPLETE_EIGHT_TEST_SOURCE = """\
final class DatasetServiceEnrichmentTest {
    // @Test
    String marker = "@Test";
    /*
    @Test
    */
    @Test void inlineAnnotationDoesNotCount() {}
    @TestFactory
    void generatedCasesDoNotCount() {}

    @Test
    void caseOne() {}

    @Test
    void caseTwo() {}

    @Test
    void caseThree() {}

    @Test
    void caseFour() {}

    @Test
    void caseFive() {}

    @Test
    void caseSix() {}

    @Test
    void caseSeven() {}

    @Test
    void notMigratedSentinelFallsBackToItemCount() {}
}
"""


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _config() -> ReviewConfig:
    return ReviewConfig(
        enabled=True,
        provider="openai",
        model="fake-model",
        limits=ReviewLimits(timeout_seconds=30.0),
    )


def _description_source(request: StructuredRequest) -> dict[str, Any]:
    return next(
        source
        for source in request.payload["sources"]
        if source["kind"] == "pull_request_description"
    )


def _count_candidate(
    source: dict[str, Any],
    *,
    evidence_hints: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source_text": COUNT_CLAIM,
        "claim_type": "implementation_claim",
        "subject": "DatasetServiceEnrichmentTest",
        "metric": "test cases covering item-count source selection",
        "direction": "not_applicable",
        "claimed_magnitude": {
            "raw": "seven test cases",
            "value": 7,
            "unit": "cases",
            "kind": "absolute",
        },
        "qualifiers": [],
        "source": {
            "source_id": source["source_id"],
            "start_line": 1,
            "end_line": 1,
        },
        "confidence": 0.9,
        "evidence_hints": evidence_hints if evidence_hints is not None else [TEST_PATH],
    }


def _unrelated_candidate(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_text": SECOND_CLAIM,
        "claim_type": "other_scientific",
        "subject": "release note organization",
        "metric": None,
        "direction": "not_applicable",
        "claimed_magnitude": None,
        "qualifiers": [],
        "source": {
            "source_id": source["source_id"],
            "start_line": 2,
            "end_line": 2,
        },
        "confidence": 0.7,
        "evidence_hints": [],
    }


def _provider_response(
    request: StructuredRequest,
    output: str,
    *,
    index: int,
) -> ProviderResponse:
    return ProviderResponse(
        output_text=output,
        provider="fake",
        model="fake-model",
        request_id=f"fake-{index}",
        usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _ordinary_row(
    claim_id: str,
    *,
    citations: list[str],
    interpretation: str = FALSE_AFFIRMATION,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "interpretation": interpretation,
        "citations": citations,
        "missing_evidence": [],
        "unsupported_inferences": [],
        "confidence": 0.9,
    }


class _ConstraintAwareProvider:
    """Mirror the frozen false affirmation unless ClaimCI issues a fixed guard."""

    def __init__(self, *, include_second_claim: bool = False) -> None:
        self.include_second_claim = include_second_claim
        self.calls: list[StructuredRequest] = []

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        source = _description_source(request)
        candidates = [_count_candidate(source)]
        if self.include_second_claim:
            candidates.append(_unrelated_candidate(source))
        self.calls.append(request)
        return _provider_response(
            request,
            json.dumps({"claims": candidates}),
            index=len(self.calls),
        )

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        rows: list[dict[str, Any]] = []
        constraints = request.payload.get("interpretation_constraints_by_claim_id", {})
        evidence_by_claim = request.payload["evidence_ids_by_claim_id"]
        for claim in request.payload["claims"]:
            claim_id = claim["claim_id"]
            constraint = constraints.get(claim_id)
            if constraint is None:
                rows.append(
                    _ordinary_row(
                        claim_id,
                        citations=list(evidence_by_claim[claim_id]),
                    )
                )
                continue
            rows.append(
                _ordinary_row(
                    claim_id,
                    interpretation=constraint["required_interpretation"],
                    citations=list(constraint["required_citations"]),
                )
            )
        self.calls.append(request)
        return _provider_response(
            request,
            json.dumps({"interpretations": rows}),
            index=len(self.calls),
        )


class _ConstraintIgnoringProvider(_ConstraintAwareProvider):
    """Return structurally valid but semantically forbidden synthesis output."""

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        claim = request.payload["claims"][0]
        claim_id = claim["claim_id"]
        evidence_ids = request.payload["evidence_ids_by_claim_id"][claim_id]
        self.calls.append(request)
        return _provider_response(
            request,
            json.dumps(
                {
                    "interpretations": [
                        _ordinary_row(claim_id, citations=list(evidence_ids))
                    ]
                }
            ),
            index=len(self.calls),
        )


class _CrossClaimCitationProvider(_ConstraintAwareProvider):
    """Keep the first row conforming, then cite its evidence from another claim."""

    def __init__(self) -> None:
        super().__init__(include_second_claim=True)

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        claims = request.payload["claims"]
        first_id = claims[0]["claim_id"]
        second_id = claims[1]["claim_id"]
        first_evidence = request.payload["evidence_ids_by_claim_id"][first_id][0]
        constraint = request.payload.get(
            "interpretation_constraints_by_claim_id", {}
        ).get(first_id)
        first_row = _ordinary_row(
            first_id,
            citations=(
                list(constraint["required_citations"])
                if constraint is not None
                else [first_evidence]
            ),
            interpretation=(
                constraint["required_interpretation"]
                if constraint is not None
                else FALSE_AFFIRMATION
            ),
        )
        second_row = _ordinary_row(
            second_id,
            citations=[first_evidence],
            interpretation="The unrelated claim borrows another claim's citation.",
        )
        self.calls.append(request)
        return _provider_response(
            request,
            json.dumps({"interpretations": [first_row, second_row]}),
            index=len(self.calls),
        )


def _run(
    repository_root: Path,
    provider: _ConstraintAwareProvider,
    *,
    base_root: Path | None = None,
    description: str = COUNT_CLAIM,
):
    route_path = (
        "docs/review.md"
        if (repository_root / "docs" / "review.md").is_file()
        else TEST_PATH
    )
    result = run_review(
        ReviewInputs(
            repository_root=repository_root,
            base_root=base_root,
            pr_description=(
                f"{description}\nBenchmark evidence: 7 runs in {route_path}"
            ),
        ),
        _config(),
        provider=provider,
    )
    assert [call.task for call in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]
    assert len(result.provider_calls) == 2
    return result


def test_complete_owned_java_test_source_contradicts_false_seven_case_affirmation(
    tmp_path: Path,
) -> None:
    """Removing the fixed constraint must expose the provider's false affirmation."""

    repository = tmp_path / "head"
    repository.mkdir()
    _write(repository, TEST_PATH, COMPLETE_EIGHT_TEST_SOURCE)
    provider = _ConstraintAwareProvider()

    result = _run(repository, provider)

    assert result.status is ReviewStatus.COMPLETE
    assert len(result.claims) == len(result.interpretations) == 1
    reference = result.evidence.references[0]
    interpretation = result.interpretations[0]
    assert reference.path == TEST_PATH
    assert reference.kind is EvidenceKind.TEST
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert reference.claim_ids == (result.claims[0].claim_id,)
    assert interpretation.citations == (reference.evidence_id,)
    assert interpretation.interpretation != FALSE_AFFIRMATION
    normalized = interpretation.interpretation.casefold()
    assert "complete source" in normalized
    assert "contradict" in normalized
    assert "8" in normalized
    assert "7" in normalized


def test_fixed_java_lexical_grammar_counts_only_standalone_test_annotations(
    tmp_path: Path,
) -> None:
    """Counting comments, strings, or inline annotations must corrupt the fixed fact."""

    repository = tmp_path / "head"
    repository.mkdir()
    _write(repository, TEST_PATH, COMPLETE_EIGHT_TEST_SOURCE)
    provider = _ConstraintAwareProvider()

    _run(repository, provider)

    synthesis = provider.calls[1]
    claim_id = synthesis.payload["claims"][0]["claim_id"]
    assert "interpretation_constraints_by_claim_id" in synthesis.payload
    constraint = synthesis.payload["interpretation_constraints_by_claim_id"][claim_id]
    assert constraint["kind"] == "java_test_cardinality"
    assert constraint["state"] == "contradicted"
    assert constraint["claimed_count"] == 7
    assert constraint["observed_count"] == 8
    assert constraint["evidence_id"] == synthesis.payload["evidence"][0]["evidence_id"]
    assert constraint["required_citations"] == [constraint["evidence_id"]]
    assert synthesis.payload["evidence"][0]["excerpt_complete"] is True
    assert synthesis.payload["evidence"][0]["excerpt_locality"] == "complete_file"


def test_incomplete_unlocalized_java_test_excerpt_cannot_verify_exact_count(
    tmp_path: Path,
) -> None:
    """A visible prefix of seven must not establish a hidden whole-file count."""

    visible_seven = """\
final class DatasetServiceEnrichmentTest {
    @Test
    void caseOne() {}
    @Test
    void caseTwo() {}
    @Test
    void caseThree() {}
    @Test
    void caseFour() {}
    @Test
    void caseFive() {}
    @Test
    void caseSix() {}
    @Test
    void caseSeven() {}
"""
    hidden_eighth = """\
    @Test
    void notMigratedSentinelFallsBackToItemCount() {}
}
"""
    source = visible_seven + ("    // bounded filler line\n" * 800) + hidden_eighth
    head = tmp_path / "head"
    head.mkdir()
    _write(head, TEST_PATH, source)
    _write(head, "docs/review.md", "Bounded review context.\n")
    provider = _ConstraintAwareProvider()

    # The added file is explicitly issued, but its bounded prefix cannot stand
    # in for complete-file cardinality evidence.
    result = _run(head, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "EVIDENCE_ROUTING_INCOMPLETE"
    assert result.evidence.routing_incomplete is True
    assert len(result.claims) == len(result.interpretations) == 1
    assert result.evidence.references
    reference = result.evidence.references[0]
    assert reference.path == TEST_PATH
    assert len(reference.excerpt) <= 16_000
    assert "caseSeven" in reference.excerpt
    assert "notMigratedSentinelFallsBackToItemCount" not in reference.excerpt
    synthesis = provider.calls[1]
    assert synthesis.payload["evidence"] == []
    claim_id = synthesis.payload["claims"][0]["claim_id"]
    constraint = synthesis.payload["interpretation_constraints_by_claim_id"][claim_id]
    assert constraint["state"] == "incomplete"
    assert constraint["claimed_count"] == 7
    assert constraint["observed_count"] is None
    assert constraint["required_citations"] == []
    interpretation = result.interpretations[0]
    assert interpretation.citations == ()
    assert interpretation.interpretation != FALSE_AFFIRMATION
    normalized = interpretation.interpretation.casefold()
    assert "cannot" in normalized
    assert "verify" in normalized


def test_unavailable_exact_java_test_evidence_cannot_verify_exact_count(
    tmp_path: Path,
) -> None:
    """No issued bytes must not leave an exact test count provider-controlled."""

    repository = tmp_path / "head"
    repository.mkdir()
    _write(repository, TEST_PATH, "")
    _write(repository, "docs/review.md", "Bounded review context.\n")
    provider = _ConstraintAwareProvider()

    result = _run(repository, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "EVIDENCE_ROUTING_INCOMPLETE"
    assert result.evidence.routing_incomplete is True
    assert result.evidence.references == ()
    synthesis = provider.calls[1]
    claim_id = synthesis.payload["claims"][0]["claim_id"]
    constraint = synthesis.payload["interpretation_constraints_by_claim_id"][claim_id]
    assert constraint["kind"] == "java_test_cardinality"
    assert constraint["state"] == "incomplete"
    assert constraint["claimed_count"] == 7
    assert constraint["observed_count"] is None
    assert constraint["evidence_id"] is None
    assert constraint["required_citations"] == []
    interpretation = result.interpretations[0]
    assert interpretation.citations == ()
    assert interpretation.interpretation != FALSE_AFFIRMATION
    assert "cannot verify" in interpretation.interpretation.casefold()


def test_nonconforming_cardinality_synthesis_fails_closed(
    tmp_path: Path,
) -> None:
    """Deleting constraint validation must accept a known-false provider row."""

    repository = tmp_path / "head"
    repository.mkdir()
    _write(repository, TEST_PATH, COMPLETE_EIGHT_TEST_SOURCE)
    provider = _ConstraintIgnoringProvider()

    result = _run(repository, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert result.interpretations == ()


def test_java_unicode_escape_ambiguity_yields_no_deterministic_count(
    tmp_path: Path,
) -> None:
    """Java's pre-lexing escapes must not create a false standalone count."""

    repository = tmp_path / "head"
    repository.mkdir()
    source = """\
final class DatasetServiceEnrichmentTest {
    \\u002f\\u002a
    @Test
    \\u002a\\u002f
    @Test
    void actualCase() {}
}
"""
    _write(repository, TEST_PATH, source)
    provider = _ConstraintAwareProvider()

    result = _run(repository, provider)

    synthesis = provider.calls[1]
    claim_id = synthesis.payload["claims"][0]["claim_id"]
    constraint = synthesis.payload["interpretation_constraints_by_claim_id"][claim_id]
    assert constraint["state"] == "incomplete"
    assert constraint["observed_count"] is None
    assert constraint["required_citations"] == []
    assert result.status is ReviewStatus.COMPLETE
    assert "cannot verify" in result.interpretations[0].interpretation.casefold()


def test_cardinality_guard_does_not_weaken_cross_claim_citation_ownership(
    tmp_path: Path,
) -> None:
    """Another claim must never borrow the complete test reference."""

    repository = tmp_path / "head"
    repository.mkdir()
    _write(repository, TEST_PATH, COMPLETE_EIGHT_TEST_SOURCE)
    provider = _CrossClaimCitationProvider()

    result = _run(
        repository,
        provider,
        description=f"{COUNT_CLAIM}\n{SECOND_CLAIM}",
    )

    first_claim_id = result.claims[0].claim_id
    reference = result.evidence.references[0]
    assert reference.claim_ids == (first_claim_id,)
    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert result.interpretations == ()
    assert result.error_message is not None
    assert "citation ownership" in result.error_message.casefold()
