"""Validated claim conversion and one-call advisory analysis review."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from claimci.analysis import (
    AdvisoryResearchInterpretation,
    AnalysisState,
    ArtifactKind,
    AuditClaimSpec,
    ClaimReference,
    Confidence,
    DeterministicAuditOutcome,
    ExperimentRole,
    FieldProvenance,
    MissingEvidence,
    ProvenanceKind,
    RepositoryPath,
    UnifiedAnalysisResult,
)
from claimci.models import AuditResult, Direction, Verdict
from claimci.review import (
    AnalysisReviewContext,
    ClaimDirection,
    ClaimMagnitude,
    ClaimType,
    MagnitudeKind,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
    ScientificClaim,
    SourceBundle,
    SourceKind,
    SourceLocation,
    SourceRecord,
    run_analysis_review,
    validate_scientific_claim_for_audit,
)
from claimci.review.provider import ProviderResponse, StructuredRequest


SOURCE_TEXT = "Accuracy improved by at least 0.05."


def _sources(text: str = SOURCE_TEXT) -> SourceBundle:
    record = SourceRecord(
        source_id="source-1",
        kind=SourceKind.REPOSITORY_FILE,
        path="CLAIM.md",
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return SourceBundle(
        sources=(record,),
        repository_paths=("CLAIM.md",),
        changed_paths=("CLAIM.md",),
        total_chars=len(text),
    )


def _scientific_claim(
    *,
    source_text: str = SOURCE_TEXT,
    magnitude: ClaimMagnitude | None = None,
    direction: ClaimDirection = ClaimDirection.HIGHER,
    claim_type: ClaimType = ClaimType.METRIC_IMPROVEMENT,
) -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim-1",
        source_text=source_text,
        claim_type=claim_type,
        subject="candidate",
        source=SourceLocation(
            "source-1",
            SourceKind.REPOSITORY_FILE,
            "CLAIM.md",
            1,
            1,
        ),
        metric="accuracy",
        direction=direction,
        claimed_magnitude=magnitude
        or ClaimMagnitude("0.05", 0.05, None, MagnitudeKind.ABSOLUTE),
        confidence=0.95,
    )


def _deterministic() -> DeterministicAuditOutcome:
    return DeterministicAuditOutcome.from_audit_result(
        AuditResult(
            verdict=Verdict.NOT_SUPPORTED,
            findings=(),
            metric="accuracy",
            minimum_improvement=0.05,
            direction=Direction.HIGHER,
        )
    )


def _context() -> AnalysisReviewContext:
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "validated source-anchored provider claim",
        RepositoryPath("CLAIM.md"),
        "source-1",
    )
    audit_claim = AuditClaimSpec(
        "claim-1",
        "accuracy",
        Direction.HIGHER,
        0.05,
        provenance,
        provenance,
        provenance,
    )
    return AnalysisReviewContext(
        claim=ClaimReference(
            "claim-1",
            SOURCE_TEXT,
            RepositoryPath("CLAIM.md"),
            Confidence(0.95),
            provenance,
        ),
        audit_claim=audit_claim,
        evidence_provenance=(provenance,),
        deterministic=_deterministic(),
        missing_evidence=(
            MissingEvidence(
                kind=ArtifactKind.DATASET,
                role=ExperimentRole.CANDIDATE,
                description="candidate dataset has only one run",
                claim_id="claim-1",
            ),
        ),
    )


class FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[StructuredRequest] = []
        self.extract_calls = 0

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        self.extract_calls += 1
        raise AssertionError("analysis review must not perform claim extraction")

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            json.dumps(self.payload),
            "fake",
            "fake-model",
            request_id="request-1",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def test_source_anchored_scientific_claim_converts_to_audit_claim_spec() -> None:
    claim = _scientific_claim()

    converted = validate_scientific_claim_for_audit(claim, _sources())

    assert type(converted) is AuditClaimSpec
    assert converted.claim_id == claim.claim_id
    assert converted.metric == "accuracy"
    assert converted.direction is Direction.HIGHER
    assert converted.minimum_absolute_improvement == 0.05
    assert converted.metric_provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL


def test_from_to_claimed_values_do_not_become_a_minimum_threshold() -> None:
    text = "Accuracy improved from 0.71 to 0.79."
    claim = _scientific_claim(
        source_text=text,
        magnitude=ClaimMagnitude("0.08", 0.08, None, MagnitudeKind.ABSOLUTE),
    )

    converted = validate_scientific_claim_for_audit(claim, _sources(text))

    assert converted.minimum_absolute_improvement is None
    assert converted.threshold_provenance is None


@pytest.mark.parametrize(
    "claim,sources,match",
    [
        (
            _scientific_claim(source_text="A paraphrase."),
            _sources(),
            "source|quote|match",
        ),
        (
            _scientific_claim(claim_type=ClaimType.RESOURCE_REDUCTION),
            _sources(),
            "type|metric improvement",
        ),
        (
            _scientific_claim(direction=ClaimDirection.NOT_APPLICABLE),
            _sources(),
            "direction",
        ),
        (
            _scientific_claim(
                magnitude=ClaimMagnitude(
                    "5 percent",
                    5,
                    "%",
                    MagnitudeKind.ABSOLUTE,
                )
            ),
            _sources(),
            "unit|magnitude|threshold",
        ),
    ],
)
def test_scientific_claim_conversion_rejects_unvalidated_semantics(
    claim: ScientificClaim,
    sources: SourceBundle,
    match: str,
) -> None:
    with pytest.raises(ReviewError, match=match):
        validate_scientific_claim_for_audit(claim, sources)


def test_analysis_review_makes_exactly_one_synthesis_call() -> None:
    provider = FakeProvider(
        {
            "summary": "The deterministic Audit remains not supported.",
            "interpretations": ["The submitted evidence has known limitations."],
            "missing_evidence": ["More candidate runs are needed."],
            "confidence": 0.8,
        }
    )

    interpretation = run_analysis_review(
        _context(),
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert type(interpretation) is AdvisoryResearchInterpretation
    assert provider.extract_calls == 0
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.task == "synthesize_review"
    assert request.payload["policy"]["deterministic_authority"] == "read_only"
    assert request.payload["deterministic"]["verdict"] == "NOT_SUPPORTED"
    assert "manifest" not in request.payload["deterministic"]["payload"]
    assert (
        request.payload["policy"]["compatibility_manifest"]
        == "internal_not_user_authored"
    )
    assert interpretation.authority.value == "advisory"


def test_provider_disagreement_cannot_override_deterministic_authority() -> None:
    provider = FakeProvider(
        {
            "summary": "I disagree and call this supported.",
            "interpretations": ["SUPPORTED"],
            "missing_evidence": [],
            "confidence": 1.0,
        }
    )
    advisory = run_analysis_review(
        _context(),
        ReviewConfig(enabled=True),
        provider=provider,
    )

    result = UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=_context().deterministic,
        research_interpretation=advisory,
    )

    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert not hasattr(advisory, "verdict")


def test_analysis_review_rejects_verdict_bearing_provider_output() -> None:
    provider = FakeProvider(
        {
            "summary": "advisory",
            "interpretations": ["text"],
            "missing_evidence": [],
            "confidence": 0.5,
            "verdict": "SUPPORTED",
        }
    )

    with pytest.raises(ReviewError, match="field|schema|verdict|output"):
        run_analysis_review(
            _context(),
            ReviewConfig(enabled=True),
            provider=provider,
        )


def test_analysis_review_does_not_fall_back_to_existing_two_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.review.orchestrator as orchestrator

    monkeypatch.setattr(
        orchestrator,
        "run_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_review fallback is forbidden")
        ),
    )
    provider = FakeProvider(
        {
            "summary": "advisory",
            "interpretations": ["bounded interpretation"],
            "missing_evidence": [],
            "confidence": 0.5,
        }
    )

    result = run_analysis_review(
        _context(),
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.summary == "advisory"
    assert len(provider.requests) == 1
