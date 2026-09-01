"""Real-boundary regressions for per-request synthesis contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from claimci.review.evidence import EvidenceBundle, EvidenceKind, EvidenceReference
from claimci.review.models import ProviderUsage, ReviewConfig, ReviewLimits, ReviewStatus
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest


FIRST_CLAIM = "The rollup reads 4.0x fewer rows."
SECOND_CLAIM = "The training budget is unchanged."
PROVIDER_TEXT = "ATTACKER_PROVIDER_TEXT_MUST_NOT_LEAK"


def _inputs(tmp_path: Path) -> ReviewInputs:
    (tmp_path / "README.md").write_text("# Synthesis fixture\n", encoding="utf-8")
    return ReviewInputs(
        repository_root=tmp_path,
        pr_title="Synthesis contract fixture",
        pr_description=f"{FIRST_CLAIM}\n{SECOND_CLAIM}",
    )


def _config() -> ReviewConfig:
    return ReviewConfig(
        enabled=True,
        provider="openai",
        model="fake-model",
        limits=ReviewLimits(),
    )


def _source(request: StructuredRequest) -> dict[str, Any]:
    return next(
        source
        for source in request.payload["sources"]
        if FIRST_CLAIM in source["text"] and SECOND_CLAIM in source["text"]
    )


def _extraction_output(request: StructuredRequest) -> str:
    source = _source(request)
    common = {
        "source": {
            "source_id": source["source_id"],
            "start_line": 1,
            "end_line": 1,
        },
        "qualifiers": [],
        "confidence": 0.9,
        "evidence_hints": [],
    }
    first = {
        **common,
        "source_text": FIRST_CLAIM,
        "claim_type": "resource_reduction",
        "subject": "rollup",
        "metric": "rows read",
        "direction": "lower",
        "claimed_magnitude": {
            "raw": "4.0x fewer rows",
            "value": 4.0,
            "unit": "x",
            "kind": "relative",
        },
    }
    second = {
        **common,
        "source_text": SECOND_CLAIM,
        "claim_type": "compute_equivalence",
        "subject": "training",
        "metric": "training budget",
        "direction": "not_applicable",
        "claimed_magnitude": None,
        "source": {
            "source_id": source["source_id"],
            "start_line": 2,
            "end_line": 2,
        },
    }
    return json.dumps({"claims": [first, second]})


def _asymmetric_evidence(
    _root: Path,
    claims: Any,
    _paths: Any,
    **_kwargs: Any,
) -> EvidenceBundle:
    """Issue one globally valid reference owned only by the first claim."""

    return EvidenceBundle(
        references=(
            EvidenceReference(
                evidence_id="evidence-first-only",
                claim_ids=(claims[0].claim_id,),
                kind=EvidenceKind.RESULTS,
                path="results.json",
                start_line=1,
                end_line=1,
                sha256="a" * 64,
                size=2,
                excerpt="{}",
            ),
        ),
        total_chars=2,
    )


def _valid_synthesis(request: StructuredRequest) -> str:
    rows = []
    for claim in request.payload["claims"]:
        claim_id = claim["claim_id"]
        citations = [
            reference["evidence_id"]
            for reference in request.payload["evidence"]
            if claim_id in reference["claim_ids"]
        ]
        rows.append(
            {
                "claim_id": claim_id,
                "interpretation": "Advisory interpretation.",
                "citations": citations,
                "missing_evidence": [],
                "unsupported_inferences": [],
                "confidence": 0.5,
            }
        )
    return json.dumps({"interpretations": rows})


class _TwoCallProvider:
    """A fake provider that records exactly the two structured requests."""

    def __init__(
        self,
        synthesis: Callable[[StructuredRequest], str] | None = None,
    ) -> None:
        self.synthesis = synthesis or _valid_synthesis
        self.calls: list[StructuredRequest] = []

    def _response(self, request: StructuredRequest, output: str) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            output_text=output,
            provider="fake",
            model="fake-model",
            request_id=f"fake-{len(self.calls)}",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        return self._response(request, _extraction_output(request))

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        return self._response(request, self.synthesis(request))


def _run(
    tmp_path: Path,
    provider: _TwoCallProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    import claimci.review.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "discover_evidence", _asymmetric_evidence)
    return run_review(_inputs(tmp_path), _config(), provider=provider)


def test_synthesis_payload_exposes_exact_evidence_ids_owned_by_each_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _TwoCallProvider()

    result = _run(tmp_path, provider, monkeypatch)

    assert result.status is ReviewStatus.COMPLETE
    assert [request.task for request in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]
    synthesis = provider.calls[1]
    claim_ids = [claim["claim_id"] for claim in synthesis.payload["claims"]]
    assert synthesis.payload["evidence_ids_by_claim_id"] == {
        claim_ids[0]: ["evidence-first-only"],
        claim_ids[1]: [],
    }


def test_synthesis_schema_has_one_owned_citation_alternative_per_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _TwoCallProvider()

    result = _run(tmp_path, provider, monkeypatch)

    assert result.status is ReviewStatus.COMPLETE
    synthesis = provider.calls[1]
    claim_ids = [claim["claim_id"] for claim in synthesis.payload["claims"]]
    interpretations = synthesis.schema["properties"]["interpretations"]
    assert interpretations["minItems"] == len(claim_ids) == 2
    assert interpretations["maxItems"] == len(claim_ids) == 2

    alternatives = interpretations["items"]["anyOf"]
    assert len(alternatives) == len(claim_ids)
    by_claim = {
        row["properties"]["claim_id"]["enum"][0]: row for row in alternatives
    }
    assert set(by_claim) == set(claim_ids)
    first_citations = by_claim[claim_ids[0]]["properties"]["citations"]
    assert first_citations["items"]["enum"] == ["evidence-first-only"]
    assert by_claim[claim_ids[1]]["properties"]["citations"] == {
        "type": "array",
        "maxItems": 0,
        "items": {"type": "string"},
    }


def test_claim_mismatched_issued_citation_is_partial_with_safe_ownership_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mismatched_synthesis(request: StructuredRequest) -> str:
        claims = request.payload["claims"]
        return json.dumps(
            {
                "interpretations": [
                    {
                        "claim_id": claims[0]["claim_id"],
                        "interpretation": "Advisory interpretation.",
                        "citations": ["evidence-first-only"],
                        "missing_evidence": [],
                        "unsupported_inferences": [],
                        "confidence": 0.5,
                    },
                    {
                        "claim_id": claims[1]["claim_id"],
                        "interpretation": PROVIDER_TEXT,
                        "citations": ["evidence-first-only"],
                        "missing_evidence": [],
                        "unsupported_inferences": [],
                        "confidence": 0.5,
                    },
                ]
            }
        )

    provider = _TwoCallProvider(synthesis=mismatched_synthesis)

    result = _run(tmp_path, provider, monkeypatch)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert result.interpretations == ()
    assert result.error_message is not None
    assert "citation ownership" in result.error_message.lower()
    assert PROVIDER_TEXT not in result.error_message
    assert len(provider.calls) == 2
