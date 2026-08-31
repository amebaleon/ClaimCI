"""Apply semantic claim-candidate recovery and partial-result hardening once."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def main() -> None:
    sources_path = ROOT / "claimci/review/sources.py"
    if "class ClaimCandidateValidation:" in sources_path.read_text(encoding="utf-8"):
        print("semantic review recovery patch already applied")
        return

    replace_once(
        "claimci/review/sources.py",
        "from collections.abc import Mapping\nfrom pathlib import Path\n",
        "from collections.abc import Mapping\nfrom dataclasses import dataclass\nfrom pathlib import Path\n",
    )
    replace_once(
        "claimci/review/sources.py",
        '''def validate_claim_candidates(
''',
        '''@dataclass(frozen=True)
class ClaimCandidateValidation:
    """Trusted claims retained from one untrusted extraction response."""

    claims: tuple[ScientificClaim, ...]
    rejection_reasons: tuple[str, ...] = ()

    @property
    def rejected_count(self) -> int:
        return len(self.rejection_reasons)


def _recover_exact_quote_location(
    candidate_value: object,
    records: Mapping[str, SourceRecord],
) -> object:
    """Repair only a uniquely recoverable exact full-line quote location.

    The provider remains unable to invent source text: recovery succeeds only
    when its exact quote occurs once as complete consecutive lines inside the
    already-issued source. Paraphrases and ambiguous repeated text stay invalid.
    """

    if not isinstance(candidate_value, Mapping):
        return candidate_value
    source_value = candidate_value.get("source")
    source_text = candidate_value.get("source_text")
    if not isinstance(source_value, Mapping) or not isinstance(source_text, str):
        return candidate_value
    source_id = source_value.get("source_id")
    if not isinstance(source_id, str) or source_id not in records or not source_text:
        return candidate_value

    record = records[source_id]
    start_line = source_value.get("start_line")
    end_line = source_value.get("end_line")
    if (
        isinstance(start_line, int)
        and not isinstance(start_line, bool)
        and isinstance(end_line, int)
        and not isinstance(end_line, bool)
    ):
        try:
            if source_text == _quote(record, start_line, end_line):
                return candidate_value
        except ReviewError:
            pass

    source_lines = record.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    quote_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not quote_lines or any(line == "" for line in quote_lines):
        return candidate_value
    width = len(quote_lines)
    matches = [
        index
        for index in range(0, len(source_lines) - width + 1)
        if source_lines[index : index + width] == quote_lines
    ]
    if len(matches) != 1:
        return candidate_value

    recovered = dict(candidate_value)
    recovered_source = dict(source_value)
    recovered_source["start_line"] = matches[0] + 1
    recovered_source["end_line"] = matches[0] + width
    recovered["source"] = recovered_source
    return recovered


def _claim_rejection_reason(exc: ReviewError) -> str:
    message = str(exc)
    if "source quote does not match" in message:
        return "source_quote_mismatch"
    if (
        "outside the selected source" in message
        or "cites an unknown source" in message
        or "claim source" in message
    ):
        return "source_location_invalid"
    if "duplicates an earlier claim" in message:
        return "duplicate_claim"
    return "invalid_claim_candidate"


def validate_claim_candidates_best_effort(
    payload: object,
    sources: SourceBundle,
    *,
    max_claims: int = MAX_CLAIMS,
) -> ClaimCandidateValidation:
    """Retain valid claims while excluding malformed provider candidates.

    Top-level response corruption remains an error. Individual candidates are
    independently checked by the strict validator, so one malformed advisory
    candidate cannot erase unrelated valid claims from the same response.
    """

    if not isinstance(sources, SourceBundle):
        raise ReviewError("sources must be a SourceBundle")
    if (
        isinstance(max_claims, bool)
        or not isinstance(max_claims, int)
        or not 1 <= max_claims <= MAX_CLAIMS
    ):
        raise ReviewError("max_claims must be an integer from 1 through 16")
    root = _strict_fields(payload, {"claims"}, "claim extraction response")
    candidates = root["claims"]
    if not isinstance(candidates, list) or len(candidates) > max_claims:
        raise ReviewError("claims must be a bounded list")

    records = {source.source_id: source for source in sources.sources}
    accepted: list[ScientificClaim] = []
    accepted_ids: set[str] = set()
    rejection_reasons: list[str] = []
    for candidate in candidates:
        recovered = _recover_exact_quote_location(candidate, records)
        try:
            validated = validate_claim_candidates(
                {"claims": [recovered]},
                sources,
                max_claims=1,
            )
        except ReviewError as exc:
            rejection_reasons.append(_claim_rejection_reason(exc))
            continue
        claim = validated[0]
        if claim.claim_id in accepted_ids:
            rejection_reasons.append("duplicate_claim")
            continue
        accepted.append(claim)
        accepted_ids.add(claim.claim_id)

    return ClaimCandidateValidation(
        claims=tuple(accepted),
        rejection_reasons=tuple(rejection_reasons),
    )


def validate_claim_candidates(
''',
    )

    replace_once(
        "claimci/review/orchestrator.py",
        "from .sources import collect_review_sources, validate_claim_candidates\n",
        "from .sources import collect_review_sources, validate_claim_candidates_best_effort\n",
    )
    replace_once(
        "claimci/review/orchestrator.py",
        '''        claims = validate_claim_candidates(
            _parse_json(extraction_response.output_text),
            sources,
            max_claims=config.limits.max_claims,
        )
        deterministic_audits = run_manifest_audits(
''',
        '''        try:
            claim_validation = validate_claim_candidates_best_effort(
                _parse_json(extraction_response.output_text),
                sources,
                max_claims=config.limits.max_claims,
            )
        except ReviewError:
            return _result(
                ReviewStatus.PARTIAL,
                calls=calls,
                error_code="CLAIM_EXTRACTION_INVALID",
                error_message=(
                    "Claim extraction returned a complete structured response whose "
                    "top-level semantics failed deterministic validation."
                ),
            )
        claims = claim_validation.claims
        rejected_claim_candidates = claim_validation.rejected_count
        if rejected_claim_candidates and not claims:
            return _result(
                ReviewStatus.PARTIAL,
                calls=calls,
                error_code="CLAIM_CANDIDATES_REJECTED",
                error_message=(
                    f"All {rejected_claim_candidates} extracted claim candidates failed "
                    "deterministic source validation and were excluded."
                ),
            )
        deterministic_audits = run_manifest_audits(
''',
    )
    replace_once(
        "claimci/review/orchestrator.py",
        '''        interpretations = _parse_interpretations(
            _parse_json(synthesis_response.output_text), claims, evidence
        )
        return _result(
            ReviewStatus.COMPLETE,
            claims=claims,
            interpretations=interpretations,
            evidence=evidence,
            deterministic_audits=deterministic_audits,
            calls=calls,
        )
''',
        '''        try:
            interpretations = _parse_interpretations(
                _parse_json(synthesis_response.output_text), claims, evidence
            )
        except ReviewError:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_INVALID",
                error_message=(
                    "Review synthesis returned complete structured output that failed "
                    "deterministic claim or citation validation."
                ),
            )
        if rejected_claim_candidates:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                interpretations=interpretations,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CLAIM_CANDIDATES_REJECTED",
                error_message=(
                    f"{rejected_claim_candidates} extracted claim candidate(s) failed "
                    "deterministic source validation and were excluded; every retained "
                    "claim was reviewed."
                ),
            )
        return _result(
            ReviewStatus.COMPLETE,
            claims=claims,
            interpretations=interpretations,
            evidence=evidence,
            deterministic_audits=deterministic_audits,
            calls=calls,
        )
''',
    )

    replace_once(
        "tests/test_review_hardening_day3.py",
        '''    assert result.status is ReviewStatus.UNAVAILABLE
    assert [request.task for request in provider.calls] == ["extract_claims", "synthesize_review"]


def test_aggregate_usage_does_not_report_partial_fields_or_cost_and_status_is_advisory(
''',
        '''    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert len(result.claims) == 1
    assert [request.task for request in provider.calls] == ["extract_claims", "synthesize_review"]


def test_aggregate_usage_does_not_report_partial_fields_or_cost_and_status_is_advisory(
''',
    )

    replace_once(
        "README.md",
        '''The review check is always `neutral` and is never a quality gate. Its Markdown
labels distinguish deterministic evidence, LLM interpretation, missing
evidence, and unsupported inference. Only the separate deterministic
''',
        '''The review check is always `neutral` and is never a quality gate. A malformed
individual provider claim candidate is excluded without discarding unrelated
valid claims; the resulting review is explicitly `PARTIAL`. A uniquely
recoverable exact full-line quote may have its line location repaired
deterministically, while paraphrases and ambiguous occurrences remain invalid.
Complete synthesis output that fails claim/citation validation also preserves
extracted claims and evidence as an explicit `PARTIAL` result rather than a
generic unavailable review. Its Markdown labels distinguish deterministic
evidence, LLM interpretation, missing evidence, and unsupported inference.
Only the separate deterministic
''',
    )

    print("semantic review recovery patch applied")


if __name__ == "__main__":
    main()
