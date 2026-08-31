"""Preserve trusted extraction/evidence when advisory synthesis is invalid."""

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


def replace_status_in_function(relative: str, function_name: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    marker = f"def {function_name}("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{relative}: missing {function_name}")
    next_function = text.find("\ndef ", start + len(marker))
    end = len(text) if next_function < 0 else next_function
    block = text[start:end]
    old = "assert result.status is ReviewStatus.UNAVAILABLE"
    if block.count(old) != 1:
        raise RuntimeError(
            f"{relative}:{function_name}: expected one unavailable assertion, "
            f"found {block.count(old)}"
        )
    block = block.replace(old, "assert result.status is ReviewStatus.PARTIAL", 1)
    path.write_text(text[:start] + block + text[end:], encoding="utf-8", newline="\n")


def main() -> None:
    orchestrator = (ROOT / "claimci/review/orchestrator.py").read_text(encoding="utf-8")
    if 'error_code="SYNTHESIS_INVALID"' in orchestrator:
        print("synthesis partial-state patch already applied")
        return

    replace_once(
        "claimci/review/orchestrator.py",
        '''        if sum(call.output_chars for call in calls) > config.limits.max_output_chars:
            raise ReviewError("provider output exceeds configured total limit")
        interpretations = _parse_interpretations(
            _parse_json(synthesis_response.output_text), claims, evidence
        )
''',
        '''        if sum(call.output_chars for call in calls) > config.limits.max_output_chars:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_OUTPUT_LIMIT",
                error_message=(
                    "Review synthesis exceeded the configured aggregate output limit; "
                    "no synthesis interpretation was accepted."
                ),
            )
        try:
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
                    "Review synthesis returned complete output that failed deterministic "
                    "claim or citation validation; no synthesis interpretation was accepted."
                ),
            )
''',
    )

    orchestrator_tests = [
        "test_unknown_claim_id_in_synthesis_is_rejected_without_authority_mutation",
        "test_unknown_evidence_citation_is_rejected",
        "test_evidence_citation_must_be_issued_for_the_interpreted_claim",
        "test_malformed_synthesis_output_is_unavailable_without_a_third_call",
        "test_synthesis_rejects_lone_unicode_surrogates_before_rendering",
        "test_authority_field_injection_in_synthesis_is_rejected",
        "test_oversized_synthesis_output_is_unavailable_without_retry",
    ]
    for name in orchestrator_tests:
        replace_status_in_function("tests/test_review_orchestrator_day3.py", name)

    replace_status_in_function(
        "tests/test_review_hardening_day3.py",
        "test_synthesis_must_cover_each_accepted_claim_exactly_once",
    )
    replace_status_in_function(
        "tests/test_review_candidate_recovery_v1.py",
        "test_invalid_synthesis_remains_unavailable_and_fail_closed",
    )
    replace_once(
        "tests/test_review_candidate_recovery_v1.py",
        "def test_invalid_synthesis_remains_unavailable_and_fail_closed(\n",
        "def test_invalid_synthesis_is_partial_and_fail_closed(\n",
    )
    replace_once(
        "tests/test_review_candidate_recovery_v1.py",
        '''    assert result.error_code == "REVIEW_UNAVAILABLE"
    assert result.claims == ()
    assert len(result.provider_calls) == 2
''',
        '''    assert result.error_code == "SYNTHESIS_INVALID"
    assert len(result.claims) == 1
    assert result.interpretations == ()
    assert len(result.provider_calls) == 2
''',
    )

    replace_once(
        "README.md",
        '''deterministically; paraphrases, ambiguous occurrences, all-invalid extraction,
and malformed synthesis still fail closed. Its Markdown labels distinguish
''',
        '''deterministically; paraphrases and ambiguous occurrences remain invalid, and
all-invalid extraction still fails closed. Once extraction and deterministic
evidence discovery have succeeded, malformed or semantically invalid synthesis
is reported as `PARTIAL` with no invalid interpretation accepted. Its Markdown
labels distinguish
''',
    )

    print("synthesis partial-state patch applied")


if __name__ == "__main__":
    main()
