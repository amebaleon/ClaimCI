"""Apply the reviewed research-review output-budget patch once.

This script exists only to let the isolated GitHub runner edit and verify the
private repository without exposing repository credentials outside GitHub.
It is idempotent and is removed after the resulting source commit is verified.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, text: str) -> None:
    (ROOT / relative).write_text(text, encoding="utf-8", newline="\n")


def _replace_once(relative: str, old: str, new: str) -> None:
    text = _read(relative)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one replacement, found {count}")
    _write(relative, text.replace(old, new, 1))


def _replace_all(relative: str, old: str, new: str) -> int:
    text = _read(relative)
    count = text.count(old)
    if count:
        _write(relative, text.replace(old, new))
    return count


def _patch_models() -> None:
    _replace_once(
        "claimci/review/models.py",
        '''@dataclass(frozen=True)
class ReviewLimits:
    max_calls: int = 2
    max_context_chars: int = 60_000
    max_output_chars: int = 12_000
    max_files: int = 24
    max_file_chars: int = 16_000
    max_output_tokens_per_call: int = 2_000
    timeout_seconds: float = 30.0
    retries: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.max_calls, "max_calls", 2)
        _positive_int(self.max_context_chars, "max_context_chars", 60_000)
        _positive_int(self.max_output_chars, "max_output_chars", 12_000)
        _positive_int(self.max_files, "max_files", 24)
        _positive_int(self.max_file_chars, "max_file_chars", 16_000)
        _positive_int(
            self.max_output_tokens_per_call,
            "max_output_tokens_per_call",
            2_000,
        )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
''',
        '''@dataclass(frozen=True)
class ReviewLimits:
    max_calls: int = 2
    max_context_chars: int = 60_000
    max_output_chars: int = 24_000
    max_files: int = 24
    max_file_chars: int = 16_000
    max_claims: int = 16
    extraction_max_output_tokens: int = 5_000
    synthesis_max_output_tokens: int = 4_000
    timeout_seconds: float = 30.0
    retries: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.max_calls, "max_calls", 2)
        _positive_int(self.max_context_chars, "max_context_chars", 60_000)
        _positive_int(self.max_output_chars, "max_output_chars", 24_000)
        _positive_int(self.max_files, "max_files", 24)
        _positive_int(self.max_file_chars, "max_file_chars", 16_000)
        _positive_int(self.max_claims, "max_claims", 16)
        _positive_int(
            self.extraction_max_output_tokens,
            "extraction_max_output_tokens",
            5_000,
        )
        _positive_int(
            self.synthesis_max_output_tokens,
            "synthesis_max_output_tokens",
            4_000,
        )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
''',
    )


def _patch_config() -> None:
    _replace_once(
        "claimci/review/config.py",
        '''_LIMIT_FIELDS = {
    "max_calls",
    "max_context_chars",
    "max_output_chars",
    "max_files",
    "max_file_chars",
    "max_output_tokens_per_call",
    "timeout_seconds",
}
''',
        '''_COMMON_LIMIT_FIELDS = {
    "max_calls",
    "max_context_chars",
    "max_output_chars",
    "max_files",
    "max_file_chars",
    "timeout_seconds",
}
_TASK_OUTPUT_LIMIT_FIELDS = {
    "max_claims",
    "extraction_max_output_tokens",
    "synthesis_max_output_tokens",
}
_LEGACY_OUTPUT_LIMIT_FIELD = "max_output_tokens_per_call"
_LIMIT_FIELDS = (
    _COMMON_LIMIT_FIELDS
    | _TASK_OUTPUT_LIMIT_FIELDS
    | {_LEGACY_OUTPUT_LIMIT_FIELD}
)
''',
    )
    _replace_once(
        "claimci/review/config.py",
        '''def load_review_config(
''',
        '''def _normalize_limits(
    raw_limits: Mapping[object, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    fields = set(raw_limits)
    unknown_limits = fields - _LIMIT_FIELDS
    if unknown_limits:
        raise ReviewError(
            f"review limits have unknown fields: {sorted(map(str, unknown_limits))}"
        )

    uses_legacy = _LEGACY_OUTPUT_LIMIT_FIELD in fields
    task_fields = fields & _TASK_OUTPUT_LIMIT_FIELDS
    if uses_legacy and task_fields:
        raise ReviewError(
            "legacy max_output_tokens_per_call cannot be combined with "
            "task-specific output limits"
        )

    if enabled:
        required = _COMMON_LIMIT_FIELDS | (
            {_LEGACY_OUTPUT_LIMIT_FIELD}
            if uses_legacy
            else _TASK_OUTPUT_LIMIT_FIELDS
        )
        missing_limits = required - fields
        if missing_limits:
            raise ReviewError(
                f"enabled review limits are missing fields: {sorted(missing_limits)}"
            )

    normalized = dict(raw_limits)
    if uses_legacy:
        legacy_value = normalized.pop(_LEGACY_OUTPUT_LIMIT_FIELD)
        normalized.setdefault("max_claims", 16)
        normalized["extraction_max_output_tokens"] = legacy_value
        normalized["synthesis_max_output_tokens"] = legacy_value
    return normalized


def load_review_config(
''',
    )
    _replace_once(
        "claimci/review/config.py",
        '''    unknown_limits = set(raw_limits) - _LIMIT_FIELDS
    if unknown_limits:
        raise ReviewError(f"review limits have unknown fields: {sorted(map(str, unknown_limits))}")
    if enabled:
        missing_limits = _LIMIT_FIELDS - set(raw_limits)
        if missing_limits:
            raise ReviewError(f"enabled review limits are missing fields: {sorted(missing_limits)}")
    try:
        limits = ReviewLimits(**dict(raw_limits))
''',
        '''    normalized_limits = _normalize_limits(raw_limits, enabled=enabled)
    try:
        limits = ReviewLimits(**normalized_limits)
''',
    )


def _patch_provider_models() -> None:
    _replace_once(
        "claimci/review/provider.py",
        '''class ProviderResponse:
    output_text: str
    provider: str
    model: str
    request_id: str | None = None
    usage: ProviderUsage = field(default_factory=ProviderUsage)

    def __post_init__(self) -> None:
''',
        '''class ProviderResponse:
    output_text: str
    provider: str
    model: str
    request_id: str | None = None
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    complete: bool = True
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
''',
    )
    _replace_once(
        "claimci/review/provider.py",
        '''        if not isinstance(self.usage, ProviderUsage):
            raise ReviewError("provider usage must be ProviderUsage")
''',
        '''        if not isinstance(self.usage, ProviderUsage):
            raise ReviewError("provider usage must be ProviderUsage")
        if not isinstance(self.complete, bool):
            raise ReviewError("provider completion state must be a boolean")
        if self.incomplete_reason is not None and (
            not isinstance(self.incomplete_reason, str)
            or not self.incomplete_reason.strip()
            or len(self.incomplete_reason) > 128
        ):
            raise ReviewError("provider incomplete reason must be bounded text or null")
        if self.complete and self.incomplete_reason is not None:
            raise ReviewError("a complete provider response cannot have an incomplete reason")
''',
    )


def _patch_openai_adapter() -> None:
    _replace_once(
        "claimci/review/openai_provider.py",
        '''        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ReviewError("OpenAI provider returned no structured text output")
        raw_usage = getattr(response, "usage", None)
''',
        '''        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ReviewError("OpenAI provider returned no structured text output")

        status = getattr(response, "status", None)
        if status in (None, "completed"):
            complete = True
            incomplete_reason = None
        elif status == "incomplete":
            complete = False
            details = getattr(response, "incomplete_details", None)
            raw_reason = getattr(details, "reason", None)
            incomplete_reason = (
                raw_reason
                if isinstance(raw_reason, str) and raw_reason.strip() and len(raw_reason) <= 128
                else "unknown"
            )
        else:
            raise ReviewError("OpenAI provider returned a non-completed response")

        raw_usage = getattr(response, "usage", None)
''',
    )
    _replace_once(
        "claimci/review/openai_provider.py",
        '''            request_id=getattr(response, "id", None),
            usage=usage,
        )
''',
        '''            request_id=getattr(response, "id", None),
            usage=usage,
            complete=complete,
            incomplete_reason=incomplete_reason,
        )
''',
    )


def _patch_sources() -> None:
    _replace_once("claimci/review/sources.py", "MAX_CLAIMS = 64\n", "MAX_CLAIMS = 16\n")
    _replace_once(
        "claimci/review/sources.py",
        '''def validate_claim_candidates(
    payload: object,
    sources: SourceBundle,
) -> tuple[ScientificClaim, ...]:
''',
        '''def validate_claim_candidates(
    payload: object,
    sources: SourceBundle,
    *,
    max_claims: int = MAX_CLAIMS,
) -> tuple[ScientificClaim, ...]:
''',
    )
    _replace_once(
        "claimci/review/sources.py",
        '''    if not isinstance(sources, SourceBundle):
        raise ReviewError("sources must be a SourceBundle")
    root = _strict_fields(payload, {"claims"}, "claim extraction response")
    candidates = root["claims"]
    if not isinstance(candidates, list) or len(candidates) > MAX_CLAIMS:
''',
        '''    if not isinstance(sources, SourceBundle):
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
''',
    )


def _patch_orchestrator() -> None:
    _replace_once(
        "claimci/review/orchestrator.py",
        "from __future__ import annotations\n\nimport json\n",
        "from __future__ import annotations\n\nimport copy\nimport json\n",
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''    "claim_selection": (
        "Extract only explicit, scientifically verifiable statements present "
        "in an issued source. Do not emit duplicate claims."
    ),
''',
        '''    "claim_selection": (
        "Extract only explicit, scientifically verifiable statements present "
        "in an issued source. Do not emit duplicate claims."
    ),
    "claim_prioritization": (
        "Return at most max_claims material claims. Prioritize claims affecting "
        "correctness, benchmark or performance conclusions, experimental fairness, "
        "production or deployment conclusions, and causal conclusions. Ignore minor "
        "implementation statements unless they materially support one of those claims."
    ),
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '            "maxItems": 64,\n',
        '            "maxItems": 16,\n',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        "\n_SYNTHESIS_SCHEMA: dict[str, Any] = {\n",
        '''

def _extraction_schema(max_claims: int) -> dict[str, Any]:
    if (
        isinstance(max_claims, bool)
        or not isinstance(max_claims, int)
        or not 1 <= max_claims <= 16
    ):
        raise ReviewError("max_claims must be an integer from 1 through 16")
    schema = copy.deepcopy(_EXTRACTION_SCHEMA)
    schema["properties"]["claims"]["maxItems"] = max_claims
    return schema


_SYNTHESIS_SCHEMA: dict[str, Any] = {
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''        provider = OpenAIReviewerProvider(
            model=config.model,
            timeout_seconds=config.limits.timeout_seconds,
            max_output_tokens=config.limits.max_output_tokens_per_call,
        )
''',
        '''        provider = OpenAIReviewerProvider(
            model=config.model,
            timeout_seconds=config.limits.timeout_seconds,
            max_output_tokens=max(
                config.limits.extraction_max_output_tokens,
                config.limits.synthesis_max_output_tokens,
            ),
        )
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''        extraction_payload = {
            "policy": {
''',
        '''        extraction_schema = _extraction_schema(config.limits.max_claims)
        extraction_payload = {
            "policy": {
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''            "extraction_contract": dict(_EXTRACTION_CONTRACT),
            "sources": _plain(sources.sources),
''',
        '''            "extraction_contract": dict(_EXTRACTION_CONTRACT),
            "max_claims": config.limits.max_claims,
            "sources": _plain(sources.sources),
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''            "extract_claims", extraction_payload, _EXTRACTION_SCHEMA
''',
        '''            "extract_claims", extraction_payload, extraction_schema
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''            schema=_EXTRACTION_SCHEMA,
            max_output_tokens=config.limits.max_output_tokens_per_call,
''',
        '''            schema=extraction_schema,
            max_output_tokens=config.limits.extraction_max_output_tokens,
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''        calls.append(extraction_call)
        if extraction_call.output_chars > config.limits.max_output_chars:
''',
        '''        calls.append(extraction_call)
        if not extraction_response.complete:
            truncated = extraction_response.incomplete_reason == "max_output_tokens"
            return _result(
                ReviewStatus.PARTIAL,
                calls=calls,
                error_code=(
                    "CLAIM_EXTRACTION_TRUNCATED"
                    if truncated
                    else "CLAIM_EXTRACTION_INCOMPLETE"
                ),
                error_message=(
                    "Claim extraction reached the provider output-token limit before "
                    "a complete structured response was returned."
                    if truncated
                    else "Claim extraction did not return a complete structured response."
                ),
            )
        if extraction_call.output_chars > config.limits.max_output_chars:
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''        claims = validate_claim_candidates(
            _parse_json(extraction_response.output_text), sources
        )
''',
        '''        claims = validate_claim_candidates(
            _parse_json(extraction_response.output_text),
            sources,
            max_claims=config.limits.max_claims,
        )
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''            schema=_SYNTHESIS_SCHEMA,
            max_output_tokens=config.limits.max_output_tokens_per_call,
''',
        '''            schema=_SYNTHESIS_SCHEMA,
            max_output_tokens=config.limits.synthesis_max_output_tokens,
''',
    )
    _replace_once(
        "claimci/review/orchestrator.py",
        '''        calls.append(synthesis_call)
        if sum(call.output_chars for call in calls) > config.limits.max_output_chars:
''',
        '''        calls.append(synthesis_call)
        if not synthesis_response.complete:
            truncated = synthesis_response.incomplete_reason == "max_output_tokens"
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code=("SYNTHESIS_TRUNCATED" if truncated else "SYNTHESIS_INCOMPLETE"),
                error_message=(
                    "Review synthesis reached the provider output-token limit before "
                    "a complete structured response was returned."
                    if truncated
                    else "Review synthesis did not return a complete structured response."
                ),
            )
        if sum(call.output_chars for call in calls) > config.limits.max_output_chars:
''',
    )


def _patch_checked_in_config_and_docs() -> None:
    _replace_once(
        ".claimci/review.yaml",
        '''  max_output_chars: 12000
  max_files: 24
  max_file_chars: 16000
  max_output_tokens_per_call: 2000
''',
        '''  max_output_chars: 24000
  max_files: 24
  max_file_chars: 16000
  max_claims: 16
  extraction_max_output_tokens: 5000
  synthesis_max_output_tokens: 4000
''',
    )
    _replace_once(
        "README.md",
        '''  max_output_chars: 12000
  max_files: 24
  max_file_chars: 16000
  max_output_tokens_per_call: 2000
''',
        '''  max_output_chars: 24000
  max_files: 24
  max_file_chars: 16000
  max_claims: 16
  extraction_max_output_tokens: 5000
  synthesis_max_output_tokens: 4000
''',
    )
    _replace_once(
        "README.md",
        '''Each review uses at most two provider calls (claim extraction and evidence-
grounded synthesis), a 60,000-character total context budget, and a 12,000-
character output budget. Provider input/output/total tokens and estimated
cost, when supplied, are recorded for observability only.
''',
        '''Each review uses at most two provider calls (claim extraction and evidence-
grounded synthesis), a 60,000-character total context budget, a 16-claim
materiality cap, task-specific output budgets of 5,000 and 4,000 tokens, and a
24,000-character audit-wide output budget. Legacy trusted configurations using
`max_output_tokens_per_call` remain readable and map that value to both calls.
Provider input/output/total tokens and estimated cost, when supplied, are
recorded for observability only.
''',
    )


def _patch_existing_tests() -> None:
    test_paths = [
        path
        for path in (ROOT / "tests").glob("test_*.py")
        if path.name != "test_review_output_budget_v1.py"
    ]
    for path in test_paths:
        text = path.read_text(encoding="utf-8")
        original = text
        text = text.replace('"max_output_chars": 12_000,', '"max_output_chars": 24_000,')
        text = text.replace(
            'assert limits.max_output_chars == 12_000',
            'assert limits.max_output_chars == 24_000',
        )
        text = text.replace(
            'assert limits.max_output_tokens_per_call == 2_000',
            'assert limits.max_claims == 16\n    assert limits.extraction_max_output_tokens == 5_000\n    assert limits.synthesis_max_output_tokens == 4_000',
        )
        text = text.replace(
            '("max_output_tokens_per_call", 0),',
            '("max_claims", 0),\n        ("extraction_max_output_tokens", 0),\n        ("synthesis_max_output_tokens", 0),',
        )
        text = text.replace(
            'assert all(request.max_output_tokens == 2_000 for request in provider.calls)',
            'assert [request.max_output_tokens for request in provider.calls] == [5_000, 4_000]',
        )
        text = re.sub(
            r'(?m)^(?P<indent>\s*)"max_output_tokens_per_call": 2_000,\s*$',
            lambda match: (
                f'{match.group("indent")}"max_claims": 16,\n'
                f'{match.group("indent")}"extraction_max_output_tokens": 5_000,\n'
                f'{match.group("indent")}"synthesis_max_output_tokens": 4_000,'
            ),
            text,
        )
        text = re.sub(
            r'(?m)^(?P<indent>\s*)max_output_tokens_per_call: 2000\s*$',
            lambda match: (
                f'{match.group("indent")}max_claims: 16\n'
                f'{match.group("indent")}extraction_max_output_tokens: 5000\n'
                f'{match.group("indent")}synthesis_max_output_tokens: 4000'
            ),
            text,
        )
        text = text.replace("max_output_chars: 12000", "max_output_chars: 24000")
        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")


def _report_legacy_references() -> None:
    allowed = {
        Path("claimci/review/config.py"),
        Path("tests/test_review_output_budget_v1.py"),
        Path("README.md"),
        Path("docs/superpowers/specs/2026-08-15-claimci-day3-design.md"),
    }
    leftovers: list[str] = []
    for base in (ROOT / "claimci", ROOT / "tests"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".md"}:
                continue
            relative = path.relative_to(ROOT)
            if relative in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            if "max_output_tokens_per_call" in text:
                leftovers.append(relative.as_posix())
    if leftovers:
        raise RuntimeError(f"stale direct output-budget references: {leftovers}")


def main() -> None:
    models = _read("claimci/review/models.py")
    if "extraction_max_output_tokens: int = 5_000" in models:
        print("review output-budget patch already applied")
        return

    _patch_models()
    _patch_config()
    _patch_provider_models()
    _patch_openai_adapter()
    _patch_sources()
    _patch_orchestrator()
    _patch_checked_in_config_and_docs()
    _patch_existing_tests()
    _report_legacy_references()
    print("review output-budget patch applied")


if __name__ == "__main__":
    main()
