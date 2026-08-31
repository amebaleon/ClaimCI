"""Second-pass one-shot driver including the analysis synthesis bridge."""

from __future__ import annotations

import apply_review_output_budget_patch as patch


def _patch_analysis_bridge() -> None:
    replaced = patch._replace_all(
        "claimci/review/analysis_bridge.py",
        "max_output_tokens=config.limits.max_output_tokens_per_call,",
        "max_output_tokens=config.limits.synthesis_max_output_tokens,",
    )
    if replaced != 2:
        raise RuntimeError(
            "claimci/review/analysis_bridge.py: expected two output budget references, "
            f"found {replaced}"
        )
    patch._replace_once(
        "claimci/review/analysis_bridge.py",
        '''    if type(response) is not ProviderResponse:
        raise ReviewError("analysis synthesis provider returned an invalid response")
    if len(response.output_text) > config.limits.max_output_chars:
''',
        '''    if type(response) is not ProviderResponse:
        raise ReviewError("analysis synthesis provider returned an invalid response")
    if not response.complete:
        if response.incomplete_reason == "max_output_tokens":
            raise ReviewError("analysis synthesis reached the provider output-token limit")
        raise ReviewError("analysis synthesis did not return a complete structured response")
    if len(response.output_text) > config.limits.max_output_chars:
''',
    )


def main() -> None:
    models = patch._read("claimci/review/models.py")
    if "extraction_max_output_tokens: int = 5_000" in models:
        print("review output-budget patch already applied")
        return

    patch._patch_models()
    patch._patch_config()
    patch._patch_provider_models()
    patch._patch_openai_adapter()
    patch._patch_sources()
    patch._patch_orchestrator()
    _patch_analysis_bridge()
    patch._patch_checked_in_config_and_docs()
    patch._patch_existing_tests()
    patch._report_legacy_references()
    print("review output-budget patch applied")


if __name__ == "__main__":
    main()
