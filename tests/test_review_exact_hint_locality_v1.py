"""RED contracts for exact source hints and bounded evidence locality.

These synthetic fixtures reproduce the Opik-shaped failure without reading or
executing an external checkout.  Provider hints remain untrusted strings: only
an exact path already issued in ``repository_paths`` may select source/test
evidence, and ClaimCI remains responsible for content, provenance, locality,
and claim ownership.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from claimci.review.evidence import (
    EvidenceKind,
    EvidenceProvenance,
    EvidenceReference,
    discover_evidence,
)
from claimci.review.models import (
    ClaimMagnitude,
    ClaimType,
    GateDisposition,
    MagnitudeKind,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
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
    "apps/opik-backend/src/main/java/com/comet/opik/domain/"
    "DatasetService.java"
)
DAO = (
    "apps/opik-backend/src/main/java/com/comet/opik/domain/"
    "DatasetItemDAO.java"
)
SERVICE_TEST = (
    "apps/opik-backend/src/test/java/com/comet/opik/domain/"
    "DatasetServiceEnrichmentTest.java"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _claim(
    claim_id: str,
    claim_type: ClaimType,
    source_text: str,
    subject: str,
    *,
    hints: tuple[str, ...] = (),
    metric: str | None = None,
    magnitude: ClaimMagnitude | None = None,
) -> ScientificClaim:
    return ScientificClaim(
        claim_id=claim_id,
        source_text=source_text,
        claim_type=claim_type,
        subject=subject,
        metric=metric,
        claimed_magnitude=magnitude,
        source=SourceLocation(
            source_id="source-pr-description",
            kind=SourceKind.PULL_REQUEST_DESCRIPTION,
            path=None,
            start_line=1,
            end_line=1,
        ),
        evidence_hints=hints,
        confidence=0.9,
    )


def _opik_shaped_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    head = tmp_path / "head"
    _write(
        base,
        SERVICE,
        "final class DatasetService { long legacyCount() { return 1; } }\n",
    )
    _write(
        head,
        SERVICE,
        "final class DatasetService { long narrowedCount() { return 1; } }\n",
    )
    # This dependency was present at both frozen commits and was not changed by
    # the PR.  It must nevertheless remain available to an exact indexed hint.
    dao = "interface DatasetItemDAO { long countDistinct(); }\n"
    _write(base, DAO, dao)
    _write(head, DAO, dao)
    _write(
        head,
        SERVICE_TEST,
        (
            "final class DatasetServiceEnrichmentTest {\n"
            "  @Test void skipsCountForVersionedData() {}\n"
            "  @Test void keepsMixedBatching() {}\n"
            "}\n"
        ),
    )
    return base, head


def _sentinel_snapshot(tmp_path: Path) -> tuple[Path, Path, str]:
    base = tmp_path / "base"
    head = tmp_path / "head"
    prefix = "// UNRELATED_PREFIX_MUST_NOT_BE_ISSUED\n"
    filler = "".join(
        f"  int unchangedField{index:04d} = {index}; // deterministic filler\n"
        for index in range(1_200)
    )
    old_material = "  return version.itemsTotal();\n"
    new_material = (
        "  if (version.itemsTotal() == ITEMS_TOTAL_NOT_MIGRATED) {\n"
        "    return datasetItemDAO.countDistinct(datasetId); // MATERIAL_SENTINEL\n"
        "  }\n"
    )
    suffix = "}\n"
    _write(base, SERVICE, prefix + filler + old_material + suffix)
    head_text = prefix + filler + new_material + suffix
    _write(head, SERVICE, head_text)
    return base, head, head_text


def _reference_for(
    references: tuple[EvidenceReference, ...],
    path: str,
) -> EvidenceReference:
    matches = [reference for reference in references if reference.path == path]
    assert len(matches) == 1
    return matches[0]


def _locality(reference: EvidenceReference) -> object:
    value = getattr(reference, "excerpt_locality", None)
    return getattr(value, "value", value)


_OBSERVED_ROUTE_MISMATCHES = (
    pytest.param(
        "resource-complexity-service",
        ClaimType.RESOURCE_REDUCTION,
        (
            "The count is O(N) in dataset item count because all ids in the "
            "range must be read and hashed."
        ),
        "dataset item-count enrichment query",
        SERVICE,
        id="resource-complexity-changed-service",
    ),
    pytest.param(
        "resource-complexity-dao",
        ClaimType.RESOURCE_REDUCTION,
        (
            "The count is O(N) in dataset item count because all ids in the "
            "range must be read and hashed."
        ),
        "dataset item-count enrichment query",
        DAO,
        id="resource-complexity-unchanged-dao",
    ),
    pytest.param(
        "resource-zero-service",
        ClaimType.RESOURCE_REDUCTION,
        "A fully-versioned workspace issues no count query at all.",
        "dataset item-count query",
        SERVICE,
        id="resource-zero-query-changed-service",
    ),
    pytest.param(
        "resource-zero-dao",
        ClaimType.RESOURCE_REDUCTION,
        "A fully-versioned workspace issues no count query at all.",
        "dataset item-count query",
        DAO,
        id="resource-zero-query-unchanged-dao",
    ),
    pytest.param(
        "resource-never-called-test",
        ClaimType.RESOURCE_REDUCTION,
        "The fully-versioned batch never calls the item summary query.",
        "fully-versioned batch item-count lookup",
        SERVICE_TEST,
        id="resource-never-called-changed-test",
    ),
    pytest.param(
        "compute-mixed-test",
        ClaimType.COMPUTE_EQUIVALENCE,
        "A mixed batch queries exactly the fallback id subset in one call.",
        "mixed batch item-count enrichment",
        SERVICE_TEST,
        id="compute-mixed-batch-changed-test",
    ),
    pytest.param(
        "compute-page-service",
        ClaimType.COMPUTE_EQUIVALENCE,
        "Batch list paths use one narrowed query per page, never per dataset.",
        "batch list call sites",
        SERVICE,
        id="compute-page-batching-changed-service",
    ),
    pytest.param(
        "compute-behavior-service",
        ClaimType.COMPUTE_EQUIVALENCE,
        "The extracted fallback is behaviorally equivalent to the old logic.",
        "dataset item-count enrichment behavior",
        SERVICE,
        id="compute-equivalence-changed-service",
    ),
    pytest.param(
        "compute-behavior-dao",
        ClaimType.COMPUTE_EQUIVALENCE,
        "The extracted fallback is behaviorally equivalent to the old logic.",
        "dataset item-count enrichment behavior",
        DAO,
        id="compute-equivalence-unchanged-dao",
    ),
)


@pytest.mark.parametrize(
    ("claim_id", "claim_type", "source_text", "subject", "hint"),
    _OBSERVED_ROUTE_MISMATCHES,
)
def test_exact_indexed_source_or_test_hint_bypasses_only_semantic_route_heuristics(
    tmp_path: Path,
    claim_id: str,
    claim_type: ClaimType,
    source_text: str,
    subject: str,
    hint: str,
) -> None:
    """Restoring `_matches()` here must recreate one of nine false gaps."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    claim = _claim(
        claim_id,
        claim_type,
        source_text,
        subject,
        hints=(hint,),
    )

    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    reference = _reference_for(bundle.references, hint)
    assert reference.claim_ids == (claim_id,)
    assert reference.kind is (
        EvidenceKind.TEST if hint == SERVICE_TEST else EvidenceKind.SOURCE
    )
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert not any(
        missing.claim_id == claim_id
        and missing.requested_path == hint
        and missing.reason == "route_mismatch"
        for missing in bundle.missing
    )
    assert bundle.routing_incomplete is False


def test_complete_unchanged_exact_source_hint_is_issued_as_complete_support(
    tmp_path: Path,
) -> None:
    """Exact identity must reach a small unchanged source without broad scanning."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    claim = _claim(
        "claim-unchanged-dao",
        ClaimType.RESOURCE_REDUCTION,
        "The count operation scans dataset item identifiers.",
        "item-count operation",
        hints=(DAO,),
    )

    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    reference = _reference_for(bundle.references, DAO)
    expected = (head / DAO).read_text(encoding="utf-8")
    assert reference.excerpt == expected
    assert reference.start_line == 1
    assert reference.end_line == 1
    assert getattr(reference, "excerpt_complete", None) is True
    assert _locality(reference) == "complete_file"


def test_unchanged_source_without_an_exact_hint_stays_out_of_autonomous_discovery(
    tmp_path: Path,
) -> None:
    """Removing changed-file confinement would leak unrelated unchanged source."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    claim = _claim(
        "claim-autonomous",
        ClaimType.IMPLEMENTATION_CLAIM,
        "Dataset enrichment uses a shared implementation.",
        "dataset enrichment",
    )

    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    assert DAO not in {
        reference.path
        for reference in bundle.references
        if claim.claim_id in reference.claim_ids
    }


def test_exact_changed_source_hint_routes_despite_semantically_unrelated_filename(
    tmp_path: Path,
) -> None:
    """An exact indexed source path must not be re-scored by filename tokens."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    claim = _claim(
        "claim-semantic-mismatch",
        ClaimType.RESOURCE_REDUCTION,
        "The legacy range scan is skipped for authoritative version totals.",
        "database scan avoidance",
        hints=(SERVICE,),
    )

    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    reference = _reference_for(bundle.references, SERVICE)
    assert reference.claim_ids == (claim.claim_id,)
    assert not any(item.reason == "route_mismatch" for item in bundle.missing)


@pytest.mark.parametrize(
    ("hint", "reason"),
    [
        pytest.param(
            "apps/opik-backend/src/main/java/Descriptive service implementation",
            "unresolved_provider_hint",
            id="safe-but-unindexed",
        ),
        pytest.param("../outside/DatasetService.java", "unsafe_path", id="parent"),
        pytest.param(
            "C:\\outside\\DatasetService.java",
            "unsafe_path",
            id="absolute-windows",
        ),
    ],
)
def test_non_exact_or_unsafe_hint_remains_unresolved_without_absence_authority(
    tmp_path: Path,
    hint: str,
    reason: str,
) -> None:
    """Exact-hint routing must not weaken indexing or confinement boundaries."""

    _, head = _opik_shaped_snapshot(tmp_path)
    claim = _claim(
        "claim-untrusted-hint",
        ClaimType.COMPUTE_EQUIVALENCE,
        "The implementation preserves behavior.",
        "behavior",
        hints=(hint,),
    )

    bundle = discover_evidence(
        head,
        [claim],
        (SERVICE, DAO, SERVICE_TEST),
        changed_paths=(),
    )

    assert bundle.references == ()
    missing = next(item for item in bundle.missing if item.requested_path == hint)
    assert missing.reason == reason
    assert "not available in the analyzed snapshot" not in missing.description.casefold()
    assert "does not exist" not in missing.description.casefold()
    assert bundle.routing_incomplete is True


def test_exact_hint_ownership_does_not_attach_evidence_to_another_claim(
    tmp_path: Path,
) -> None:
    """A hint may issue evidence only for the claim that supplied that hint."""

    base, head = _opik_shaped_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    owner = _claim(
        "claim-owner",
        ClaimType.COMPUTE_EQUIVALENCE,
        "The DAO contract preserves count behavior.",
        "count behavior",
        hints=(DAO,),
    )
    other = _claim(
        "claim-other",
        ClaimType.IMPLEMENTATION_CLAIM,
        "The service centralizes enrichment.",
        "service enrichment",
    )

    bundle = discover_evidence(
        head,
        [owner, other],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    reference = _reference_for(bundle.references, DAO)
    assert reference.claim_ids == (owner.claim_id,)


def test_changed_exact_source_uses_bounded_contiguous_changed_region_not_prefix(
    tmp_path: Path,
) -> None:
    """A prefix-only implementation would hide the material sentinel branch."""

    base, head, head_text = _sentinel_snapshot(tmp_path)
    sources = collect_review_sources(head, base_root=base)
    claim = _claim(
        "claim-sentinel",
        ClaimType.COMPUTE_EQUIVALENCE,
        "The new fallback is behaviorally equivalent for every version total.",
        "fallback equivalence",
        hints=(SERVICE,),
    )

    if "base_root" not in inspect.signature(discover_evidence).parameters:
        pytest.fail(
            "discover_evidence needs the trusted base_root to derive changed locality"
        )
    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
        base_root=base,
    )

    reference = _reference_for(bundle.references, SERVICE)
    assert len(reference.excerpt) <= ReviewLimits().max_file_chars
    assert "MATERIAL_SENTINEL" in reference.excerpt
    assert "ITEMS_TOTAL_NOT_MIGRATED" in reference.excerpt
    assert "UNRELATED_PREFIX_MUST_NOT_BE_ISSUED" not in reference.excerpt
    assert reference.start_line > 1
    head_lines = head_text.splitlines(keepends=True)
    assert reference.excerpt == "".join(
        head_lines[reference.start_line - 1 : reference.end_line]
    )
    assert getattr(reference, "excerpt_complete", None) is False
    assert _locality(reference) == "changed_region"


def test_changed_exact_config_without_comparison_locality_is_non_citable(
    tmp_path: Path,
) -> None:
    """A changed supporting artifact cannot turn an arbitrary prefix into proof."""

    path = "configs/opaque.ini"
    prefix = "PREFIX_IS_NOT_THE_CHANGED_CONFIGURATION\n" + ("x" * 20_000) + "\n"
    _write(tmp_path, path, prefix + "MATERIAL_CONFIG_AT_END=true\n")
    claim = _claim(
        "claim-config-locality",
        ClaimType.METRIC_IMPROVEMENT,
        "The candidate improves accuracy by five percent.",
        "candidate accuracy",
        hints=(path,),
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
    )

    reference = _reference_for(bundle.references, path)
    assert reference.kind is EvidenceKind.CONFIG
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert _locality(reference) == "unlocalized_prefix"
    assert "MATERIAL_CONFIG_AT_END" not in reference.excerpt
    assert any(
        item.claim_id == claim.claim_id
        and item.requested_path == path
        and item.reason == "excerpt_locality_unavailable"
        for item in bundle.missing
    )
    assert bundle.routing_incomplete is True


def test_oversized_unchanged_exact_source_is_explicitly_unlocalized_and_incomplete(
    tmp_path: Path,
) -> None:
    """A hollow prefix citation must never masquerade as localized support."""

    head = tmp_path / "head"
    prefix = "// PREFIX_IS_NOT_THE_MATERIAL_REGION\n" + ("x" * 20_000) + "\n"
    material = "long materialCountContract(); // MATERIAL_AT_END\n"
    _write(head, DAO, prefix + material)
    claim = _claim(
        "claim-oversized-unchanged",
        ClaimType.RESOURCE_REDUCTION,
        "The unchanged DAO exposes the material count contract.",
        "material count contract",
        hints=(DAO,),
    )

    bundle = discover_evidence(
        head,
        [claim],
        (DAO,),
        changed_paths=(),
    )

    reference = _reference_for(bundle.references, DAO)
    assert reference.claim_ids == (claim.claim_id,)
    assert len(reference.excerpt) <= ReviewLimits().max_file_chars
    assert "MATERIAL_AT_END" not in reference.excerpt
    assert getattr(reference, "excerpt_complete", None) is False
    assert _locality(reference) == "unlocalized_prefix"
    assert _locality(reference) != "complete_file"
    assert bundle.routing_incomplete is True
    assert any(
        item.claim_id == claim.claim_id
        and item.requested_path == DAO
        and item.reason == "excerpt_locality_unavailable"
        for item in bundle.missing
    )


class _EvidenceAwareFakeProvider:
    """No-network provider whose synthesis depends only on issued evidence."""

    def __init__(
        self,
        *,
        claim_text: str,
        claim_type: ClaimType,
        subject: str,
        hint: str,
        synthesis_mode: str,
    ) -> None:
        self.claim_text = claim_text
        self.claim_type = claim_type
        self.subject = subject
        self.hint = hint
        self.synthesis_mode = synthesis_mode
        self.calls: list[StructuredRequest] = []

    def _response(self, request: StructuredRequest, output: object) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            output_text=json.dumps(output),
            provider="fake",
            model="fake-model",
            request_id=f"fake-{len(self.calls)}",
            usage=ProviderUsage(),
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        source = next(
            item
            for item in request.payload["sources"]
            if item["kind"] == SourceKind.PULL_REQUEST_DESCRIPTION.value
        )
        return self._response(
            request,
            {
                "claims": [
                    {
                        "source_text": self.claim_text,
                        "claim_type": self.claim_type.value,
                        "subject": self.subject,
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
                        "evidence_hints": [self.hint],
                    }
                ]
            },
        )

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        claim_id = request.payload["claims"][0]["claim_id"]
        owned = [
            reference
            for reference in request.payload["evidence"]
            if claim_id in reference["claim_ids"]
        ]
        if self.synthesis_mode == "sentinel_from_excerpt":
            material = next(
                (
                    reference
                    for reference in owned
                    if "ITEMS_TOTAL_NOT_MIGRATED" in reference["excerpt"]
                ),
                None,
            )
            if material is None:
                interpretation = "Behavioral equivalence is not established."
                citations: list[str] = []
            else:
                interpretation = (
                    "The -1 not-migrated sentinel changes fallback behavior, so "
                    "the no-behavioral-change claim is contradicted."
                )
                citations = [material["evidence_id"]]
        elif self.synthesis_mode == "affirm_unlocalized_prefix":
            interpretation = (
                "The cited source positively establishes the material whole-file "
                "implementation claim."
            )
            citations = [owned[0]["evidence_id"]] if owned else []
        else:  # pragma: no cover - test construction is closed over two modes.
            raise AssertionError("unsupported fake synthesis mode")
        return self._response(
            request,
            {
                "interpretations": [
                    {
                        "claim_id": claim_id,
                        "interpretation": interpretation,
                        "citations": citations,
                        "missing_evidence": [],
                        "unsupported_inferences": [],
                        "confidence": 0.9,
                    }
                ]
            },
        )


def _review_config() -> ReviewConfig:
    return ReviewConfig(
        enabled=True,
        provider="openai",
        model="fake-model",
        limits=ReviewLimits(timeout_seconds=30.0),
    )


def test_changed_region_alone_surfaces_sentinel_behavior_in_final_interpretation(
    tmp_path: Path,
) -> None:
    """Routing/locality must suffice; no sentinel-specific product rule is needed."""

    claim_text = (
        "Performance only—no behavioral change; the extracted fallback is "
        "equivalent for every version total."
    )
    base, head, _ = _sentinel_snapshot(tmp_path)
    provider = _EvidenceAwareFakeProvider(
        claim_text=claim_text,
        claim_type=ClaimType.COMPUTE_EQUIVALENCE,
        subject="dataset item-count enrichment behavior",
        hint=SERVICE,
        synthesis_mode="sentinel_from_excerpt",
    )

    result = run_review(
        ReviewInputs(
            repository_root=head,
            base_root=base,
            pr_description=claim_text,
        ),
        _review_config(),
        provider=provider,
    )

    assert result.status is ReviewStatus.PARTIAL
    assert result.preflight is not None and result.preflight.scope is not None
    assert result.preflight.scope.mode == "legacy_pairwise_v1"
    assert result.preflight.scope.complete is False
    assert any(
        issue.code == "PREFLIGHT_G3_SELECTED_SOURCE_CHAR_LIMIT"
        for issue in result.preflight.scope.issues
    )
    assert result.error_code is None
    assert len(result.interpretations) == 1
    interpretation = result.interpretations[0]
    assert "-1 not-migrated sentinel changes fallback behavior" in (
        interpretation.interpretation
    )
    assert len(interpretation.citations) == 1
    reference = _reference_for(result.evidence.references, SERVICE)
    assert interpretation.citations == (reference.evidence_id,)
    assert "ITEMS_TOTAL_NOT_MIGRATED" in reference.excerpt
    assert _locality(reference) == "changed_region"
    assert [request.task for request in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def test_preflight_path_allocations_preserve_changed_region_after_earlier_exact_hints(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    earlier_paths = (
        "aa/oversized.py",
        "ab/oversized.py",
        "ac/oversized.py",
    )
    target = "zz/zz_target.py"
    for path in earlier_paths:
        _write(head, path, "x" * 17_000)
    unchanged_prefix = "".join(
        f"unchanged_{index:04d} = {index}\n" for index in range(150)
    )
    base_line = "target = '" + ("a" * 13_900) + "'\n"
    head_line = "target = '" + ("b" * 13_900) + "'\n"
    _write(base, target, unchanged_prefix + base_line)
    _write(head, target, unchanged_prefix + head_line)
    claim_text = (
        "Accuracy improves by 5% because the changed implementation in "
        f"{target} preserves the bounded target region."
    )

    class FourPathProvider:
        def __init__(self) -> None:
            self.calls: list[StructuredRequest] = []

        def _response(
            self, request: StructuredRequest, payload: dict[str, object]
        ) -> ProviderResponse:
            self.calls.append(request)
            return ProviderResponse(
                output_text=json.dumps(payload),
                provider="fake",
                model="fake-model",
            )

        def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
            source = next(
                item
                for item in request.payload["sources"]
                if item["kind"] == "pull_request_description"
            )
            return self._response(
                request,
                {
                    "claims": [
                        {
                            "source_text": claim_text,
                            "claim_type": "implementation_claim",
                            "subject": "bounded target region",
                            "metric": "accuracy",
                            "direction": "higher",
                            "claimed_magnitude": {
                                "raw": "5%",
                                "value": 5.0,
                                "unit": "%",
                                "kind": "relative",
                            },
                            "qualifiers": [],
                            "source": {
                                "source_id": source["source_id"],
                                "start_line": 1,
                                "end_line": 1,
                            },
                            "confidence": 0.9,
                            "evidence_hints": [*earlier_paths, target],
                        }
                    ]
                },
            )

        def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
            claim_id = request.payload["claims"][0]["claim_id"]
            target_evidence = next(
                (
                    item
                    for item in request.payload["evidence"]
                    if item["path"] == target
                ),
                None,
            )
            return self._response(
                request,
                {
                    "interpretations": [
                        {
                            "claim_id": claim_id,
                            "interpretation": "The bounded changed region is citable.",
                            "citations": (
                                []
                                if target_evidence is None
                                else [target_evidence["evidence_id"]]
                            ),
                            "missing_evidence": [],
                            "unsupported_inferences": [],
                            "confidence": 0.9,
                        }
                    ]
                },
            )

    provider = FourPathProvider()
    result = run_review(
        ReviewInputs(
            repository_root=head,
            base_root=base,
            pr_description=claim_text,
        ),
        _review_config(),
        provider=provider,
    )

    assert result.preflight is not None and result.preflight.scope is not None
    assert result.preflight.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert result.preflight.gates[2].disposition is GateDisposition.PASS_PARTIAL
    allocated_chars = dict(result.preflight.scope.materialized_path_chars)[target]
    assert 0 < allocated_chars <= 16_000
    target_reference = _reference_for(result.evidence.references, target)
    assert _locality(target_reference) == "changed_region"
    assert len(target_reference.excerpt) == allocated_chars
    assert [request.task for request in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]
    synthesis = provider.calls[1]
    assert [item["path"] for item in synthesis.payload["evidence"]] == [target]
    assert result.interpretations[0].citations == (target_reference.evidence_id,)
    assert result.status is ReviewStatus.PARTIAL


@pytest.mark.parametrize(
    "allocation",
    [
        [],
        (),
        ((1, 1),),
        (("other.py", 1),),
        (("evidence.py", True),),
        (("evidence.py", 16_001),),
    ],
)
def test_evidence_rejects_non_frozen_or_out_of_bounds_path_allocations(
    tmp_path: Path,
    allocation: object,
) -> None:
    _write(tmp_path, "evidence.py", "value = 1\n")

    with pytest.raises(ReviewError):
        discover_evidence(
            tmp_path,
            (),
            ("evidence.py",),
            materialized_path_chars=allocation,  # type: ignore[arg-type]
        )


def test_table_and_regular_excerpts_share_one_frozen_path_allowance(
    tmp_path: Path,
) -> None:
    path = "benchmarks/results.md"
    table = (
        "| Benchmark | Metric | Value |\n"
        "| --- | --- | ---: |\n"
        "| Rollup | Rows read | 4.0x |\n"
    )
    _write(tmp_path, path, table + ("supporting implementation detail\n" * 40))
    table_claim = _claim(
        "claim-table-budget",
        ClaimType.RESOURCE_REDUCTION,
        "The rollup reads 4.0x fewer rows.",
        "rollup",
        metric="rows read",
        magnitude=ClaimMagnitude(
            raw="4.0x",
            value=4.0,
            unit="x",
            kind=MagnitudeKind.RELATIVE,
        ),
    )
    regular_claim = _claim(
        "claim-regular-budget",
        ClaimType.RESOURCE_REDUCTION,
        "The benchmark implementation reduces rows read.",
        "benchmark implementation",
        metric="rows read",
    )

    bundle = discover_evidence(
        tmp_path,
        (table_claim, regular_claim),
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
        materialized_path_chars=((path, 200),),
    )

    path_references = tuple(
        reference for reference in bundle.references if reference.path == path
    )
    assert len(path_references) == 2
    assert sum(len(reference.excerpt) for reference in path_references) <= 200


def test_positive_synthesis_citation_to_unlocalized_prefix_fails_closed(
    tmp_path: Path,
) -> None:
    """An owned citation is insufficient when its material locality is unknown."""

    head = tmp_path / "head"
    oversized = (
        "// UNRELATED_PREFIX\n"
        + ("x" * 20_000)
        + "\nlong materialCountContract(); // MATERIAL_AT_END\n"
    )
    _write(head, DAO, oversized)
    routed_artifact = "scripts/run-eval.sh"
    _write(head, routed_artifact, "#!/bin/sh\nexit 0\n")
    claim_text = "The DAO establishes the material count contract."
    provider = _EvidenceAwareFakeProvider(
        claim_text=claim_text,
        claim_type=ClaimType.RESOURCE_REDUCTION,
        subject="material count contract",
        hint=DAO,
        synthesis_mode="affirm_unlocalized_prefix",
    )
    routed_description = (
        f"{claim_text}\nEvaluation configuration: {routed_artifact}"
    )

    result = run_review(
        ReviewInputs(
            repository_root=head,
            pr_description=routed_description,
        ),
        _review_config(),
        provider=provider,
    )

    reference = _reference_for(result.evidence.references, DAO)
    assert getattr(reference, "excerpt_complete", None) is False
    assert _locality(reference) == "unlocalized_prefix"
    synthesis = provider.calls[1]
    claim_id = synthesis.payload["claims"][0]["claim_id"]
    assert all(item["path"] != DAO for item in synthesis.payload["evidence"])
    assert synthesis.payload["evidence_ids_by_claim_id"][claim_id] == []
    constraint = synthesis.payload["interpretation_constraints_by_claim_id"][claim_id]
    assert constraint["kind"] == "unlocalized_evidence"
    assert constraint["state"] == "incomplete"
    assert constraint["required_citations"] == []
    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert result.interpretations == ()
    assert [request.task for request in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def test_legitimate_zero_evidence_has_no_routing_degradation(tmp_path: Path) -> None:
    """Zero evidence remains valid when no provider route failed."""

    _write(tmp_path, DAO, "interface DatasetItemDAO {}\n")
    claim = _claim(
        "claim-zero-evidence",
        ClaimType.IMPLEMENTATION_CLAIM,
        "The pull request describes an implementation property.",
        "implementation property",
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (DAO,),
        changed_paths=(),
    )

    assert bundle.references == ()
    assert bundle.routing_incomplete is False
    assert [item.reason for item in bundle.missing] == ["no_matching_evidence"]


@pytest.mark.parametrize("suffix", [".py", ".js", ".ts", ".rs", ".go", ".java"])
def test_existing_source_suffixes_keep_changed_heuristic_supporting_provenance(
    tmp_path: Path,
    suffix: str,
) -> None:
    """The exact-hint seam must not replace existing changed-file discovery."""

    path = f"apps/component/src/main/source/Worker{suffix}"
    _write(tmp_path, path, "implementation source\n")
    claim = _claim(
        f"claim-suffix-{suffix[1:]}",
        ClaimType.IMPLEMENTATION_CLAIM,
        "The worker implementation changed.",
        "worker implementation",
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (path,),
        changed_paths=(path,),
    )

    reference = _reference_for(bundle.references, path)
    assert reference.kind is EvidenceKind.SOURCE
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT


def test_markdown_and_runtime_artifacts_keep_distinct_provenance_and_locality(
    tmp_path: Path,
) -> None:
    """Source locality metadata must not blur measurement trust classes."""

    table = (
        "| Benchmark | Metric | Row fold |\n"
        "| --- | --- | ---: |\n"
        "| Rollup | Rows read | 4.0x |\n"
    )
    _write(tmp_path, "README.md", "# Results\n\n" + table)
    _write(
        tmp_path,
        "benchmarks/row_fold.sql",
        "-- row-fold benchmark\nSELECT count(*) FROM dataset_items;\n",
    )
    _write(
        tmp_path,
        "results/benchmark_results.json",
        (
            '{"executed": true, "run_id": "run-1", "measurements": '
            '[{"metric": "rows read", "value": "4.0x"}]}\n'
        ),
    )
    claim = _claim(
        "claim-row-fold",
        ClaimType.RESOURCE_REDUCTION,
        "The rollup benchmark reports a 4.0x reduction in rows read.",
        "rollup benchmark",
        metric="rows read",
        magnitude=ClaimMagnitude(
            raw="4.0x",
            value=4.0,
            unit="x",
            kind=MagnitudeKind.RELATIVE,
        ),
    )
    paths = (
        "README.md",
        "benchmarks/row_fold.sql",
        "results/benchmark_results.json",
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        paths,
        changed_paths=paths,
    )

    by_path = {reference.path: reference for reference in bundle.references}
    assert by_path["README.md"].provenance is EvidenceProvenance.REPORTED_MEASUREMENT
    assert _locality(by_path["README.md"]) == "selected_region"
    assert getattr(by_path["README.md"], "excerpt_complete", None) is False
    assert (
        by_path["benchmarks/row_fold.sql"].provenance
        is EvidenceProvenance.EXECUTABLE_BENCHMARK_DEFINITION
    )
    assert (
        by_path["results/benchmark_results.json"].provenance
        is EvidenceProvenance.EXECUTED_RESULT_ARTIFACT
    )
