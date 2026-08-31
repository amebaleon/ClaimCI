"""Regression tests for keeping review evidence inside the active pull request."""

from __future__ import annotations

from pathlib import Path

from claimci.review.evidence import discover_evidence
from claimci.review.models import (
    ClaimType,
    ScientificClaim,
    SourceKind,
    SourceLocation,
)
from claimci.review.sources import collect_review_sources
from claimci.review.tools import (
    ManifestAuditPlan,
    select_relevant_manifest_audit_plans,
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resource_claim() -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim-resource",
        source_text="The rollup reads 4.0x fewer rows for the benchmark.",
        claim_type=ClaimType.RESOURCE_REDUCTION,
        subject="rollup benchmark row reduction",
        metric="rows read",
        source=SourceLocation(
            source_id="source-description",
            kind=SourceKind.PULL_REQUEST_DESCRIPTION,
            path=None,
            start_line=1,
            end_line=1,
        ),
    )


def test_changed_index_includes_structured_research_evidence(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()

    changed = {
        "bench/gold.sql": "SELECT 1;\n",
        "results/metrics.json": '{"rows": 100}\n',
        "configs/run.yaml": "seed: 1\n",
        "data/runs.csv": "seed,rows\n1,100\n",
    }
    for relative, content in changed.items():
        _write(base, relative, "old\n")
        _write(head, relative, content)
    _write(base, "results/unchanged.json", '{"rows": 999}\n')
    _write(head, "results/unchanged.json", '{"rows": 999}\n')

    sources = collect_review_sources(head, base_root=base)

    assert set(changed) <= set(sources.changed_paths)
    assert "results/unchanged.json" not in sources.changed_paths


def test_pr_scoped_heuristics_do_not_pull_unrelated_unchanged_evidence(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a-unrelated/benchmark-cost.json", '{"rows": 999999}\n')
    _write(tmp_path, "external/benchmark-cost.json", '{"rows": 100}\n')
    paths = (
        "a-unrelated/benchmark-cost.json",
        "external/benchmark-cost.json",
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        paths,
        changed_paths=("external/benchmark-cost.json",),
    )

    assert [reference.path for reference in bundle.references] == [
        "external/benchmark-cost.json"
    ]
    assert all("999999" not in reference.excerpt for reference in bundle.references)


def test_explicit_valid_hint_can_reach_an_unchanged_indexed_path(tmp_path: Path) -> None:
    _write(tmp_path, "external/benchmark-cost.json", '{"rows": 100}\n')
    _write(tmp_path, "baseline/benchmark-cost.json", '{"rows": 400}\n')
    claim = _resource_claim()

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (
            "external/benchmark-cost.json",
            "baseline/benchmark-cost.json",
        ),
        changed_paths=("external/benchmark-cost.json",),
        suggested_paths={claim.claim_id: ("baseline/benchmark-cost.json",)},
    )

    assert {reference.path for reference in bundle.references} == {
        "external/benchmark-cost.json",
        "baseline/benchmark-cost.json",
    }


def test_only_manifest_plans_touched_by_the_pr_remain_relevant() -> None:
    plans = (
        ManifestAuditPlan(
            manifest_path="research.yaml",
            paths=(
                "research.yaml",
                "baseline/results.json",
                "candidate/results.json",
            ),
        ),
        ManifestAuditPlan(
            manifest_path="studies/latency/research.yaml",
            paths=(
                "studies/latency/research.yaml",
                "studies/latency/baseline/results.json",
                "studies/latency/candidate/results.json",
            ),
        ),
    )

    selected = select_relevant_manifest_audit_plans(
        plans,
        changed_paths=("studies/latency/candidate/results.json",),
    )

    assert selected == (plans[1],)
