"""RED contract tests for bounded Markdown benchmark evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from claimci.review.evidence import discover_evidence
from claimci.review.models import (
    ClaimMagnitude,
    ClaimType,
    MagnitudeKind,
    ReviewLimits,
    ScientificClaim,
    SourceKind,
    SourceLocation,
)


_TABLE = (
    "| Benchmark | Metric | Row fold |\n"
    "| --- | --- | ---: |\n"
    "| Rollup | Rows read | 4.0x |\n"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resource_claim() -> ScientificClaim:
    return ScientificClaim(
        claim_id="claim-row-fold",
        source_text=(
            "The PR description reports a 4.0x row-fold reduction for the "
            "rollup benchmark."
        ),
        claim_type=ClaimType.RESOURCE_REDUCTION,
        subject="rollup benchmark",
        metric="rows read",
        claimed_magnitude=ClaimMagnitude(
            raw="4.0x",
            value=4.0,
            unit="x",
            kind=MagnitudeKind.RELATIVE,
        ),
        source=SourceLocation(
            source_id="source-description",
            kind=SourceKind.PULL_REQUEST_DESCRIPTION,
            path=None,
            start_line=1,
            end_line=1,
        ),
    )


def _provenance_value(reference: object) -> object:
    provenance = getattr(reference, "provenance", None)
    return getattr(provenance, "value", provenance)


def test_changed_readme_table_is_reported_measurement_with_snapshot_gap(
    tmp_path: Path,
) -> None:
    """A PR-description claim may cite a changed README table, not execution."""

    readme = "# Benchmark results\n\n" + _TABLE + "\n"
    _write(tmp_path, "README.md", readme)

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    references = [
        reference
        for reference in bundle.references
        if reference.path == "README.md"
    ]
    assert len(references) == 1
    reference = references[0]
    assert reference.claim_ids == ("claim-row-fold",)
    assert (reference.start_line, reference.end_line) == (3, 5)
    assert reference.excerpt == _TABLE
    assert _provenance_value(reference) == "reported_measurement"

    gaps = [
        item
        for item in bundle.missing
        if item.claim_id == "claim-row-fold"
        and "independent" in item.description.casefold()
    ]
    assert len(gaps) == 1
    description = gaps[0].description.casefold()
    assert "analyzed snapshot" in description
    assert "executed result" in description


def test_markdown_sql_and_structured_results_keep_distinct_provenance(
    tmp_path: Path,
) -> None:
    """Summary Markdown, executable SQL, and result artifacts are not conflated."""

    _write(tmp_path, "README.md", _TABLE)
    _write(
        tmp_path,
        "benchmarks/row_fold.sql",
        "-- executable row-fold benchmark definition\nSELECT count(*) FROM rows;\n",
    )
    _write(
        tmp_path,
        "results/benchmark_results.json",
        (
            '{"executed": true, "run_id": "run-1", "measurements": '
            '[{"benchmark": "rollup", "metric": "rows read", '
            '"row_fold": "4.0x"}]}\n'
        ),
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        (
            "README.md",
            "benchmarks/row_fold.sql",
            "results/benchmark_results.json",
        ),
        changed_paths=(
            "README.md",
            "benchmarks/row_fold.sql",
            "results/benchmark_results.json",
        ),
    )

    by_path = {reference.path: reference for reference in bundle.references}
    assert _provenance_value(by_path["README.md"]) == "reported_measurement"
    assert (
        _provenance_value(by_path["benchmarks/row_fold.sql"])
        == "executable_benchmark_definition"
    )
    assert (
        _provenance_value(by_path["results/benchmark_results.json"])
        == "executed_result_artifact"
    )


def test_unchanged_readme_is_not_newly_routed_by_pr_scoped_markdown_heuristics(
    tmp_path: Path,
) -> None:
    """Markdown routing remains confined to paths changed by the PR."""

    _write(tmp_path, "README.md", _TABLE)
    _write(tmp_path, "docs/README.md", _TABLE)

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md", "docs/README.md"),
        changed_paths=("README.md",),
    )

    paths = {reference.path for reference in bundle.references}
    assert "README.md" in paths
    assert "docs/README.md" not in paths


def test_unrelated_markdown_cannot_starve_routed_results_at_file_limit(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "README.md", "# Unrelated\n\nNo benchmark table here.\n")
    _write(
        tmp_path,
        "results/benchmark_results.json",
        '{"executed": true, "rows_read": 100}\n',
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md", "results/benchmark_results.json"),
        changed_paths=("README.md", "results/benchmark_results.json"),
        limits=ReviewLimits(max_files=1),
    )

    assert [reference.path for reference in bundle.references] == [
        "results/benchmark_results.json"
    ]
    assert not any(item.reason == "file_limit" for item in bundle.missing)


def test_reported_measurement_matches_value_and_semantics_in_the_same_row(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "README.md",
        (
            "| Benchmark | Metric | Row fold |\n"
            "| --- | --- | ---: |\n"
            "| Rollup | Rows read | 2.0x |\n"
            "| Other | Latency | 4.0x |\n"
        ),
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    assert bundle.references == ()


def test_expected_value_summary_is_not_an_executed_result_artifact(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "README.md", _TABLE)
    _write(
        tmp_path,
        "results/benchmark_results.json",
        '{"expected": "4.0x"}\n',
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md", "results/benchmark_results.json"),
        changed_paths=("README.md", "results/benchmark_results.json"),
    )

    result = next(
        reference
        for reference in bundle.references
        if reference.path == "results/benchmark_results.json"
    )
    assert _provenance_value(result) == "supporting_artifact"
    assert any(item.reason == "executed_result_not_available" for item in bundle.missing)


def test_markdown_table_without_outer_pipes_is_supported(tmp_path: Path) -> None:
    table = (
        "Benchmark | Metric | Row fold\n"
        "--- | --- | ---:\n"
        "Rollup | Rows read | 4.0x\n"
    )
    _write(tmp_path, "README.md", table)

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    reference = next(reference for reference in bundle.references)
    assert reference.excerpt == table
    assert _provenance_value(reference) == "reported_measurement"


def test_repository_source_must_overlap_the_specific_measurement_row(
    tmp_path: Path,
) -> None:
    table = (
        "| Benchmark | Metric | Row fold |\n"
        "| --- | --- | ---: |\n"
        "| Rollup | Rows read | 2.0x |\n"
        "| Other | Latency | 4.0x |\n"
    )
    _write(tmp_path, "README.md", table)
    claim = replace(
        _resource_claim(),
        source_text="| Rollup | Rows read | 2.0x |",
        source=SourceLocation(
            source_id="source-readme",
            kind=SourceKind.REPOSITORY_FILE,
            path="README.md",
            start_line=3,
            end_line=3,
        ),
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        ("README.md",),
        changed_paths=("README.md",),
    )

    assert bundle.references == ()


def test_completed_summary_is_not_a_raw_executed_result(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", _TABLE)
    _write(
        tmp_path,
        "results/benchmark_results.json",
        '{"status": "complete", "summary": {"row_fold": "4.0x"}}\n',
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md", "results/benchmark_results.json"),
        changed_paths=("README.md", "results/benchmark_results.json"),
    )

    result = next(
        reference
        for reference in bundle.references
        if reference.path == "results/benchmark_results.json"
    )
    assert _provenance_value(result) == "supporting_artifact"
    assert any(item.reason == "executed_result_not_available" for item in bundle.missing)


def test_indented_code_block_is_not_reported_as_a_markdown_measurement(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "README.md",
        "    | Benchmark | Metric | Row fold |\n"
        "    | --- | --- | ---: |\n"
        "    | Rollup | Rows read | 4.0x |\n",
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    assert bundle.references == ()


def test_aggregate_measurements_mapping_is_not_a_raw_result(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", _TABLE)
    _write(
        tmp_path,
        "results/benchmark_results.json",
        '{"status": "complete", "measurements": {"row_fold": "4.0x"}}\n',
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md", "results/benchmark_results.json"),
        changed_paths=("README.md", "results/benchmark_results.json"),
    )

    result = next(
        reference
        for reference in bundle.references
        if reference.path == "results/benchmark_results.json"
    )
    assert _provenance_value(result) == "supporting_artifact"


def test_fence_prefix_with_text_does_not_close_a_markdown_code_block(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "README.md",
        "```text\n"
        "```not-a-close\n"
        "| Benchmark | Metric | Row fold |\n"
        "| --- | --- | ---: |\n"
        "| Rollup | Rows read | 4.0x |\n"
        "```\n",
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    assert bundle.references == ()


def test_generic_benchmark_token_does_not_route_an_unrelated_row(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "README.md",
        "| Benchmark | Metric | Value |\n"
        "| --- | --- | ---: |\n"
        "| Other benchmark | Latency | 4.0x |\n",
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("README.md",),
        changed_paths=("README.md",),
    )

    assert bundle.references == ()


def test_executable_definition_alone_keeps_independent_verification_gap(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "benchmarks/row_fold.sql",
        "-- benchmark fold query\nSELECT count(*) FROM rows;\n",
    )

    bundle = discover_evidence(
        tmp_path,
        [_resource_claim()],
        ("benchmarks/row_fold.sql",),
        changed_paths=("benchmarks/row_fold.sql",),
    )

    assert [
        _provenance_value(reference) for reference in bundle.references
    ] == ["executable_benchmark_definition"]
    assert any(item.reason == "executed_result_not_available" for item in bundle.missing)
