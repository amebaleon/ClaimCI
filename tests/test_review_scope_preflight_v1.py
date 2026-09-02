"""Deterministic bounded material-scope regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

import claimci.review.preflight as preflight_module
from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    MaterialClaimSeed,
    ReviewConfig,
    ReviewLimits,
    ScientificClaim,
    SourceKind,
)
from claimci.review.orchestrator import ReviewInputs
from claimci.review.preflight import build_review_scope, scope_source_bundle


BASE = "a" * 40
HEAD = "b" * 40


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _inventory(*entries: tuple[str, ChangeStatus]) -> ChangeInventory:
    changes = tuple(
        sorted(
            (ChangeEntry(path, status) for path, status in entries),
            key=lambda entry: (entry.path, entry.status.value),
        )
    )
    return ChangeInventory(
        schema_version=1,
        requested_base_sha=BASE,
        comparison_base_sha=BASE,
        head_sha=HEAD,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=len(changes),
        complete=True,
        entries=changes,
    )


def _scope(
    root: Path,
    inventory: ChangeInventory,
    *,
    title: str = "",
    description: str = "",
    limits: ReviewLimits = ReviewLimits(),
):
    return build_review_scope(
        ReviewInputs(
            repository_root=root,
            pr_title=title,
            pr_description=description,
        ),
        ReviewConfig(enabled=True, limits=limits),
        inventory,
    )


def test_material_seed_contract_is_routing_metadata_not_a_scientific_claim() -> None:
    seed = MaterialClaimSeed(
        origin_source_id="source-1234567890abcdef",
        category="benchmark",
        route_terms=("accuracy", "benchmark"),
    )
    assert seed.category == "benchmark"
    assert not isinstance(seed, ScientificClaim)


def test_quantitative_comparative_benchmark_reproducibility_and_model_dataset_seeds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/model_eval.py", "def evaluate():\n    return 1\n")
    scope = _scope(
        root,
        _inventory(("benchmarks/model_eval.py", ChangeStatus.MODIFIED)),
        title="Qwen model benchmark improves WER by 12%",
        description=(
            "Evaluation is 1.5x faster in 42 seconds because batching changed. "
            "Reproduce with dataset revision abc and batch count 16."
        ),
    )
    assert {seed.category for seed in scope.seeds} == {
        "benchmark",
        "comparative_causal",
        "model_dataset",
        "quantitative",
        "reproducibility",
    }
    assert all(isinstance(seed, MaterialClaimSeed) for seed in scope.seeds)
    assert scope.selected_paths == ("benchmarks/model_eval.py",)


@pytest.mark.parametrize(
    ("description", "category"),
    [
        ("Processed 42 files in the held-out run.", "quantitative"),
        ("Model A versus model B on the same corpus.", "comparative_causal"),
    ],
)
def test_count_and_comparison_terms_create_normalized_material_seeds(
    tmp_path: Path,
    description: str,
    category: str,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/eval.py", "score = 1\n")
    scope = _scope(
        root,
        _inventory(("benchmarks/eval.py", ChangeStatus.MODIFIED)),
        description=description,
    )
    assert category in {seed.category for seed in scope.seeds}


def test_exact_path_then_category_overlap_kind_priority_and_path_order_are_deterministic(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    paths = (
        "config/train.yaml",
        "results/final.json",
        "src/model.py",
        "src/z_model.py",
    )
    for root, creation_order in zip(roots, (paths, tuple(reversed(paths))), strict=True):
        root.mkdir()
        for path in creation_order:
            _write(root, path, f"material for {path}\n")
    inventory = _inventory(*( (path, ChangeStatus.MODIFIED) for path in paths))
    scopes = tuple(
        _scope(
            root,
            inventory,
            description=(
                "The benchmark result improves model accuracy by 10%; "
                "see results/final.json and reproduce with train config."
            ),
        )
        for root in roots
    )
    assert scopes[0].selected_paths[0] == "results/final.json"
    assert scopes[0].issued_paths == scopes[1].issued_paths
    assert scopes[0].selected_paths == scopes[1].selected_paths
    assert scopes[0].seeds == scopes[1].seeds
    assert scopes[0].issues == scopes[1].issues
    bundles = tuple(scope_source_bundle(scope) for scope in scopes)
    assert [record.source_id for record in bundles[0].sources] == [
        record.source_id for record in bundles[1].sources
    ]
    assert bundles[0].total_chars == bundles[1].total_chars


def test_prose_seed_without_a_validated_local_changed_candidate_is_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    scope = _scope(
        root,
        _inventory(),
        title="Accuracy improves by 20%",
        description="External benchmark results are faster.",
    )
    assert scope.seeds
    assert not scope.complete
    assert {issue.code for issue in scope.issues} >= {
        "PREFLIGHT_G2_NO_ROUTABLE_CHANGED_PATH"
    }


def test_external_only_required_route_is_explicit_and_cannot_complete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    scope = _scope(
        root,
        _inventory(),
        description=(
            "Accuracy improves by 20%; results are only at "
            "https://example.invalid/results.json."
        ),
    )
    assert not scope.complete
    assert "PREFLIGHT_G2_EXTERNAL_EVIDENCE_ONLY" in {
        issue.code for issue in scope.issues
    }


def test_comparison_explosion_only_opens_declared_changed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    for index in range(513):
        _write(root, f"irrelevant/file_{index:04d}.py", "x = 1\n")
    _write(root, "zz_target/benchmark.py", "score = 0.95\n")
    opened: list[str] = []
    real_capture = preflight_module.capture_confined_regular_file

    def traced_capture(root_path: Path, relative: str, *, max_bytes: int):
        opened.append(relative)
        return real_capture(root_path, relative, max_bytes=max_bytes)

    monkeypatch.setattr(preflight_module, "capture_confined_regular_file", traced_capture)
    scope = _scope(
        root,
        _inventory(("zz_target/benchmark.py", ChangeStatus.MODIFIED)),
        title="Benchmark score improves by 5%",
    )
    assert scope.issued_changed_paths == ("zz_target/benchmark.py",)
    assert set(opened) == {"zz_target/benchmark.py"}


def test_changed_entries_survive_a_legacy_2048_path_lexical_prefix(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    for index in range(2_049):
        _write(root, f"aaa/file_{index:04d}.txt", "irrelevant\n")
    targets = ("src/models/glm/a.py", "src/models/glm/b.py")
    for target in targets:
        _write(root, target, "model_accuracy = 0.9\n")
    scope = _scope(
        root,
        _inventory(*( (target, ChangeStatus.MODIFIED) for target in targets)),
        title="GLM model accuracy improves by 9%",
    )
    assert scope.issued_changed_paths == targets


def test_huge_unchanged_repository_file_is_never_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "src/changed.py", "throughput = 2\n")
    huge = _write(root, "cassettes/huge.yaml", b"x" * 2_187_560)
    real_capture = preflight_module.capture_confined_regular_file

    def guarded_capture(root_path: Path, relative: str, *, max_bytes: int):
        if relative == "cassettes/huge.yaml":
            raise AssertionError("unchanged huge file was touched")
        return real_capture(root_path, relative, max_bytes=max_bytes)

    monkeypatch.setattr(preflight_module, "capture_confined_regular_file", guarded_capture)
    scope = _scope(
        root,
        _inventory(("src/changed.py", ChangeStatus.MODIFIED)),
        title="Throughput improves by 2x",
    )
    assert scope.selected_paths == ("src/changed.py",)
    assert huge.stat().st_size == 2_187_560


def test_openasr_submission_script_is_passive_config_and_missing_results_are_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    script = "MODEL=Qwen3-ASR\npython run_eval.py --dataset librispeech\n"
    _write(root, "transformers/submit_jobs_qwen3asr.sh", script)
    scope = _scope(
        root,
        _inventory(("transformers/submit_jobs_qwen3asr.sh", ChangeStatus.ADDED)),
        title="Evaluate Qwen3-ASR",
        description="The model benchmark reports 8% WER and 3x RTFx.",
    )
    assert scope.selected_paths == ("transformers/submit_jobs_qwen3asr.sh",)
    assert any(seed.category == "benchmark" for seed in scope.seeds)
    assert "PREFLIGHT_G2_MATERIAL_SEED_UNROUTED" in {
        issue.code for issue in scope.issues
    }
    assert all(
        record.path != "transformers/submit_jobs_qwen3asr.sh"
        for record in scope.sources
    )
    assert not scope.complete


def test_selected_huge_file_is_recorded_and_a_safe_route_keeps_partial_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "results/required.json", b"x" * (16 * 1024 * 1024 + 1))
    _write(root, "benchmarks/safe.py", "accuracy = 95\n")
    scope = _scope(
        root,
        _inventory(
            ("benchmarks/safe.py", ChangeStatus.MODIFIED),
            ("results/required.json", ChangeStatus.MODIFIED),
        ),
        description="Benchmark accuracy is 95%; see results/required.json.",
    )
    assert "benchmarks/safe.py" in scope.selected_paths
    assert "results/required.json" not in scope.selected_paths
    assert any(
        issue.code == "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"
        and issue.path == "results/required.json"
        for issue in scope.issues
    )
    assert not scope.complete


@pytest.mark.parametrize("status", [ChangeStatus.DELETED])
def test_deleted_only_required_route_never_produces_complete_scope(
    tmp_path: Path,
    status: ChangeStatus,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    scope = _scope(
        root,
        _inventory(("results/deleted.json", status)),
        description="The deleted benchmark result improved accuracy by 10%.",
    )
    assert not scope.issued_changed_paths
    assert not scope.complete
    assert "PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH" in {
        issue.code for issue in scope.issues
    }


def test_exact_deleted_route_remains_explicit_when_another_safe_route_exists(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/safe.py", "accuracy = 95\n")
    scope = _scope(
        root,
        _inventory(
            ("benchmarks/safe.py", ChangeStatus.MODIFIED),
            ("results/deleted.json", ChangeStatus.DELETED),
        ),
        description=(
            "Benchmark accuracy improved by 10%; the required result was "
            "results/deleted.json."
        ),
    )
    assert "benchmarks/safe.py" in scope.selected_paths
    assert not scope.complete
    assert any(
        issue.code == "PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH"
        and issue.path == "results/deleted.json"
        for issue in scope.issues
    )


def test_unsafe_required_path_literal_is_reported_without_becoming_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/safe.py", "accuracy = 95\n")
    scope = _scope(
        root,
        _inventory(("benchmarks/safe.py", ChangeStatus.MODIFIED)),
        description="Accuracy improved by 10%; see ../../outside/results.json.",
    )
    assert scope.issued_paths == ("benchmarks/safe.py",)
    assert not scope.complete
    assert any(
        issue.code == "PREFLIGHT_G2_OUT_OF_SCOPE_PATH" and issue.path is None
        for issue in scope.issues
    )


def test_symlinked_head_root_is_rejected_before_any_material_read(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    _write(target, "benchmarks/eval.py", "score = 1\n")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="head repository root is invalid"):
        _scope(
            linked,
            _inventory(("benchmarks/eval.py", ChangeStatus.MODIFIED)),
            title="Benchmark score improves by 5%",
        )


def test_required_changed_document_truncation_is_an_explicit_locality_omission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "docs/large-study.md", "Accuracy improves by 10%.\n" + "x" * 100)
    scope = _scope(
        root,
        _inventory(("docs/large-study.md", ChangeStatus.MODIFIED)),
        title="Review docs/large-study.md accuracy benchmark",
        limits=ReviewLimits(max_file_chars=20),
    )
    assert not scope.complete
    assert any(
        issue.code == "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE"
        and issue.path == "docs/large-study.md"
        and issue.observed == 126
        and issue.limit == 20
        for issue in scope.issues
    )


def test_undecodable_required_candidate_is_explicit_and_not_complete(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "results/metrics.json", b"\xff\xfe")
    scope = _scope(
        root,
        _inventory(("results/metrics.json", ChangeStatus.MODIFIED)),
        description="Accuracy improves by 10%; see results/metrics.json.",
    )
    assert not scope.complete
    assert any(
        issue.code == "PREFLIGHT_G2_CANDIDATE_UNREADABLE"
        for issue in scope.issues
    )


def test_source_bundle_adapter_uses_only_issued_and_changed_non_deleted_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "docs/study.md", "Accuracy improves by 10%.\n")
    _write(root, "src/model.py", "accuracy = 0.9\n")
    scope = _scope(
        root,
        _inventory(
            ("docs/study.md", ChangeStatus.MODIFIED),
            ("old/result.json", ChangeStatus.DELETED),
            ("src/model.py", ChangeStatus.MODIFIED),
        ),
    )
    bundle = scope_source_bundle(scope)
    assert bundle.sources == scope.sources
    assert bundle.repository_paths == scope.issued_paths
    assert bundle.changed_paths == scope.issued_changed_paths
    assert bundle.total_chars == sum(len(record.text) for record in scope.sources)
    assert all(record.kind is SourceKind.REPOSITORY_FILE for record in scope.sources)


def test_incomplete_legacy_inventory_cannot_claim_declared_complete_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    legacy = ChangeInventory(
        schema_version=1,
        requested_base_sha=BASE,
        comparison_base_sha=BASE,
        head_sha=HEAD,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.LEGACY_PAIRWISE,
        declared_entry_count=0,
        complete=False,
        entries=(),
    )
    scope = _scope(root, legacy, title="Accuracy improves by 10%")
    assert scope.mode == "legacy_pairwise_v1"
    assert not scope.complete
    assert "PREFLIGHT_G1_CHANGESET_INCOMPLETE" in {
        issue.code for issue in scope.issues
    }
