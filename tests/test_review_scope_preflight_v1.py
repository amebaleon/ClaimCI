"""Deterministic bounded material-scope regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

import claimci.review.preflight as preflight_module
from claimci.passive_files import PassiveFileError
from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    GateDisposition,
    MaterialClaimSeed,
    ReviewConfig,
    ReviewLimits,
    ReviewStatus,
    ScopeIssue,
    ScientificClaim,
    SourceKind,
)
from claimci.review.orchestrator import ReviewInputs
from claimci.review.preflight import (
    build_review_scope,
    preflight_review,
    scope_source_bundle,
)


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


def test_material_seed_truncation_is_deterministic_and_forces_partial_scope(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    material = (
        "Benchmark model accuracy improves by 5% with reproducible seed "
        "configuration; see docs/one.md."
    )
    inventory = _inventory(
        ("docs/one.md", ChangeStatus.MODIFIED),
        ("docs/two.md", ChangeStatus.MODIFIED),
    )
    scopes = []
    for root in roots:
        root.mkdir()
        _write(root, "docs/one.md", material)
        _write(root, "docs/two.md", material)
        scopes.append(
            _scope(
                root,
                inventory,
                title=material,
                description=material,
            )
        )

    scope = scopes[0]
    result = preflight_module._evaluate_material_gates(
        scope,
        ReviewConfig(enabled=True),
        requested_sha=BASE,
        comparison_sha=BASE,
        head_sha=HEAD,
    )
    truncation = tuple(
        issue
        for issue in scope.issues
        if issue.code == "PREFLIGHT_G2_MATERIAL_SEED_TRUNCATED"
    )

    assert len(scope.sources) == 4
    assert len(scope.seeds) == 16
    assert scope.seeds == scopes[1].seeds
    assert truncation == (
        ScopeIssue(
            code="PREFLIGHT_G2_MATERIAL_SEED_TRUNCATED",
            observed=20,
            limit=16,
        ),
    )
    assert truncation[0].as_dict() == {
        "code": "PREFLIGHT_G2_MATERIAL_SEED_TRUNCATED",
        "observed": 20,
        "limit": 16,
    }
    assert scope.complete is False
    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert result.gates[1].metrics["material_seed_count"] == 16
    assert result.gates[1].metrics["material_seed_omitted_count"] == 4
    assert result.ready_for_provider is True
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


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


def test_failed_captures_consume_the_fixed_read_budget_without_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "results/z_safe.json", "{\"accuracy\": 95}\n")
    inventory = _inventory(
        ("results/a_missing.json", ChangeStatus.MODIFIED),
        ("results/b_missing.json", ChangeStatus.MODIFIED),
        ("results/z_safe.json", ChangeStatus.MODIFIED),
    )
    opened: list[str] = []
    real_capture = preflight_module.capture_confined_regular_file

    def traced_capture(root_path: Path, relative: str, *, max_bytes: int):
        opened.append(relative)
        return real_capture(root_path, relative, max_bytes=max_bytes)

    monkeypatch.setattr(preflight_module, "capture_confined_regular_file", traced_capture)
    scope = _scope(
        root,
        inventory,
        title="Benchmark accuracy improves by 5%",
        limits=ReviewLimits(max_files=2),
    )
    assert opened == ["results/a_missing.json", "results/b_missing.json"]
    assert not scope.selected_paths
    assert any(
        issue.code == "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED"
        and issue.path == "results/z_safe.json"
        for issue in scope.issues
    )


def test_changed_document_seeds_rerank_the_remaining_fixed_budget(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(
        root,
        "docs/study.md",
        "Accuracy improves by 10%; use zz_target/result.json.\n",
    )
    _write(root, "aa/result.json", "{\"accuracy\": 90}\n")
    _write(root, "zz_target/result.json", "{\"accuracy\": 95}\n")
    scope = _scope(
        root,
        _inventory(
            ("aa/result.json", ChangeStatus.MODIFIED),
            ("docs/study.md", ChangeStatus.MODIFIED),
            ("zz_target/result.json", ChangeStatus.MODIFIED),
        ),
        limits=ReviewLimits(max_files=2),
    )
    assert scope.selected_paths == (
        "docs/study.md",
        "zz_target/result.json",
    )
    assert any(
        issue.code == "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED"
        and issue.path == "aa/result.json"
        for issue in scope.issues
    )


def test_cross_metadata_exact_result_survives_document_flood_at_file_cap(
    tmp_path: Path,
) -> None:
    """PR title seeds and body paths jointly route exact changed results."""

    document_paths = tuple(f"docs/note-{index:02d}.md" for index in range(24))
    result_path = "results/exact.json"
    entries = tuple(
        (path, ChangeStatus.MODIFIED) for path in (*document_paths, result_path)
    )
    inventory = _inventory(*entries)
    roots = (tmp_path / "first", tmp_path / "second")
    scopes = []
    for root, creation_order in zip(
        roots,
        (document_paths + (result_path,), tuple(reversed(document_paths + (result_path,)))),
        strict=True,
    ):
        root.mkdir()
        for path in creation_order:
            _write(
                root,
                path,
                '{"accuracy": 0.95}\n'
                if path == result_path
                else f"supporting note for {path}\n",
            )
        scopes.append(
            _scope(
                root,
                inventory,
                title="Benchmark accuracy improves by 5%.",
                description="See results/exact.json.",
                limits=ReviewLimits(max_files=24),
            )
        )

    gates = tuple(
        preflight_module._evaluate_material_gates(
            scope,
            ReviewConfig(enabled=True, limits=ReviewLimits(max_files=24)),
            requested_sha=BASE,
            comparison_sha=BASE,
            head_sha=HEAD,
        )
        for scope in scopes
    )

    assert scopes[0].selected_paths == scopes[1].selected_paths
    assert result_path in scopes[0].selected_paths
    assert len(scopes[0].selected_paths) == 24
    assert len(set(scopes[0].selected_paths) & set(document_paths)) == 23
    assert gates[0].gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert gates[1].gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert all(
        issue.code == "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED"
        for issue in scopes[0].issues
        if issue.path not in scopes[0].selected_paths
    )


def test_path_prefix_is_not_an_exact_inventory_path_mention(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "config/aaa.yaml", "seed: 1\n")
    _write(root, "config/train.yaml", "seed: 2\n")
    scope = _scope(
        root,
        _inventory(
            ("config/aaa.yaml", ChangeStatus.MODIFIED),
            ("config/train.yaml", ChangeStatus.MODIFIED),
        ),
        description="Reproduce with config/train.yaml.bak.",
        limits=ReviewLimits(max_files=1),
    )
    assert scope.selected_paths == ("config/aaa.yaml",)


def test_exact_inventory_path_mention_that_is_not_issued_is_out_of_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/safe.py", "accuracy = 95\n")
    _write(root, "notes/claimed.bin", "not review material\n")

    scope = _scope(
        root,
        _inventory(
            ("benchmarks/safe.py", ChangeStatus.MODIFIED),
            ("notes/claimed.bin", ChangeStatus.MODIFIED),
        ),
        description=(
            "Benchmark accuracy improves by 5%; the exact supporting path is "
            "notes/claimed.bin."
        ),
    )

    assert scope.issued_paths == ("benchmarks/safe.py",)
    assert not scope.complete
    assert ScopeIssue(
        code="PREFLIGHT_G2_OUT_OF_SCOPE_PATH", path="notes/claimed.bin"
    ) in scope.issues


@pytest.mark.parametrize(
    "unsafe_reference",
    [
        "/outside/results.json",
        "/results.json",
        "../../outside/results.json",
        "../results.json",
        r"C:\outside\results.json",
        r"C:\results.json",
        r"\\server\share\results.json",
        r"outside\results.json",
    ],
)
def test_absolute_traversal_and_backslash_path_references_are_never_admitted(
    tmp_path: Path,
    unsafe_reference: str,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "benchmarks/safe.py", "accuracy = 95\n")
    scope = _scope(
        root,
        _inventory(("benchmarks/safe.py", ChangeStatus.MODIFIED)),
        description=f"Accuracy improved by 10%; see {unsafe_reference}.",
    )
    assert scope.issued_paths == ("benchmarks/safe.py",)
    assert any(
        issue.code == "PREFLIGHT_G2_OUT_OF_SCOPE_PATH" and issue.path is None
        for issue in scope.issues
    )


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


def test_deleted_material_with_another_citable_route_forces_partial_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "tests/test_result_check.py", "def test_result():\n    pass\n")
    inventory = _inventory(
        ("results/deleted.json", ChangeStatus.DELETED),
        ("tests/test_result_check.py", ChangeStatus.MODIFIED),
    )

    scope = _scope(
        root,
        inventory,
        title="Reproducibility improves with deterministic tests",
    )
    result = preflight_module._evaluate_material_gates(
        scope,
        ReviewConfig(enabled=True),
        requested_sha=BASE,
        comparison_sha=BASE,
        head_sha=HEAD,
    )

    assert ScopeIssue(
        code="PREFLIGHT_G2_DELETED_MATERIAL_UNAVAILABLE",
        path="results/deleted.json",
    ) in scope.issues
    assert scope.complete is False
    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert result.gates[1].metrics["deleted_material_unavailable_count"] == 1
    assert result.ready_for_provider is True
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


def test_exactly_mentioned_deleted_material_remains_out_of_scope_and_unavailable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    scope = _scope(
        root,
        _inventory(("results/deleted.json", ChangeStatus.DELETED)),
        title="Benchmark improves according to results/deleted.json",
    )
    issue_codes = {issue.code for issue in scope.issues}
    gate2 = preflight_module._gate2(scope)

    assert issue_codes >= {
        "PREFLIGHT_G2_DELETED_MATERIAL_UNAVAILABLE",
        "PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH",
        "PREFLIGHT_G2_OUT_OF_SCOPE_PATH",
    }
    assert gate2.disposition is GateDisposition.FAIL
    assert gate2.metrics["deleted_material_unavailable_count"] == 1


def test_other_deletion_does_not_reduce_complete_material_scope(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    _write(root, "tests/test_result_check.py", "def test_result():\n    pass\n")
    scope = _scope(
        root,
        _inventory(
            ("assets/old.bin", ChangeStatus.DELETED),
            ("tests/test_result_check.py", ChangeStatus.MODIFIED),
        ),
        title="Reproducibility improves with deterministic tests",
    )

    assert "PREFLIGHT_G2_DELETED_MATERIAL_UNAVAILABLE" not in {
        issue.code for issue in scope.issues
    }
    assert scope.complete is True


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


@pytest.mark.parametrize("deleted_path", ["results/deleted.json", "src/deleted.py"])
def test_legacy_inventory_surfaces_base_only_material_deletion(
    tmp_path: Path,
    deleted_path: str,
) -> None:
    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _write(base, deleted_path, "old material\n")

    inventory, issues = preflight_module._legacy_inventory(
        ReviewInputs(repository_root=head, base_root=base)
    )

    assert issues == ()
    assert inventory is not None
    assert inventory.entries == (ChangeEntry(deleted_path, ChangeStatus.DELETED),)


def test_legacy_case_only_rename_is_a_typed_gate1_duplicate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.review.orchestrator as orchestrator

    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _write(head, "README.md", "new\n")
    _write(base, "readme.md", "old\n")

    monkeypatch.setattr(
        orchestrator,
        "OpenAIReviewerProvider",
        lambda **_kwargs: pytest.fail("provider factory must remain untouched"),
    )
    review = orchestrator.run_review(
        ReviewInputs(repository_root=head, base_root=base),
        ReviewConfig(enabled=True),
    )
    result = review.preflight

    assert result is not None
    assert result.gates[0].disposition is GateDisposition.FAIL
    assert {reason.code for reason in result.gates[0].reasons} == {
        "PREFLIGHT_G1_CHANGE_INVENTORY_DUPLICATE_PATH"
    }
    assert result.gates[1].disposition is GateDisposition.NOT_EVALUATED
    assert result.gates[2].disposition is GateDisposition.NOT_EVALUATED
    assert result.ready_for_provider is False
    assert result.scope is None
    assert review.status is ReviewStatus.UNAVAILABLE
    assert review.provider_calls == ()


def test_legacy_invalid_generated_path_is_a_typed_gate1_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    monkeypatch.setattr(
        preflight_module,
        "_legacy_regular_paths",
        lambda *_args, **_kwargs: (("../result.json",), ()),
    )

    result = preflight_review(
        ReviewInputs(repository_root=root),
        ReviewConfig(enabled=True),
    )

    assert result.gates[0].disposition is GateDisposition.FAIL
    assert {reason.code for reason in result.gates[0].reasons} == {
        "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH"
    }
    assert result.gates[1].disposition is GateDisposition.NOT_EVALUATED
    assert result.gates[2].disposition is GateDisposition.NOT_EVALUATED
    assert result.ready_for_provider is False
    assert result.scope is None


def test_legacy_deleted_material_with_modified_route_is_partial(tmp_path: Path) -> None:
    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _write(base, "results/deleted.json", '{"accuracy": 0.8}\n')
    _write(base, "tests/test_result_check.py", "old = 1\n")
    _write(head, "tests/test_result_check.py", "new_result = 2\n")

    result = preflight_review(
        ReviewInputs(
            repository_root=head,
            base_root=base,
            pr_title="Reproducibility improves with deterministic tests",
        ),
        ReviewConfig(enabled=True),
    )

    assert result.scope is not None
    assert ChangeEntry(
        "results/deleted.json", ChangeStatus.DELETED
    ) in result.scope.inventory.entries
    assert ScopeIssue(
        code="PREFLIGHT_G2_DELETED_MATERIAL_UNAVAILABLE",
        path="results/deleted.json",
    ) in result.scope.issues
    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert result.ready_for_provider is True
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


@pytest.mark.parametrize(
    ("constant", "reason"),
    [
        ("MAX_REPOSITORY_PATHS", "PREFLIGHT_G1_REPOSITORY_PATH_LIMIT"),
        ("MAX_REPOSITORY_ENTRIES", "PREFLIGHT_G1_REPOSITORY_ENTRY_LIMIT"),
    ],
)
def test_legacy_head_and_base_share_one_traversal_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    reason: str,
) -> None:
    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _write(head, "live.py", "value = 1\n")
    _write(base, "deleted.json", "{}\n")
    monkeypatch.setattr(preflight_module, constant, 1)

    inventory, issues = preflight_module._legacy_inventory(
        ReviewInputs(repository_root=head, base_root=base)
    )

    assert inventory is None
    assert reason in {issue.code for issue in issues}


def test_legacy_identical_paths_do_not_spuriously_exhaust_union_path_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _write(head, "results/same.json", "{}\n")
    _write(base, "results/same.json", "{}\n")
    monkeypatch.setattr(preflight_module, "MAX_REPOSITORY_PATHS", 1)

    inventory, issues = preflight_module._legacy_inventory(
        ReviewInputs(repository_root=head, base_root=base)
    )

    assert issues == ()
    assert inventory is not None
    assert inventory.entries == ()


def test_legacy_base_only_symlink_is_never_reported_as_deleted_material(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    base = tmp_path / "base"
    outside = tmp_path / "outside.json"
    head.mkdir()
    base.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    link = base / "results" / "linked.json"
    link.parent.mkdir()
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    inventory, issues = preflight_module._legacy_inventory(
        ReviewInputs(repository_root=head, base_root=base)
    )

    assert issues == ()
    assert inventory is not None
    assert inventory.entries == ()


def test_legacy_rename_like_delete_add_is_deterministic_partial(
    tmp_path: Path,
) -> None:
    pairs = (
        (tmp_path / "head-one", tmp_path / "base-one"),
        (tmp_path / "head-two", tmp_path / "base-two"),
    )
    inventories = []
    for index, (head, base) in enumerate(pairs):
        head.mkdir()
        base.mkdir()
        head_paths = ("results/new.json", "tests/same.py")
        base_paths = ("results/old.json", "tests/same.py")
        if index:
            head_paths = tuple(reversed(head_paths))
            base_paths = tuple(reversed(base_paths))
        for path in head_paths:
            _write(head, path, "same\n" if path == "tests/same.py" else "new\n")
        for path in base_paths:
            _write(base, path, "same\n" if path == "tests/same.py" else "old\n")
        inventory, issues = preflight_module._legacy_inventory(
            ReviewInputs(repository_root=head, base_root=base)
        )
        assert issues == ()
        assert inventory is not None
        inventories.append(inventory)

    expected = (
        ChangeEntry("results/new.json", ChangeStatus.ADDED),
        ChangeEntry("results/old.json", ChangeStatus.DELETED),
    )
    assert inventories[0].entries == expected
    assert inventories[1].entries == expected

    result = preflight_review(
        ReviewInputs(
            repository_root=pairs[0][0],
            base_root=pairs[0][1],
            pr_title="Benchmark improves in results/new.json",
        ),
        ReviewConfig(enabled=True),
    )
    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


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


def test_legacy_gate1_failure_filters_prior_gate2_inventory_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later comparison cap cannot misfile an earlier status error in Gate 1."""

    head = (tmp_path / "head").resolve()
    base = (tmp_path / "base").resolve()
    head.mkdir()
    base.mkdir()
    for name in ("a-results.json", "b-results.json"):
        _write(head, name, "same\n")
        _write(base, name, "same\n")
    real_capture = preflight_module.capture_confined_regular_file

    def first_status_unknown(root: Path, path: str, *, max_bytes: int):
        if path == "a-results.json":
            raise PassiveFileError("identity changed", code="changed")
        return real_capture(root, path, max_bytes=max_bytes)

    monkeypatch.setattr(preflight_module, "MAX_CHANGE_COMPARISON_FILES", 1)
    monkeypatch.setattr(
        preflight_module, "capture_confined_regular_file", first_status_unknown
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=head,
            base_root=base,
            pr_title="Benchmark accuracy improves in a-results.json",
        ),
        ReviewConfig(enabled=True),
    )

    assert result.gates[0].disposition is GateDisposition.FAIL
    assert {reason.code for reason in result.gates[0].reasons} == {
        "PREFLIGHT_G1_COMPARISON_FILE_LIMIT"
    }
    assert result.gates[1].disposition is GateDisposition.NOT_EVALUATED
    assert result.gates[2].disposition is GateDisposition.NOT_EVALUATED
