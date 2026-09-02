"""Provider-free gate and exact request-accounting regressions."""

from __future__ import annotations

import builtins
import json
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claimci.passive_files import PassiveFileError

from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    GateDisposition,
    PreflightGateResult,
    ReviewConfig,
    ReviewLimits,
    ReviewPreflight,
    ReviewStatus,
    ScopeIssue,
    SnapshotIdentity,
    SnapshotRole,
)
from claimci.review.inventory import InventoryVerificationError
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.preflight import build_review_scope, preflight_review, scope_source_bundle
from claimci.review.provider import ProviderResponse, StructuredRequest
from claimci.review.request_budget import (
    RequestParts,
    build_extraction_request_parts,
    build_synthesis_request_parts,
    logical_request_chars,
    worst_valid_synthesis_request_chars,
)


SHA = "1" * 40
OTHER_SHA = "2" * 40


def _inventory(
    entries: tuple[ChangeEntry, ...],
    *,
    requested_sha: str = SHA,
    comparison_sha: str = SHA,
    head_sha: str = SHA,
    basis: ComparisonBasis = ComparisonBasis.DIRECT_BASE,
) -> ChangeInventory:
    return ChangeInventory(
        schema_version=1,
        requested_base_sha=requested_sha,
        comparison_base_sha=comparison_sha,
        head_sha=head_sha,
        comparison_basis=basis,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=len(entries),
        complete=True,
        entries=entries,
    )


def _declared_inputs(
    root: Path,
    inventory: object | None,
    *,
    title: str = "Benchmark accuracy improves by 5% in results.json",
    description: str = "The evaluation is reproducible.",
    requested_sha: str = SHA,
    comparison_sha: str = SHA,
    head_sha: str = SHA,
) -> ReviewInputs:
    root = root.resolve()
    return ReviewInputs(
        repository_root=root,
        pr_title=title,
        pr_description=description,
        requested_base=SnapshotIdentity(
            SnapshotRole.REQUESTED_BASE, root, requested_sha
        ),
        comparison_base=SnapshotIdentity(
            SnapshotRole.COMPARISON_BASE, root, comparison_sha
        ),
        head=SnapshotIdentity(SnapshotRole.HEAD, root, head_sha),
        inventory=inventory,
    )


def _ready_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str = '{"accuracy": 0.95}',
    config: ReviewConfig | None = None,
) -> tuple[Any, ReviewInputs, ReviewConfig]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "results.json").write_text(text, encoding="utf-8")
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    inputs = _declared_inputs(root, inventory)
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )
    selected_config = config or ReviewConfig(enabled=True)
    return preflight_review(inputs, selected_config), inputs, selected_config


def _issue_codes(preflight: Any, gate: int) -> set[str]:
    return {
        issue.code
        for result in preflight.gates
        if result.gate == gate
        for issue in result.reasons
    }


def test_gate_contracts_are_frozen_and_metrics_and_issues_serialize_deterministically() -> None:
    issue = ScopeIssue(code="PREFLIGHT_G2_CANDIDATE_TOO_LARGE", limit=16)
    gate = PreflightGateResult(
        gate=2,
        disposition=GateDisposition.PASS_PARTIAL,
        reasons=(issue,),
        metrics={"z": 2, "a": 1},
    )

    assert issue.as_dict() == {
        "code": "PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
        "limit": 16,
    }
    assert list(gate.metrics) == ["a", "z"]
    with pytest.raises(TypeError):
        gate.metrics["new"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        gate.gate = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("dispositions", "ready", "ceiling"),
    [
        (
            (
                GateDisposition.PASS_COMPLETE,
                GateDisposition.PASS_COMPLETE,
                GateDisposition.PASS_COMPLETE,
            ),
            True,
            ReviewStatus.COMPLETE,
        ),
        (
            (
                GateDisposition.PASS_COMPLETE,
                GateDisposition.PASS_PARTIAL,
                GateDisposition.PASS_COMPLETE,
            ),
            True,
            ReviewStatus.PARTIAL,
        ),
        (
            (
                GateDisposition.FAIL,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
            ),
            False,
            ReviewStatus.UNAVAILABLE,
        ),
    ],
)
def test_review_preflight_enforces_the_exact_status_ceiling(
    dispositions: tuple[GateDisposition, ...],
    ready: bool,
    ceiling: ReviewStatus,
) -> None:
    gates = tuple(
        PreflightGateResult(
            gate=index,
            disposition=disposition,
            reasons=(
                ScopeIssue(code="PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"),
            )
            if disposition is GateDisposition.NOT_EVALUATED
            else (
                (ScopeIssue(code="PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED"),)
                if disposition is GateDisposition.PASS_PARTIAL
                else (
                    (ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_UNAVAILABLE"),)
                    if disposition is GateDisposition.FAIL
                    else ()
                )
            ),
            metrics={},
        )
        for index, disposition in enumerate(dispositions, 1)
    )
    preflight = ReviewPreflight(
        schema_version=1,
        requested_base_sha=SHA,
        comparison_base_sha=SHA,
        head_sha=SHA,
        gates=gates,
        ready_for_provider=ready,
        review_status_ceiling=ceiling,
        scope=None,
    )

    assert preflight.ready_for_provider is ready
    assert preflight.review_status_ceiling is ceiling


@pytest.mark.parametrize(
    "dispositions",
    [
        (
            GateDisposition.FAIL,
            GateDisposition.PASS_COMPLETE,
            GateDisposition.NOT_EVALUATED,
        ),
        (
            GateDisposition.PASS_COMPLETE,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
        ),
        (
            GateDisposition.PASS_COMPLETE,
            GateDisposition.FAIL,
            GateDisposition.PASS_PARTIAL,
        ),
    ],
)
def test_review_preflight_rejects_impossible_gate_sequences(
    dispositions: tuple[GateDisposition, ...],
) -> None:
    gates = tuple(
        PreflightGateResult(
            gate=index,
            disposition=disposition,
            reasons=(
                ScopeIssue(code="PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"),
            )
            if disposition is GateDisposition.NOT_EVALUATED
            else (
                (ScopeIssue(code=f"PREFLIGHT_G{index}_TEST_FAILURE"),)
                if disposition in {GateDisposition.FAIL, GateDisposition.PASS_PARTIAL}
                else ()
            ),
            metrics={},
        )
        for index, disposition in enumerate(dispositions, 1)
    )

    with pytest.raises(ValueError, match="gate sequence"):
        ReviewPreflight(
            schema_version=1,
            requested_base_sha=SHA,
            comparison_base_sha=SHA,
            head_sha=SHA,
            gates=gates,
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=None,
        )


def test_gate_1_absent_inventory_fails_and_downstream_gates_are_not_evaluated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    result = preflight_review(
        _declared_inputs(root, None), ReviewConfig(enabled=True)
    )

    assert [gate.disposition for gate in result.gates] == [
        GateDisposition.FAIL,
        GateDisposition.NOT_EVALUATED,
        GateDisposition.NOT_EVALUATED,
    ]
    assert _issue_codes(result, 1) == {"PREFLIGHT_G1_CHANGE_INVENTORY_UNAVAILABLE"}
    assert _issue_codes(result, 2) == {
        "PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"
    }
    assert result.ready_for_provider is False
    assert result.review_status_ceiling is ReviewStatus.UNAVAILABLE


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("head", "PREFLIGHT_G1_HEAD_ROOT_INVALID"),
        ("requested_base", "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"),
        ("comparison_base", "PREFLIGHT_G1_COMPARISON_BASE_ROOT_INVALID"),
    ],
)
def test_gate_1_reports_each_invalid_coordinate_root(
    tmp_path: Path, field: str, expected: str
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    inventory = _inventory(())
    inputs = _declared_inputs(root, inventory)
    object.__setattr__(inputs, field, SimpleNamespace(root=tmp_path / "missing", sha=SHA))

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert expected in _issue_codes(result, 1)


def test_gate_1_reports_undeclared_comparison_base(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    inventory = _inventory(())
    inputs = _declared_inputs(root, inventory)
    object.__setattr__(inputs, "comparison_base", None)

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert "PREFLIGHT_G1_COMPARISON_BASE_UNDECLARED" in _issue_codes(result, 1)


def test_gate_1_rejects_snapshot_sha_mismatch_before_adapter_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inventory = _inventory(
        (ChangeEntry("results.json", ChangeStatus.ADDED),), head_sha=OTHER_SHA
    )
    inputs = _declared_inputs(root, inventory)
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: pytest.fail("adapter must not run after identity mismatch"),
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert "PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH" in _issue_codes(result, 1)
    assert result.ready_for_provider is False


@pytest.mark.parametrize(
    "code",
    [
        "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
        "PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH",
        "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
        "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
    ],
)
def test_gate_1_maps_typed_inventory_verification_failures_without_message_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    inputs = _declared_inputs(root, inventory)
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InventoryVerificationError(code, "deliberately unrelated text")
        ),
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert _issue_codes(result, 1) == {code}


def test_gate_1_classifies_verified_inventory_content_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    declared = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    verified = _inventory((ChangeEntry("other.json", ChangeStatus.ADDED),))
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: verified,
    )

    result = preflight_review(
        _declared_inputs(root, declared), ReviewConfig(enabled=True)
    )

    assert _issue_codes(result, 1) == {
        "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"
    }


@pytest.mark.parametrize(
    ("inventory", "expected"),
    [
        (
            SimpleNamespace(
                schema_version=2,
                requested_base_sha=SHA,
                comparison_base_sha=SHA,
                head_sha=SHA,
                comparison_basis=ComparisonBasis.DIRECT_BASE,
                source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
                declared_entry_count=0,
                complete=True,
                entries=(),
            ),
            "PREFLIGHT_G1_CHANGE_INVENTORY_SCHEMA_UNSUPPORTED",
        ),
        (
            SimpleNamespace(
                schema_version=1,
                requested_base_sha=SHA,
                comparison_base_sha=SHA,
                head_sha=SHA,
                comparison_basis=ComparisonBasis.DIRECT_BASE,
                source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
                declared_entry_count=1,
                complete=True,
                entries=(SimpleNamespace(path="../escape.json", status=ChangeStatus.ADDED),),
            ),
            "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
        ),
        (
            SimpleNamespace(
                schema_version=1,
                requested_base_sha=SHA,
                comparison_base_sha=SHA,
                head_sha=SHA,
                comparison_basis=ComparisonBasis.DIRECT_BASE,
                source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
                declared_entry_count=1,
                complete=True,
                entries=(SimpleNamespace(path="results.json", status="unknown"),),
            ),
            "PREFLIGHT_G1_CHANGE_INVENTORY_STATUS_INVALID",
        ),
    ],
)
def test_gate_1_reports_granular_inventory_failures(
    tmp_path: Path, inventory: object, expected: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    result = preflight_review(
        _declared_inputs(root, inventory), ReviewConfig(enabled=True)
    )

    assert expected in _issue_codes(result, 1)
    assert "PREFLIGHT_FAILED" not in _issue_codes(result, 1)


def test_gate_1_reports_every_inventory_shape_bound() -> None:
    import claimci.review.preflight as module

    common = {
        "schema_version": 1,
        "requested_base_sha": SHA,
        "comparison_base_sha": SHA,
        "head_sha": SHA,
        "comparison_basis": ComparisonBasis.DIRECT_BASE,
        "source": ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        "declared_entry_count": 2,
        "complete": False,
        "entries": (
            SimpleNamespace(path="same.json", status=ChangeStatus.ADDED),
            SimpleNamespace(path="SAME.JSON", status=ChangeStatus.MODIFIED),
        ),
    }
    codes = {issue.code for issue in module._inventory_shape_issues(SimpleNamespace(**common))}

    assert {
        "PREFLIGHT_G1_CHANGE_INVENTORY_DUPLICATE_PATH",
        "PREFLIGHT_G1_CHANGESET_INCOMPLETE",
    }.issubset(codes)

    mismatched = SimpleNamespace(**{**common, "complete": True, "declared_entry_count": 1})
    assert "PREFLIGHT_G1_CHANGED_FILE_COUNT_MISMATCH" in {
        issue.code for issue in module._inventory_shape_issues(mismatched)
    }

    large_entries = tuple(
        ChangeEntry(path=f"{index:04}/" + "x" * 4091, status=ChangeStatus.ADDED)
        for index in range(260)
    )

    @dataclass(frozen=True)
    class RawInventory:
        schema_version: int = 1
        requested_base_sha: str = SHA
        comparison_base_sha: str = SHA
        head_sha: str = SHA
        comparison_basis: ComparisonBasis = ComparisonBasis.DIRECT_BASE
        source: ChangeInventorySource = ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH
        declared_entry_count: int = len(large_entries)
        complete: bool = True
        entries: tuple[ChangeEntry, ...] = large_entries

    oversized = RawInventory()
    assert "PREFLIGHT_G1_CHANGESET_METADATA_BYTES_LIMIT" in {
        issue.code for issue in module._inventory_shape_issues(oversized)
    }

    too_many = SimpleNamespace(
        **{
            **common,
            "complete": True,
            "declared_entry_count": 8_193,
            "entries": tuple(
                SimpleNamespace(path=f"p/{index:05}.json", status=ChangeStatus.ADDED)
                for index in range(8_193)
            ),
        }
    )
    assert "PREFLIGHT_G1_CHANGESET_PATH_LIMIT" in {
        issue.code for issue in module._inventory_shape_issues(too_many)
    }


@pytest.mark.parametrize(
    ("passive_code", "expected"),
    [
        ("outside", "PREFLIGHT_G1_MATERIAL_PATH_OUTSIDE_ROOT"),
        ("symlink", "PREFLIGHT_G1_MATERIAL_PATH_SYMLINK"),
        ("not_regular", "PREFLIGHT_G1_MATERIAL_PATH_NOT_REGULAR"),
        ("changed", "PREFLIGHT_G1_MATERIAL_PATH_IDENTITY_CHANGED"),
        ("too_large", "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"),
    ],
)
def test_material_capture_failures_have_frozen_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passive_code: str,
    expected: str,
) -> None:
    import claimci.review.preflight as module

    root = (tmp_path / "repo").resolve()
    root.mkdir()
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    monkeypatch.setattr(
        module,
        "capture_confined_regular_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PassiveFileError("sentinel", code=passive_code)
        ),
    )

    result = module.build_review_scope(
        _declared_inputs(root, inventory), ReviewConfig(enabled=True), inventory
    )

    assert expected in {issue.code for issue in result.issues}


@pytest.mark.parametrize(
    ("title", "path", "status", "expected"),
    [
        ("Documentation cleanup", "results.json", ChangeStatus.ADDED, "PREFLIGHT_G2_NO_MATERIAL_CLAIM_SEED"),
        ("Accuracy improves by 5%", "scripts/helper.sh", ChangeStatus.ADDED, "PREFLIGHT_G2_UNSUPPORTED_MATERIAL_TYPE"),
        ("Accuracy improves by 5% in results.json", "results.json", ChangeStatus.DELETED, "PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH"),
    ],
)
def test_gate_2_rejects_each_no_route_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str,
    path: str,
    status: ChangeStatus,
    expected: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    if status is not ChangeStatus.DELETED:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("benchmark accuracy 5%", encoding="utf-8")
    inventory = _inventory((ChangeEntry(path, status),))
    inputs = _declared_inputs(root, inventory, title=title, description="")
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.gates[1].disposition is GateDisposition.FAIL
    assert expected in _issue_codes(result, 2)
    assert result.gates[2].disposition is GateDisposition.NOT_EVALUATED


def test_gate_2_fails_when_every_material_seed_is_unrouted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs.md").write_text("release notes", encoding="utf-8")
    inventory = _inventory((ChangeEntry("docs.md", ChangeStatus.ADDED),))
    inputs = _declared_inputs(
        root,
        inventory,
        title="The model accuracy improves by 5%",
        description="",
    )
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.gates[1].disposition is GateDisposition.FAIL
    assert "PREFLIGHT_G2_MATERIAL_SEED_UNROUTED" in _issue_codes(result, 2)
    assert result.gates[1].metrics["routed_seed_count"] == 0
    assert result.ready_for_provider is False


@pytest.mark.parametrize(
    ("constant", "value", "expected"),
    [
        ("MAX_REPOSITORY_ENTRIES", 1, "PREFLIGHT_G1_REPOSITORY_ENTRY_LIMIT"),
        ("MAX_REPOSITORY_PATHS", 1, "PREFLIGHT_G1_REPOSITORY_PATH_LIMIT"),
    ],
)
def test_legacy_traversal_limits_fail_with_exact_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    value: int,
    expected: str,
) -> None:
    import claimci.review.preflight as module

    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "a.json").write_text("{}", encoding="utf-8")
    (root / "b.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, constant, value)

    result = module.legacy_preflight_failure(
        ReviewInputs(repository_root=root), ReviewConfig(enabled=True)
    )

    assert result is not None
    assert expected in _issue_codes(result, 1)


def test_legacy_depth_limit_fails_with_exact_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.preflight as module

    root = (tmp_path / "repo").resolve()
    nested = root / "one"
    nested.mkdir(parents=True)
    (nested / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "MAX_REPOSITORY_DEPTH", 0)

    result = module.legacy_preflight_failure(
        ReviewInputs(repository_root=root), ReviewConfig(enabled=True)
    )

    assert result is not None
    assert "PREFLIGHT_G1_REPOSITORY_DEPTH_LIMIT" in _issue_codes(result, 1)


@pytest.mark.parametrize(
    ("constant", "value", "expected"),
    [
        ("MAX_CHANGE_COMPARISON_FILES", 0, "PREFLIGHT_G1_COMPARISON_FILE_LIMIT"),
        ("MAX_CHANGE_COMPARISON_BYTES", 1, "PREFLIGHT_G1_COMPARISON_BYTES_LIMIT"),
        (
            "MAX_CHANGE_COMPARISON_FILE_BYTES",
            0,
            "PREFLIGHT_G1_COMPARISON_PER_FILE_BYTES_LIMIT",
        ),
    ],
)
def test_legacy_comparison_limits_fail_with_exact_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    value: int,
    expected: str,
) -> None:
    import claimci.review.preflight as module

    head = (tmp_path / "head").resolve()
    base = (tmp_path / "base").resolve()
    head.mkdir()
    base.mkdir()
    (head / "result.json").write_text("x", encoding="utf-8")
    (base / "result.json").write_text("x", encoding="utf-8")
    monkeypatch.setattr(module, constant, value)

    result = module.legacy_preflight_failure(
        ReviewInputs(repository_root=head, base_root=base),
        ReviewConfig(enabled=True),
    )

    assert result is not None
    assert expected in _issue_codes(result, 1)


def test_legacy_unknown_change_status_fails_gate_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.preflight as module

    head = (tmp_path / "head").resolve()
    base = (tmp_path / "base").resolve()
    head.mkdir()
    base.mkdir()
    (head / "result.json").write_text("x", encoding="utf-8")
    (base / "result.json").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "capture_confined_regular_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PassiveFileError("sentinel", code="changed")
        ),
    )

    result = module.legacy_preflight_failure(
        ReviewInputs(repository_root=head, base_root=base),
        ReviewConfig(enabled=True),
    )

    assert result is not None
    assert result.gates[1].disposition is GateDisposition.FAIL
    assert "PREFLIGHT_G2_CHANGE_STATUS_UNKNOWN" in _issue_codes(result, 2)


def test_gate_2_unreadable_candidate_fails_when_no_other_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "results.json").write_bytes(b"\xff")
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(
        _declared_inputs(root, inventory), ReviewConfig(enabled=True)
    )

    assert result.gates[1].disposition is GateDisposition.FAIL
    assert "PREFLIGHT_G2_CANDIDATE_UNREADABLE" in _issue_codes(result, 2)
    assert "PREFLIGHT_G2_NO_ROUTABLE_CHANGED_PATH" in _issue_codes(result, 2)


def test_gate_2_locality_external_and_out_of_scope_omissions_are_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "results.json").write_text("x" * 101, encoding="utf-8")
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )
    inputs = _declared_inputs(
        root,
        inventory,
        title=(
            "Benchmark accuracy improves by 5% in results.json; see docs/missing.json "
            "and https://example.invalid/report"
        ),
        description="",
    )
    limits = ReviewLimits(max_file_chars=100)

    result = preflight_review(
        inputs, ReviewConfig(enabled=True, limits=limits)
    )

    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert {
        "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE",
        "PREFLIGHT_G2_EXTERNAL_EVIDENCE_ONLY",
        "PREFLIGHT_G2_OUT_OF_SCOPE_PATH",
    }.issubset(_issue_codes(result, 2))
    assert "PREFLIGHT_G3_SELECTED_SOURCE_CHAR_LIMIT" in _issue_codes(result, 3)


def test_selection_truncation_is_partial_and_provider_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    entries = []
    for index in range(25):
        path = f"benchmarks/result-{index:02}.json"
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"accuracy": 0.95}', encoding="utf-8")
        entries.append(ChangeEntry(path, ChangeStatus.ADDED))
    inventory = _inventory(tuple(entries))
    inputs = _declared_inputs(root, inventory)
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.gates[1].disposition is GateDisposition.PASS_PARTIAL
    assert "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED" in _issue_codes(result, 2)
    assert result.gates[2].disposition is GateDisposition.PASS_PARTIAL
    assert "PREFLIGHT_G3_SELECTED_FILE_LIMIT" in _issue_codes(result, 3)
    assert "PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT" in _issue_codes(result, 3)
    assert result.scope is not None
    assert len(result.scope.selected_paths) < 24
    assert result.gates[2].metrics["synthesis_reserved_context_chars"] <= 60_000
    assert result.ready_for_provider is True
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


@pytest.mark.parametrize("file_count", [23, 24, 25])
def test_selected_file_limit_boundary_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_count: int,
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    entries = []
    for index in range(file_count):
        path = f"benchmarks/result-{index:02}.json"
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        entries.append(ChangeEntry(path, ChangeStatus.ADDED))
    inventory = _inventory(tuple(entries))
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(
        _declared_inputs(root, inventory), ReviewConfig(enabled=True)
    )

    assert (
        "PREFLIGHT_G3_SELECTED_FILE_LIMIT" in _issue_codes(result, 3)
    ) is (file_count == 25)
    assert (
        "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED" in _issue_codes(result, 2)
    ) is (file_count == 25)
    if file_count == 25:
        issue = next(
            issue
            for issue in result.gates[2].reasons
            if issue.code == "PREFLIGHT_G3_SELECTED_FILE_LIMIT"
        )
        assert (issue.observed, issue.limit) == (25, 24)


@pytest.mark.parametrize("source_chars", [15_999, 16_000, 16_001])
def test_selected_source_character_limit_boundary_is_exact(
    tmp_path: Path,
    source_chars: int,
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "results.json").write_text("x" * source_chars, encoding="utf-8")
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))

    scope = build_review_scope(
        _declared_inputs(root, inventory), ReviewConfig(enabled=True), inventory
    )

    assert (
        "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE"
        in {issue.code for issue in scope.issues}
    ) is (source_chars == 16_001)
    assert dict(scope.materialized_path_chars)["results.json"] == min(
        source_chars, 16_000
    )


def test_gate_3_exact_extraction_projection_matches_runtime_request_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _inputs, config = _ready_preflight(tmp_path, monkeypatch)
    assert result.scope is not None
    bundle = scope_source_bundle(result.scope)
    parts = build_extraction_request_parts(bundle, config.limits.max_claims)
    request = StructuredRequest(
        task=parts.task,
        payload=parts.payload,
        schema=parts.schema,
        max_output_tokens=config.limits.extraction_max_output_tokens,
    )

    assert result.gates[2].metrics["extraction_context_chars"] == logical_request_chars(
        request.task, request.payload, request.schema
    )
    assert result.ready_for_provider is True


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"max_calls": 1}, "PREFLIGHT_G3_CALL_LIMIT_LT_TWO"),
        ({"max_output_chars": 23_999}, "PREFLIGHT_G3_OUTPUT_RESERVE_LIMIT"),
    ],
)
def test_gate_3_rejects_fixed_state_machine_budget_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, int],
    expected: str,
) -> None:
    defaults = dict(vars(ReviewLimits()))
    defaults.update(updates)
    result, _inputs, _config = _ready_preflight(
        tmp_path,
        monkeypatch,
        config=ReviewConfig(enabled=True, limits=ReviewLimits(**defaults)),
    )

    assert result.gates[2].disposition is GateDisposition.FAIL
    assert expected in _issue_codes(result, 3)
    assert result.ready_for_provider is False


def test_gate_3_reports_both_fixed_overhead_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.preflight as module

    result, _inputs, _config = _ready_preflight(tmp_path, monkeypatch)
    assert result.scope is not None

    gate = module._gate3(
        result.scope,
        ReviewConfig(enabled=True, limits=ReviewLimits(max_context_chars=1)),
    )

    assert {
        "PREFLIGHT_G3_EXTRACTION_FIXED_OVERHEAD_LIMIT",
        "PREFLIGHT_G3_SYNTHESIS_FIXED_OVERHEAD_LIMIT",
    }.issubset({issue.code for issue in gate.reasons})


def test_gate_3_reports_exact_extraction_context_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.preflight as module

    result, _inputs, config = _ready_preflight(tmp_path, monkeypatch)
    assert result.scope is not None
    projected = int(result.gates[2].metrics["extraction_context_chars"])
    fixed = int(result.gates[2].metrics["extraction_fixed_overhead_chars"])
    assert projected > fixed
    monkeypatch.setattr(module, "worst_valid_synthesis_request_chars", lambda *_args: 1)

    gate = module._gate3(
        result.scope,
        ReviewConfig(
            enabled=True,
            limits=ReviewLimits(max_context_chars=projected - 1),
        ),
    )

    assert "PREFLIGHT_G3_EXTRACTION_CONTEXT_LIMIT" in {
        issue.code for issue in gate.reasons
    }


def test_gate_3_reports_synthesis_fixed_overhead_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.preflight as module

    result, _inputs, config = _ready_preflight(tmp_path, monkeypatch)
    assert result.scope is not None
    original = module.build_synthesis_request_parts

    def oversized(*args: Any, **kwargs: Any) -> RequestParts:
        parts = original(*args, **kwargs)
        return RequestParts(task="x" * config.limits.max_context_chars, payload=parts.payload, schema=parts.schema)

    monkeypatch.setattr(module, "build_synthesis_request_parts", oversized)
    monkeypatch.setattr(module, "worst_valid_synthesis_request_chars", lambda *_args: 1)

    gate = module._gate3(result.scope, config)

    assert "PREFLIGHT_G3_SYNTHESIS_FIXED_OVERHEAD_LIMIT" in {
        issue.code for issue in gate.reasons
    }


def test_gate_3_blocks_call_one_when_extraction_fits_but_synthesis_reserve_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "results.json").write_text("r" * 16_000, encoding="utf-8")
    inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
    inputs = _declared_inputs(
        root,
        inventory,
        description="The reproducible benchmark reports " + ("x" * 15_000),
    )
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.gates[2].metrics["extraction_context_chars"] <= 60_000
    assert result.gates[2].metrics["synthesis_reserved_context_chars"] > 60_000
    assert "PREFLIGHT_G3_SYNTHESIS_RESERVED_CONTEXT_LIMIT" in _issue_codes(result, 3)
    assert result.ready_for_provider is False


def test_synthesis_reserve_covers_all_owner_references_missing_rows_and_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = tuple(f"benchmarks/result-{index:02}.json" for index in range(24))
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x" * 100, encoding="utf-8")
    inventory = _inventory(tuple(ChangeEntry(path, ChangeStatus.ADDED) for path in paths))
    config = ReviewConfig(enabled=True)
    scope = build_review_scope(_declared_inputs(root, inventory), config, inventory)
    claim_ids = tuple(f"claim-{index:016x}" for index in range(16))
    claims = [
        {
            "claim_id": claim_id,
            "source_text": "x" * 400,
            "claim_type": "other_scientific",
            "subject": "x" * 100,
            "source": {
                "source_id": "source-0000000000000000",
                "kind": "pull_request_title",
                "path": None,
                "start_line": 1,
                "end_line": 1,
            },
            "metric": None,
            "direction": "not_applicable",
            "claimed_magnitude": None,
            "qualifiers": [],
            "confidence": 1.0,
            "evidence_hints": list(paths),
        }
        for claim_id in claim_ids
    ]
    owners = {claim_id: [f"evidence-{index:016x}" for index in range(24)] for claim_id in claim_ids}
    evidence = [
        {
            "evidence_id": f"evidence-{index:016x}",
            "claim_ids": list(claim_ids),
            "kind": "document",
            "path": path,
            "start_line": 1,
            "end_line": 1,
            "sha256": "0" * 64,
            "size": 100,
            "excerpt": "x" * 100,
            "provenance": "supporting_artifact",
            "excerpt_complete": True,
            "excerpt_locality": "complete_file",
        }
        for index, path in enumerate(paths)
    ]
    missing = [
        {
            "claim_id": claim_id,
            "reason": "unresolved_provider_hint",
            "requested_path": path,
            "description": "Provider hint was unresolved in the analyzed snapshot.",
        }
        for claim_id in claim_ids
        for path in paths
    ]
    findings = [
        {
            "rule_id": "RESULT.TEST",
            "severity": "WARNING",
            "impact": "INSUFFICIENT",
            "title": "Finding",
            "explanation": "x" * 100,
            "evidence": {"path": path},
        }
        for path in paths
    ]
    audits = [
        {
            "manifest_path": f"audit-{index}.yaml",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "metric": "accuracy",
            "minimum_improvement": 0.0,
            "direction": "higher",
            "findings": findings,
        }
        for index in range(4)
    ]
    actual = build_synthesis_request_parts(claims, evidence, owners, {}, missing, audits)
    actual_chars = logical_request_chars(actual.task, actual.payload, actual.schema)

    assert worst_valid_synthesis_request_chars(scope, config.limits) >= actual_chars


def test_runtime_obeys_preflight_synthesis_allocation_with_many_exact_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    paths = tuple(f"benchmarks/result-{index:02}.json" for index in range(24))
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"accuracy": 0.95}', encoding="utf-8")
    inventory = _inventory(tuple(ChangeEntry(path, ChangeStatus.ADDED) for path in paths))
    inputs = _declared_inputs(root, inventory)
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    class Provider:
        def __init__(self) -> None:
            self.calls: list[StructuredRequest] = []

        def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
            self.calls.append(request)
            title = next(source for source in request.payload["sources"] if source["kind"] == "pull_request_title")
            hints = list(request.payload["repository_paths"])
            claims = [
                {
                    "source_text": inputs.pr_title,
                    "claim_type": "other_scientific",
                    "subject": f"subject-{index}",
                    "metric": None,
                    "direction": "not_applicable",
                    "claimed_magnitude": None,
                    "qualifiers": [],
                    "source": {
                        "source_id": title["source_id"],
                        "start_line": 1,
                        "end_line": 1,
                    },
                    "confidence": 0.5,
                    "evidence_hints": hints,
                }
                for index in range(16)
            ]
            return ProviderResponse(
                output_text=json.dumps({"claims": claims}),
                provider="fake",
                model="fake",
            )

        def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
            self.calls.append(request)
            rows = [
                {
                    "claim_id": claim["claim_id"],
                    "interpretation": "Advisory interpretation.",
                    "citations": [],
                    "missing_evidence": ["No matching routed evidence."],
                    "unsupported_inferences": [],
                    "confidence": 0.5,
                }
                for claim in request.payload["claims"]
            ]
            return ProviderResponse(
                output_text=json.dumps({"interpretations": rows}),
                provider="fake",
                model="fake",
            )

    provider = Provider()
    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert result.preflight is not None and result.preflight.scope is not None
    assert "PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT" in _issue_codes(result.preflight, 3)
    assert len(provider.calls) == 2
    synthesis = provider.calls[1]
    actual_chars = logical_request_chars(
        synthesis.task, synthesis.payload, synthesis.schema
    )
    reserved = result.preflight.gates[2].metrics["synthesis_reserved_context_chars"]
    assert isinstance(reserved, int)
    assert actual_chars <= reserved <= 60_000


@pytest.mark.parametrize(
    ("failure", "setup"),
    [
        ("PREFLIGHT_G1_CHANGE_INVENTORY_UNAVAILABLE", "gate1"),
        ("PREFLIGHT_G2_NO_MATERIAL_CLAIM_SEED", "gate2"),
        ("PREFLIGHT_G3_CALL_LIMIT_LT_TWO", "gate3"),
    ],
)
def test_run_review_gate_failures_never_import_or_construct_a_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    setup: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inventory: ChangeInventory | None = None
    title = "Benchmark accuracy improves by 5% in results.json"
    config = ReviewConfig(enabled=True)
    if setup != "gate1":
        (root / "results.json").write_text('{"accuracy": 0.95}', encoding="utf-8")
        inventory = _inventory((ChangeEntry("results.json", ChangeStatus.ADDED),))
        monkeypatch.setattr(
            "claimci.review.preflight.build_git_change_inventory",
            lambda *_args, **_kwargs: inventory,
        )
    if setup == "gate2":
        title = "Documentation cleanup"
    if setup == "gate3":
        config = ReviewConfig(enabled=True, limits=ReviewLimits(max_calls=1))
    inputs = _declared_inputs(root, inventory, title=title, description="")

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.endswith("openai_provider"):
            raise AssertionError("provider module imported before preflight passed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = run_review(inputs, config)

    assert result.status is ReviewStatus.UNAVAILABLE
    assert result.preflight is not None
    assert failure in {
        issue.code for gate in result.preflight.gates for issue in gate.reasons
    }
    assert result.provider_calls == ()
    assert result.usage == result.usage.__class__()


def test_preflight_directly_never_imports_provider_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inputs = _declared_inputs(root, None)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "provider" in name:
            raise AssertionError("preflight must be provider-free")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.ready_for_provider is False
    assert result.gates[0].disposition is GateDisposition.FAIL


def test_default_provider_factory_symbol_is_reached_only_after_ready_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.orchestrator as orchestrator

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result, inputs, _config = _ready_preflight(tmp_path, monkeypatch)
    assert result.ready_for_provider is True
    constructions = 0

    def bomb(**_kwargs: Any) -> Any:
        nonlocal constructions
        constructions += 1
        raise AssertionError("factory sentinel")

    monkeypatch.setattr(orchestrator, "OpenAIReviewerProvider", bomb, raising=False)

    failed_gate = run_review(
        inputs,
        ReviewConfig(enabled=True, limits=ReviewLimits(max_calls=1)),
    )
    assert failed_gate.status is ReviewStatus.UNAVAILABLE
    assert constructions == 0

    after_ready_gate = run_review(inputs, ReviewConfig(enabled=True))
    assert after_ready_gate.status is ReviewStatus.UNAVAILABLE
    assert constructions == 1
