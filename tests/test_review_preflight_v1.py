"""Provider-free gate and exact request-accounting regressions."""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.preflight import preflight_review, scope_source_bundle
from claimci.review.provider import StructuredRequest
from claimci.review.request_budget import (
    build_extraction_request_parts,
    logical_request_chars,
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
    assert result.ready_for_provider is True
    assert result.review_status_ceiling is ReviewStatus.PARTIAL


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
