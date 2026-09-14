"""Wave 4 CLI contracts for opt-in advisory research review.

The review subcommand must be additive: it can render an advisory view and
write both formats in one orchestration run, while the existing ``audit``
command remains byte-for-byte and exit-code compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import claimci.cli as cli_module
import claimci.review as review_package
import claimci.review.orchestrator as orchestrator
from claimci.cli import main
from claimci.review.inventory import InventoryVerificationError
from claimci.review.models import (
    ChangeInventory,
    ChangeInventorySource,
    ComparisonBasis,
    GateDisposition,
    PreflightGateResult,
    ReviewConfig,
    ReviewPreflight,
    ReviewScope,
    ReviewStatus,
    SnapshotRole,
)
from claimci.review.orchestrator import ResearchReview


_REQUESTED_SHA = "1" * 40
_COMPARISON_SHA = "2" * 40
_HEAD_SHA = "3" * 40


def _write_enabled_config(root: Path, *, filename: str = ".claimci/review.yaml") -> Path:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "enabled": True,
                "policy": "advisory",
                "provider": "openai",
                "model": "fake-model",
                "limits": {
                    "max_calls": 2,
                    "max_context_chars": 60_000,
                    "max_output_chars": 24_000,
                    "max_files": 24,
                    "max_file_chars": 16_000,
                    "max_claims": 16,
                    "extraction_max_output_tokens": 5_000,
                    "synthesis_max_output_tokens": 4_000,
                    "timeout_seconds": 30,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _complete_review() -> ResearchReview:
    return ResearchReview(status=ReviewStatus.COMPLETE)


def _inventory(*, basis: ComparisonBasis) -> ChangeInventory:
    comparison_sha = (
        _REQUESTED_SHA if basis is ComparisonBasis.DIRECT_BASE else _COMPARISON_SHA
    )
    return ChangeInventory(
        schema_version=1,
        requested_base_sha=_REQUESTED_SHA,
        comparison_base_sha=comparison_sha,
        head_sha=_HEAD_SHA,
        comparison_basis=basis,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=0,
        complete=True,
        entries=(),
    )


def _ready_preflight(inventory: ChangeInventory) -> ReviewPreflight:
    scope = ReviewScope(
        mode="declared_changed_v1",
        inventory=inventory,
        issued_paths=(),
        issued_changed_paths=(),
        selected_paths=(),
        sources=(),
        seeds=(),
        complete=True,
        issues=(),
    )
    gates = tuple(
        PreflightGateResult(
            gate=gate,
            disposition=GateDisposition.PASS_COMPLETE,
            reasons=(),
            metrics={},
        )
        for gate in (1, 2, 3)
    )
    return ReviewPreflight(
        schema_version=1,
        requested_base_sha=inventory.requested_base_sha,
        comparison_base_sha=inventory.comparison_base_sha,
        head_sha=inventory.head_sha,
        gates=gates,
        ready_for_provider=True,
        review_status_ceiling=ReviewStatus.COMPLETE,
        scope=scope,
    )


def _coordinate_arguments(
    requested_root: Path,
    comparison_root: Path,
    *,
    basis: ComparisonBasis = ComparisonBasis.MERGE_BASE,
) -> list[str]:
    comparison_sha = (
        _REQUESTED_SHA if basis is ComparisonBasis.DIRECT_BASE else _COMPARISON_SHA
    )
    return [
        "--requested-base-root",
        str(requested_root),
        "--comparison-base-root",
        str(comparison_root),
        "--requested-base-sha",
        _REQUESTED_SHA,
        "--comparison-base-sha",
        comparison_sha,
        "--comparison-basis",
        basis.value,
        "--head-sha",
        _HEAD_SHA,
    ]


def _patch_run_review(
    monkeypatch: pytest.MonkeyPatch,
    result: ResearchReview,
    calls: list[tuple[Any, ...]],
) -> None:
    """Patch either CLI import style without touching the real provider."""

    def fake_run_review(*args: Any, **kwargs: Any) -> ResearchReview:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(cli_module, "run_review", fake_run_review, raising=False)
    monkeypatch.setattr(orchestrator, "run_review", fake_run_review)


def _spy_on_provider_free_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Keep the real preflight runtime while recording the CLI boundary call."""

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_run_review = orchestrator.run_review

    def recording_run_review(*args: Any, **kwargs: Any) -> ResearchReview:
        calls.append((args, kwargs))
        return real_run_review(*args, **kwargs)

    monkeypatch.setattr(cli_module, "run_review", recording_run_review)
    return calls


def test_review_package_exposes_the_provider_free_preflight_contract() -> None:
    assert review_package.preflight_review is cli_module.preflight_review
    assert review_package.ReviewPreflight is ReviewPreflight


def test_review_cli_default_is_disabled_and_never_constructs_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()

    class NoProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("default-disabled review must not construct an LLM provider")

    monkeypatch.setattr(orchestrator, "OpenAIReviewerProvider", NoProvider)

    exit_code = main(["review", str(repository), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["review"] == {
        "status": "DISABLED",
        "policy": "advisory",
        "blocking": False,
    }
    assert payload["provider"]["calls"] == []
    assert payload["provider"]["usage"] == {
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert "OPENAI_API_KEY" not in captured.out


def test_review_cli_reads_event_metadata_from_trusted_config_root_and_writes_both_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    trusted = tmp_path / "trusted-base"
    trusted.mkdir()
    config_path = _write_enabled_config(trusted, filename="review-config.yaml")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "Event title",
                    "body": "Event body with \"quoted\" data",
                }
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "review.json"
    markdown_output = tmp_path / "review.md"
    calls: list[tuple[Any, ...]] = []
    _patch_run_review(monkeypatch, _complete_review(), calls)

    exit_code = main(
        [
            "review",
            str(repository),
            "--base-root",
            str(base),
            "--config-root",
            str(trusted),
            "--config",
            str(config_path),
            "--event-json",
            str(event_path),
            "--pr-title",
            "Explicit title wins",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert len(calls) == 1, "one CLI invocation must orchestrate review only once"
    args, kwargs = calls[0]
    assert kwargs == {}
    inputs, config = args
    assert inputs.repository_root == repository.resolve()
    assert inputs.base_root == base.resolve()
    assert inputs.requested_base is None
    assert inputs.comparison_base is None
    assert inputs.head is None
    assert inputs.inventory is None
    assert inputs.pr_title == "Explicit title wins"
    assert inputs.pr_description == 'Event body with "quoted" data'
    assert isinstance(config, ReviewConfig)
    assert config.enabled is True
    assert json.loads(json_output.read_text(encoding="utf-8"))["review"]["status"] == "COMPLETE"
    markdown = markdown_output.read_text(encoding="utf-8")
    assert markdown.startswith("## ClaimCI Research Review (Advisory)\n")


@pytest.mark.parametrize(
    "missing_option",
    [
        "--requested-base-root",
        "--comparison-base-root",
        "--requested-base-sha",
        "--comparison-base-sha",
        "--comparison-basis",
        "--head-sha",
    ],
)
def test_review_cli_requires_the_declared_coordinate_contract_as_one_group(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing_option: str,
) -> None:
    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    for root in (repository, requested, comparison):
        root.mkdir()
    group = _coordinate_arguments(requested, comparison)
    index = group.index(missing_option)
    del group[index : index + 2]

    exit_code = main(["review", str(repository), "--preflight-only", "--json", *group])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "all six coordinate options are required together" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("basis", [ComparisonBasis.DIRECT_BASE, ComparisonBasis.MERGE_BASE])
def test_review_cli_preflight_only_builds_inventory_then_calls_provider_free_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    basis: ComparisonBasis,
) -> None:
    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = requested if basis is ComparisonBasis.DIRECT_BASE else tmp_path / "comparison"
    for root in {repository, requested, comparison}:
        root.mkdir()
    _write_enabled_config(requested)
    inventory = _inventory(basis=basis)
    inventory_calls: list[tuple[Any, ...]] = []
    review_calls: list[tuple[Any, ...]] = []
    json_output = tmp_path / "preflight.json"
    markdown_output = tmp_path / "preflight.md"

    def fake_inventory(*args: Any) -> ChangeInventory:
        inventory_calls.append(args)
        return inventory

    monkeypatch.setattr(cli_module, "build_git_change_inventory", fake_inventory)
    _patch_run_review(
        monkeypatch,
        ResearchReview(
            status=ReviewStatus.COMPLETE,
            preflight=_ready_preflight(inventory),
        ),
        review_calls,
    )
    monkeypatch.setattr(
        orchestrator,
        "OpenAIReviewerProvider",
        lambda **_kwargs: pytest.fail("preflight-only must not construct a provider"),
    )

    exit_code = main(
        [
            "review",
            str(repository),
            "--config-root",
            str(requested),
            "--preflight-only",
            "--json",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            *_coordinate_arguments(requested, comparison, basis=basis),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert len(inventory_calls) == 1
    requested_id, comparison_id, head_id, passed_basis = inventory_calls[0]
    assert requested_id.role is SnapshotRole.REQUESTED_BASE
    assert requested_id.root == requested.resolve()
    assert comparison_id.role is SnapshotRole.COMPARISON_BASE
    assert comparison_id.root == comparison.resolve()
    assert head_id.role is SnapshotRole.HEAD
    assert head_id.root == repository.resolve()
    assert passed_basis is basis
    assert len(review_calls) == 1
    args, kwargs = review_calls[0]
    assert kwargs == {"preflight_only": True}
    inputs, config = args
    assert inputs.inventory is inventory
    assert inputs.requested_base is requested_id
    assert inputs.comparison_base is comparison_id
    assert inputs.head is head_id
    assert inputs.coordinates.requested_base_sha == _REQUESTED_SHA
    assert inputs.coordinates.comparison_base_sha == comparison_id.sha
    assert inputs.coordinates.head_sha == _HEAD_SHA
    assert inputs.coordinates.comparison_basis is basis
    assert isinstance(config, ReviewConfig)
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 2
    assert payload["preflight"]["ready_for_provider"] is True
    assert payload["provider"]["lifecycle"] == "not_attempted"
    assert payload["provider"]["attempted_call_count"] == 0
    assert payload["provider"]["calls"] == []
    assert payload["provider"]["usage"] == {
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert json.loads(json_output.read_text(encoding="utf-8")) == payload
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "Provider lifecycle: not&#95;attempted." in markdown
    assert "Provider attempts: 0; completed response records: 0." in markdown


def test_review_cli_preflight_inventory_failure_is_structured_and_does_not_leak_git_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    for root in (repository, requested, comparison):
        root.mkdir()
    _write_enabled_config(requested)
    review_calls = _spy_on_provider_free_runtime(monkeypatch)

    def fail_inventory(*_args: Any) -> ChangeInventory:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "fatal: hostile-git-output sk-secret",
        )

    monkeypatch.setattr(cli_module, "build_git_change_inventory", fail_inventory)

    exit_code = main(
        [
            "review",
            str(repository),
            "--config-root",
            str(requested),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(requested, comparison),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert len(review_calls) == 1
    args, kwargs = review_calls[0]
    assert kwargs == {"preflight_only": True}
    inputs, config = args
    assert inputs.inventory is None
    assert inputs.inventory_failure.code == "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"
    assert isinstance(config, ReviewConfig)
    assert payload["review"]["status"] == "UNAVAILABLE"
    assert payload["preflight"]["ready_for_provider"] is False
    assert payload["preflight"]["gates"][0]["reasons"] == [
        {"code": "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"}
    ]
    assert payload["preflight"]["coordinates"] == {
        "requested_base_sha": _REQUESTED_SHA,
        "comparison_base_sha": _COMPARISON_SHA,
        "head_sha": _HEAD_SHA,
        "comparison_basis": "merge_base",
    }
    assert payload["provider"]["calls"] == []
    assert payload["provider"]["usage"] == {
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert "hostile-git-output" not in captured.out + captured.err
    assert "sk-secret" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_review_cli_declared_symlink_root_is_a_structured_gate_one_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    for root in (repository, requested, comparison):
        root.mkdir()
    _write_enabled_config(requested)
    requested_link = tmp_path / "requested-link"
    path_type = type(requested_link)
    original_resolve = path_type.resolve
    original_is_symlink = path_type.is_symlink
    requested_resolved = requested.resolve()

    def pretend_symlink_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == requested_link:
            return requested_resolved
        return original_resolve(path, *args, **kwargs)

    def pretend_is_symlink(path: Path) -> bool:
        return path == requested_link or original_is_symlink(path)

    monkeypatch.setattr(path_type, "resolve", pretend_symlink_resolve)
    monkeypatch.setattr(path_type, "is_symlink", pretend_is_symlink)
    monkeypatch.setattr(
        cli_module,
        "build_git_change_inventory",
        lambda *_args: pytest.fail("a symlinked declared root must not reach Git"),
    )
    review_calls = _spy_on_provider_free_runtime(monkeypatch)

    exit_code = main(
        [
            "review",
            str(repository),
            "--config-root",
            str(requested),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(requested_link, comparison),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    reasons = payload["preflight"]["gates"][0]["reasons"]
    assert exit_code == 0
    assert len(review_calls) == 1
    assert review_calls[0][1] == {"preflight_only": True}
    assert reasons == [{"code": "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"}]
    assert payload["preflight"]["coordinates"] == {
        "requested_base_sha": _REQUESTED_SHA,
        "comparison_base_sha": _COMPARISON_SHA,
        "head_sha": _HEAD_SHA,
        "comparison_basis": "merge_base",
    }
    assert payload["provider"]["calls"] == []
    assert "Traceback" not in captured.out + captured.err


def test_review_cli_invalid_requested_root_without_config_root_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid default trusted-config root cannot bypass report rendering."""

    repository = tmp_path / "head"
    comparison = tmp_path / "comparison"
    missing_requested = tmp_path / "missing-requested"
    repository.mkdir()
    comparison.mkdir()
    monkeypatch.setattr(
        cli_module,
        "build_git_change_inventory",
        lambda *_args: pytest.fail("an invalid requested root must not reach Git"),
    )
    review_calls = _spy_on_provider_free_runtime(monkeypatch)

    exit_code = main(
        [
            "review",
            str(repository),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(missing_requested, comparison),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    reasons = payload["preflight"]["gates"][0]["reasons"]
    assert exit_code == 0
    assert len(review_calls) == 1
    assert review_calls[0][1] == {"preflight_only": True}
    assert payload["review"]["status"] == "UNAVAILABLE"
    assert reasons == [{"code": "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"}]
    assert payload["preflight"]["coordinates"]["comparison_basis"] == "merge_base"
    assert payload["provider"]["calls"] == []
    assert str(missing_requested) not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("root_role", "reason_code"),
    [
        ("requested", "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"),
        ("comparison", "PREFLIGHT_G1_COMPARISON_BASE_ROOT_INVALID"),
        ("head", "PREFLIGHT_G1_HEAD_ROOT_INVALID"),
    ],
)
@pytest.mark.parametrize("invalid_kind", ["missing", "file", "symlink"])
def test_review_cli_declared_invalid_roots_preserve_raw_coordinates_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root_role: str,
    reason_code: str,
    invalid_kind: str,
) -> None:
    """Every invalid declared-root kind is a content-free Gate 1 result."""

    roots = {
        "requested": tmp_path / "requested",
        "comparison": tmp_path / "comparison",
        "head": tmp_path / "head",
    }
    for root in roots.values():
        root.mkdir()
    config_root = tmp_path / "trusted-config"
    config_root.mkdir()
    _write_enabled_config(config_root)

    invalid = tmp_path / f"invalid-{root_role}-{invalid_kind}"
    if invalid_kind == "file":
        invalid.write_text("not a directory", encoding="utf-8")
    elif invalid_kind == "symlink":
        target = tmp_path / f"target-{root_role}"
        target.mkdir()
        target_resolved = target.resolve()
        path_type = type(invalid)
        original_resolve = path_type.resolve
        original_is_symlink = path_type.is_symlink

        def pretend_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
            if path == invalid:
                return target_resolved
            return original_resolve(path, *args, **kwargs)

        def pretend_is_symlink(path: Path) -> bool:
            return path == invalid or original_is_symlink(path)

        monkeypatch.setattr(path_type, "resolve", pretend_resolve)
        monkeypatch.setattr(path_type, "is_symlink", pretend_is_symlink)
    roots[root_role] = invalid

    monkeypatch.setattr(
        cli_module,
        "build_git_change_inventory",
        lambda *_args: pytest.fail("invalid declared roots must not reach Git"),
    )
    review_calls = _spy_on_provider_free_runtime(monkeypatch)

    exit_code = main(
        [
            "review",
            str(roots["head"]),
            "--config-root",
            str(config_root),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(roots["requested"], roots["comparison"]),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert len(review_calls) == 1
    assert review_calls[0][1] == {"preflight_only": True}
    assert payload["review"]["status"] == "UNAVAILABLE"
    assert payload["preflight"]["gates"][0]["reasons"] == [
        {"code": reason_code}
    ]
    assert payload["preflight"]["coordinates"] == {
        "requested_base_sha": _REQUESTED_SHA,
        "comparison_base_sha": _COMPARISON_SHA,
        "head_sha": _HEAD_SHA,
        "comparison_basis": "merge_base",
    }
    assert payload["provider"]["calls"] == []
    assert str(invalid) not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_review_cli_preflight_only_short_circuits_disabled_config_before_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A disabled free path mirrors runtime and cannot become provider-ready."""

    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    for root in (repository, requested, comparison):
        root.mkdir()
    monkeypatch.setattr(
        cli_module,
        "build_git_change_inventory",
        lambda *_args: pytest.fail("disabled preflight must not invoke Git"),
    )
    monkeypatch.setattr(
        cli_module,
        "run_review",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled preflight must short-circuit before run_review"
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "OpenAIReviewerProvider",
        lambda **_kwargs: pytest.fail("disabled preflight must not construct a provider"),
    )

    exit_code = main(
        [
            "review",
            str(repository),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(requested, comparison),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["review"] == {
        "status": "DISABLED",
        "policy": "advisory",
        "blocking": False,
    }
    assert payload["preflight"] is None
    assert payload["provider"]["calls"] == []
    assert payload["provider"]["usage"] == {
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert captured.err == ""


def test_review_cli_rejects_symlinked_explicit_config_root_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trusted configuration roots are lexical non-symlink directories."""

    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    config_target = tmp_path / "config-target"
    for root in (repository, requested, comparison, config_target):
        root.mkdir()
    _write_enabled_config(config_target)
    config_link = tmp_path / "config-link"
    target_resolved = config_target.resolve()
    path_type = type(config_link)
    original_resolve = path_type.resolve
    original_is_symlink = path_type.is_symlink

    def pretend_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == config_link:
            return target_resolved
        return original_resolve(path, *args, **kwargs)

    def pretend_is_symlink(path: Path) -> bool:
        return path == config_link or original_is_symlink(path)

    monkeypatch.setattr(path_type, "resolve", pretend_resolve)
    monkeypatch.setattr(path_type, "is_symlink", pretend_is_symlink)
    monkeypatch.setattr(
        cli_module,
        "load_review_config",
        lambda *_args: pytest.fail("a symlinked config root must not be followed"),
    )

    exit_code = main(
        [
            "review",
            str(repository),
            "--config-root",
            str(config_link),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(requested, comparison),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "config root is invalid" in captured.err
    assert str(config_link) not in captured.err
    assert str(config_target) not in captured.err
    assert "Traceback" not in captured.err


def test_review_cli_preflight_event_json_keeps_the_one_mebibyte_input_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "head"
    requested = tmp_path / "requested"
    comparison = tmp_path / "comparison"
    for root in (repository, requested, comparison):
        root.mkdir()
    _write_enabled_config(requested)
    event = tmp_path / "event.json"
    event.write_bytes(b"{" + b"x" * (cli_module.MAX_EVENT_METADATA_BYTES + 1))
    monkeypatch.setattr(
        cli_module,
        "build_git_change_inventory",
        lambda *_args: _inventory(basis=ComparisonBasis.MERGE_BASE),
    )
    monkeypatch.setattr(
        cli_module,
        "run_review",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized event metadata must fail before run_review"
        ),
    )

    exit_code = main(
        [
            "review",
            str(repository),
            "--config-root",
            str(requested),
            "--event-json",
            str(event),
            "--preflight-only",
            "--json",
            *_coordinate_arguments(requested, comparison),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "could not load event metadata" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "provider_error",
    [
        "OpenAI SDK is not installed; install the llm extra",
        "OPENAI_API_KEY=sk-live-secret is missing",
    ],
)
def test_review_cli_provider_or_key_failure_is_safe_advisory_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_error: str,
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()
    _write_enabled_config(repository)

    from claimci.review.openai_provider import OpenAIReviewerProvider

    def fail_without_network(self: OpenAIReviewerProvider) -> object:
        raise RuntimeError(provider_error)

    monkeypatch.setattr(OpenAIReviewerProvider, "_client", fail_without_network)

    exit_code = main(["review", str(repository), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["review"]["status"] == "UNAVAILABLE"
    assert payload["review"]["blocking"] is False
    assert payload["provider"]["calls"] == []
    assert "sk-live-secret" not in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_review_cli_invalid_invocation_config_and_path_are_controlled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["review"])
    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err

    missing_repository = tmp_path / "does-not-exist"
    assert main(["review", str(missing_repository)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err

    repository = tmp_path / "repo"
    repository.mkdir()
    bad_config = repository / ".claimci" / "review.yaml"
    bad_config.parent.mkdir()
    bad_config.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    assert main(["review", str(repository), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.startswith("ClaimCI error:")


def test_review_cli_rejects_conflicting_view_flags_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(SystemExit) as raised:
        main(["review", str(repository), "--json", "--markdown"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "mutually exclusive" in captured.err
    assert "Traceback" not in captured.err


def test_review_parser_does_not_change_existing_audit_bytes_or_exit(
    study_factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = study_factory(candidate_config_updates={"training_steps": 300})

    before_exit = main(["audit", str(manifest), "--json"])
    before = capsys.readouterr()

    # Exercise the newly added subcommand between two audit invocations.  A
    # parser/global-state regression must not alter deterministic report bytes.
    repository = tmp_path / "review-repository"
    repository.mkdir()
    assert main(["review", str(repository), "--json"]) == 0
    capsys.readouterr()

    after_exit = main(["audit", str(manifest), "--json"])
    after = capsys.readouterr()

    assert after_exit == before_exit == 1
    assert after.out == before.out
    assert after.err == before.err == ""
