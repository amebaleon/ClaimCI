"""Contracts for the coordinate-bound, metadata-only Git change inventory."""

from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from claimci.review import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    SnapshotIdentity,
    SnapshotRole,
    build_git_change_inventory,
)


BASE = "1" * 40
MERGE_BASE = "2" * 40
HEAD = "3" * 40
SHA256 = "a" * 64


def _inventory(**overrides: object) -> ChangeInventory:
    values: dict[str, object] = {
        "schema_version": 1,
        "requested_base_sha": BASE,
        "comparison_base_sha": MERGE_BASE,
        "head_sha": HEAD,
        "comparison_basis": ComparisonBasis.MERGE_BASE,
        "source": ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        "declared_entry_count": 3,
        "complete": True,
        "entries": (
            ChangeEntry("deleted.txt", ChangeStatus.DELETED),
            ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
            ChangeEntry("tests/test_model.py", ChangeStatus.ADDED),
        ),
    }
    values.update(overrides)
    return ChangeInventory(**values)  # type: ignore[arg-type]


def test_inventory_contract_has_exact_vocabulary_and_shape() -> None:
    assert {item.name: item.value for item in ComparisonBasis} == {
        "DIRECT_BASE": "direct_base",
        "MERGE_BASE": "merge_base",
    }
    assert {item.name: item.value for item in ChangeStatus} == {
        "ADDED": "added",
        "MODIFIED": "modified",
        "DELETED": "deleted",
    }
    assert {item.name: item.value for item in ChangeInventorySource} == {
        "TRUSTED_GIT_OBJECT_GRAPH": "trusted_git_object_graph",
        "LEGACY_PAIRWISE": "legacy_pairwise",
    }
    assert {item.name: item.value for item in SnapshotRole} == {
        "REQUESTED_BASE": "requested_base",
        "COMPARISON_BASE": "comparison_base",
        "HEAD": "head",
    }

    inventory = _inventory()

    assert inventory.entries == (
        ChangeEntry("deleted.txt", ChangeStatus.DELETED),
        ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
        ChangeEntry("tests/test_model.py", ChangeStatus.ADDED),
    )
    assert inventory.declared_entry_count == 3
    assert inventory.complete is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        inventory.complete = False  # type: ignore[misc]


def test_snapshot_identity_contract_binds_exact_role_resolved_root_and_full_sha(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    identity = SnapshotIdentity(SnapshotRole.HEAD, root, SHA256)
    assert identity == SnapshotIdentity(role=SnapshotRole.HEAD, root=root, sha=SHA256)
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.sha = HEAD  # type: ignore[misc]


@pytest.mark.parametrize(
    "sha",
    [
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "A" * 40,
        "g" * 40,
        "0x" + "a" * 40,
        "",
        None,
    ],
)
def test_snapshot_identity_rejects_invalid_or_abbreviated_sha(
    tmp_path: Path, sha: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SnapshotIdentity(SnapshotRole.HEAD, tmp_path.resolve(), sha)  # type: ignore[arg-type]


@pytest.mark.parametrize("role", ["head", "HEAD", None, 3])
def test_snapshot_identity_rejects_invalid_role(tmp_path: Path, role: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SnapshotIdentity(role, tmp_path.resolve(), HEAD)  # type: ignore[arg-type]


def test_snapshot_identity_rejects_invalid_or_unresolved_root(tmp_path: Path) -> None:
    relative = Path("relative-root")
    missing = tmp_path / "missing"
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("not a directory", encoding="utf-8")

    for root in (relative, missing, regular_file, str(tmp_path)):
        with pytest.raises((TypeError, ValueError)):
            SnapshotIdentity(SnapshotRole.HEAD, root, HEAD)  # type: ignore[arg-type]

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-root"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises((TypeError, ValueError)):
            SnapshotIdentity(SnapshotRole.HEAD, link.absolute(), HEAD)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "/absolute.py",
        "../outside.py",
        "src/../outside.py",
        "src\\model.py",
        "C:/outside.py",
        "//server/share.py",
        "src//model.py",
        "src/./model.py",
        "src/model.py/",
        "src/\x00model.py",
        "src/\tmodel.py",
        "src/\nmodel.py",
        "src/\x7fmodel.py",
    ],
)
def test_change_entry_rejects_invalid_nonportable_path(path: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChangeEntry(path, ChangeStatus.MODIFIED)


@pytest.mark.parametrize("status", ["added", "M", "renamed", None, 1])
def test_change_entry_rejects_invalid_status(status: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChangeEntry("src/model.py", status)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("declared_entry_count", True),
        ("declared_entry_count", -1),
        ("complete", 1),
        ("comparison_basis", "merge_base"),
        ("source", "trusted_git_object_graph"),
        ("entries", []),
        ("requested_base_sha", "1" * 39),
        ("comparison_base_sha", "B" * 40),
        ("head_sha", "3" * 64 + "0"),
    ],
)
def test_inventory_rejects_invalid_contract_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _inventory(**{field: value})


def test_trusted_inventory_rejects_incomplete_or_wrong_declared_count() -> None:
    with pytest.raises((TypeError, ValueError)):
        _inventory(complete=False)
    with pytest.raises((TypeError, ValueError)):
        _inventory(declared_entry_count=2)


def test_inventory_rejects_duplicate_and_case_preserving_aliases() -> None:
    for entries in (
        (
            ChangeEntry("src/model.py", ChangeStatus.ADDED),
            ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
        ),
        (
            ChangeEntry("SRC/model.py", ChangeStatus.ADDED),
            ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            _inventory(declared_entry_count=2, entries=entries)


def test_inventory_rejects_noncanonical_ordering() -> None:
    entries = (
        ChangeEntry("tests/test_model.py", ChangeStatus.ADDED),
        ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
    )
    with pytest.raises((TypeError, ValueError)):
        _inventory(declared_entry_count=2, entries=entries)


def test_direct_base_inventory_requires_equal_requested_and_comparison_sha() -> None:
    with pytest.raises((TypeError, ValueError)):
        _inventory(comparison_basis=ComparisonBasis.DIRECT_BASE)
    direct = _inventory(
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        comparison_base_sha=BASE,
    )
    assert direct.requested_base_sha == direct.comparison_base_sha


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=False,
    )
    return result.stdout.strip()


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def divergent_git_graph(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    tmp_path = tmp_path_factory.mktemp("review-inventory-git")
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@claimci.invalid")
    _git(repository, "config", "user.name", "ClaimCI Tests")
    _write(repository, "deleted.txt", "delete me\n")
    _write(repository, "modified.txt", "old\n")
    _write(repository, "rename-old.txt", "rename me\n")
    _write(repository, ".gitattributes", "*.txt diff=danger\n")
    _write(repository, "must-not-run.py", "raise AssertionError('checkout code executed')\n")
    original = _commit(repository, "original")

    _git(repository, "branch", "requested", original)
    _write(repository, "early.txt", "early\n")
    direct_base = _commit(repository, "direct base")
    _write(repository, "modified.txt", "new\n")
    (repository / "deleted.txt").unlink()
    _git(repository, "mv", "rename-old.txt", "rename-new.txt")
    _write(repository, "added.txt", "added\n")
    head = _commit(repository, "head")

    _git(repository, "checkout", "-q", "requested")
    _write(repository, "requested-only.txt", "requested\n")
    requested = _commit(repository, "requested base tip")

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    roots: dict[str, Path] = {}
    for name, sha in (
        ("original", original),
        ("direct", direct_base),
        ("requested", requested),
        ("head", head),
    ):
        root = (snapshots / name).resolve()
        _git(repository, "worktree", "add", "-q", "--detach", str(root), sha)
        roots[name] = root

    return {
        "repository": repository,
        "roots": roots,
        "original": original,
        "direct": direct_base,
        "requested": requested,
        "head": head,
    }


def _identity(role: SnapshotRole, root: Path, sha: str) -> SnapshotIdentity:
    return SnapshotIdentity(role=role, root=root.resolve(), sha=sha)


def _direct_inventory(graph: dict[str, object]) -> ChangeInventory:
    roots = graph["roots"]
    assert isinstance(roots, dict)
    direct_root = roots["direct"]
    head_root = roots["head"]
    assert isinstance(direct_root, Path) and isinstance(head_root, Path)
    direct_sha = graph["direct"]
    head_sha = graph["head"]
    assert isinstance(direct_sha, str) and isinstance(head_sha, str)
    return build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, direct_root, direct_sha),
        _identity(SnapshotRole.COMPARISON_BASE, direct_root, direct_sha),
        _identity(SnapshotRole.HEAD, head_root, head_sha),
        ComparisonBasis.DIRECT_BASE,
    )


def _same_snapshot_inventory(root: Path, sha: str) -> ChangeInventory:
    return build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, root, sha),
        _identity(SnapshotRole.COMPARISON_BASE, root, sha),
        _identity(SnapshotRole.HEAD, root, sha),
        ComparisonBasis.DIRECT_BASE,
    )


def test_git_inventory_verifies_all_roots_and_emits_rename_as_delete_add(
    divergent_git_graph: dict[str, object],
) -> None:
    inventory = _direct_inventory(divergent_git_graph)
    assert inventory.entries == (
        ChangeEntry("added.txt", ChangeStatus.ADDED),
        ChangeEntry("deleted.txt", ChangeStatus.DELETED),
        ChangeEntry("modified.txt", ChangeStatus.MODIFIED),
        ChangeEntry("rename-new.txt", ChangeStatus.ADDED),
        ChangeEntry("rename-old.txt", ChangeStatus.DELETED),
    )
    assert inventory.declared_entry_count == 5
    assert inventory.complete is True
    assert inventory.source is ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH


def test_merge_base_inventory_requires_unique_declared_merge_base(
    divergent_git_graph: dict[str, object],
) -> None:
    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    requested_root = roots["requested"]
    comparison_root = roots["original"]
    head_root = roots["head"]
    assert all(isinstance(root, Path) for root in (requested_root, comparison_root, head_root))
    requested = divergent_git_graph["requested"]
    comparison = divergent_git_graph["original"]
    head = divergent_git_graph["head"]
    assert all(isinstance(sha, str) for sha in (requested, comparison, head))

    inventory = build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, requested_root, requested),  # type: ignore[arg-type]
        _identity(SnapshotRole.COMPARISON_BASE, comparison_root, comparison),  # type: ignore[arg-type]
        _identity(SnapshotRole.HEAD, head_root, head),  # type: ignore[arg-type]
        ComparisonBasis.MERGE_BASE,
    )

    assert inventory.comparison_base_sha == comparison
    assert inventory.entries[0] == ChangeEntry("added.txt", ChangeStatus.ADDED)
    assert ChangeEntry("early.txt", ChangeStatus.ADDED) in inventory.entries


def test_same_head_different_coordinates_have_distinct_inventory_identity_and_entries(
    divergent_git_graph: dict[str, object],
) -> None:
    direct = _direct_inventory(divergent_git_graph)
    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    requested = divergent_git_graph["requested"]
    original = divergent_git_graph["original"]
    head = divergent_git_graph["head"]
    assert all(isinstance(value, str) for value in (requested, original, head))
    merge = build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, roots["requested"], requested),  # type: ignore[arg-type]
        _identity(SnapshotRole.COMPARISON_BASE, roots["original"], original),  # type: ignore[arg-type]
        _identity(SnapshotRole.HEAD, roots["head"], head),  # type: ignore[arg-type]
        ComparisonBasis.MERGE_BASE,
    )
    assert direct.head_sha == merge.head_sha
    assert (
        direct.requested_base_sha,
        direct.comparison_base_sha,
        direct.comparison_basis,
    ) != (
        merge.requested_base_sha,
        merge.comparison_base_sha,
        merge.comparison_basis,
    )
    assert direct.entries != merge.entries


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_git_inventory_rejects_dirty_snapshot(
    divergent_git_graph: dict[str, object], dirty_kind: str
) -> None:
    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    head_root = roots["head"]
    assert isinstance(head_root, Path)
    if dirty_kind == "tracked":
        _write(head_root, "modified.txt", "dirty\n")
    else:
        _write(head_root, "untracked.txt", "dirty\n")
    try:
        with pytest.raises((TypeError, ValueError), match="dirty"):
            _direct_inventory(divergent_git_graph)
    finally:
        if dirty_kind == "tracked":
            _write(head_root, "modified.txt", "new\n")
        else:
            (head_root / "untracked.txt").unlink()


def test_git_inventory_rejects_wrong_sha_role_basis_and_inconsistent_roots(
    divergent_git_graph: dict[str, object],
) -> None:
    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    direct = divergent_git_graph["direct"]
    head = divergent_git_graph["head"]
    assert isinstance(direct, str) and isinstance(head, str)
    requested = _identity(SnapshotRole.REQUESTED_BASE, roots["direct"], direct)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, roots["direct"], direct)
    head_identity = _identity(SnapshotRole.HEAD, roots["head"], head)

    with pytest.raises((TypeError, ValueError)):
        build_git_change_inventory(
            _identity(SnapshotRole.HEAD, roots["direct"], direct),
            comparison,
            head_identity,
            ComparisonBasis.DIRECT_BASE,
        )
    with pytest.raises((TypeError, ValueError)):
        build_git_change_inventory(requested, comparison, head_identity, "direct_base")  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError), match="SHA|sha|identity"):
        build_git_change_inventory(
            requested,
            comparison,
            _identity(SnapshotRole.HEAD, roots["head"], "f" * len(head)),
            ComparisonBasis.DIRECT_BASE,
        )
    with pytest.raises((TypeError, ValueError)):
        build_git_change_inventory(
            requested,
            _identity(SnapshotRole.COMPARISON_BASE, roots["head"], head),
            head_identity,
            ComparisonBasis.DIRECT_BASE,
        )


def test_git_inventory_rejects_missing_cross_repository_object(tmp_path: Path) -> None:
    def new_repo(name: str, text: str) -> tuple[Path, str]:
        root = (tmp_path / name).resolve()
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "tests@claimci.invalid")
        _git(root, "config", "user.name", "ClaimCI Tests")
        _write(root, "file.txt", text)
        return root, _commit(root, name)

    base_root, base_sha = new_repo("base", "base\n")
    head_root, head_sha = new_repo("head", "head\n")
    with pytest.raises((TypeError, ValueError)):
        build_git_change_inventory(
            _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha),
            _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha),
            _identity(SnapshotRole.HEAD, head_root, head_sha),
            ComparisonBasis.DIRECT_BASE,
        )


def test_git_inventory_rejects_wrong_or_multiple_merge_base(
    divergent_git_graph: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    requested = divergent_git_graph["requested"]
    original = divergent_git_graph["original"]
    direct = divergent_git_graph["direct"]
    head = divergent_git_graph["head"]
    assert all(isinstance(value, str) for value in (requested, original, direct, head))

    with pytest.raises((TypeError, ValueError)):
        build_git_change_inventory(
            _identity(SnapshotRole.REQUESTED_BASE, roots["requested"], requested),  # type: ignore[arg-type]
            _identity(SnapshotRole.COMPARISON_BASE, roots["direct"], direct),  # type: ignore[arg-type]
            _identity(SnapshotRole.HEAD, roots["head"], head),  # type: ignore[arg-type]
            ComparisonBasis.MERGE_BASE,
        )

    real_run_git = inventory_module._run_git

    def multiple(root: Path, arguments: tuple[str, ...], *, stdout_limit: int) -> bytes:
        if arguments and arguments[0] == "merge-base":
            return f"{original}\n{direct}\n".encode("ascii")
        return real_run_git(root, arguments, stdout_limit=stdout_limit)

    monkeypatch.setattr(inventory_module, "_run_git", multiple)
    with pytest.raises((TypeError, ValueError), match="merge base"):
        build_git_change_inventory(
            _identity(SnapshotRole.REQUESTED_BASE, roots["requested"], requested),  # type: ignore[arg-type]
            _identity(SnapshotRole.COMPARISON_BASE, roots["original"], original),  # type: ignore[arg-type]
            _identity(SnapshotRole.HEAD, roots["head"], head),  # type: ignore[arg-type]
            ComparisonBasis.MERGE_BASE,
        )


@pytest.mark.parametrize("hostile_name", ["tab\tname.txt", "line\nbreak.txt"])
def test_git_inventory_nul_safe_metadata_rejects_control_filename(
    divergent_git_graph: dict[str, object],
    hostile_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.review.inventory as inventory_module

    real_run_git = inventory_module._run_git

    def hostile(root: Path, arguments: tuple[str, ...], *, stdout_limit: int) -> bytes:
        if arguments and arguments[0] == "diff-tree":
            return b"A\0" + hostile_name.encode("utf-8") + b"\0"
        return real_run_git(root, arguments, stdout_limit=stdout_limit)

    monkeypatch.setattr(inventory_module, "_run_git", hostile)

    with pytest.raises((TypeError, ValueError)) as raised:
        _direct_inventory(divergent_git_graph)
    assert "metadata only" not in str(raised.value)
    assert hostile_name not in str(raised.value)


def test_git_inventory_rejects_metadata_above_one_mib_before_parsing(
    divergent_git_graph: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    real_run_git = inventory_module._run_git

    def oversized(root: Path, arguments: tuple[str, ...], *, stdout_limit: int) -> bytes:
        if arguments and arguments[0] == "diff-tree":
            raise inventory_module._GitOutputLimitError("bounded")
        return real_run_git(root, arguments, stdout_limit=stdout_limit)

    monkeypatch.setattr(inventory_module, "_run_git", oversized)
    with pytest.raises((TypeError, ValueError), match="metadata limit"):
        _direct_inventory(divergent_git_graph)


def test_git_metadata_stream_reads_only_limit_plus_one_sentinel_byte() -> None:
    import claimci.review.inventory as inventory_module

    class TrackingStream(io.BytesIO):
        consumed = 0

        def read(self, size: int = -1) -> bytes:
            value = super().read(size)
            self.consumed += len(value)
            return value

    stream = TrackingStream(b"x" * 100)
    sink = bytearray()
    overflow = inventory_module.threading.Event()
    inventory_module._drain_bounded(stream, 10, sink, overflow)

    assert stream.consumed == 11
    assert bytes(sink) == b"x" * 11
    assert overflow.is_set()


def test_git_inventory_rejects_more_than_8192_entries(
    divergent_git_graph: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    real_run_git = inventory_module._run_git
    encoded = b"".join(
        b"A\0entry-%04d.txt\0" % index for index in range(8_193)
    )

    def too_many(root: Path, arguments: tuple[str, ...], *, stdout_limit: int) -> bytes:
        if arguments and arguments[0] == "diff-tree":
            return encoded
        return real_run_git(root, arguments, stdout_limit=stdout_limit)

    monkeypatch.setattr(inventory_module, "_run_git", too_many)
    with pytest.raises((TypeError, ValueError), match="entry limit"):
        _direct_inventory(divergent_git_graph)


def test_git_inventory_never_reads_checkout_files_and_does_not_execute_extensions(
    divergent_git_graph: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.review.inventory as inventory_module
    import claimci.review.sources as sources_module

    marker = tmp_path / "must-not-run.txt"
    roots = divergent_git_graph["roots"]
    assert isinstance(roots, dict)
    head_root = roots["head"]
    assert isinstance(head_root, Path)
    external = tmp_path / "external-diff.cmd"
    external.write_text(f"@echo ran>{marker}\n", encoding="utf-8")
    _git(head_root, "config", "diff.external", str(external))
    _git(head_root, "config", "diff.danger.textconv", str(external))
    common_git_dir = Path(_git(head_root, "rev-parse", "--git-common-dir"))
    if not common_git_dir.is_absolute():
        common_git_dir = (head_root / common_git_dir).resolve()
    hook = common_git_dir / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"#!/bin/sh\necho ran > {str(marker)!r}\n", encoding="utf-8")
    os.chmod(hook, 0o700)

    def forbidden_capture(*args: object, **kwargs: object) -> object:
        raise AssertionError("inventory read checkout content")

    def forbidden_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("inventory used Path.read_*")

    monkeypatch.setattr(sources_module, "capture_confined_regular_file", forbidden_capture)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object):
        argv = args[0]
        assert isinstance(argv, tuple)
        calls.append((argv, kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(inventory_module.subprocess, "Popen", recording_popen)
    inventory = _direct_inventory(divergent_git_graph)
    assert inventory.entries
    assert not marker.exists()
    assert calls
    assert all(call_kwargs.get("shell") is False for _, call_kwargs in calls)
    assert any("--no-ext-diff" in argv for argv, _ in calls)
    assert all(call_kwargs.get("env", {}).get("GIT_TERMINAL_PROMPT") == "0" for _, call_kwargs in calls)
    assert all(call_kwargs.get("env", {}).get("GIT_OPTIONAL_LOCKS") == "0" for _, call_kwargs in calls)


def test_git_inventory_is_byte_for_byte_deterministic(
    divergent_git_graph: dict[str, object],
) -> None:
    first = _direct_inventory(divergent_git_graph)
    second = _direct_inventory(divergent_git_graph)

    def serialized(value: ChangeInventory) -> bytes:
        return json.dumps(
            dataclasses.asdict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    assert first == second
    assert serialized(first) == serialized(second)


def _filter_sentinel_repository(tmp_path: Path) -> tuple[Path, str]:
    root = (tmp_path / "filtered-repository").resolve()
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")
    _write(root, ".gitattributes", "filtered.txt filter=danger\n")
    _write(root, "filtered.txt", "clean\n")
    sha = _commit(root, "filtered base")
    return root, sha


def _command_argv(*values: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(values)
    return shlex.join(values)


def _remove_disposable_tree(path: Path) -> None:
    def make_writable_and_retry(function: object, name: str, _: object) -> None:
        os.chmod(name, 0o700)
        function(name)  # type: ignore[operator]

    shutil.rmtree(path, onerror=make_writable_and_retry)


@pytest.mark.parametrize("filter_mode", ["clean", "process"])
def test_git_cleanliness_never_invokes_configured_clean_or_process_filter(
    tmp_path: Path, filter_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    root, sha = _filter_sentinel_repository(tmp_path)
    marker = tmp_path / f"{filter_mode}-filter-ran.txt"
    helper = tmp_path / f"{filter_mode}-filter.py"
    helper.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('invoked', encoding='utf-8')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    command = _command_argv(sys.executable, str(helper), str(marker))
    _git(root, "config", f"filter.danger.{filter_mode}", command)
    _write(root, "filtered.txt", "dirty\n")
    calls: list[tuple[str, ...]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object):
        argv = args[0]
        assert isinstance(argv, tuple)
        calls.append(argv)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(inventory_module.subprocess, "Popen", recording_popen)

    with pytest.raises((TypeError, ValueError), match="dirty"):
        _same_snapshot_inventory(root, sha)
    assert not marker.exists()
    assert not any("status" in argv for argv in calls)


def test_git_cleanliness_does_not_traverse_dirty_submodule_worktree(
    tmp_path: Path,
) -> None:
    submodule_source = (tmp_path / "submodule-source").resolve()
    submodule_source.mkdir()
    _git(submodule_source, "init", "-q")
    _git(submodule_source, "config", "user.email", "tests@claimci.invalid")
    _git(submodule_source, "config", "user.name", "ClaimCI Tests")
    _write(submodule_source, ".gitattributes", "filtered.txt filter=danger\n")
    _write(submodule_source, "filtered.txt", "clean\n")
    _commit(submodule_source, "submodule base")

    root = (tmp_path / "superproject").resolve()
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(submodule_source),
        "deps/submodule",
    )
    sha = _commit(root, "superproject base")

    marker = tmp_path / "submodule-filter-ran.txt"
    helper = tmp_path / "submodule-filter.py"
    helper.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('invoked', encoding='utf-8')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    command = _command_argv(sys.executable, str(helper), str(marker))
    submodule_root = root / "deps" / "submodule"
    _git(submodule_root, "config", "filter.danger.clean", command)
    _write(submodule_root, "filtered.txt", "dirty\n")

    inventory = _same_snapshot_inventory(root, sha)
    assert inventory.entries == ()
    assert not marker.exists()


def test_git_inventory_ignores_malicious_replace_refs(
    divergent_git_graph: dict[str, object],
) -> None:
    repository = divergent_git_graph["repository"]
    direct = divergent_git_graph["direct"]
    requested = divergent_git_graph["requested"]
    assert isinstance(repository, Path)
    assert isinstance(direct, str) and isinstance(requested, str)
    expected = _direct_inventory(divergent_git_graph)

    _git(repository, "replace", direct, requested)
    try:
        assert _direct_inventory(divergent_git_graph) == expected
    finally:
        _git(repository, "replace", "-d", direct)


def test_git_inventory_never_lazy_fetches_a_promised_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    root = (tmp_path / "promisor-repository").resolve()
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")
    _write(root, "tracked.txt", "tracked\n")
    sha = _commit(root, "promised commit")

    marker = tmp_path / "remote-helper-ran.txt"
    helper_directory = tmp_path / "helpers"
    helper_directory.mkdir()
    helper = helper_directory / (
        "git-remote-sentinel.cmd" if os.name == "nt" else "git-remote-sentinel"
    )
    if os.name == "nt":
        helper.write_text(f"@echo invoked>{marker}\n@exit /b 1\n", encoding="utf-8")
    else:
        helper.write_text(f"#!/bin/sh\nprintf invoked > {shlex.quote(str(marker))}\nexit 1\n", encoding="utf-8")
        os.chmod(helper, 0o700)
    monkeypatch.setenv("PATH", str(helper_directory) + os.pathsep + os.environ["PATH"])
    _git(root, "config", "core.repositoryformatversion", "1")
    _git(root, "config", "extensions.partialclone", "origin")
    _git(root, "config", "remote.origin.url", "sentinel::missing")
    _git(root, "config", "remote.origin.promisor", "true")
    _git(root, "config", "remote.origin.partialclonefilter", "blob:none")

    object_path = root / ".git" / "objects" / sha[:2] / sha[2:]
    backup = tmp_path / "promised-object-backup"
    object_path.replace(backup)
    environments: list[dict[str, str]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object):
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        environments.append(environment)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(inventory_module.subprocess, "Popen", recording_popen)
    try:
        with pytest.raises((TypeError, ValueError)):
            _same_snapshot_inventory(root, sha)
        assert not marker.exists()
        assert environments
        assert all(env.get("GIT_NO_REPLACE_OBJECTS") == "1" for env in environments)
        assert all(env.get("GIT_NO_LAZY_FETCH") == "1" for env in environments)
    finally:
        backup.replace(object_path)


def test_git_runner_rejects_reader_that_remains_alive_after_join(
    divergent_git_graph: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    roots = divergent_git_graph["roots"]
    direct = divergent_git_graph["direct"]
    assert isinstance(roots, dict) and isinstance(direct, str)
    root = roots["direct"]
    assert isinstance(root, Path)
    release = inventory_module.threading.Event()
    original_drain = inventory_module._drain_bounded

    def delayed_drain(
        stream: object,
        limit: int,
        sink: bytearray,
        overflow: object,
    ) -> None:
        release.wait(timeout=5.0)
        original_drain(stream, limit, sink, overflow)  # type: ignore[arg-type]

    monkeypatch.setattr(inventory_module, "_drain_bounded", delayed_drain)
    try:
        with pytest.raises(inventory_module._GitCommandError, match="reader"):
            inventory_module._run_git(
                root,
                ("cat-file", "-e", f"{direct}^{{commit}}"),
                stdout_limit=0,
            )
    finally:
        release.set()


def test_git_inventory_retains_added_modified_deleted_gitlinks_without_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claimci.review.inventory as inventory_module

    submodule_source = (tmp_path / "gitlink-source").resolve()
    submodule_source.mkdir()
    _git(submodule_source, "init", "-q")
    _git(submodule_source, "config", "user.email", "tests@claimci.invalid")
    _git(submodule_source, "config", "user.name", "ClaimCI Tests")
    _write(submodule_source, "tracked.txt", "version one\n")
    gitlink_v1 = _commit(submodule_source, "gitlink v1")
    _write(submodule_source, "tracked.txt", "version two\n")
    gitlink_v2 = _commit(submodule_source, "gitlink v2")

    root = (tmp_path / "superproject").resolve()
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")
    for relative in ("deps/deleted", "deps/modified"):
        _git(
            root,
            "-c",
            "protocol.file.allow=always",
            "clone",
            "-q",
            str(submodule_source),
            relative,
        )
        _git(root / relative, "checkout", "-q", gitlink_v1)
    _write(
        root,
        ".gitmodules",
        "[submodule \"deps/added\"]\n"
        "\tpath = deps/added\n"
        f"\turl = {submodule_source.as_posix()}\n"
        "\tignore = all\n"
        "[submodule \"deps/deleted\"]\n"
        "\tpath = deps/deleted\n"
        f"\turl = {submodule_source.as_posix()}\n"
        "\tignore = all\n"
        "[submodule \"deps/modified\"]\n"
        "\tpath = deps/modified\n"
        f"\turl = {submodule_source.as_posix()}\n"
        "\tignore = all\n",
    )
    base = _commit(root, "base gitlinks")
    base_root = (tmp_path / "base-snapshot").resolve()
    _git(root, "worktree", "add", "-q", "--detach", str(base_root), base)

    _git(root / "deps" / "modified", "checkout", "-q", gitlink_v2)
    _remove_disposable_tree(root / "deps" / "deleted")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "clone",
        "-q",
        str(submodule_source),
        "deps/added",
    )
    _git(root / "deps" / "added", "checkout", "-q", gitlink_v2)
    head = _commit(root, "head gitlinks")

    marker = tmp_path / "gitlink-worktree-executed.txt"
    filter_helper = tmp_path / "gitlink-filter.py"
    filter_helper.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('filter', encoding='utf-8')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    filter_command = _command_argv(sys.executable, str(filter_helper), str(marker))
    nested_roots = tuple((root / relative).resolve() for relative in ("deps/added", "deps/modified"))
    for nested_root in nested_roots:
        _git(nested_root, "config", "filter.danger.clean", filter_command)
        _write(nested_root, ".gitattributes", "tracked.txt filter=danger\n")
        _write(nested_root, "tracked.txt", "dirty worktree\n")
        hook = nested_root / ".git" / "hooks" / "pre-commit"
        hook.write_text(
            f"#!/bin/sh\nprintf hook > {shlex.quote(str(marker))}\nexit 91\n",
            encoding="utf-8",
        )
        os.chmod(hook, 0o700)

    git_cwds: list[Path] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object):
        cwd = kwargs.get("cwd")
        assert isinstance(cwd, str)
        git_cwds.append(Path(cwd).resolve())
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(inventory_module.subprocess, "Popen", recording_popen)
    inventory = build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, base_root, base),
        _identity(SnapshotRole.COMPARISON_BASE, base_root, base),
        _identity(SnapshotRole.HEAD, root, head),
        ComparisonBasis.DIRECT_BASE,
    )

    assert inventory.complete is True
    assert inventory.entries == (
        ChangeEntry("deps/added", ChangeStatus.ADDED),
        ChangeEntry("deps/deleted", ChangeStatus.DELETED),
        ChangeEntry("deps/modified", ChangeStatus.MODIFIED),
    )
    assert not marker.exists()
    assert not set(git_cwds).intersection(nested_roots)
