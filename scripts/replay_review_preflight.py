"""Replay the five frozen Research Review preflights without a provider.

Only trusted local Git object graphs are copied.  Repository files are never
executed, no provider adapter is imported, and the frozen study tree remains
read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

# A directly executed script otherwise gives its ``scripts`` directory import
# priority, which can accidentally resolve an unrelated editable checkout.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT_TEXT = str(_WORKSPACE_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _WORKSPACE_ROOT_TEXT]
sys.path.insert(0, _WORKSPACE_ROOT_TEXT)

from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    ProviderLifecycle,
    ReviewConfig,
    ReviewError,
    ReviewPreflight,
    SnapshotIdentity,
    SnapshotRole,
)
from claimci.review.config import load_review_config
from claimci.review.inventory import build_git_change_inventory
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.preflight import plan_review_supplement_paths


MAX_MANIFEST_BYTES = 1_048_576
MAX_METADATA_BYTES = 1_048_576
MAX_GIT_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_GIT_METADATA_BYTES = 256 * 1024 * 1024
MAX_GIT_METADATA_ENTRIES = 250_000
MAX_GIT_METADATA_PATH_BYTES = 64 * 1024 * 1024
MAX_GIT_OBJECT_ENTRIES = 250_000
MAX_GIT_OBJECT_PATH_BYTES = 64 * 1024 * 1024
MAX_GIT_OBJECT_INFO_BYTES = 16 * 1024 * 1024
MAX_MATERIAL_BLOB_BYTES = 16 * 1024 * 1024
MAX_GIT_ERROR_BYTES = 65_536
GIT_TIMEOUT_SECONDS = 120.0
_STUDY_NAME = re.compile(r"ClaimCI-External-Study-\d{4}-\d{2}-\d{2}\Z")
_HEX_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CANDIDATE_NAME = re.compile(r"[A-Z]-[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_REPOSITORY_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
@dataclass(frozen=True)
class FrozenCandidate:
    schema_version: int
    candidate: str
    repository: str
    pull_request: int
    requested_base_sha: str
    comparison_base_sha: str
    head_sha: str
    comparison_basis: ComparisonBasis
    declared_entry_count: int
    entries: tuple[ChangeEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("frozen candidate schema_version must be 1")
        if not isinstance(self.candidate, str) or not _CANDIDATE_NAME.fullmatch(
            self.candidate
        ):
            raise ValueError("frozen candidate name is invalid")
        if not isinstance(self.repository, str) or not _REPOSITORY_NAME.fullmatch(
            self.repository
        ):
            raise ValueError("frozen repository name is invalid")
        if (
            isinstance(self.pull_request, bool)
            or not isinstance(self.pull_request, int)
            or self.pull_request < 1
        ):
            raise ValueError("frozen pull request number is invalid")
        if not isinstance(self.comparison_basis, ComparisonBasis):
            raise ValueError("frozen comparison basis is invalid")
        if (
            isinstance(self.declared_entry_count, bool)
            or not isinstance(self.declared_entry_count, int)
            or self.declared_entry_count != len(self.entries)
        ):
            raise ValueError("frozen entry count is inconsistent")
        # Constructing the production contract supplies all SHA, path, status,
        # ordering, uniqueness, and direct-base consistency validation.
        self.inventory

    @property
    def inventory(self) -> ChangeInventory:
        return ChangeInventory(
            schema_version=1,
            requested_base_sha=self.requested_base_sha,
            comparison_base_sha=self.comparison_base_sha,
            head_sha=self.head_sha,
            comparison_basis=self.comparison_basis,
            source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
            declared_entry_count=self.declared_entry_count,
            complete=True,
            entries=self.entries,
        )


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("study layout path must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("study layout path must be a portable relative path")
    if path.as_posix() != value:
        raise ValueError("study layout path must be canonical")
    return value


@dataclass(frozen=True)
class CandidateInputLayout:
    source_relative: str
    metadata_relative: str
    metadata_kind: str
    body_relative: str | None
    body_capture: str
    body_kind: str = "text"
    body_start_line: int | None = None
    body_end_line: int | None = None

    def __post_init__(self) -> None:
        for value in (
            self.source_relative,
            self.metadata_relative,
            self.body_relative,
        ):
            if value is not None:
                _safe_relative(value)
        if self.metadata_kind not in {
            "capture_json",
            "frozen_coordinates",
            "phase1_markdown",
            "pr_json",
        }:
            raise ValueError("candidate metadata kind is invalid")
        if self.body_kind not in {"text", "pr_json"}:
            raise ValueError("candidate body kind is invalid")
        if self.body_capture not in {"raw_pr_body", "analyst_summary"}:
            raise ValueError("candidate body capture provenance is invalid")
        if (self.body_start_line is None) != (self.body_end_line is None):
            raise ValueError("candidate body line range is incomplete")
        if self.body_start_line is not None and (
            self.body_start_line < 1 or self.body_end_line < self.body_start_line
        ):
            raise ValueError("candidate body line range is invalid")


DEFAULT_INPUT_LAYOUTS: Mapping[str, CandidateInputLayout] = {
    "A-InferenceX-2630": CandidateInputLayout(
        source_relative="phase1/A-InferenceX-2630/repo",
        metadata_relative="phase1/A-InferenceX-2630/capture.json",
        metadata_kind="capture_json",
        body_relative="phase1/A-InferenceX-2630/phase1.md",
        body_capture="analyst_summary",
        body_start_line=43,
        body_end_line=75,
    ),
    "B-Transformers-48455": CandidateInputLayout(
        source_relative="paid/B-Transformers-48455/head",
        metadata_relative="paid/B-Transformers-48455/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/B-Transformers-48455/inputs/pr-description.md",
        body_capture="raw_pr_body",
        body_start_line=5,
        body_end_line=153,
    ),
    "C-Weave-7801": CandidateInputLayout(
        source_relative="paid/C-Weave-7801/head",
        metadata_relative="phase1/C-Weave-7801/pr.json",
        metadata_kind="pr_json",
        body_relative="phase1/C-Weave-7801/pr.json",
        body_capture="raw_pr_body",
        body_kind="pr_json",
    ),
    "D-OpenASR-205": CandidateInputLayout(
        source_relative="phase1/D-OpenASR-205/repo",
        metadata_relative="phase1/D-OpenASR-205/phase1.md",
        metadata_kind="phase1_markdown",
        body_relative="phase1/D-OpenASR-205/phase1.md",
        body_capture="analyst_summary",
        body_start_line=5,
        body_end_line=60,
    ),
    "E-NVCF-1425": CandidateInputLayout(
        source_relative="paid/E-NVCF-1425/head",
        metadata_relative="paid/E-NVCF-1425/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/E-NVCF-1425/inputs/pr-description.md",
        body_capture="raw_pr_body",
        body_start_line=1,
        body_end_line=87,
    ),
}


def _read_bounded(path: Path, *, limit: int = MAX_METADATA_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen metadata file is missing or unsafe")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("frozen metadata file exceeds its fixed bound")
    return value


def _unique_json(raw: bytes) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("frozen JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen JSON is invalid") from exc


def _candidate_from_payload(payload: object, filename: str) -> FrozenCandidate:
    if not isinstance(payload, dict):
        raise ValueError("frozen candidate manifest must be an object")
    expected = {
        "schema_version",
        "candidate",
        "repository",
        "pull_request",
        "requested_base_sha",
        "comparison_base_sha",
        "head_sha",
        "comparison_basis",
        "declared_entry_count",
        "entries",
    }
    if set(payload) != expected:
        raise ValueError("frozen candidate manifest fields are invalid")
    if payload["candidate"] != filename.removesuffix(".json"):
        raise ValueError("frozen candidate filename is inconsistent")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("frozen candidate entries must be a list")
    entries: list[ChangeEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"path", "status"}:
            raise ValueError("frozen change entry fields are invalid")
        try:
            entries.append(
                ChangeEntry(
                    path=raw_entry["path"],
                    status=ChangeStatus(raw_entry["status"]),
                )
            )
        except (ReviewError, TypeError, ValueError) as exc:
            raise ValueError("frozen change entry is invalid") from exc
    try:
        basis = ComparisonBasis(payload["comparison_basis"])
        return FrozenCandidate(
            schema_version=payload["schema_version"],
            candidate=payload["candidate"],
            repository=payload["repository"],
            pull_request=payload["pull_request"],
            requested_base_sha=payload["requested_base_sha"],
            comparison_base_sha=payload["comparison_base_sha"],
            head_sha=payload["head_sha"],
            comparison_basis=basis,
            declared_entry_count=payload["declared_entry_count"],
            entries=tuple(entries),
        )
    except (ReviewError, TypeError, ValueError) as exc:
        raise ValueError("frozen candidate manifest is invalid") from exc


def load_frozen_manifests(directory: Path) -> tuple[FrozenCandidate, ...]:
    try:
        root = Path(directory).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("frozen manifest directory is invalid") from exc
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen manifest directory is invalid")
    files = tuple(sorted(root.glob("*.json"), key=lambda item: item.name))
    if not files:
        raise ValueError("no frozen candidate manifests were found")
    candidates = tuple(
        _candidate_from_payload(
            _unique_json(_read_bounded(path, limit=MAX_MANIFEST_BYTES)), path.name
        )
        for path in files
    )
    if len({candidate.candidate for candidate in candidates}) != len(candidates):
        raise ValueError("frozen candidate manifests are duplicated")
    return candidates


def _looks_remote(value: object) -> bool:
    text = os.fspath(value)
    lowered = text.casefold()
    return (
        "://" in lowered
        or lowered.startswith(("git@", "ssh:", "git:", "file:"))
        or text.startswith(("\\\\", "//"))
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_replay_paths(
    study_root: str | os.PathLike[str], output: str | os.PathLike[str]
) -> tuple[Path, Path]:
    if _looks_remote(study_root):
        raise ValueError("study root must be a local filesystem path")
    if _looks_remote(output):
        raise ValueError("output must be a local filesystem path")
    try:
        study = Path(study_root)
        if study.is_symlink():
            raise ValueError
        study = study.resolve(strict=True)
        if not study.is_dir() or not _STUDY_NAME.fullmatch(study.name):
            raise ValueError
        destination = Path(output)
        if destination.exists() and destination.is_symlink():
            raise ValueError
        unresolved_destination = destination.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("study root or output local path is invalid") from exc
    if _is_within(unresolved_destination, study):
        raise ValueError("replay output must be outside the frozen study tree")
    try:
        parent = destination.parent.resolve(strict=True)
        destination = parent / destination.name
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("study root or output local path is invalid") from exc
    return study, destination


def _confined_study_path(study: Path, relative: str, *, directory: bool) -> Path:
    _safe_relative(relative)
    candidate = study / Path(relative)
    if candidate.is_symlink():
        raise ValueError("frozen study path is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(study)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("frozen study path is unavailable") from exc
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise ValueError("frozen study path has the wrong type")
    return resolved


def _git_environment(
    config_pairs: Sequence[tuple[str, str]], *, global_config: Path
) -> dict[str, str]:
    env = {
        name: value
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
        )
        if (value := os.environ.get(name)) is not None
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(global_config),
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_COUNT": str(len(config_pairs)),
        }
    )
    for index, (key, value) in enumerate(config_pairs):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


def _trusted_global_git_config(empty_hooks: Path) -> Path:
    """Create the server-visible filter capability config in disposable space."""

    path = empty_hooks.parent / "git-global-config"
    expected = "[uploadpack]\n\tallowFilter = true\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != expected:
            raise ValueError("temporary Git configuration is invalid")
    else:
        path.write_text(expected, encoding="utf-8", newline="\n")
    return path.resolve(strict=True)


def _git_config(empty_hooks: Path) -> tuple[tuple[str, str], ...]:
    return (
        ("protocol.allow", "never"),
        ("protocol.file.allow", "always"),
        ("uploadpack.allowFilter", "true"),
        ("core.hooksPath", str(empty_hooks)),
        ("core.attributesFile", os.devnull),
        ("core.fsmonitor", "false"),
        ("core.untrackedCache", "false"),
        ("core.preloadIndex", "false"),
        ("submodule.recurse", "false"),
        ("fetch.recurseSubmodules", "false"),
        ("filter.lfs.required", "false"),
        ("filter.lfs.smudge", ""),
        ("filter.lfs.clean", ""),
        ("filter.lfs.process", ""),
    )


def _run_git(
    cwd: Path,
    arguments: Sequence[str],
    *,
    empty_hooks: Path,
    output_limit: int = MAX_GIT_OUTPUT_BYTES,
    input_bytes: bytes | None = None,
) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("local Git executable is unavailable")
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def drain(stream: Any, limit: int, sink: bytearray) -> None:
        try:
            while len(sink) <= limit:
                chunk = stream.read(min(65_536, limit + 1 - len(sink)))
                if not chunk:
                    return
                sink.extend(chunk)
                if len(sink) > limit:
                    overflow.set()
                    return
        finally:
            stream.close()

    def feed(stream: Any, value: bytes) -> None:
        try:
            stream.write(value)
            stream.flush()
        except BrokenPipeError:
            pass
        finally:
            stream.close()

    try:
        process = subprocess.Popen(
            [str(Path(executable).resolve()), "--no-pager", *arguments],
            cwd=cwd,
            env=_git_environment(
                _git_config(empty_hooks),
                global_config=_trusted_global_git_config(empty_hooks),
            ),
            stdin=(subprocess.DEVNULL if input_bytes is None else subprocess.PIPE),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("local Git operation failed") from exc
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise ValueError("local Git pipes are unavailable")
    readers = (
        threading.Thread(target=drain, args=(process.stdout, output_limit, stdout), daemon=True),
        threading.Thread(
            target=drain,
            args=(process.stderr, MAX_GIT_ERROR_BYTES, stderr),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    writer: threading.Thread | None = None
    if input_bytes is not None:
        if process.stdin is None:
            process.kill()
            raise ValueError("local Git input pipe is unavailable")
        writer = threading.Thread(
            target=feed, args=(process.stdin, input_bytes), daemon=True
        )
        writer.start()

    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    timed_out = False
    while process.poll() is None:
        if overflow.is_set() or time.monotonic() >= deadline:
            timed_out = time.monotonic() >= deadline
            process.kill()
            break
        time.sleep(0.01)
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait(timeout=1.0)
        timed_out = True
    for reader in readers:
        reader.join(timeout=1.0)
    if writer is not None:
        writer.join(timeout=1.0)
    if any(reader.is_alive() for reader in readers) or (
        writer is not None and writer.is_alive()
    ):
        raise ValueError("local Git pipe did not terminate")
    if overflow.is_set():
        raise ValueError("local Git output exceeded its fixed bound")
    if timed_out or return_code != 0:
        raise ValueError("local Git operation failed")
    return bytes(stdout)


def _copy_snapshot_blobs(
    source: Path,
    clone: Path,
    candidate: FrozenCandidate,
    paths: Sequence[str],
    empty_hooks: Path,
) -> dict[str, tuple[str, ...]]:
    """Hydrate bounded exact blobs needed by each disposable snapshot."""

    role_shas = (
        ("requested_base", candidate.requested_base_sha),
        ("comparison_base", candidate.comparison_base_sha),
        ("head", candidate.head_sha),
    )
    role_paths: dict[str, list[str]] = {role: [] for role, _sha in role_shas}
    copied_objects: set[str] = set()
    material_paths = tuple(dict.fromkeys(paths))
    for role, sha in role_shas:
        for path in material_paths:
            _safe_relative(path)
            raw = _run_git(
                source,
                ("ls-tree", "-z", "--full-tree", sha, "--", path),
                empty_hooks=empty_hooks,
                output_limit=8_192,
            )
            if not raw:
                continue
            records = tuple(record for record in raw.split(b"\0") if record)
            if len(records) != 1:
                raise ValueError("declared material tree identity is ambiguous")
            try:
                header, raw_path = records[0].split(b"\t", 1)
                mode, object_type, raw_object_id = header.split(b" ", 2)
                actual_path = raw_path.decode("utf-8", errors="strict")
                object_id = raw_object_id.decode("ascii", errors="strict")
            except (UnicodeError, ValueError) as exc:
                raise ValueError("declared material tree identity is invalid") from exc
            if (
                actual_path != path
                or object_type != b"blob"
                or mode not in {b"100644", b"100755"}
                or not _HEX_OBJECT.fullmatch(object_id)
            ):
                raise ValueError("declared material is not a regular Git blob")
            role_paths[role].append(path)
            if object_id in copied_objects:
                continue
            content = _run_git(
                source,
                ("cat-file", "blob", object_id),
                empty_hooks=empty_hooks,
                output_limit=MAX_MATERIAL_BLOB_BYTES + 1,
            )
            if len(content) > MAX_MATERIAL_BLOB_BYTES:
                raise ValueError("declared material blob exceeds its fixed clone bound")
            copied = _run_git(
                clone,
                ("hash-object", "-w", "--stdin"),
                empty_hooks=empty_hooks,
                output_limit=256,
                input_bytes=content,
            ).decode("ascii", errors="strict").strip()
            if copied != object_id:
                raise ValueError("declared material blob identity changed during copy")
            copied_objects.add(object_id)
    return {role: tuple(paths) for role, paths in role_paths.items()}


def _sparse_patterns(paths: Sequence[str]) -> bytes:
    values = tuple(paths) or (".claimci-no-material",)
    patterns: list[str] = []
    for value in values:
        _safe_relative(value)
        escaped = re.sub(r"([\\*?\[\]#!])", r"\\\1", value)
        patterns.append(f"/{escaped}")
    return ("\n".join(patterns) + "\n").encode("utf-8")


def _stabilize_disposable_index(root: Path, empty_hooks: Path) -> None:
    encoded_path = _run_git(
        root,
        ("rev-parse", "--path-format=absolute", "--git-path", "index"),
        empty_hooks=empty_hooks,
        output_limit=4_096,
    )
    try:
        index_path = Path(encoded_path.decode("utf-8", errors="strict").strip())
        metadata = index_path.lstat()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("disposable snapshot index is unavailable") from exc
    if (
        not index_path.is_absolute()
        or index_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ValueError("disposable snapshot index is unavailable")
    stabilized_time = max(metadata.st_mtime_ns, time.time_ns()) + 1_000_000_000
    try:
        timestamps = (metadata.st_atime_ns, stabilized_time)
        if os.utime in os.supports_follow_symlinks:
            os.utime(index_path, ns=timestamps, follow_symlinks=False)
        else:
            os.utime(index_path, ns=timestamps)
    except (NotImplementedError, OSError) as exc:
        raise ValueError("disposable snapshot index is unavailable") from exc


def _add_sparse_worktree(
    clone: Path,
    root: Path,
    sha: str,
    paths: Sequence[str],
    empty_hooks: Path,
) -> None:
    _run_git(
        clone,
        ("worktree", "add", "--detach", "--no-checkout", "--", str(root), sha),
        empty_hooks=empty_hooks,
    )
    _run_git(
        root,
        ("sparse-checkout", "set", "--no-cone", "--stdin"),
        empty_hooks=empty_hooks,
        input_bytes=_sparse_patterns(paths),
    )
    _run_git(
        root,
        ("checkout", "--detach", sha),
        empty_hooks=empty_hooks,
    )
    _stabilize_disposable_index(root, empty_hooks)


def _set_sparse_worktree_paths(
    root: Path,
    paths: Sequence[str],
    empty_hooks: Path,
) -> None:
    """Expand one disposable sparse worktree to an exact bounded path set."""

    _run_git(
        root,
        ("sparse-checkout", "set", "--no-cone", "--stdin"),
        empty_hooks=empty_hooks,
        input_bytes=_sparse_patterns(paths),
    )
    _stabilize_disposable_index(root, empty_hooks)


def _reject_unsafe_source_config(source: Path, empty_hooks: Path) -> None:
    raw = _run_git(
        source,
        ("config", "--local", "--no-includes", "--null", "--list"),
        empty_hooks=empty_hooks,
        output_limit=MAX_METADATA_BYTES,
    )
    text = raw.decode("utf-8", errors="strict")
    unsafe_prefixes = (
        "include.",
        "includeif.",
        "uploadpack.packobjectshook",
        "core.alternaterefscommand",
        "core.fsmonitor",
        "core.sshcommand",
        "credential.",
        "filter.",
    )
    for record in text.split("\0"):
        key = record.split("\n", 1)[0].split("=", 1)[0].casefold()
        if key.startswith(unsafe_prefixes):
            raise ValueError("frozen source Git configuration is not passive")


def _git_head(root: Path, empty_hooks: Path) -> str:
    value = _run_git(
        root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        empty_hooks=empty_hooks,
        output_limit=256,
    ).decode("ascii", errors="strict").strip()
    if not _HEX_OBJECT.fullmatch(value):
        raise ValueError("local Git HEAD identity is invalid")
    return value


def _git_dir(root: Path, argument: str, empty_hooks: Path) -> Path:
    raw = _run_git(
        root,
        ("rev-parse", argument),
        empty_hooks=empty_hooks,
        output_limit=8_192,
    ).decode("utf-8", errors="strict").strip()
    try:
        candidate = Path(raw)
        path = (
            candidate.resolve(strict=True)
            if candidate.is_absolute()
            else (root / candidate).resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("local Git metadata path is invalid") from exc
    if not path.is_dir():
        raise ValueError("local Git metadata path is invalid")
    return path


def _digest_metadata_tree(root: Path, *, namespace: str, sink: Any) -> tuple[int, int]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError("Git metadata is unavailable") from exc
    root_attributes = getattr(root_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    root_is_junction = getattr(root, "is_junction", lambda: False)()
    if (
        root.is_symlink()
        or root_is_junction
        or (reparse_flag and root_attributes & reparse_flag)
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise ValueError("Git metadata root is unsafe")

    files: list[tuple[str, Path]] = []
    directories = [root]
    entry_count = 0
    path_bytes = 0
    while directories:
        directory = directories.pop()
        children: list[Path] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    path_bytes += len(relative.encode("utf-8", errors="strict"))
                    entry_count += 1
                    if entry_count > MAX_GIT_METADATA_ENTRIES:
                        raise ValueError("Git metadata exceeds its entry bound")
                    if path_bytes > MAX_GIT_METADATA_PATH_BYTES:
                        raise ValueError("Git metadata exceeds its path bound")
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise ValueError("Git metadata is unavailable") from exc
                    attributes = getattr(entry_stat, "st_file_attributes", 0)
                    is_junction = getattr(path, "is_junction", lambda: False)()
                    if (
                        entry.is_symlink()
                        or is_junction
                        or (reparse_flag and attributes & reparse_flag)
                    ):
                        raise ValueError("Git metadata contains a link")
                    if directory == root and entry.name == "objects":
                        if not stat.S_ISDIR(entry_stat.st_mode):
                            raise ValueError("Git object store is unsafe")
                        continue
                    if stat.S_ISDIR(entry_stat.st_mode):
                        children.append(path)
                    elif stat.S_ISREG(entry_stat.st_mode):
                        files.append((relative, path))
                    else:
                        raise ValueError("Git metadata contains a special entry")
        except OSError as exc:
            raise ValueError("Git metadata is unavailable") from exc
        directories.extend(
            reversed(sorted(children, key=lambda child: child.name))
        )

    count = 0
    total = 0
    for relative, path in sorted(files, key=lambda item: item[0]):
        raw = _read_bounded(path, limit=MAX_GIT_METADATA_BYTES - total)
        total += len(raw)
        sink.update(namespace.encode("utf-8") + b"\0")
        sink.update(relative.encode("utf-8") + b"\0")
        sink.update(str(len(raw)).encode("ascii") + b"\0")
        sink.update(raw)
        count += 1
    return count, total


def _digest_object_store(root: Path) -> tuple[int, int, int, str]:
    """Passively inventory Git objects without opening object payloads.

    Paths, entry types, and sizes detect loose-object and pack additions while
    keeping reads independent of repository size.  Only ``objects/info`` files
    are content-hashed because their contents can redirect or alter object
    lookup behavior.
    """

    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError("Git object store is unavailable") from exc
    root_attributes = getattr(root_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    root_is_junction = getattr(root, "is_junction", lambda: False)()
    if (
        root.is_symlink()
        or root_is_junction
        or (reparse_flag and root_attributes & reparse_flag)
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise ValueError("Git object store is unsafe")

    records: list[tuple[str, str, int, Path]] = []
    directories = [root]
    path_bytes = 0
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise ValueError("Git object inventory is unavailable") from exc
                    attributes = getattr(entry_stat, "st_file_attributes", 0)
                    is_junction = getattr(path, "is_junction", lambda: False)()
                    if (
                        entry.is_symlink()
                        or is_junction
                        or (reparse_flag and attributes & reparse_flag)
                    ):
                        raise ValueError("Git object store contains a link")
                    if stat.S_ISDIR(entry_stat.st_mode):
                        kind = "directory"
                        directories.append(path)
                    elif stat.S_ISREG(entry_stat.st_mode):
                        kind = "file"
                    else:
                        raise ValueError("Git object store contains a special entry")
                    relative = path.relative_to(root).as_posix()
                    encoded_relative = relative.encode("utf-8", errors="strict")
                    path_bytes += len(encoded_relative)
                    records.append((relative, kind, entry_stat.st_size, path))
                    if len(records) > MAX_GIT_OBJECT_ENTRIES:
                        raise ValueError("Git object inventory exceeds its entry bound")
                    if path_bytes > MAX_GIT_OBJECT_PATH_BYTES:
                        raise ValueError("Git object inventory exceeds its path bound")
        except OSError as exc:
            raise ValueError("Git object inventory is unavailable") from exc

    digest = hashlib.sha256()
    info_bytes = 0
    for relative, kind, size, path in sorted(records, key=lambda record: record[0]):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(kind.encode("ascii") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        if kind == "file" and relative.startswith("info/"):
            remaining = MAX_GIT_OBJECT_INFO_BYTES - info_bytes
            if remaining < 0 or size > remaining:
                raise ValueError("Git object info exceeds its content bound")
            raw = _read_bounded(path, limit=remaining)
            if len(raw) != size:
                raise ValueError("Git object info changed during identity capture")
            info_bytes += len(raw)
            digest.update(b"info-content\0" + raw)
    return len(records), path_bytes, info_bytes, digest.hexdigest()


def _source_identity(source: Path, empty_hooks: Path) -> dict[str, Any]:
    top = _run_git(
        source,
        ("rev-parse", "--show-toplevel"),
        empty_hooks=empty_hooks,
        output_limit=8_192,
    ).decode("utf-8", errors="strict").strip()
    try:
        if Path(top).resolve(strict=True) != source:
            raise ValueError
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("frozen source must be a Git worktree root") from exc
    status = _run_git(
        source,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        empty_hooks=empty_hooks,
    )
    if status:
        raise ValueError("frozen source worktree is dirty")
    tracked = _run_git(
        source,
        ("ls-files", "--stage", "-z"),
        empty_hooks=empty_hooks,
    )
    tracked_rows = tuple(row for row in tracked.split(b"\0") if row)
    git_dir = _git_dir(source, "--absolute-git-dir", empty_hooks)
    common_dir = _git_dir(source, "--git-common-dir", empty_hooks)
    marker = source / ".git"
    marker_bytes = (
        _read_bounded(marker, limit=8_192)
        if marker.is_file() and not marker.is_symlink()
        else b"directory"
    )
    digest = hashlib.sha256()
    digest.update(b"git-marker\0" + marker_bytes)
    roots = (("common", common_dir),)
    if git_dir != common_dir:
        roots += (("worktree", git_dir),)
    metadata_files = 0
    metadata_bytes = 0
    for namespace, root in roots:
        count, size = _digest_metadata_tree(root, namespace=namespace, sink=digest)
        metadata_files += count
        metadata_bytes += size
    (
        object_entries,
        object_path_bytes,
        object_info_bytes,
        object_digest,
    ) = _digest_object_store(common_dir / "objects")
    return {
        "head_sha": _git_head(source, empty_hooks),
        "porcelain_status_sha256": hashlib.sha256(status).hexdigest(),
        "porcelain_status_entries": len(tuple(row for row in status.split(b"\0") if row)),
        "tracked_file_count": len(tracked_rows),
        "tracked_file_identity_sha256": hashlib.sha256(tracked).hexdigest(),
        "git_metadata_file_count": metadata_files,
        "git_metadata_bytes": metadata_bytes,
        "git_metadata_sha256": digest.hexdigest(),
        "git_object_store_entry_count": object_entries,
        "git_object_store_path_bytes": object_path_bytes,
        "git_object_store_info_bytes": object_info_bytes,
        "git_object_store_sha256": object_digest,
    }


def _read_pr_metadata(
    study: Path, layout: CandidateInputLayout
) -> tuple[str, str, dict[str, Any]]:
    metadata_path = _confined_study_path(
        study, layout.metadata_relative, directory=False
    )
    metadata_raw = _read_bounded(metadata_path)
    metadata_text = metadata_raw.decode("utf-8", errors="strict")
    metadata_json: dict[str, Any] | None = None
    if layout.metadata_kind in {"capture_json", "pr_json"}:
        parsed = _unique_json(metadata_raw)
        if not isinstance(parsed, dict):
            raise ValueError("frozen PR metadata must be an object")
        metadata_json = parsed
        title = parsed.get("title")
    elif layout.metadata_kind == "frozen_coordinates":
        match = re.search(r"(?m)^- Title: `([^\r\n`]*)`\s*$", metadata_text)
        title = None if match is None else match.group(1)
    else:
        match = re.search(r"#\d+\s+[—-]\s+([^\]\r\n]+)", metadata_text)
        title = None if match is None else match.group(1).strip()
    if not isinstance(title, str) or not title.strip() or len(title) > 16_000:
        raise ValueError("frozen PR title is unavailable or invalid")

    if layout.body_relative is None:
        body = ""
        body_sha = hashlib.sha256(b"").hexdigest()
    else:
        body_path = _confined_study_path(study, layout.body_relative, directory=False)
        body_raw = _read_bounded(body_path)
        if layout.body_kind == "pr_json":
            body_payload = metadata_json if body_path == metadata_path else _unique_json(body_raw)
            if not isinstance(body_payload, dict) or not isinstance(
                body_payload.get("body"), str
            ):
                raise ValueError("frozen PR body is unavailable or invalid")
            body = body_payload["body"]
        else:
            body_text = body_raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
            lines = body_text.split("\n")
            if layout.body_start_line is None:
                body = body_text
            else:
                if layout.body_end_line > len(lines):
                    raise ValueError("frozen body line range exceeds its source")
                body = "\n".join(
                    lines[layout.body_start_line - 1 : layout.body_end_line]
                )
        if len(body) > 60_000:
            raise ValueError("frozen PR body exceeds the preflight input bound")
        body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()

    provenance: dict[str, Any] = {
        "title_source": layout.metadata_relative,
        "title_source_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "body_source": layout.body_relative,
        "body_capture": layout.body_capture,
        "body_sha256": body_sha,
    }
    if layout.body_start_line is not None:
        provenance["body_line_range"] = [
            layout.body_start_line,
            layout.body_end_line,
        ]
    return title, body, provenance


def serialize_preflight(preflight: ReviewPreflight) -> dict[str, Any]:
    scope = preflight.scope
    return {
        "schema_version": preflight.schema_version,
        "coordinates": {
            "requested_base_sha": preflight.requested_base_sha,
            "comparison_base_sha": preflight.comparison_base_sha,
            "head_sha": preflight.head_sha,
            "comparison_basis": (
                None
                if preflight.comparison_basis is None
                else preflight.comparison_basis.value
            ),
        },
        "gates": [
            {
                "gate": gate.gate,
                "disposition": gate.disposition.value,
                "issues": [reason.as_dict() for reason in gate.reasons],
                "metrics": dict(gate.metrics),
            }
            for gate in preflight.gates
        ],
        "ready_for_provider": preflight.ready_for_provider,
        "review_status_ceiling": preflight.review_status_ceiling.value,
        "scope": (
            None
            if scope is None
            else {
                "mode": scope.mode,
                "complete": scope.complete,
                "inventory_entry_count": len(scope.inventory.entries),
                "issued_paths": list(scope.issued_paths),
                "issued_changed_paths": list(scope.issued_changed_paths),
                "selected_paths": list(scope.selected_paths),
                "materialized_chars": scope.materialized_chars,
                "materialized_path_chars": [list(item) for item in scope.materialized_path_chars],
                "issues": [issue.as_dict() for issue in scope.issues],
            }
        ),
    }


def _serialize_review_config(config: ReviewConfig) -> dict[str, Any]:
    limits = config.limits
    return {
        "schema_version": config.schema_version,
        "enabled": config.enabled,
        "policy": config.policy,
        "provider": config.provider,
        "model": config.model,
        "limits": {
            "max_calls": limits.max_calls,
            "max_context_chars": limits.max_context_chars,
            "max_output_chars": limits.max_output_chars,
            "max_files": limits.max_files,
            "max_file_chars": limits.max_file_chars,
            "max_claims": limits.max_claims,
            "extraction_max_output_tokens": limits.extraction_max_output_tokens,
            "synthesis_max_output_tokens": limits.synthesis_max_output_tokens,
            "timeout_seconds": limits.timeout_seconds,
            "retries": limits.retries,
        },
    }


def _runtime_code_identity() -> dict[str, Any]:
    """Bind replay output to one clean exact Core commit and tree."""

    with tempfile.TemporaryDirectory(prefix="claimci-runtime-identity-") as raw:
        empty_hooks = Path(raw) / "empty-hooks"
        empty_hooks.mkdir()
        commit = _run_git(
            _WORKSPACE_ROOT,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            empty_hooks=empty_hooks,
            output_limit=256,
        ).decode("ascii", errors="strict").strip()
        tree = _run_git(
            _WORKSPACE_ROOT,
            ("rev-parse", "--verify", "HEAD^{tree}"),
            empty_hooks=empty_hooks,
            output_limit=256,
        ).decode("ascii", errors="strict").strip()
        status = _run_git(
            _WORKSPACE_ROOT,
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
            empty_hooks=empty_hooks,
            output_limit=MAX_METADATA_BYTES,
        )
    if not _HEX_OBJECT.fullmatch(commit) or not _HEX_OBJECT.fullmatch(tree):
        raise ValueError("runtime Core Git identity is invalid")
    if status:
        raise ValueError("provider-free replay requires a clean Core worktree")
    return {
        "commit_sha": commit,
        "tree_sha": tree,
        "worktree_clean": True,
    }


def _review_config() -> tuple[ReviewConfig, dict[str, Any]]:
    """Load and fingerprint the exact trusted workspace review configuration."""

    if os.environ.get("CLAIMCI_OPENAI_MODEL") is not None:
        raise ValueError("provider-free replay forbids review model overrides")
    config_path = _WORKSPACE_ROOT / ".claimci" / "review.yaml"
    if config_path.parent.is_symlink():
        raise ValueError("review replay configuration directory is unsafe")
    before = _read_bounded(config_path)
    config = load_review_config(
        _WORKSPACE_ROOT,
        config_path=config_path,
        content=before,
    )
    after = _read_bounded(config_path)
    if after != before:
        raise ValueError("review replay configuration changed while loading")
    if not config.enabled:
        raise ValueError("review replay configuration must be enabled")
    return config, {
        "source_path": ".claimci/review.yaml",
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "effective": _serialize_review_config(config),
    }


def _validate_temp_parent(study: Path, temp_parent: Path | None) -> Path:
    base = Path(tempfile.gettempdir()) if temp_parent is None else Path(temp_parent)
    try:
        if base.is_symlink():
            raise ValueError
        resolved = base.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("temporary replay parent is invalid") from exc
    if _is_within(resolved, study):
        raise ValueError("temporary replay parent must be outside the frozen study tree")
    return resolved


def replay_candidate(
    study_root: Path,
    candidate: FrozenCandidate,
    layout: CandidateInputLayout,
    config: ReviewConfig,
    *,
    temp_parent: Path | None = None,
) -> dict[str, Any]:
    study = Path(study_root).resolve(strict=True)
    parent = _validate_temp_parent(study, temp_parent)
    source = _confined_study_path(study, layout.source_relative, directory=True)
    title, body, provenance = _read_pr_metadata(study, layout)

    # Validate the exact temporary parent before TemporaryDirectory owns cleanup.
    with tempfile.TemporaryDirectory(
        prefix=f"claimci-{candidate.candidate}-", dir=parent
    ) as raw_temp:
        temporary = Path(raw_temp).resolve(strict=True)
        if temporary.parent != parent or _is_within(temporary, study):
            raise ValueError("temporary replay directory escaped its trusted parent")
        empty_hooks = temporary / "empty-hooks"
        empty_hooks.mkdir()
        _reject_unsafe_source_config(source, empty_hooks)
        before = _source_identity(source, empty_hooks)
        if before["head_sha"] != candidate.head_sha:
            raise ValueError("frozen source HEAD does not match the manifest")
        clone = temporary / "repository"
        snapshots = temporary / "snapshots"
        snapshots.mkdir()
        try:
            _run_git(
                temporary,
                (
                    "clone",
                    "--no-local",
                    "--no-hardlinks",
                    "--no-checkout",
                    "--filter=blob:none",
                    "--",
                    str(source),
                    str(clone),
                ),
                empty_hooks=empty_hooks,
            )
            _reject_unsafe_source_config(clone, empty_hooks)
            snapshot_paths: dict[str, tuple[str, ...]] = {
                "requested_base": (),
                "comparison_base": (),
                "head": (),
            }
            roots: dict[str, Path] = {}
            for role, sha in (
                ("requested_base", candidate.requested_base_sha),
                ("comparison_base", candidate.comparison_base_sha),
                ("head", candidate.head_sha),
            ):
                root = (snapshots / role).resolve()
                _add_sparse_worktree(
                    clone,
                    root,
                    sha,
                    snapshot_paths[role],
                    empty_hooks,
                )
                if _git_head(root, empty_hooks) != sha:
                    raise ValueError("disposable snapshot HEAD does not match the manifest")
                status = _run_git(
                    root,
                    (
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignore-submodules=none",
                    ),
                    empty_hooks=empty_hooks,
                )
                if status:
                    raise ValueError("disposable snapshot is dirty")
                roots[role] = root

            snapshot_identities = {
                "requested_base": SnapshotIdentity(
                    SnapshotRole.REQUESTED_BASE,
                    roots["requested_base"],
                    candidate.requested_base_sha,
                ),
                "comparison_base": SnapshotIdentity(
                    SnapshotRole.COMPARISON_BASE,
                    roots["comparison_base"],
                    candidate.comparison_base_sha,
                ),
                "head": SnapshotIdentity(
                    SnapshotRole.HEAD,
                    roots["head"],
                    candidate.head_sha,
                ),
            }
            verified_inventory = build_git_change_inventory(
                snapshot_identities["requested_base"],
                snapshot_identities["comparison_base"],
                snapshot_identities["head"],
                candidate.comparison_basis,
            )
            if verified_inventory != candidate.inventory:
                raise ValueError("frozen change inventory does not match exact Git metadata")

            planning_inputs = ReviewInputs(
                repository_root=roots["head"],
                base_root=roots["comparison_base"],
                pr_title=title,
                pr_description=body,
                requested_base=SnapshotIdentity(
                    SnapshotRole.REQUESTED_BASE,
                    source,
                    candidate.requested_base_sha,
                ),
                comparison_base=SnapshotIdentity(
                    SnapshotRole.COMPARISON_BASE,
                    source,
                    candidate.comparison_base_sha,
                ),
                head=SnapshotIdentity(
                    SnapshotRole.HEAD,
                    source,
                    candidate.head_sha,
                ),
                inventory=candidate.inventory,
                coordinates=None,
                inventory_failure=None,
            )
            hydration_plan = plan_review_supplement_paths(
                planning_inputs,
                config,
                candidate.inventory,
            )
            material_paths = hydration_plan.paths
            if len(material_paths) > config.limits.max_files:
                raise ValueError("hydration plan exceeds the fixed material file bound")
            snapshot_paths = _copy_snapshot_blobs(
                source,
                clone,
                candidate,
                material_paths,
                empty_hooks,
            )
            if snapshot_paths["head"] != material_paths:
                raise ValueError("hydration plan is not present at the exact head")
            for role in roots:
                _set_sparse_worktree_paths(
                    roots[role],
                    snapshot_paths[role],
                    empty_hooks,
                )
                status = _run_git(
                    roots[role],
                    (
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignore-submodules=none",
                    ),
                    empty_hooks=empty_hooks,
                )
                if status:
                    raise ValueError("hydrated disposable snapshot is dirty")

            inputs = ReviewInputs(
                repository_root=roots["head"],
                base_root=roots["comparison_base"],
                pr_title=title,
                pr_description=body,
                requested_base=snapshot_identities["requested_base"],
                comparison_base=snapshot_identities["comparison_base"],
                head=snapshot_identities["head"],
                inventory=candidate.inventory,
                coordinates=None,
                inventory_failure=None,
                exact_material_omissions=hydration_plan.omissions,
            )
            review = run_review(inputs, config, preflight_only=True)
            if (
                review.provider_lifecycle is not ProviderLifecycle.NOT_ATTEMPTED
                or review.provider_attempt_count != 0
                or review.provider_calls
                or review.preflight is None
            ):
                raise ValueError("provider-free replay returned an invalid review envelope")
            preflight = review.preflight
        finally:
            after = _source_identity(source, empty_hooks)
            if after != before:
                raise RuntimeError("frozen source identity changed during replay")

        return {
            "candidate": candidate.candidate,
            "repository": candidate.repository,
            "pull_request": candidate.pull_request,
            "metadata_provenance": provenance,
            "materialized_coordinates": {
                "requested_base_sha": candidate.requested_base_sha,
                "comparison_base_sha": candidate.comparison_base_sha,
                "head_sha": candidate.head_sha,
            },
            "preflight": serialize_preflight(preflight),
            "ready": preflight.ready_for_provider,
            "provider_calls": 0,
            "provider_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0,
            },
            "source_before": before,
            "source_after": after,
            "source_unchanged": True,
        }


def replay_study(
    study_root: Path,
    output: Path,
    *,
    fixture_directory: Path | None = None,
    temp_parent: Path | None = None,
) -> dict[str, Any]:
    study, destination = resolve_replay_paths(study_root, output)
    fixtures = (
        Path(__file__).parents[1] / "tests" / "fixtures" / "review-preflight-v1"
        if fixture_directory is None
        else fixture_directory
    )
    manifests = load_frozen_manifests(fixtures)
    if tuple(candidate.candidate for candidate in manifests) != tuple(
        DEFAULT_INPUT_LAYOUTS
    ):
        raise ValueError("replay requires exactly the five frozen A-E manifests")
    runtime_code = _runtime_code_identity()
    config, config_provenance = _review_config()
    results = [
        replay_candidate(
            study,
            candidate,
            DEFAULT_INPUT_LAYOUTS[candidate.candidate],
            config,
            temp_parent=temp_parent,
        )
        for candidate in manifests
    ]
    ready_count = sum(bool(result["ready"]) for result in results)
    runtime_code_after = _runtime_code_identity()
    config_after, config_provenance_after = _review_config()
    if runtime_code_after != runtime_code:
        raise RuntimeError("runtime Core identity changed during replay")
    if config_after != config or config_provenance_after != config_provenance:
        raise RuntimeError("review configuration changed during replay")
    payload = {
        "schema_version": 1,
        "mode": "provider_free_frozen_local_replay",
        "study_root": str(study),
        "candidate_count": len(results),
        "ready_count": ready_count,
        "acceptance_threshold": 3,
        "acceptance_met": ready_count >= 3,
        "provider_calls": 0,
        "provider_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0,
        },
        "runtime_code": runtime_code,
        "review_config": config_provenance,
        "candidates": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _publish_output(destination, rendered)
    print(rendered, end="")
    return payload


def _publish_output(destination: Path, rendered: str) -> None:
    """Atomically replace an output entry without opening its existing inode."""

    raw = rendered.encode("utf-8", errors="strict")
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".claimci-replay-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        if temporary.parent.resolve(strict=True) != destination.parent:
            raise ValueError("temporary replay output escaped its destination directory")
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("temporary replay output is not a regular file")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay all five frozen local ClaimCI preflights without a provider."
    )
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = replay_study(Path(args.study_root), Path(args.output))
    except (OSError, ReviewError, RuntimeError, UnicodeError, ValueError) as exc:
        print(f"ClaimCI replay error: {exc}", file=sys.stderr)
        return 2
    return 0 if payload["acceptance_met"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
