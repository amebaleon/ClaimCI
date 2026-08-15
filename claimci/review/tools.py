"""One-way bridge from deterministic ClaimCI audits to advisory review data."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

from claimci.audit import audit_research, load_research_spec
from claimci.models import AuditResult, ClaimCIError
from claimci.parsing import load_unique_yaml

from .models import ReviewError, ReviewLimits


_ARTIFACT_FIELDS = ("config", "results", "train_dataset", "eval_dataset")


class _FrozenList(tuple):
    """Immutable sequence that retains structural equality with JSON lists."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return False

    __hash__ = tuple.__hash__


@dataclass(frozen=True)
class DeterministicFindingSnapshot:
    rule_id: str
    severity: str
    impact: str
    title: str
    explanation: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class DeterministicAuditSnapshot:
    manifest_path: str
    verdict: str
    metric: str
    minimum_improvement: float
    direction: str
    findings: tuple[DeterministicFindingSnapshot, ...]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        return copy.deepcopy(value)
    except Exception:
        return str(value)


def _root(path: Path) -> Path:
    try:
        root = Path(path).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"repository root is invalid: {exc}") from exc
    if not root.is_dir():
        raise ReviewError(f"repository root is not a directory: {root}")
    return root


def _relative(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None
    portable = raw.replace("\\", "/")
    if (
        PurePosixPath(portable).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ".." in portable.split("/")
    ):
        return None
    normalized = PurePosixPath(portable).as_posix()
    return None if normalized in {"", "."} else normalized


def snapshot_audit_result(
    result: AuditResult,
    *,
    repository_root: Path,
) -> DeterministicAuditSnapshot:
    """Copy an actual deterministic result into immutable plain review data."""

    if not isinstance(result, AuditResult):
        raise ReviewError("deterministic snapshots require an actual AuditResult")
    root = _root(repository_root)
    try:
        manifest = result.manifest_path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ReviewError("deterministic manifest resolves outside repository") from exc
    findings = tuple(
        DeterministicFindingSnapshot(
            rule_id=finding.rule_id,
            severity=finding.severity.value,
            impact=finding.impact.value,
            title=finding.title,
            explanation=finding.explanation,
            evidence=_freeze(finding.evidence),
        )
        for finding in result.findings
    )
    return DeterministicAuditSnapshot(
        manifest_path=manifest,
        verdict=result.verdict.value,
        metric=result.metric,
        minimum_improvement=result.minimum_improvement,
        direction=result.direction.value,
        findings=findings,
    )


def discover_manifests(
    repository_root: Path,
    repository_paths: Sequence[str],
    *,
    max_manifests: int = 4,
) -> tuple[str, ...]:
    """Return sorted, indexed, confined ClaimCI manifest paths."""

    root = _root(repository_root)
    if isinstance(max_manifests, bool) or not isinstance(max_manifests, int) or max_manifests < 1:
        raise ReviewError("max_manifests must be a positive integer")
    selected: list[str] = []
    for raw in sorted(set(repository_paths)):
        relative = _relative(raw)
        if relative is None or PurePosixPath(relative).name.casefold() not in {
            "research.yaml",
            "research.yml",
        }:
            continue
        try:
            resolved = (root / Path(relative)).resolve()
            resolved.relative_to(root)
        except (OSError, ValueError, RuntimeError):
            continue
        if resolved.is_file():
            selected.append(relative)
            if len(selected) >= max_manifests:
                break
    return tuple(selected)


def run_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
) -> tuple[DeterministicAuditSnapshot, ...]:
    """Run deterministic audits only after one global bounded-file preflight."""

    root = _root(repository_root)
    if not isinstance(limits, ReviewLimits):
        raise ReviewError("manifest audit limits must be ReviewLimits")
    if not isinstance(selected_paths, Sequence) or isinstance(
        selected_paths, (str, bytes)
    ):
        raise ReviewError("selected manifest paths must be a sequence")
    selected: set[str] = set()
    for raw in selected_paths:
        relative = _relative(raw)
        if relative is None:
            raise ReviewError("selected manifest path is unsafe")
        selected.add(relative)
    if len(selected) > limits.max_files:
        raise ReviewError("selected paths exceed the global review file limit")

    def has_symlink_component(relative: str) -> bool:
        """Reject a lexical link before ``resolve`` erases its provenance."""

        cursor = root
        try:
            for part in PurePosixPath(relative).parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    return True
        except (OSError, ValueError, RuntimeError):
            return True
        return False

    def declared_artifact_paths(manifest_relative: str) -> tuple[str, ...] | None:
        """Read only declared lexical paths so links can be rejected pre-audit."""

        try:
            manifest_path = root / Path(manifest_relative)
            payload = load_unique_yaml(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                return None
            parent = PurePosixPath(manifest_relative).parent
            declared: list[str] = []
            for experiment_name in ("baseline", "candidate"):
                experiment = payload.get(experiment_name)
                if not isinstance(experiment, Mapping):
                    return None
                for field in _ARTIFACT_FIELDS:
                    raw = experiment.get(field)
                    # Backslash spellings are platform-ambiguous and are not
                    # accepted by the advisory repository path model.
                    if not isinstance(raw, str) or "\\" in raw:
                        return None
                    artifact = _relative(raw)
                    if artifact is None:
                        return None
                    joined = _relative((parent / PurePosixPath(artifact)).as_posix())
                    if joined is None:
                        return None
                    declared.append(joined)
            return tuple(declared)
        except (
            OSError,
            UnicodeError,
            TypeError,
            ValueError,
            RuntimeError,
            yaml.YAMLError,
        ):
            return None

    def bounded_text(path: Path) -> bool:
        """Allow missing evidence, but bound every existing file before audit."""

        try:
            lexical = path.relative_to(root).as_posix()
            if has_symlink_component(lexical):
                return False
            if not path.exists():
                return True
            if path.is_symlink() or not path.is_file():
                return False
            if path.stat().st_size > limits.max_file_chars * 4:
                return False
            with path.open("r", encoding="utf-8", newline=None) as handle:
                return len(handle.read(limits.max_file_chars + 1)) <= limits.max_file_chars
        except (OSError, UnicodeError, ValueError, RuntimeError):
            return False

    snapshots: list[DeterministicAuditSnapshot] = []
    for raw in manifest_paths:
        relative = _relative(raw)
        if relative is None:
            continue
        try:
            lexical_manifest = root / Path(relative)
            if has_symlink_component(relative) or not bounded_text(lexical_manifest):
                continue
            lexical_artifacts = declared_artifact_paths(relative)
            if lexical_artifacts is None or any(
                has_symlink_component(path) for path in lexical_artifacts
            ):
                continue
            manifest = lexical_manifest.resolve()
            manifest.relative_to(root)
            load_research_spec(manifest, artifact_root=root)
            # Count the declared lexical files.  Links were rejected before
            # resolution, so these names preserve the submitted provenance.
            audit_paths = {relative, *lexical_artifacts}
            if len(selected | audit_paths) > limits.max_files:
                continue
            if any(
                not bounded_text(root / Path(artifact_path))
                for artifact_path in lexical_artifacts
            ):
                continue
            selected.update(audit_paths)
            result = audit_research(manifest, artifact_root=root)
            snapshots.append(snapshot_audit_result(result, repository_root=root))
        except (ClaimCIError, ReviewError, OSError, ValueError, RuntimeError):
            # A malformed manifest is missing usable deterministic evidence;
            # it never becomes a model-invented finding.
            continue
    return tuple(snapshots)
