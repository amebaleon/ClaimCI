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


@dataclass(frozen=True)
class ManifestAuditBundle:
    """One actual deterministic audit plus the submitted files it consumed."""

    snapshot: DeterministicAuditSnapshot
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ManifestAuditPlan:
    """A path-confined, bounded manifest bundle reserved for later auditing."""

    manifest_path: str
    paths: tuple[str, ...]


def select_relevant_manifest_audit_plans(
    plans: Sequence[ManifestAuditPlan],
    *,
    changed_paths: Sequence[str],
) -> tuple[ManifestAuditPlan, ...]:
    """Keep audits whose manifest or declared artifact changed in this PR.

    An unrelated study elsewhere in a large repository must not reserve the
    review file budget, appear as deterministic authority, or enter synthesis.
    A no-base local review still marks every eligible file as changed, so this
    filter does not weaken standalone inspection.
    """

    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        raise ReviewError("manifest audit plans must be a sequence")
    if not all(isinstance(plan, ManifestAuditPlan) for plan in plans):
        raise ReviewError("manifest audit plans contain an invalid value")
    if not isinstance(changed_paths, Sequence) or isinstance(
        changed_paths, (str, bytes)
    ):
        raise ReviewError("changed manifest paths must be a sequence")

    changed: set[str] = set()
    for raw in changed_paths:
        relative = _relative(raw)
        if relative is None:
            raise ReviewError("changed manifest path is unsafe")
        changed.add(relative)

    return tuple(
        plan
        for plan in plans
        if any(path in changed for path in plan.paths)
    )


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
    """Return bounded, confined manifests with repository-root studies first."""

    root = _root(repository_root)
    if isinstance(max_manifests, bool) or not isinstance(max_manifests, int) or max_manifests < 1:
        raise ReviewError("max_manifests must be a positive integer")
    candidates: set[str] = set()
    for raw in repository_paths:
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
            candidates.add(relative)
    return tuple(
        sorted(
            candidates,
            key=lambda path: (
                len(PurePosixPath(path).parts),
                path.casefold(),
                path,
            ),
        )[:max_manifests]
    )


def _has_symlink_component(root: Path, relative: str) -> bool:
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


def _declared_artifact_paths(
    root: Path, manifest_relative: str
) -> tuple[str, ...] | None:
    """Read declared lexical paths without following provider-selected links."""

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


def _bounded_text(root: Path, path: Path, limits: ReviewLimits) -> bool:
    """Allow missing evidence, but bound every existing file before audit."""

    try:
        lexical = path.relative_to(root).as_posix()
        if _has_symlink_component(root, lexical):
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


def plan_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
) -> tuple[ManifestAuditPlan, ...]:
    """Reserve safe manifest bundles before broad repository retrieval."""

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

    plans: list[ManifestAuditPlan] = []
    for raw in manifest_paths:
        relative = _relative(raw)
        if relative is None:
            continue
        try:
            lexical_manifest = root / Path(relative)
            if _has_symlink_component(root, relative) or not _bounded_text(
                root, lexical_manifest, limits
            ):
                continue
            lexical_artifacts = _declared_artifact_paths(root, relative)
            if lexical_artifacts is None or any(
                _has_symlink_component(root, path) for path in lexical_artifacts
            ):
                continue
            manifest = lexical_manifest.resolve()
            manifest.relative_to(root)
            load_research_spec(manifest, artifact_root=root)
            # Count the declared lexical files.  Links were rejected before
            # resolution, so these names preserve the submitted provenance.
            ordered_audit_paths = tuple(
                dict.fromkeys((relative, *lexical_artifacts))
            )
            audit_paths = set(ordered_audit_paths)
            if len(selected | audit_paths) > limits.max_files:
                continue
            if any(
                not _bounded_text(root, root / Path(artifact_path), limits)
                for artifact_path in lexical_artifacts
            ):
                continue
            selected.update(audit_paths)
            plans.append(
                ManifestAuditPlan(
                    manifest_path=relative,
                    paths=ordered_audit_paths,
                )
            )
        except (ClaimCIError, ReviewError, OSError, ValueError, RuntimeError):
            # A malformed manifest is missing usable deterministic evidence;
            # it never becomes a model-invented finding.
            continue
    return tuple(plans)


def collect_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
) -> tuple[ManifestAuditBundle, ...]:
    """Run planned audits and retain their exact lexical input paths."""

    root = _root(repository_root)
    plans = plan_manifest_audits(
        root,
        manifest_paths,
        limits=limits,
        selected_paths=selected_paths,
    )
    bundles: list[ManifestAuditBundle] = []
    for plan in plans:
        try:
            result = audit_research(root / Path(plan.manifest_path), artifact_root=root)
            bundles.append(
                ManifestAuditBundle(
                    snapshot=snapshot_audit_result(result, repository_root=root),
                    paths=plan.paths,
                )
            )
        except (ClaimCIError, ReviewError, OSError, ValueError, RuntimeError):
            continue
    return tuple(bundles)


def run_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
) -> tuple[DeterministicAuditSnapshot, ...]:
    """Compatibility view returning only immutable deterministic snapshots."""

    return tuple(
        bundle.snapshot
        for bundle in collect_manifest_audits(
            repository_root,
            manifest_paths,
            limits=limits,
            selected_paths=selected_paths,
        )
    )
