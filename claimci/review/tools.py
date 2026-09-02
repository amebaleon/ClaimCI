"""One-way bridge from deterministic ClaimCI audits to advisory review data."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

from claimci.audit import audit_research, load_research_spec
from claimci.models import AuditResult, ClaimCIError
from claimci.passive_files import PassiveFileError, inspect_confined_regular_file
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
    input_sha256: tuple[tuple[str, str | None], ...] = field(
        default=(), compare=False, repr=False
    )


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


def _capture_manifest_input_sha256(
    root: Path,
    paths: Sequence[str],
    limits: ReviewLimits,
) -> tuple[tuple[str, str | None], ...] | None:
    """Capture exact bounded identities without discovering any new path."""

    identities: list[tuple[str, str | None]] = []
    for raw in paths:
        relative = _relative(raw)
        if relative is None or relative != raw:
            return None
        try:
            inspection = inspect_confined_regular_file(
                root,
                relative,
                max_bytes=limits.max_file_chars * 4,
            )
        except PassiveFileError as exc:
            if exc.code != "unavailable":
                return None
            # Missing evidence is a meaningful deterministic Audit input.  It
            # receives a stable null identity, while every other inspection
            # failure invalidates the reservation.
            try:
                (root / Path(relative)).lstat()
            except FileNotFoundError:
                identities.append((relative, None))
                continue
            except OSError:
                return None
            return None
        identities.append((relative, inspection.sha256))
    return tuple(identities)


def _valid_manifest_input_sha256(plan: ManifestAuditPlan) -> bool:
    """Validate the private identity payload carried by a reserved plan."""

    identities = plan.input_sha256
    if not isinstance(identities, tuple) or len(identities) != len(plan.paths):
        return False
    hexadecimal = frozenset("0123456789abcdef")
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and item[0] == path
        and (
            item[1] is None
            or (
                isinstance(item[1], str)
                and len(item[1]) == 64
                and set(item[1]).issubset(hexadecimal)
            )
        )
        for path, item in zip(plan.paths, identities)
    )


def plan_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
    issued_paths: Sequence[str] | None = None,
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
    issued: set[str] | None = None
    if issued_paths is not None:
        if not isinstance(issued_paths, Sequence) or isinstance(
            issued_paths, (str, bytes)
        ):
            raise ReviewError("issued manifest paths must be a sequence")
        issued = set()
        for raw in issued_paths:
            relative = _relative(raw)
            if relative is None:
                raise ReviewError("issued manifest path is unsafe")
            issued.add(relative)

    plans: list[ManifestAuditPlan] = []
    for raw in manifest_paths:
        relative = _relative(raw)
        if relative is None or (issued is not None and relative not in issued):
            continue
        try:
            lexical_manifest = root / Path(relative)
            if _has_symlink_component(root, relative) or not _bounded_text(
                root, lexical_manifest, limits
            ):
                continue
            lexical_artifacts = _declared_artifact_paths(root, relative)
            if lexical_artifacts is None:
                continue
            # Preserve the submitted lexical names. Admitted plans reject
            # links before resolution; omitted plans never touch dependencies.
            ordered_audit_paths = tuple(
                dict.fromkeys((relative, *lexical_artifacts))
            )
            audit_paths = set(ordered_audit_paths)
            if issued is not None and not audit_paths.issubset(issued):
                # Retain lexical dependency names for the caller's structured
                # omission accounting, but do not stat, open, parse, or audit
                # any dependency outside the issued scope.
                plans.append(
                    ManifestAuditPlan(
                        manifest_path=relative,
                        paths=ordered_audit_paths,
                    )
                )
                continue
            if any(
                _has_symlink_component(root, path) for path in lexical_artifacts
            ):
                continue
            manifest = lexical_manifest.resolve()
            manifest.relative_to(root)
            load_research_spec(manifest, artifact_root=root)
            if len(selected | audit_paths) > limits.max_files:
                continue
            if any(
                not _bounded_text(root, root / Path(artifact_path), limits)
                for artifact_path in lexical_artifacts
            ):
                continue
            input_sha256 = _capture_manifest_input_sha256(
                root, ordered_audit_paths, limits
            )
            if input_sha256 is None:
                continue
            selected.update(audit_paths)
            plans.append(
                ManifestAuditPlan(
                    manifest_path=relative,
                    paths=ordered_audit_paths,
                    input_sha256=input_sha256,
                )
            )
        except (ClaimCIError, ReviewError, OSError, ValueError, RuntimeError):
            # A malformed manifest is missing usable deterministic evidence;
            # it never becomes a model-invented finding.
            continue
    return tuple(plans)


def _revalidate_reserved_manifest_audit_plans(
    root: Path,
    manifest_paths: Sequence[str],
    reserved_plans: Sequence[ManifestAuditPlan],
    *,
    limits: ReviewLimits,
    selected_paths: Sequence[str],
    issued_paths: Sequence[str] | None,
) -> tuple[ManifestAuditPlan, ...]:
    """Admit only unchanged exact plans without expanding their path boundary."""

    if not isinstance(reserved_plans, Sequence) or isinstance(
        reserved_plans, (str, bytes)
    ):
        raise ReviewError("reserved manifest audit plans must be a sequence")
    if not all(isinstance(plan, ManifestAuditPlan) for plan in reserved_plans):
        raise ReviewError("reserved manifest audit plans contain an invalid value")
    candidates = {
        relative
        for raw in manifest_paths
        if (relative := _relative(raw)) is not None
    }
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
    issued: set[str] | None = None
    if issued_paths is not None:
        if not isinstance(issued_paths, Sequence) or isinstance(
            issued_paths, (str, bytes)
        ):
            raise ReviewError("issued manifest paths must be a sequence")
        issued = set()
        for raw in issued_paths:
            relative = _relative(raw)
            if relative is None:
                raise ReviewError("issued manifest path is unsafe")
            issued.add(relative)

    admitted: list[ManifestAuditPlan] = []
    for plan in reserved_plans:
        relative = _relative(plan.manifest_path)
        if not isinstance(plan.paths, Sequence) or isinstance(
            plan.paths, (str, bytes)
        ):
            continue
        normalized_paths = tuple(_relative(path) for path in plan.paths)
        if (
            relative is None
            or relative not in candidates
            or not normalized_paths
            or any(path is None for path in normalized_paths)
            or normalized_paths != plan.paths
            or normalized_paths[0] != relative
            or len(set(normalized_paths)) != len(normalized_paths)
        ):
            continue
        plan_paths = set(normalized_paths)
        # A reserved plan never grants authority to a path omitted from the
        # coordinate-bound scope. Reject it before touching even its manifest.
        if issued is not None and not plan_paths.issubset(issued):
            continue
        if len(selected | plan_paths) > limits.max_files:
            continue
        reserved_input_sha256 = plan.input_sha256
        if reserved_input_sha256 and not _valid_manifest_input_sha256(plan):
            continue
        current_input_sha256 = _capture_manifest_input_sha256(
            root, normalized_paths, limits
        )
        if current_input_sha256 is None or (
            reserved_input_sha256
            and current_input_sha256 != reserved_input_sha256
        ):
            continue
        try:
            lexical_manifest = root / Path(relative)
            if _has_symlink_component(root, relative) or not _bounded_text(
                root, lexical_manifest, limits
            ):
                continue
            lexical_artifacts = _declared_artifact_paths(root, relative)
            if lexical_artifacts is None:
                continue
            current_paths = tuple(
                dict.fromkeys((relative, *lexical_artifacts))
            )
            # Parsing the issued manifest may reveal drift, but a changed
            # declaration is never followed, statted, or substituted into the
            # exact pre-provider plan.
            if current_paths != normalized_paths:
                continue
            if any(
                _has_symlink_component(root, path)
                for path in lexical_artifacts
            ):
                continue
            manifest = lexical_manifest.resolve()
            manifest.relative_to(root)
            load_research_spec(manifest, artifact_root=root)
            if any(
                not _bounded_text(root, root / Path(artifact_path), limits)
                for artifact_path in lexical_artifacts
            ):
                continue
            if (
                _capture_manifest_input_sha256(root, normalized_paths, limits)
                != current_input_sha256
            ):
                continue
            selected.update(plan_paths)
            admitted.append(
                ManifestAuditPlan(
                    manifest_path=relative,
                    paths=normalized_paths,
                    input_sha256=current_input_sha256,
                )
            )
        except (ClaimCIError, ReviewError, OSError, ValueError, RuntimeError):
            continue
    return tuple(admitted)


def revalidate_manifest_audit_input_identities(
    repository_root: Path,
    reserved_plans: Sequence[ManifestAuditPlan],
    *,
    limits: ReviewLimits = ReviewLimits(),
    issued_paths: Sequence[str] | None = None,
) -> tuple[ManifestAuditPlan, ...]:
    """Retain exact plans whose already-issued bytes remain unchanged.

    This identity-only post-provider check never parses a manifest, discovers
    dependencies, or runs deterministic Audit.  It re-captures only the paths
    frozen into an existing reservation.
    """

    root = _root(repository_root)
    if not isinstance(limits, ReviewLimits):
        raise ReviewError("manifest audit limits must be ReviewLimits")
    if not isinstance(reserved_plans, Sequence) or isinstance(
        reserved_plans, (str, bytes)
    ):
        raise ReviewError("reserved manifest audit plans must be a sequence")
    if not all(isinstance(plan, ManifestAuditPlan) for plan in reserved_plans):
        raise ReviewError("reserved manifest audit plans contain an invalid value")
    issued: set[str] | None = None
    if issued_paths is not None:
        if not isinstance(issued_paths, Sequence) or isinstance(
            issued_paths, (str, bytes)
        ):
            raise ReviewError("issued manifest paths must be a sequence")
        issued = set()
        for raw in issued_paths:
            relative = _relative(raw)
            if relative is None:
                raise ReviewError("issued manifest path is unsafe")
            issued.add(relative)

    retained: list[ManifestAuditPlan] = []
    selected: set[str] = set()
    for plan in reserved_plans:
        if not isinstance(plan.paths, Sequence) or isinstance(
            plan.paths, (str, bytes)
        ):
            continue
        normalized_paths = tuple(_relative(path) for path in plan.paths)
        if (
            not normalized_paths
            or any(path is None for path in normalized_paths)
            or normalized_paths != plan.paths
            or len(set(normalized_paths)) != len(normalized_paths)
        ):
            continue
        plan_paths = set(normalized_paths)
        if issued is not None and not plan_paths.issubset(issued):
            continue
        if len(selected | plan_paths) > limits.max_files:
            continue
        # Empty identities preserve compatibility for manually constructed
        # legacy plans.  Production plans always carry exact captures.
        if not plan.input_sha256:
            retained.append(plan)
            selected.update(plan_paths)
            continue
        if not _valid_manifest_input_sha256(plan):
            continue
        current = _capture_manifest_input_sha256(root, plan.paths, limits)
        if current == plan.input_sha256:
            retained.append(plan)
            selected.update(plan_paths)
    return tuple(retained)


def collect_manifest_audits(
    repository_root: Path,
    manifest_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    selected_paths: Sequence[str] = (),
    reserved_plans: Sequence[ManifestAuditPlan] | None = None,
    issued_paths: Sequence[str] | None = None,
) -> tuple[ManifestAuditBundle, ...]:
    """Run planned audits and retain their exact lexical input paths.

    Existing callers may omit ``reserved_plans`` and retain discovery-time
    planning. Coordinate-bound callers pass their pre-provider plans so this
    step can only revalidate those exact paths and cannot broaden after model
    execution.
    """

    root = _root(repository_root)
    if not isinstance(limits, ReviewLimits):
        raise ReviewError("manifest audit limits must be ReviewLimits")
    if reserved_plans is None:
        plans = plan_manifest_audits(
            root,
            manifest_paths,
            limits=limits,
            selected_paths=selected_paths,
            issued_paths=issued_paths,
        )
        plans = _revalidate_reserved_manifest_audit_plans(
            root,
            manifest_paths,
            plans,
            limits=limits,
            selected_paths=selected_paths,
            issued_paths=issued_paths,
        )
    else:
        plans = _revalidate_reserved_manifest_audit_plans(
            root,
            manifest_paths,
            reserved_plans,
            limits=limits,
            selected_paths=selected_paths,
            issued_paths=issued_paths,
        )
    bundles: list[ManifestAuditBundle] = []
    for plan in plans:
        try:
            result = audit_research(root / Path(plan.manifest_path), artifact_root=root)
            if plan.input_sha256 and (
                _capture_manifest_input_sha256(root, plan.paths, limits)
                != plan.input_sha256
            ):
                continue
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
    reserved_plans: Sequence[ManifestAuditPlan] | None = None,
    issued_paths: Sequence[str] | None = None,
) -> tuple[DeterministicAuditSnapshot, ...]:
    """Compatibility view returning only immutable deterministic snapshots."""

    return tuple(
        bundle.snapshot
        for bundle in collect_manifest_audits(
            repository_root,
            manifest_paths,
            limits=limits,
            selected_paths=selected_paths,
            reserved_plans=reserved_plans,
            issued_paths=issued_paths,
        )
    )
