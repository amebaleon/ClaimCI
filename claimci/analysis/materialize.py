"""Confined ephemeral compatibility materialization for the native Audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from claimci.audit import audit_research
from claimci.models import AuditResult

from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    ConfigValue,
    DatasetSplit,
    EphemeralAuditPlan,
    ExperimentRole,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    NormalizedEvidence,
    PassiveArtifact,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
)
from .planner import derive_ephemeral_plan_id


_PLAN_ID = re.compile(r"plan-[0-9a-f]{24}\Z")
_CONFIG_SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")


class MaterializationError(RuntimeError):
    """Base class for controlled ephemeral execution failures."""


class MaterializationPartial(MaterializationError):
    """Known evidence or representation limitation before native Audit."""


class MaterializationUnavailable(MaterializationError):
    """Fail-closed runtime integrity or internal execution failure."""


@dataclass(frozen=True, slots=True)
class MaterializationLimits:
    max_file_bytes: int = 16 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for label, value in (
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AnalysisContractError(f"{label} must be a positive integer")
        if self.max_total_bytes < self.max_file_bytes:
            raise AnalysisContractError(
                "max_total_bytes must be at least max_file_bytes"
            )


def _resolved_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        if path.is_symlink():
            raise AnalysisContractError(f"{label} root must not be a symlink")
        current = Path(path.anchor) if path.is_absolute() else Path.cwd()
        for component in path.parts[1:] if path.is_absolute() else path.parts:
            current = current / component
            if current.exists() and current.is_symlink():
                raise AnalysisContractError(
                    f"{label} root path must not contain a symlink"
                )
        resolved = path.resolve(strict=False)
        if not resolved.exists() or not resolved.is_dir():
            raise AnalysisContractError(f"{label} root must be an existing directory")
        if resolved.is_symlink():
            raise AnalysisContractError(f"{label} root must not be a symlink")
        return resolved
    except AnalysisContractError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AnalysisContractError(f"{label} root is invalid: {error}") from error


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class RuntimeExecutionContext:
    """Trusted repository/head identity and invocation-private filesystem roots."""

    repository: RepositoryIdentity
    checkout_root: Path
    head_sha: GitCommitSha
    scratch_root: Path
    limits: MaterializationLimits = field(default_factory=MaterializationLimits)
    _checkout_identity: tuple[int, int, int] = field(init=False, repr=False)
    _scratch_identity: tuple[int, int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.repository) is not RepositoryIdentity:
            raise TypeError("runtime repository must be RepositoryIdentity")
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if type(self.limits) is not MaterializationLimits:
            raise TypeError("runtime limits must be MaterializationLimits")
        checkout_input = Path(self.checkout_root)
        scratch_input = Path(self.scratch_root)
        try:
            checkout_candidate = checkout_input.resolve(strict=False)
            scratch_candidate = scratch_input.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise AnalysisContractError(f"runtime roots are invalid: {error}") from error
        if _paths_overlap(checkout_candidate, scratch_candidate):
            raise AnalysisContractError(
                "runtime checkout and scratch roots must not overlap"
            )
        checkout = _resolved_directory(checkout_input, "checkout")
        scratch = _resolved_directory(scratch_input, "scratch")
        if _paths_overlap(checkout, scratch):
            raise AnalysisContractError(
                "runtime checkout and scratch roots must not overlap"
            )
        object.__setattr__(self, "checkout_root", checkout)
        object.__setattr__(self, "scratch_root", scratch)
        checkout_stat = os.lstat(checkout)
        scratch_stat = os.lstat(scratch)
        object.__setattr__(
            self,
            "_checkout_identity",
            (checkout_stat.st_dev, checkout_stat.st_ino, checkout_stat.st_mode),
        )
        object.__setattr__(
            self,
            "_scratch_identity",
            (scratch_stat.st_dev, scratch_stat.st_ino, scratch_stat.st_mode),
        )


def _validate_runtime_roots(runtime: RuntimeExecutionContext) -> None:
    for label, path, expected in (
        (
            "checkout",
            runtime.checkout_root,
            runtime._checkout_identity,
        ),
        (
            "scratch",
            runtime.scratch_root,
            runtime._scratch_identity,
        ),
    ):
        try:
            current = os.lstat(path)
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise MaterializationUnavailable(
                f"trusted {label} root is unavailable: {error}"
            ) from error
        identity = (current.st_dev, current.st_ino, current.st_mode)
        if (
            identity != expected
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or resolved != path
        ):
            raise MaterializationUnavailable(
                f"trusted {label} root identity changed"
            )
    if _paths_overlap(runtime.checkout_root, runtime.scratch_root):
        raise MaterializationUnavailable(
            "trusted checkout and scratch roots now overlap"
        )


def _open_customer_artifact(path: Path) -> int:
    """Open one passive source without following the final symlink when possible."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size)


def _capture_artifact(
    candidate: ArtifactCandidate,
    runtime: RuntimeExecutionContext,
    *,
    aggregate_before: int,
) -> PassiveArtifact:
    if candidate.size > runtime.limits.max_file_bytes:
        raise MaterializationUnavailable("artifact exceeds the runtime file bound")
    if aggregate_before + candidate.size > runtime.limits.max_total_bytes:
        raise MaterializationUnavailable("artifacts exceed the runtime aggregate bound")
    current = runtime.checkout_root
    try:
        for component in str(candidate.path).split("/"):
            current = current / component
            item_stat = os.lstat(current)
            if stat.S_ISLNK(item_stat.st_mode):
                raise MaterializationUnavailable(
                    "artifact snapshot path contains a symlink"
                )
        source = current.resolve(strict=True)
        source.relative_to(runtime.checkout_root)
        if source != current or current.is_symlink():
            raise MaterializationUnavailable(
                "artifact snapshot path changed through a symlink"
            )
        before = os.lstat(current)
        if not stat.S_ISREG(before.st_mode):
            raise MaterializationUnavailable(
                "artifact snapshot source must be a regular file"
            )
        descriptor = _open_customer_artifact(current)
    except MaterializationUnavailable:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise MaterializationUnavailable(
            f"artifact snapshot could not be opened: {error}"
        ) from error

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise MaterializationUnavailable(
                "artifact snapshot descriptor is not a regular file"
            )
        if _stat_identity(opened) != _stat_identity(before):
            raise MaterializationUnavailable(
                "artifact snapshot identity changed during open"
            )
        chunks: list[bytes] = []
        remaining = runtime.limits.max_file_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > runtime.limits.max_file_bytes:
            raise MaterializationUnavailable(
                "artifact changed beyond the runtime file bound"
            )
    except MaterializationUnavailable:
        raise
    except OSError as error:
        raise MaterializationUnavailable(
            f"artifact snapshot read failed: {error}"
        ) from error
    finally:
        os.close(descriptor)

    try:
        after = os.lstat(current)
    except OSError as error:
        raise MaterializationUnavailable(
            f"artifact snapshot identity could not be rechecked: {error}"
        ) from error
    if _stat_identity(after) != _stat_identity(opened):
        raise MaterializationUnavailable(
            "artifact snapshot identity changed during capture"
        )
    if len(content) != candidate.size:
        raise MaterializationUnavailable("artifact snapshot size no longer matches")
    if hashlib.sha256(content).hexdigest() != candidate.sha256:
        raise MaterializationUnavailable("artifact snapshot hash no longer matches")
    try:
        return PassiveArtifact(candidate, content)
    except (TypeError, ValueError) as error:
        raise MaterializationUnavailable(
            f"artifact snapshot validation failed: {error}"
        ) from error


def _evidence_by_binding(
    plan: EphemeralAuditPlan,
) -> tuple[tuple[NormalizedEvidence, ArtifactBinding], ...]:
    if plan.audit_claim is None or plan.selected_mapping is None:
        raise MaterializationPartial(
            "ephemeral execution requires an audit claim and selected mapping"
        )
    selected = plan.selected_mapping
    if type(selected) is MappingCandidate:
        if selected.confidence.value < 0.90:
            raise MaterializationUnavailable(
                "selected mapping confidence is below the runtime trust floor"
            )
        if (
            selected.trust is MappingTrust.INFERRED
            and selected.provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL
        ):
            raise MaterializationUnavailable(
                "selected inferred mapping does not satisfy deterministic trust"
            )
        candidate_provenance = (
            selected.provenance,
            *(binding.provenance for binding in selected.bindings),
            *(
                mapping.provenance
                for binding in selected.bindings
                for mapping in binding.mappings
            ),
            *(
                mapping.selector.provenance
                for binding in selected.bindings
                for mapping in binding.mappings
            ),
        )
        if any(
            item.kind is ProvenanceKind.PROVIDER_PROPOSAL
            for item in candidate_provenance
        ):
            raise MaterializationUnavailable(
                "provider mapping requires explicit RepoMapping approval"
            )
    elif type(selected) is RepoMapping:
        if selected.repository != plan.repository:
            raise MaterializationUnavailable(
                "approved mapping repository does not match the plan"
            )
    else:
        raise MaterializationUnavailable("selected mapping type is not trusted")

    evidence = (*plan.baseline_evidence, *plan.candidate_evidence)
    by_key = {
        (item.artifact.path, item.artifact.kind): item for item in evidence
    }
    if len(by_key) != len(evidence):
        raise MaterializationUnavailable("plan evidence paths are not unique")
    bound: list[tuple[NormalizedEvidence, ArtifactBinding]] = []
    for binding in selected.bindings:
        item = by_key.get((binding.path, binding.kind))
        if item is None:
            raise MaterializationUnavailable(
                "selected binding has no issued normalized evidence"
            )
        if binding.adapter_id != item.adapter_match.adapter_id:
            raise MaterializationUnavailable("selected adapter identity changed")
        if binding.mappings != item.adapter_match.mappings:
            raise MaterializationUnavailable("selected field selectors changed")
        _validate_normalized_provenance(item)
        expected = (
            ExperimentRole.BASELINE
            if item in plan.baseline_evidence
            else ExperimentRole.CANDIDATE
        )
        if binding.role is not expected:
            raise MaterializationUnavailable("selected experiment role changed")
        if binding.kind is ArtifactKind.DATASET and binding.dataset_split not in {
            DatasetSplit.TRAIN,
            DatasetSplit.EVAL,
        }:
            raise MaterializationUnavailable(
                "selected dataset binding has no validated train/eval split"
            )
        bound.append((item, binding))
    if {item.evidence_id for item, _binding in bound} != {
        item.evidence_id for item in evidence
    }:
        raise MaterializationUnavailable(
            "selected mapping does not bind exactly the planned evidence"
        )
    return tuple(bound)


def _validate_normalized_provenance(evidence: NormalizedEvidence) -> None:
    if any(
        item.kind is ProvenanceKind.PROVIDER_PROPOSAL
        for item in evidence.adapter_match.match_evidence
    ):
        raise MaterializationUnavailable(
            "provider-originated adapter evidence is not deterministic extraction"
        )
    for observation in evidence.observations:
        provenance = (
            observation.provenance,
            *(item.provenance for item in observation.config_values),
            *(item.provenance for item in observation.dataset_references),
            *(item.provenance for item in observation.compute_evidence),
        )
        if any(
            item.kind is not ProvenanceKind.ADAPTER_EXTRACTION
            for item in provenance
        ):
            raise MaterializationUnavailable(
                "only adapter-validated normalized observations may be materialized"
            )


def _capture_plan_artifacts(
    runtime: RuntimeExecutionContext,
    bound: tuple[tuple[NormalizedEvidence, ArtifactBinding], ...],
) -> Mapping[str, PassiveArtifact]:
    captured: dict[str, PassiveArtifact] = {}
    captured_kinds: dict[str, ArtifactKind] = {}
    revalidated_datasets: set[str] = set()
    total = 0
    for evidence, binding in sorted(
        bound,
        key=lambda item: (
            str(item[0].artifact.path),
            item[1].dataset_split.value if item[1].dataset_split else "",
        ),
    ):
        path = str(evidence.artifact.path)
        existing_kind = captured_kinds.get(path)
        if existing_kind is not None and existing_kind is not evidence.artifact.kind:
            raise MaterializationUnavailable(
                "one captured repository path cannot have conflicting artifact kinds"
            )
        passive = captured.get(path)
        if passive is None:
            passive = _capture_artifact(
                evidence.artifact,
                runtime,
                aggregate_before=total,
            )
            captured[path] = passive
            captured_kinds[path] = evidence.artifact.kind
            total += len(passive.content)
        elif passive.candidate != evidence.artifact:
            raise MaterializationUnavailable(
                "duplicate selected path has conflicting artifact identity"
            )
        if (
            evidence.artifact.kind is ArtifactKind.DATASET
            and path not in revalidated_datasets
        ):
            _revalidate_dataset_identity(evidence, binding, passive)
            revalidated_datasets.add(path)
    return captured


def _revalidate_dataset_identity(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    passive: PassiveArtifact,
) -> None:
    if not str(evidence.artifact.path).casefold().endswith(".jsonl"):
        raise MaterializationPartial(
            "dataset evidence is not a losslessly supported JSONL artifact"
        )
    if binding.adapter_id != "claimci-jsonl-dataset-v1":
        raise MaterializationUnavailable(
            "selected dataset adapter is not the fixed passive JSONL adapter"
        )
    if binding.mappings or evidence.adapter_match.mappings:
        raise MaterializationUnavailable(
            "passive dataset identity must not contain executable selectors"
        )
    try:
        from .adapters import get_adapter

        adapter = get_adapter(binding.adapter_id)
        match = adapter.probe(passive)
        if match is None:
            raise MaterializationUnavailable(
                "selected dataset adapter no longer matches captured bytes"
            )
        fresh = adapter.extract(passive, match)
    except MaterializationUnavailable:
        raise
    except Exception as error:
        raise MaterializationUnavailable(
            f"fresh passive dataset adapter revalidation failed: {error}"
        ) from error
    if type(fresh) is not NormalizedEvidence or fresh != evidence:
        raise MaterializationUnavailable(
            "planned dataset evidence no longer matches fresh adapter identity"
        )


def _results_payload(
    evidence: tuple[NormalizedEvidence, ...],
    metric: str,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        for observation in item.observations:
            if observation.metric_name != metric:
                continue
            run: dict[str, object] = {metric: observation.metric_value}
            if observation.seed is not None:
                if isinstance(observation.seed, bool) or not isinstance(
                    observation.seed, int
                ):
                    raise MaterializationPartial(
                        "a present seed cannot be represented by the native Audit"
                    )
                run["seed"] = observation.seed
            runs.append(run)
    if not runs:
        raise MaterializationPartial(
            f"no normalized result observations match metric {metric!r}"
        )
    try:
        json.dumps(runs, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise MaterializationPartial(
            f"normalized result observations are not JSON representable: {error}"
        ) from error
    return {"runs": runs}


def _insert_config(root: dict[str, object], value: ConfigValue) -> None:
    parts = value.key.split(".")
    if any(not _CONFIG_SEGMENT.fullmatch(part) for part in parts):
        raise MaterializationPartial("config key cannot be represented as a native path")
    current = root
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            child: dict[str, object] = {}
            current[part] = child
            current = child
        elif isinstance(existing, dict):
            current = existing
        else:
            raise MaterializationPartial("config scalar/mapping paths conflict")
    leaf = parts[-1]
    if leaf in current:
        raise MaterializationPartial("duplicate normalized config key")
    try:
        json.dumps(value.value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise MaterializationPartial(
            f"config value is not safely representable: {error}"
        ) from error
    current[leaf] = value.value


def _config_payload(evidence: tuple[NormalizedEvidence, ...]) -> dict[str, object]:
    payload: dict[str, object] = {}
    found = False
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        for observation in item.observations:
            for config_value in observation.config_values:
                found = True
                _insert_config(payload, config_value)
    if not found:
        raise MaterializationPartial("normalized config evidence is unavailable")
    return payload


def _dataset_artifacts(
    bound: tuple[tuple[NormalizedEvidence, ArtifactBinding], ...],
    role: ExperimentRole,
    captured: Mapping[str, PassiveArtifact],
) -> dict[str, bytes]:
    splits: dict[str, bytes] = {}
    selected = tuple(
        (item, binding)
        for item, binding in bound
        if binding.role is role and binding.kind is ArtifactKind.DATASET
    )
    for item, binding in selected:
        if not str(item.artifact.path).casefold().endswith(".jsonl"):
            raise MaterializationPartial(
                "dataset evidence is not a losslessly supported JSONL artifact"
            )
        references = tuple(
            reference
            for observation in item.observations
            for reference in observation.dataset_references
        )
        if len(references) != 1 or references[0].path != item.artifact.path:
            raise MaterializationPartial(
                "dataset evidence does not identify one selected passive artifact"
            )
        split = binding.dataset_split
        if split not in {DatasetSplit.TRAIN, DatasetSplit.EVAL}:
            raise MaterializationUnavailable(
                "selected dataset mapping lost its validated train/eval split"
            )
        if split.value in splits:
            raise MaterializationUnavailable(
                "selected dataset mapping contains a duplicate split"
            )
        splits[split.value] = captured[str(item.artifact.path)].content
    if set(splits) != {"train", "eval"}:
        raise MaterializationUnavailable(
            "selected dataset mapping must contain one train and one eval split"
        )
    return splits


def _exclusive_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _exclusive_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def _write_native_tree(
    plan_root: Path,
    plan: EphemeralAuditPlan,
    captured: Mapping[str, PassiveArtifact],
    bound: tuple[tuple[NormalizedEvidence, ArtifactBinding], ...],
) -> Path:
    if plan.audit_claim is None:
        raise MaterializationPartial("plan has no deterministic audit claim")
    for role in ("baseline", "candidate"):
        (plan_root / role).mkdir(exist_ok=False)
    by_role = {
        ExperimentRole.BASELINE: plan.baseline_evidence,
        ExperimentRole.CANDIDATE: plan.candidate_evidence,
    }
    for role, evidence in by_role.items():
        role_name = role.value
        result_evidence = tuple(
            item for item in evidence if item.artifact.kind is ArtifactKind.RESULTS
        )
        config_evidence = tuple(
            item for item in evidence if item.artifact.kind is ArtifactKind.CONFIG
        )
        dataset_evidence = tuple(
            item for item in evidence if item.artifact.kind is ArtifactKind.DATASET
        )
        if not result_evidence or not config_evidence or not dataset_evidence:
            raise MaterializationPartial(
                f"{role_name} has an entirely missing native artifact category"
            )
        results = _results_payload(result_evidence, plan.audit_claim.metric)
        config = _config_payload(config_evidence)
        datasets = _dataset_artifacts(bound, role, captured)
        _exclusive_text(
            plan_root / role_name / "results.json",
            json.dumps(
                results,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )
        _exclusive_text(
            plan_root / role_name / "config.yaml",
            yaml.safe_dump(config, sort_keys=True, allow_unicode=False),
        )
        _exclusive_bytes(plan_root / role_name / "train.jsonl", datasets["train"])
        _exclusive_bytes(plan_root / role_name / "eval.jsonl", datasets["eval"])
    manifest = {
        "claim": {
            "metric": plan.audit_claim.metric,
            "direction": plan.audit_claim.direction.value,
            "minimum_improvement": plan.audit_claim.minimum_absolute_improvement,
        },
        "baseline": {
            "config": "baseline/config.yaml",
            "results": "baseline/results.json",
            "train_dataset": "baseline/train.jsonl",
            "eval_dataset": "baseline/eval.jsonl",
        },
        "candidate": {
            "config": "candidate/config.yaml",
            "results": "candidate/results.json",
            "train_dataset": "candidate/train.jsonl",
            "eval_dataset": "candidate/eval.jsonl",
        },
    }
    manifest_path = plan_root / "research.yaml"
    _exclusive_text(
        manifest_path,
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
    )
    return manifest_path


def _cleanup_plan_tree(plan_root: Path, created_identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(plan_root)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise MaterializationUnavailable(
            "created plan tree identity changed before cleanup"
        )
    if (current.st_dev, current.st_ino) != created_identity:
        raise MaterializationUnavailable(
            "created plan tree was substituted before cleanup"
        )
    try:
        shutil.rmtree(plan_root)
    except OSError as error:
        raise MaterializationUnavailable(
            f"created plan tree cleanup failed: {error}"
        ) from error


def execute_ephemeral_audit(
    plan: EphemeralAuditPlan,
    runtime: RuntimeExecutionContext,
) -> AuditResult:
    """Capture, materialize, and run exactly one existing deterministic Audit."""

    if type(plan) is not EphemeralAuditPlan:
        raise TypeError("ephemeral execution requires EphemeralAuditPlan")
    if type(runtime) is not RuntimeExecutionContext:
        raise TypeError("ephemeral execution requires RuntimeExecutionContext")
    _validate_runtime_roots(runtime)
    if not _PLAN_ID.fullmatch(plan.plan_id):
        raise MaterializationUnavailable("plan identity is not canonical")
    try:
        expected_plan_id = derive_ephemeral_plan_id(plan)
    except Exception as error:
        raise MaterializationUnavailable(
            "plan identity cannot be revalidated"
        ) from error
    if plan.plan_id != expected_plan_id:
        raise MaterializationUnavailable("plan identity does not match its inputs")
    if plan.repository != runtime.repository or plan.head_sha != runtime.head_sha:
        raise MaterializationUnavailable(
            "trusted repository or head snapshot no longer matches the plan"
        )
    if plan.missing_evidence:
        raise MaterializationPartial("plan still contains missing evidence")
    if plan.audit_claim is None or plan.audit_claim.minimum_absolute_improvement is None:
        raise MaterializationPartial(
            "the native Audit requires an explicit improvement threshold"
        )
    if plan.selected_mapping is None:
        raise MaterializationPartial("plan has no selected mapping")
    expected_provenance = (
        (plan.selected_mapping.approval_provenance,)
        if type(plan.selected_mapping) is RepoMapping
        else (plan.selected_mapping.provenance,)
    )
    if plan.mapping_provenance != expected_provenance:
        raise MaterializationUnavailable(
            "plan mapping provenance does not match the selected mapping"
        )

    plan_root = runtime.scratch_root / plan.plan_id
    if _paths_overlap(plan_root.resolve(strict=False), runtime.checkout_root):
        raise MaterializationUnavailable(
            "plan tree overlaps the trusted customer repository"
        )
    created = False
    created_identity: tuple[int, int] | None = None
    try:
        bound = _evidence_by_binding(plan)
        captured = _capture_plan_artifacts(runtime, bound)
        try:
            plan_root.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise MaterializationUnavailable(
                "exclusive scratch plan tree already exists"
            ) from error
        created = True
        root_stat = os.lstat(plan_root)
        created_identity = (root_stat.st_dev, root_stat.st_ino)
        manifest = _write_native_tree(plan_root, plan, captured, bound)
        result = audit_research(manifest, artifact_root=plan_root)
        if type(result) is not AuditResult:
            raise MaterializationUnavailable(
                "native Audit did not return an actual AuditResult"
            )
        return result
    except (MaterializationPartial, MaterializationUnavailable):
        raise
    except Exception as error:
        raise MaterializationUnavailable(
            f"ephemeral materialization or Audit failed closed: {error}"
        ) from error
    finally:
        if created and created_identity is not None:
            _cleanup_plan_tree(plan_root, created_identity)


__all__ = [
    "MaterializationError",
    "MaterializationLimits",
    "MaterializationPartial",
    "MaterializationUnavailable",
    "RuntimeExecutionContext",
    "execute_ephemeral_audit",
]
