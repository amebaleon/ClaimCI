"""Passive artifact classification, ranking, and legacy manifest hints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import yaml

from claimci.analysis.confidence import Confidence
from claimci.analysis.contracts import (
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    DatasetSplit,
    ExperimentRole,
    FieldProvenance,
    MappingCandidate,
    MappingTrust,
    ProvenanceKind,
    RepoMapping,
    RepositoryPath,
)
from claimci.models import coerce_direction
from claimci.parsing import load_unique_yaml
from claimci.review.evidence import EvidenceKind, discover_evidence
from claimci.review.models import ClaimType, ReviewError, ReviewLimits
from claimci.review.tools import discover_manifests

from .claims import as_scientific_claim
from .models import ArtifactIssue, DiscoveredClaim, DiscoveryError, DiscoveryLimits
from .repository import (
    ArtifactInspectionBudget,
    ChangeComparisonBudget,
    RepositoryContext,
    artifact_changed,
    inspect_artifact,
    read_artifact_text,
)


@dataclass(frozen=True, slots=True)
class ArtifactDiscovery:
    artifacts: tuple[ArtifactCandidate, ...]
    manifest_mappings: tuple[MappingCandidate, ...]
    issues: tuple[ArtifactIssue, ...]


_EVIDENCE_KINDS = {
    EvidenceKind.MANIFEST: ArtifactKind.MANIFEST,
    EvidenceKind.RESULTS: ArtifactKind.RESULTS,
    EvidenceKind.CONFIG: ArtifactKind.CONFIG,
    EvidenceKind.DATASET: ArtifactKind.DATASET,
    EvidenceKind.BENCHMARK: ArtifactKind.BENCHMARK,
    EvidenceKind.SOURCE: ArtifactKind.SOURCE,
    EvidenceKind.TEST: ArtifactKind.TEST,
    EvidenceKind.DOCUMENT: ArtifactKind.DOCUMENT,
}
_ARTIFACT_FIELDS = {
    "config": (ArtifactKind.CONFIG, None),
    "results": (ArtifactKind.RESULTS, None),
    "train_dataset": (ArtifactKind.DATASET, DatasetSplit.TRAIN),
    "eval_dataset": (ArtifactKind.DATASET, DatasetSplit.EVAL),
}
_TOKENS = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ROLE_TOKENS = frozenset(
    {"baseline", "base", "control", "reference", "candidate", "proposed", "new"}
)
_KIND_BASE_SCORE = {
    ArtifactKind.RESULTS: 0.55,
    ArtifactKind.CONFIG: 0.50,
    ArtifactKind.BENCHMARK: 0.46,
    ArtifactKind.DATASET: 0.42,
    ArtifactKind.TEST: 0.40,
    ArtifactKind.SOURCE: 0.38,
    ArtifactKind.DOCUMENT: 0.30,
    ArtifactKind.MANIFEST: 0.95,
}
_CLAIM_KIND_ROUTES = {
    ClaimType.METRIC_IMPROVEMENT: {
        ArtifactKind.RESULTS,
        ArtifactKind.CONFIG,
        ArtifactKind.DATASET,
        ArtifactKind.MANIFEST,
    },
    ClaimType.COMPUTE_EQUIVALENCE: {
        ArtifactKind.CONFIG,
        ArtifactKind.RESULTS,
        ArtifactKind.BENCHMARK,
        ArtifactKind.MANIFEST,
    },
    ClaimType.HELD_OUT_EVALUATION: {
        ArtifactKind.DATASET,
        ArtifactKind.RESULTS,
        ArtifactKind.MANIFEST,
    },
    ClaimType.RESOURCE_REDUCTION: {
        ArtifactKind.BENCHMARK,
        ArtifactKind.CONFIG,
        ArtifactKind.RESULTS,
    },
    ClaimType.COMPONENT_CAUSALITY: {
        ArtifactKind.RESULTS,
        ArtifactKind.CONFIG,
        ArtifactKind.SOURCE,
        ArtifactKind.TEST,
        ArtifactKind.MANIFEST,
    },
    ClaimType.NO_EXTERNAL_REWARD: {
        ArtifactKind.CONFIG,
        ArtifactKind.SOURCE,
        ArtifactKind.TEST,
    },
    ClaimType.IMPLEMENTATION_CLAIM: {ArtifactKind.SOURCE, ArtifactKind.TEST},
    ClaimType.OTHER_SCIENTIFIC: {
        ArtifactKind.DOCUMENT,
        ArtifactKind.RESULTS,
        ArtifactKind.MANIFEST,
    },
}


def _tokens(value: str) -> set[str]:
    return set(_TOKENS.findall(value.casefold()))


def _classify_path(path: RepositoryPath) -> ArtifactKind | None:
    pure = PurePosixPath(path)
    lowered = str(path).casefold()
    name = pure.name.casefold()
    tokens = _tokens(lowered)
    segments = set(pure.parts)
    suffix = pure.suffix.casefold()
    if name in {"research.yaml", "research.yml"}:
        return ArtifactKind.MANIFEST
    if "tests" in segments or name.startswith(("test_", "test-")):
        return ArtifactKind.TEST
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go"}:
        return ArtifactKind.SOURCE
    if tokens & {"result", "results", "metric", "metrics", "score", "scores"}:
        return ArtifactKind.RESULTS
    if tokens & {"config", "configs", "configuration", "configurations"} or suffix in {
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    }:
        return ArtifactKind.CONFIG
    if tokens & {"benchmark", "benchmarks", "latency", "memory", "cost", "profile"}:
        return ArtifactKind.BENCHMARK
    if tokens & {"data", "dataset", "datasets", "train", "eval", "evaluation"} or suffix == ".jsonl":
        return ArtifactKind.DATASET
    if suffix in {".csv", ".tsv"}:
        return ArtifactKind.BENCHMARK
    if suffix in {".md", ".markdown", ".tex", ".txt"}:
        return ArtifactKind.DOCUMENT
    return None


def _resolve_manifest_path(manifest: RepositoryPath, raw: object) -> RepositoryPath:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or "\x00" in raw
        or "\\" in raw
        or PurePosixPath(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or bool(PureWindowsPath(raw).drive)
    ):
        raise DiscoveryError("manifest artifact path is unsafe")
    parts = list(PurePosixPath(manifest).parent.parts)
    for part in PurePosixPath(raw).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise DiscoveryError("manifest artifact escapes repository root")
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise DiscoveryError("manifest artifact path is empty")
    return RepositoryPath("/".join(parts))


def _manifest_hint(
    context: RepositoryContext,
    manifest_path: RepositoryPath,
    *,
    limits: DiscoveryLimits,
    inspection_budget: ArtifactInspectionBudget,
) -> tuple[MappingCandidate | None, ArtifactIssue | None]:
    text_result = read_artifact_text(context, manifest_path, limits=limits)
    if isinstance(text_result, ArtifactIssue):
        return None, text_result
    try:
        payload = load_unique_yaml(text_result[1])
        if not isinstance(payload, Mapping):
            raise DiscoveryError("manifest root must be a mapping")
        claim = payload.get("claim")
        if not isinstance(claim, Mapping):
            raise DiscoveryError("manifest claim must be a mapping")
        metric = claim.get("metric")
        threshold = claim.get("minimum_improvement")
        if not isinstance(metric, str) or not metric.strip():
            raise DiscoveryError("manifest metric is invalid")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0
        ):
            raise DiscoveryError("manifest threshold is invalid")
        coerce_direction(claim.get("direction", "higher"))
        provenance = FieldProvenance(
            kind=ProvenanceKind.MANIFEST_HINT,
            detail=f"passive mapping declared by {manifest_path}",
            source_path=manifest_path,
            source_id=f"manifest:{manifest_path}",
        )
        issued = set(context.repository_paths)
        bindings: list[ArtifactBinding] = []
        for role_name, role in (
            ("baseline", ExperimentRole.BASELINE),
            ("candidate", ExperimentRole.CANDIDATE),
        ):
            experiment = payload.get(role_name)
            if not isinstance(experiment, Mapping):
                raise DiscoveryError("manifest experiment mapping is invalid")
            for field, (kind, dataset_split) in _ARTIFACT_FIELDS.items():
                path = _resolve_manifest_path(manifest_path, experiment.get(field))
                if path not in issued:
                    raise DiscoveryError("manifest artifact is not indexed")
                inspection = inspect_artifact(
                    context,
                    path,
                    limits=limits,
                    budget=inspection_budget,
                )
                if isinstance(inspection, ArtifactIssue):
                    raise DiscoveryError("manifest artifact is not usable")
                bindings.append(
                    ArtifactBinding(
                        path=path,
                        kind=kind,
                        role=role,
                        adapter_id=None,
                        mappings=(),
                        provenance=provenance,
                        dataset_split=dataset_split,
                    )
                )
        material = json.dumps(
            {
                "manifest": str(manifest_path),
                "bindings": [
                    (
                        str(binding.path),
                        binding.kind.value,
                        binding.role.value,
                        (
                            binding.dataset_split.value
                            if binding.dataset_split is not None
                            else None
                        ),
                    )
                    for binding in bindings
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        mapping = MappingCandidate(
            mapping_id="mapping-manifest-" + hashlib.sha256(material).hexdigest()[:16],
            bindings=tuple(bindings),
            confidence=Confidence(0.99),
            trust=MappingTrust.MANIFEST_HINT,
            provenance=provenance,
        )
        return mapping, None
    except (
        DiscoveryError,
        TypeError,
        ValueError,
        RecursionError,
        yaml.YAMLError,
    ):
        return None, ArtifactIssue(
            manifest_path,
            "manifest is not a valid confined mapping hint",
        )


def _claim_ids_for_artifact(
    claims: tuple[DiscoveredClaim, ...],
    kind: ArtifactKind,
    path: RepositoryPath,
) -> set[str]:
    routed = tuple(
        claim for claim in claims if kind in _CLAIM_KIND_ROUTES[claim.claim_type]
    )
    path_tokens = _tokens(str(path))
    metric_matched = {
        claim.reference.claim_id
        for claim in routed
        if claim.metric and path_tokens & _tokens(claim.metric)
    }
    if metric_matched:
        return metric_matched
    matched = {
        claim.reference.claim_id
        for claim in routed
        if any(
            literal and path_tokens & _tokens(literal)
            for literal in (claim.subject, *claim.qualifiers)
        )
    }
    if matched:
        return matched
    return {claim.reference.claim_id for claim in routed}


def _path_has_claim_literal(path: RepositoryPath, claims: tuple[DiscoveredClaim, ...]) -> bool:
    path_tokens = _tokens(str(path))
    for claim in claims:
        literals = [claim.metric, claim.subject, *claim.qualifiers]
        for literal in literals:
            if literal and path_tokens & _tokens(literal):
                return True
    return False


def discover_artifacts(
    context: RepositoryContext,
    claims: tuple[DiscoveredClaim, ...],
    *,
    approved_mapping: RepoMapping | None = None,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> ArtifactDiscovery:
    """Return hash-bound passive candidates and non-authoritative manifest hints."""

    if not isinstance(context, RepositoryContext):
        raise TypeError("repository context must be RepositoryContext")
    if not isinstance(claims, tuple) or not all(
        isinstance(claim, DiscoveredClaim) for claim in claims
    ):
        raise TypeError("claims must be a tuple of DiscoveredClaim values")
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("discovery limits must be DiscoveryLimits")
    if approved_mapping is not None and not isinstance(approved_mapping, RepoMapping):
        raise TypeError("approved_mapping must be RepoMapping or null")

    manifest_mappings: list[MappingCandidate] = []
    issue_by_path: dict[RepositoryPath, ArtifactIssue] = {}
    valid_manifests: set[RepositoryPath] = set()
    inspection_budget = ArtifactInspectionBudget.from_limits(limits)
    for raw in discover_manifests(
        context.head_root,
        tuple(str(path) for path in context.repository_paths),
        max_manifests=min(4, limits.max_artifact_candidates),
    ):
        path = RepositoryPath(raw)
        mapping, issue = _manifest_hint(
            context,
            path,
            limits=limits,
            inspection_budget=inspection_budget,
        )
        if issue is not None:
            issue_by_path.setdefault(issue.path, issue)
        elif mapping is not None:
            valid_manifests.add(path)
            manifest_mappings.append(mapping)

    scientific = tuple(as_scientific_claim(claim) for claim in claims)
    suggested = {
        claim.reference.claim_id: tuple(str(path) for path in claim.evidence_hints)
        for claim in claims
        if claim.evidence_hints
    }
    try:
        evidence = discover_evidence(
            context.head_root,
            scientific,
            tuple(str(path) for path in context.repository_paths),
            limits=ReviewLimits(),
            suggested_paths=suggested or None,
            changed_paths=tuple(str(path) for path in context.changed_paths),
            base_root=context.base_root,
        )
    except ReviewError as error:
        raise DiscoveryError("bounded evidence routing failed") from error

    kinds: dict[RepositoryPath, ArtifactKind] = {}
    claim_ids: dict[RepositoryPath, set[str]] = defaultdict(set)
    evidence_paths: set[RepositoryPath] = set()
    for reference in evidence.references:
        path = RepositoryPath(reference.path)
        kind = _EVIDENCE_KINDS[reference.kind]
        kinds[path] = kind
        routed = _claim_ids_for_artifact(claims, kind, path)
        referenced = set(reference.claim_ids)
        claim_ids[path].update(referenced & routed or referenced)
        evidence_paths.add(path)

    supplement_kinds = {
        ArtifactKind.RESULTS,
        ArtifactKind.CONFIG,
        ArtifactKind.DATASET,
        ArtifactKind.BENCHMARK,
        ArtifactKind.MANIFEST,
    }
    for path in context.repository_paths:
        kind = _classify_path(path)
        if kind in supplement_kinds:
            kinds.setdefault(path, kind)
            claim_ids[path].update(_claim_ids_for_artifact(claims, kind, path))
    approved_paths: set[RepositoryPath] = set()
    if approved_mapping is not None:
        for binding in approved_mapping.bindings:
            if binding.path in set(context.repository_paths):
                approved_paths.add(binding.path)
                kinds[binding.path] = binding.kind
                claim_ids[binding.path].update(
                    _claim_ids_for_artifact(claims, binding.kind, binding.path)
                )

    manifest_paths = {
        path for path, kind in kinds.items() if kind is ArtifactKind.MANIFEST
    }
    for invalid in manifest_paths - valid_manifests:
        kinds.pop(invalid, None)
        claim_ids.pop(invalid, None)

    preliminary = sorted(
        kinds,
        key=lambda path: (
            0 if path in approved_paths else 1,
            0 if path in valid_manifests else 1,
            0 if path in evidence_paths else 1,
            -_KIND_BASE_SCORE[kinds[path]],
            str(path).casefold(),
            str(path),
        ),
    )[: limits.max_artifact_candidates]
    comparison_budget = ChangeComparisonBudget.from_limits(limits)
    artifacts: list[ArtifactCandidate] = []
    hint_paths = {
        path for claim in claims for path in claim.evidence_hints
    }
    for path in preliminary:
        inspection = inspect_artifact(
            context,
            path,
            limits=limits,
            budget=inspection_budget,
        )
        if isinstance(inspection, ArtifactIssue):
            issue_by_path.setdefault(path, inspection)
            continue
        kind = kinds[path]
        changed = artifact_changed(context, inspection, budget=comparison_budget)
        reasons: list[str] = []
        score = _KIND_BASE_SCORE[kind]
        provenance_kind = ProvenanceKind.DETERMINISTIC_DISCOVERY
        if path in approved_paths:
            score += 0.18
            reasons.append("approved repository mapping path")
        if path in evidence_paths:
            score += 0.15
            reasons.append("claim-routed evidence match")
        if changed is True:
            score += 0.08
            reasons.append("head artifact differs from base")
        elif changed is None:
            reasons.append("base comparison unavailable within bounds")
        path_tokens = _tokens(str(path))
        if path_tokens & _ROLE_TOKENS:
            score += 0.08
            reasons.append("explicit experiment-role path token")
        if _path_has_claim_literal(path, claims):
            score += 0.04
            reasons.append("claim literal appears in path")
        if path in hint_paths:
            score += 0.05
            reasons.append("validated evidence hint")
        if path in valid_manifests:
            score = 0.99
            provenance_kind = ProvenanceKind.MANIFEST_HINT
            reasons = ["valid passive research manifest hint"]
            claim_ids[path].update(claim.reference.claim_id for claim in claims)
        confidence = Confidence(min(0.99, max(0.0, score)))
        provenance = FieldProvenance(
            kind=provenance_kind,
            detail="; ".join(reasons) or "deterministic artifact path classification",
            source_path=path,
        )
        artifacts.append(
            ArtifactCandidate(
                path=path,
                kind=kind,
                sha256=inspection.sha256,
                size=inspection.size,
                confidence=confidence,
                discovery_reason=provenance.detail,
                relevant_claim_ids=tuple(sorted(claim_ids[path])),
                provenance=provenance,
            )
        )

    artifacts.sort(
        key=lambda item: (
            -item.confidence.value,
            str(item.path).casefold(),
            str(item.path),
            item.kind.value,
            item.relevant_claim_ids,
        )
    )
    manifest_mappings.sort(
        key=lambda item: (-item.confidence.value, item.mapping_id)
    )
    issues = tuple(
        issue_by_path[path]
        for path in sorted(issue_by_path, key=lambda item: (str(item).casefold(), str(item)))
    )
    return ArtifactDiscovery(
        artifacts=tuple(artifacts),
        manifest_mappings=tuple(manifest_mappings),
        issues=issues,
    )


__all__ = ["ArtifactDiscovery", "discover_artifacts"]
