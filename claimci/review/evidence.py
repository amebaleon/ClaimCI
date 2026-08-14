"""Deterministic, path-confined evidence discovery for extracted claims."""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

from .models import ClaimType, ReviewError, ReviewLimits, ScientificClaim


MAX_EVIDENCE_FILE_BYTES = 16 * 1024 * 1024


class EvidenceKind(str, Enum):
    MANIFEST = "manifest"
    RESULTS = "results"
    CONFIG = "config"
    DATASET = "dataset"
    BENCHMARK = "benchmark"
    SOURCE = "source"
    TEST = "test"
    DOCUMENT = "document"


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    claim_ids: tuple[str, ...]
    kind: EvidenceKind
    path: str
    start_line: int
    end_line: int
    sha256: str
    size: int
    excerpt: str


@dataclass(frozen=True)
class MissingEvidence:
    claim_id: str
    reason: str
    requested_path: str | None = None
    description: str = ""


@dataclass(frozen=True)
class EvidenceBundle:
    references: tuple[EvidenceReference, ...] = ()
    missing: tuple[MissingEvidence, ...] = ()
    total_chars: int = 0


_ROUTES: dict[ClaimType, tuple[str, ...]] = {
    ClaimType.METRIC_IMPROVEMENT: (
        "research.yaml",
        "research.yml",
        "result",
        "metric",
        "score",
        "config",
    ),
    ClaimType.COMPUTE_EQUIVALENCE: (
        "research.yaml",
        "research.yml",
        "config",
        "metadata",
        "run",
        "train",
        "step",
        "compute",
    ),
    ClaimType.HELD_OUT_EVALUATION: (
        "research.yaml",
        "research.yml",
        "eval",
        "held",
        "test",
        "dataset",
        "data/",
    ),
    ClaimType.RESOURCE_REDUCTION: (
        "benchmark",
        "latency",
        "memory",
        "cost",
        "resource",
        "profile",
        "config",
    ),
    ClaimType.COMPONENT_CAUSALITY: (
        "ablation",
        "component",
        "result",
        "experiment",
        "research.yaml",
    ),
    ClaimType.NO_EXTERNAL_REWARD: (
        "reward",
        "config",
        "objective",
        "trainer",
        "loss",
    ),
    ClaimType.IMPLEMENTATION_CLAIM: (
        "src/",
        "lib/",
        "claimci/",
        "tests/",
        "test_",
        ".py",
        ".js",
        ".ts",
        ".rs",
        ".go",
    ),
    ClaimType.OTHER_SCIENTIFIC: (
        "research.yaml",
        "research.yml",
        "readme",
        "paper",
        "note",
    ),
}

_CHANGED_SOURCE_CLAIMS = {
    ClaimType.RESOURCE_REDUCTION,
    ClaimType.COMPONENT_CAUSALITY,
    ClaimType.NO_EXTERNAL_REWARD,
    ClaimType.IMPLEMENTATION_CLAIM,
}

_LITERAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "candidate",
    "experiment",
    "for",
    "from",
    "in",
    "method",
    "model",
    "of",
    "on",
    "or",
    "study",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "under",
    "with",
}


def _tokens(value: str) -> set[str]:
    """Unicode-aware alphanumeric path/claim tokens without underscore glue."""

    return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _root(path: Path) -> Path:
    try:
        resolved = Path(path).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"repository root is invalid: {exc}") from exc
    if not resolved.is_dir():
        raise ReviewError(f"repository root is not a directory: {resolved}")
    return resolved


def _safe_relative(raw: object) -> str | None:
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


def _kind(path: str) -> EvidenceKind:
    lowered = path.casefold().replace("\\", "/")
    name = PurePosixPath(path).name.casefold()
    tokens = _tokens(lowered)
    segments = set(PurePosixPath(lowered).parts)
    if name in {"research.yaml", "research.yml"}:
        return EvidenceKind.MANIFEST
    # File format is authoritative for implementation evidence.  A source
    # filename such as ``config.py`` must not bypass the changed-file policy
    # merely because its stem also names another evidence category.
    if lowered.startswith("tests/") or name.startswith("test_"):
        return EvidenceKind.TEST
    if PurePosixPath(path).suffix.casefold() in {".py", ".js", ".ts", ".rs", ".go"}:
        return EvidenceKind.SOURCE
    if tokens & {"result", "results", "metric", "metrics", "score", "scores"}:
        return EvidenceKind.RESULTS
    if tokens & {"config", "configs", "configuration", "configurations"} or name.endswith((".yaml", ".yml", ".toml")):
        return EvidenceKind.CONFIG
    if tokens & {"benchmark", "benchmarks", "latency", "memory", "cost", "costs"}:
        return EvidenceKind.BENCHMARK
    if tokens & {"data", "dataset", "datasets"} or "data" in segments or name.endswith(".jsonl"):
        return EvidenceKind.DATASET
    return EvidenceKind.DOCUMENT


def _matches(
    claim: ScientificClaim,
    path: str,
    changed_paths: set[str],
) -> bool:
    lowered = path.casefold().replace("\\", "/")
    name = PurePosixPath(lowered).name
    segments = tuple(part for part in PurePosixPath(lowered).parts if part not in {"/", ""})
    tokens = _tokens(lowered)

    if (
        claim.claim_type in _CHANGED_SOURCE_CLAIMS
        and _kind(path) in {EvidenceKind.SOURCE, EvidenceKind.TEST}
        and path not in changed_paths
    ):
        return False

    def route_term_matches(term: str) -> bool:
        term = term.casefold()
        if term.endswith("/"):
            return term[:-1] in segments
        if term.startswith("."):
            return name.endswith(term)
        if term.endswith("_"):
            return name.startswith(term)
        if "." in term:
            return name == term
        aliases = {
            "config": {"configuration", "configurations", "configs"},
            "eval": {"evaluation", "evaluations", "evals"},
            "train": {"training", "trains"},
        }
        forms = {
            term,
            f"{term}s",
            f"{term}es",
            f"{term}ing",
            *aliases.get(term, set()),
        }
        return bool(tokens & forms)

    if any(route_term_matches(term) for term in _ROUTES[claim.claim_type]):
        return True
    source_tokens = _tokens(claim.source_text)
    literal_terms = [claim.metric, claim.subject, *claim.qualifiers]
    for term in literal_terms:
        if not isinstance(term, str):
            continue
        literal_tokens = {
            token
            for token in _tokens(term)
            if len(token) >= 3 or any(character.isdigit() for character in token)
        } & source_tokens - _LITERAL_STOPWORDS
        if tokens & literal_tokens:
            return True
    return False


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _excerpt(path: Path, limit: int) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read(limit)


def _validate_regular(root: Path, relative: str) -> tuple[Path | None, str]:
    try:
        candidate = root / Path(relative)
        cursor = root
        for part in Path(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None, "symbolic links are not eligible evidence"
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return None, "path resolves outside the repository"
    try:
        stat = resolved.stat()
    except OSError:
        return None, "path does not exist"
    if not resolved.is_file() or not os.path.isfile(resolved):
        return None, "path is not a regular file"
    if not stat.st_size >= 0:
        return None, "path has invalid file metadata"
    return resolved, ""


def discover_evidence(
    repository_root: Path,
    claims: Sequence[ScientificClaim],
    repository_paths: Sequence[str],
    *,
    limits: ReviewLimits = ReviewLimits(),
    suggested_paths: Mapping[str, Sequence[str]] | None = None,
    selected_paths: Sequence[str] = (),
    changed_paths: Sequence[str] | None = None,
) -> EvidenceBundle:
    """Select bounded indexed paths within one audit-wide unique-file budget.

    ``selected_paths`` contains repository files already sent during claim
    extraction.  They may be reused without consuming another slot; every new
    evidence path consumes one of the same ``limits.max_files`` slots.
    """

    root = _root(repository_root)
    if not isinstance(limits, ReviewLimits):
        raise ReviewError("limits must be ReviewLimits")
    if not isinstance(claims, Sequence) or not all(
        isinstance(claim, ScientificClaim) for claim in claims
    ):
        raise ReviewError("claims must contain ScientificClaim values")
    normalized_index = {
        normalized
        for raw in repository_paths
        if (normalized := _safe_relative(raw)) is not None
    }
    if not isinstance(selected_paths, Sequence) or isinstance(
        selected_paths, (str, bytes)
    ):
        raise ReviewError("selected evidence paths must be a sequence")
    selected: set[str] = set()
    for raw in selected_paths:
        normalized = _safe_relative(raw)
        if normalized is None or normalized not in normalized_index:
            raise ReviewError("selected evidence path is unsafe or not indexed")
        selected.add(normalized)
    if len(selected) > limits.max_files:
        raise ReviewError("selected evidence paths exceed the global file limit")
    sorted_index = tuple(sorted(normalized_index))
    if changed_paths is None:
        changed: set[str] = set(sorted_index)
    else:
        if not isinstance(changed_paths, Sequence) or isinstance(
            changed_paths, (str, bytes)
        ):
            raise ReviewError("changed evidence paths must be a sequence")
        changed = set()
        for raw in changed_paths:
            normalized = _safe_relative(raw)
            if normalized is None or normalized not in normalized_index:
                raise ReviewError("changed evidence path is unsafe or not indexed")
            changed.add(normalized)
    by_path: dict[str, set[str]] = defaultdict(set)
    missing: list[MissingEvidence] = []

    for claim in claims:
        for path in sorted_index:
            if _matches(claim, path, changed):
                by_path[path].add(claim.claim_id)
        hints = tuple(claim.evidence_hints)
        if suggested_paths is not None:
            raw_hints = suggested_paths.get(claim.claim_id, ())
            if not isinstance(raw_hints, Sequence) or isinstance(raw_hints, (str, bytes)):
                raise ReviewError("suggested evidence paths must be a sequence")
            hints += tuple(raw_hints)
        for raw_hint in hints:
            normalized = _safe_relative(raw_hint)
            if normalized is None:
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="unsafe_path",
                        description="Suggested evidence path is not a confined relative path.",
                    )
                )
                continue
            if not _matches(claim, normalized, changed):
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="route_mismatch",
                        description=(
                            "Suggested evidence path does not match the trusted "
                            "routing rules for this claim type."
                        ),
                    )
                )
                continue
            resolved, reason = _validate_regular(root, normalized)
            if normalized not in normalized_index or resolved is None:
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="not_available" if resolved is None else "not_indexed",
                        description=reason or "Suggested path was not issued by the repository index.",
                    )
                )
                continue
            by_path[normalized].add(claim.claim_id)

    references: list[EvidenceReference] = []
    file_limit_counts: dict[str, int] = defaultdict(int)
    remaining = limits.max_context_chars
    for relative in sorted(by_path):
        if remaining <= 0:
            break
        if relative not in selected and len(selected) >= limits.max_files:
            for claim_id in sorted(by_path[relative]):
                file_limit_counts[claim_id] += 1
            continue
        resolved, reason = _validate_regular(root, relative)
        if resolved is None:
            for claim_id in sorted(by_path[relative]):
                missing.append(
                    MissingEvidence(
                        claim_id=claim_id,
                        requested_path=relative,
                        reason="not_available",
                        description=reason,
                    )
                )
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            size = -1
        if size < 0 or size > MAX_EVIDENCE_FILE_BYTES:
            for claim_id in sorted(by_path[relative]):
                missing.append(
                    MissingEvidence(
                        claim_id=claim_id,
                        requested_path=relative,
                        reason="too_large",
                        description=(
                            "Evidence file exceeds the bounded inspection size."
                        ),
                    )
                )
            continue
        char_limit = min(limits.max_file_chars, remaining)
        try:
            excerpt = _excerpt(resolved, char_limit)
            digest = _digest(resolved)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            for claim_id in sorted(by_path[relative]):
                missing.append(
                    MissingEvidence(
                        claim_id=claim_id,
                        requested_path=relative,
                        reason="unreadable",
                        description=f"Evidence file could not be read: {type(exc).__name__}.",
                    )
                )
            continue
        if not excerpt:
            continue
        evidence_id = "evidence-" + hashlib.sha256(
            f"{relative}\0{digest}".encode("utf-8")
        ).hexdigest()[:16]
        references.append(
            EvidenceReference(
                evidence_id=evidence_id,
                claim_ids=tuple(sorted(by_path[relative])),
                kind=_kind(relative),
                path=relative,
                start_line=1,
                end_line=max(1, excerpt.count("\n") + (0 if excerpt.endswith("\n") else 1)),
                sha256=digest,
                size=size,
                excerpt=excerpt,
            )
        )
        selected.add(relative)
        remaining -= len(excerpt)

    for claim_id, omitted_count in sorted(file_limit_counts.items()):
        missing.append(
            MissingEvidence(
                claim_id=claim_id,
                reason="file_limit",
                description=(
                    f"{omitted_count} matching evidence path(s) were not inspected "
                    "because the global selected-file limit was reached."
                ),
            )
        )

    covered = {claim_id for reference in references for claim_id in reference.claim_ids}
    for claim in claims:
        if claim.claim_id not in covered and not any(
            item.claim_id == claim.claim_id for item in missing
        ):
            missing.append(
                MissingEvidence(
                    claim_id=claim.claim_id,
                    reason="no_matching_evidence",
                    description="No bounded indexed artifact matched this claim type.",
                )
            )
    return EvidenceBundle(
        references=tuple(references),
        missing=tuple(
            sorted(
                missing,
                key=lambda item: (
                    item.claim_id,
                    item.requested_path or "",
                    item.reason,
                ),
            )
        ),
        total_chars=sum(len(reference.excerpt) for reference in references),
    )
