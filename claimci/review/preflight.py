"""Provider-free deterministic planning for a bounded review material scope."""

from __future__ import annotations

import re
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from claimci.passive_files import PassiveFileError, capture_confined_regular_file

from .models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    DeclaredReviewCoordinates,
    GateDisposition,
    MaterialClaimSeed,
    PreflightGateResult,
    ReviewConfig,
    ReviewError,
    ReviewInventoryFailure,
    ReviewMaterialKind,
    ReviewPreflight,
    ReviewScope,
    ReviewStatus,
    ScopeIssue,
    SnapshotIdentity,
    SnapshotRole,
    SourceBundle,
    SourceKind,
    SourceRecord,
)
from .path_policy import classify_review_material
from .inventory import (
    MAX_CHANGESET_METADATA_BYTES,
    InventoryVerificationError,
    build_git_change_inventory,
)
from .request_budget import (
    build_extraction_request_parts,
    build_synthesis_request_parts,
    logical_request_chars,
    output_budget_ready,
    plain,
    worst_valid_synthesis_request_chars,
)
from .sources import (
    MAX_CHANGE_COMPARISON_BYTES,
    MAX_CHANGE_COMPARISON_FILE_BYTES,
    MAX_CHANGE_COMPARISON_FILES,
    MAX_REPOSITORY_DEPTH,
    MAX_REPOSITORY_ENTRIES,
    MAX_REPOSITORY_PATHS,
    MAX_SOURCE_FILE_BYTES,
    _record,
    source_bundle_from_scope,
)


class _ScopeInputs(Protocol):
    repository_root: Path
    pr_title: str
    pr_description: str
    requested_base: SnapshotIdentity | None
    comparison_base: SnapshotIdentity | None
    head: SnapshotIdentity | None
    inventory: ChangeInventory | None
    coordinates: DeclaredReviewCoordinates | None
    inventory_failure: ReviewInventoryFailure | None


_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "benchmark",
        re.compile(
            r"\b(?:benchmark|evaluation|evaluate|eval|wer|rtfx|accuracy|latency|throughput|score)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "comparative_causal",
        re.compile(
            r"\b(?:improv(?:e|es|ed|ement)|reduc(?:e|es|ed|tion)|faster|slower|higher|lower|"
            r"outperform(?:s|ed)?|because|caus(?:e|es|ed|al)|enabl(?:e|es|ed)|"
            r"versus|compared?|comparison|than)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "model_dataset",
        re.compile(
            r"\b(?:model|dataset|corpus|checkpoint|transformer|parameters?|qwen|glm|asr)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "quantitative",
        re.compile(
            r"(?:\b\d+(?:\.\d+)?\s*(?:%|x\b|ms\b|s\b|sec(?:ond)?s?\b|min(?:ute)?s?\b|"
            r"hours?\b|kb\b|mb\b|gb\b|tb\b|tokens?\b|params?\b|parameters?\b|samples?\b|"
            r"runs?\b|epochs?\b|files?\b|examples?\b|datasets?\b|models?\b|batches?\b|"
            r"wer\b|rtfx\b))|(?:\b\d+\s*[:/]\s*\d+\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "reproducibility",
        re.compile(
            r"\b(?:reproduc(?:e|es|ed|ible|ibility)|seed|revision|commit|deterministic|"
            r"configuration|config|batch|hardware|environment|version)\b",
            re.IGNORECASE,
        ),
    ),
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[_.+-][a-z0-9]+)*")
_PATH_REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.+\\/-])"
    r"(?:"
    r"((?:[A-Za-z]:[\\/]|\\\\|/|(?:\.\.?[\\/])+))"
    r"([A-Za-z0-9_.+-]+(?:[\\/][A-Za-z0-9_.+-]+)*)"
    r"|"
    r"([A-Za-z0-9_.+-]+(?:[\\/][A-Za-z0-9_.+-]+)+)"
    r")"
    r"(?![A-Za-z0-9_+\\/-])"
)
_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "see",
        "the",
        "this",
        "to",
        "with",
    }
)

_CATEGORY_KINDS: dict[str, frozenset[ReviewMaterialKind]] = {
    "benchmark": frozenset(
        {
            ReviewMaterialKind.BENCHMARK,
            ReviewMaterialKind.RESULT,
            ReviewMaterialKind.SUBMISSION_CONFIG,
        }
    ),
    "comparative_causal": frozenset(
        {
            ReviewMaterialKind.BENCHMARK,
            ReviewMaterialKind.RESULT,
            ReviewMaterialKind.SOURCE,
            ReviewMaterialKind.TEST,
        }
    ),
    "model_dataset": frozenset(
        {
            ReviewMaterialKind.CONFIG,
            ReviewMaterialKind.MANIFEST,
            ReviewMaterialKind.SOURCE,
            ReviewMaterialKind.SUBMISSION_CONFIG,
        }
    ),
    "quantitative": frozenset(
        {ReviewMaterialKind.BENCHMARK, ReviewMaterialKind.RESULT}
    ),
    "reproducibility": frozenset(
        {
            ReviewMaterialKind.CONFIG,
            ReviewMaterialKind.MANIFEST,
            ReviewMaterialKind.SOURCE,
            ReviewMaterialKind.SUBMISSION_CONFIG,
            ReviewMaterialKind.TEST,
        }
    ),
}

_KIND_PRIORITY = {
    ReviewMaterialKind.DOCUMENT: 0,
    ReviewMaterialKind.BENCHMARK: 1,
    ReviewMaterialKind.RESULT: 2,
    ReviewMaterialKind.MANIFEST: 3,
    ReviewMaterialKind.SUBMISSION_CONFIG: 4,
    ReviewMaterialKind.TEST: 5,
    ReviewMaterialKind.SOURCE: 6,
    ReviewMaterialKind.CONFIG: 7,
    ReviewMaterialKind.OTHER: 8,
}


def _normalized_text(value: str, limit: int) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")[:limit]


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_PATTERN.findall(value.casefold())
        if token not in _STOP_TERMS and len(token) <= 128
    )


def _path_tokens(path: str) -> frozenset[str]:
    return _tokens(" ".join(PurePosixPath(path).parts))


@dataclass(frozen=True)
class _PathReferences:
    canonical: frozenset[str]
    unsafe_count: int


def _path_references(value: str) -> _PathReferences:
    canonical: set[str] = set()
    unsafe_count = 0
    for match in _PATH_REFERENCE_PATTERN.finditer(value):
        candidate = "".join(group or "" for group in match.groups()).rstrip(
            ".,;:!?)]}"
        )
        parts = candidate.split("/")
        if (
            not candidate
            or "\\" in candidate
            or candidate.startswith("/")
            or re.match(r"^[A-Za-z]:", candidate)
            or any(part in {"", ".", ".."} for part in parts)
            or PurePosixPath(candidate).is_absolute()
        ):
            unsafe_count += 1
            continue
        if PurePosixPath(candidate).as_posix() == candidate:
            canonical.add(candidate)
        else:
            unsafe_count += 1
    return _PathReferences(frozenset(canonical), unsafe_count)


def _seeds(records: tuple[SourceRecord, ...], max_seeds: int) -> tuple[MaterialClaimSeed, ...]:
    seeds: list[MaterialClaimSeed] = []
    for record in records:
        route_terms = tuple(sorted(_tokens(record.text)))[:24]
        if not route_terms:
            continue
        for category, pattern in _CATEGORY_PATTERNS:
            if pattern.search(record.text):
                seeds.append(
                    MaterialClaimSeed(
                        origin_source_id=record.source_id,
                        category=category,
                        route_terms=route_terms,
                    )
                )
                if len(seeds) >= max_seeds:
                    return tuple(seeds)
    return tuple(seeds)


def _source_path_references(
    records: tuple[SourceRecord, ...],
) -> dict[str, _PathReferences]:
    return {record.source_id: _path_references(record.text) for record in records}


def _exact_mentions(
    path: str,
    seed: MaterialClaimSeed,
    references: dict[str, _PathReferences],
) -> bool:
    source_references = references.get(seed.origin_source_id)
    return source_references is not None and path in source_references.canonical


def _kind_agrees(kind: ReviewMaterialKind, seed: MaterialClaimSeed) -> bool:
    return kind in _CATEGORY_KINDS.get(seed.category, frozenset())


def _rank_key(
    path: str,
    kind: ReviewMaterialKind,
    seeds: tuple[MaterialClaimSeed, ...],
    references: dict[str, _PathReferences],
) -> tuple[int, int, int, int, str]:
    exact = any(_exact_mentions(path, seed, references) for seed in seeds)
    category = any(_kind_agrees(kind, seed) for seed in seeds)
    path_terms = _path_tokens(path)
    overlap = max(
        (len(path_terms.intersection(seed.route_terms)) for seed in seeds),
        default=0,
    )
    return (-int(exact), -int(category), -overlap, _KIND_PRIORITY[kind], path)


def _issue_sort_key(issue: ScopeIssue) -> tuple[str, str, int, int]:
    return (
        issue.code,
        issue.path or "",
        -1 if issue.observed is None else issue.observed,
        -1 if issue.limit is None else issue.limit,
    )


def build_review_scope(
    inputs: _ScopeInputs,
    config: ReviewConfig,
    inventory: ChangeInventory,
) -> ReviewScope:
    """Plan and validate only a bounded subset of a declared change inventory."""

    if not isinstance(config, ReviewConfig):
        raise ReviewError("config must be ReviewConfig")
    if not isinstance(inventory, ChangeInventory):
        raise ReviewError("inventory must be ChangeInventory")
    try:
        requested_root = Path(inputs.repository_root)
        if requested_root.is_symlink():
            raise ReviewError("head repository root is invalid")
        root = requested_root.resolve(strict=True)
    except ReviewError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReviewError("head repository root is invalid") from exc
    if not root.is_dir() or root.is_symlink():
        raise ReviewError("head repository root is invalid")
    if not isinstance(inputs.pr_title, str) or not isinstance(inputs.pr_description, str):
        raise ReviewError("pull-request title and description must be strings")

    limits = config.limits
    records: list[SourceRecord] = []
    remaining_chars = limits.max_context_chars
    for kind, value in (
        (SourceKind.PULL_REQUEST_TITLE, inputs.pr_title),
        (SourceKind.PULL_REQUEST_DESCRIPTION, inputs.pr_description),
    ):
        text = _normalized_text(value, remaining_chars)
        if text:
            records.append(_record(kind, None, text))
            remaining_chars -= len(text)

    metadata_records = tuple(records)
    initial_seeds = _seeds(metadata_records, limits.max_claims)
    initial_references = _source_path_references(metadata_records)
    classified = tuple(
        (entry, classify_review_material(entry.path)) for entry in inventory.entries
    )
    candidates = [
        (entry, kind)
        for entry, kind in classified
        if entry.status is not ChangeStatus.DELETED
        and kind is not ReviewMaterialKind.OTHER
    ]

    selected: list[str] = []
    materialized_path_chars: dict[str, int] = {}
    issues: list[ScopeIssue] = []
    materialized_chars = sum(len(record.text) for record in records)
    if not inventory.complete:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGESET_INCOMPLETE"))

    def materialize(shortlist: list[tuple[ChangeEntry, ReviewMaterialKind]]) -> None:
        nonlocal materialized_chars, remaining_chars
        for entry, kind in shortlist:
            if remaining_chars <= 0:
                issues.append(
                    ScopeIssue(
                        code="PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE",
                        path=entry.path,
                        observed=0,
                        limit=0,
                    )
                )
                continue
            try:
                capture = capture_confined_regular_file(
                    root,
                    entry.path,
                    max_bytes=MAX_SOURCE_FILE_BYTES,
                )
                decoded = capture.content.decode("utf-8", errors="strict")
            except PassiveFileError as exc:
                code = {
                    "outside": "PREFLIGHT_G1_MATERIAL_PATH_OUTSIDE_ROOT",
                    "unsafe_path": "PREFLIGHT_G1_MATERIAL_PATH_OUTSIDE_ROOT",
                    "symlink": "PREFLIGHT_G1_MATERIAL_PATH_SYMLINK",
                    "not_regular": "PREFLIGHT_G1_MATERIAL_PATH_NOT_REGULAR",
                    "unavailable": "PREFLIGHT_G1_MATERIAL_PATH_NOT_REGULAR",
                    "changed": "PREFLIGHT_G1_MATERIAL_PATH_IDENTITY_CHANGED",
                    "too_large": "PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
                }.get(exc.code, "PREFLIGHT_G2_CANDIDATE_UNREADABLE")
                issues.append(
                    ScopeIssue(
                        code=code,
                        path=entry.path,
                        observed=(
                            MAX_SOURCE_FILE_BYTES + 1
                            if exc.code == "too_large"
                            else None
                        ),
                        limit=(MAX_SOURCE_FILE_BYTES if exc.code == "too_large" else None),
                    )
                )
                continue
            except (UnicodeError, ValueError, RecursionError):
                issues.append(
                    ScopeIssue(
                        code="PREFLIGHT_G2_CANDIDATE_UNREADABLE",
                        path=entry.path,
                    )
                )
                continue

            normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
            excerpt_limit = min(limits.max_file_chars, remaining_chars)
            text = normalized[:excerpt_limit]
            if len(normalized) > len(text):
                issues.append(
                    ScopeIssue(
                        code="PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE",
                        path=entry.path,
                        observed=len(normalized),
                        limit=excerpt_limit,
                    )
                )
            selected.append(entry.path)
            materialized_path_chars[entry.path] = len(text)
            materialized_chars += len(text)
            remaining_chars -= len(text)
            if kind is ReviewMaterialKind.DOCUMENT:
                records.append(
                    _record(
                        SourceKind.REPOSITORY_FILE,
                        entry.path,
                        text,
                    )
                )

    documents = [
        item for item in candidates if item[1] is ReviewMaterialKind.DOCUMENT
    ]
    documents.sort(
        key=lambda item: _rank_key(
            item[0].path,
            item[1],
            initial_seeds,
            initial_references,
        )
    )
    document_shortlist = documents[: limits.max_files]
    materialize(document_shortlist)

    document_seed_records = tuple(records)
    planning_seeds = _seeds(document_seed_records, limits.max_claims)
    planning_references = _source_path_references(document_seed_records)
    remaining_candidates = [
        item for item in candidates if item[1] is not ReviewMaterialKind.DOCUMENT
    ]
    remaining_candidates.sort(
        key=lambda item: _rank_key(
            item[0].path,
            item[1],
            planning_seeds,
            planning_references,
        )
    )
    remaining_slots = limits.max_files - len(document_shortlist)
    material_shortlist = remaining_candidates[:remaining_slots]
    materialize(material_shortlist)

    shortlisted_paths = {
        entry.path for entry, _kind in document_shortlist + material_shortlist
    }
    omitted = [
        entry.path
        for entry, _kind in candidates
        if entry.path not in shortlisted_paths
    ]
    for path in omitted:
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED",
                path=path,
                observed=len(candidates),
                limit=limits.max_files,
            )
        )

    all_records = tuple(records)
    seeds = _seeds(all_records, limits.max_claims)
    references = _source_path_references(all_records)
    selected_kinds = {
        path: classify_review_material(path) for path in selected
    }
    unrouted_seed_count = 0
    for seed in seeds:
        if any(
            _exact_mentions(path, seed, references) or _kind_agrees(kind, seed)
            for path, kind in selected_kinds.items()
        ):
            continue
        else:
            unrouted_seed_count += 1
    if unrouted_seed_count:
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G2_MATERIAL_SEED_UNROUTED",
                observed=unrouted_seed_count,
                limit=len(seeds),
            )
        )

    nondeleted_entries = tuple(
        entry for entry, _kind in classified if entry.status is not ChangeStatus.DELETED
    )
    eligible_deleted = tuple(
        entry
        for entry, kind in classified
        if entry.status is ChangeStatus.DELETED and kind is not ReviewMaterialKind.OTHER
    )
    if not seeds:
        issues.append(ScopeIssue(code="PREFLIGHT_G2_NO_MATERIAL_CLAIM_SEED"))
    if seeds and not selected:
        issues.append(ScopeIssue(code="PREFLIGHT_G2_NO_ROUTABLE_CHANGED_PATH"))
    all_path_mentions = frozenset(
        path
        for source_references in references.values()
        for path in source_references.canonical
    )
    mentioned_deleted = tuple(
        entry for entry in eligible_deleted if entry.path in all_path_mentions
    )
    for entry in mentioned_deleted:
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH",
                path=entry.path,
            )
        )
    if eligible_deleted and not candidates and not mentioned_deleted:
        issues.append(ScopeIssue(code="PREFLIGHT_G2_ONLY_DELETED_ROUTABLE_PATH"))
    if nondeleted_entries and not candidates:
        issues.append(ScopeIssue(code="PREFLIGHT_G2_UNSUPPORTED_MATERIAL_TYPE"))

    inventory_paths = {entry.path for entry in inventory.entries}
    for mention in sorted(all_path_mentions):
        if mention not in inventory_paths:
            issues.append(
                ScopeIssue(code="PREFLIGHT_G2_OUT_OF_SCOPE_PATH", path=mention)
            )
    unsafe_mention_count = sum(
        source_references.unsafe_count for source_references in references.values()
    )
    if unsafe_mention_count:
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G2_OUT_OF_SCOPE_PATH",
                observed=unsafe_mention_count,
            )
        )
    if seeds and any(
        "http://" in record.text.casefold() or "https://" in record.text.casefold()
        for record in all_records
    ):
        issues.append(ScopeIssue(code="PREFLIGHT_G2_EXTERNAL_EVIDENCE_ONLY"))
    unique_issues = tuple(sorted(set(issues), key=_issue_sort_key))
    selected_paths = tuple(selected)
    return ReviewScope(
        mode=(
            "declared_changed_v1"
            if inventory.source is ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH
            else "legacy_pairwise_v1"
        ),
        inventory=inventory,
        issued_paths=selected_paths,
        issued_changed_paths=selected_paths,
        selected_paths=selected_paths,
        sources=all_records,
        seeds=seeds,
        complete=bool(seeds and selected_paths and not unique_issues),
        issues=unique_issues,
        materialized_chars=materialized_chars,
        materialized_path_chars=tuple(sorted(materialized_path_chars.items())),
    )


def scope_source_bundle(scope: ReviewScope) -> SourceBundle:
    """Adapt a planned scope to the unchanged extraction SourceBundle contract."""

    return source_bundle_from_scope(scope)


def _legacy_regular_paths(root: Path) -> tuple[tuple[str, ...], tuple[ScopeIssue, ...]]:
    """Enumerate a complete legacy index while surfacing every fixed bound."""

    paths: list[str] = []
    entry_count = 0
    issues: list[ScopeIssue] = []
    seen = {root}

    def visit(directory: Path, depth: int) -> bool:
        nonlocal entry_count
        if depth > MAX_REPOSITORY_DEPTH:
            issues.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_REPOSITORY_DEPTH_LIMIT",
                    observed=depth,
                    limit=MAX_REPOSITORY_DEPTH,
                )
            )
            return False
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError:
            issues.append(ScopeIssue(code="PREFLIGHT_G1_HEAD_ROOT_INVALID"))
            return False
        for entry in sorted(entries, key=lambda item: (item.name.casefold(), item.name)):
            entry_count += 1
            if entry_count > MAX_REPOSITORY_ENTRIES:
                issues.append(
                    ScopeIssue(
                        code="PREFLIGHT_G1_REPOSITORY_ENTRY_LIMIT",
                        observed=entry_count,
                        limit=MAX_REPOSITORY_ENTRIES,
                    )
                )
                return False
            candidate = Path(entry.path)
            try:
                if entry.name.casefold() in {".git", "__pycache__"}:
                    continue
                if entry.is_symlink() or candidate.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(root)
                    if resolved not in seen:
                        seen.add(resolved)
                        if not visit(candidate, depth + 1):
                            return False
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                relative = candidate.resolve(strict=True).relative_to(root).as_posix()
                if "\\" in relative:
                    continue
            except (OSError, RuntimeError, ValueError):
                continue
            if len(paths) >= MAX_REPOSITORY_PATHS:
                issues.append(
                    ScopeIssue(
                        code="PREFLIGHT_G1_REPOSITORY_PATH_LIMIT",
                        observed=len(paths) + 1,
                        limit=MAX_REPOSITORY_PATHS,
                    )
                )
                return False
            paths.append(relative)
        return True

    visit(root, 0)
    return tuple(sorted(paths)), tuple(sorted(set(issues), key=_issue_sort_key))


def _legacy_inventory(
    inputs: _ScopeInputs,
) -> tuple[ChangeInventory | None, tuple[ScopeIssue, ...]]:
    """Build a provider-free compatibility inventory with explicit exhaustion."""

    try:
        requested_head = Path(inputs.repository_root)
        if requested_head.is_symlink():
            raise OSError
        head = requested_head.resolve(strict=True)
        if not head.is_dir():
            raise OSError
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None, (ScopeIssue(code="PREFLIGHT_G1_HEAD_ROOT_INVALID"),)
    raw_base = getattr(inputs, "base_root", None)
    base: Path | None = None
    if raw_base is not None:
        try:
            requested_base = Path(raw_base)
            if requested_base.is_symlink():
                raise OSError
            base = requested_base.resolve(strict=True)
            if not base.is_dir():
                raise OSError
        except (OSError, RuntimeError, TypeError, ValueError):
            return None, (
                ScopeIssue(code="PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"),
                ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_ROOT_INVALID"),
            )
    paths, traversal_issues = _legacy_regular_paths(head)
    if traversal_issues:
        return None, traversal_issues
    comparison_files = 0
    comparison_bytes = 0
    entries: list[ChangeEntry] = []
    issues: list[ScopeIssue] = []
    for path in paths:
        if classify_review_material(path) is ReviewMaterialKind.OTHER:
            continue
        if base is None:
            entries.append(ChangeEntry(path, ChangeStatus.ADDED))
            continue
        head_path = head / Path(path)
        base_path = base / Path(path)
        try:
            head_size = head_path.stat().st_size
            if not base_path.is_file() or base_path.is_symlink():
                entries.append(ChangeEntry(path, ChangeStatus.ADDED))
                continue
            base_size = base_path.stat().st_size
        except OSError:
            issues.append(
                ScopeIssue(code="PREFLIGHT_G2_CHANGE_STATUS_UNKNOWN", path=path)
            )
            continue
        if base_size != head_size:
            entries.append(ChangeEntry(path, ChangeStatus.MODIFIED))
            continue
        if head_size > MAX_CHANGE_COMPARISON_FILE_BYTES:
            issues.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_COMPARISON_PER_FILE_BYTES_LIMIT",
                    path=path,
                    observed=head_size,
                    limit=MAX_CHANGE_COMPARISON_FILE_BYTES,
                )
            )
            break
        if comparison_files >= MAX_CHANGE_COMPARISON_FILES:
            issues.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_COMPARISON_FILE_LIMIT",
                    observed=comparison_files + 1,
                    limit=MAX_CHANGE_COMPARISON_FILES,
                )
            )
            break
        required = head_size + base_size
        if comparison_bytes + required > MAX_CHANGE_COMPARISON_BYTES:
            issues.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_COMPARISON_BYTES_LIMIT",
                    observed=comparison_bytes + required,
                    limit=MAX_CHANGE_COMPARISON_BYTES,
                )
            )
            break
        comparison_files += 1
        comparison_bytes += required
        try:
            head_capture = capture_confined_regular_file(
                head, path, max_bytes=head_size
            )
            base_capture = capture_confined_regular_file(
                base, path, max_bytes=base_size
            )
        except PassiveFileError:
            issues.append(
                ScopeIssue(code="PREFLIGHT_G2_CHANGE_STATUS_UNKNOWN", path=path)
            )
            continue
        if head_capture.sha256 != base_capture.sha256:
            entries.append(ChangeEntry(path, ChangeStatus.MODIFIED))
    if any(issue.code.startswith("PREFLIGHT_G1_") for issue in issues):
        return None, tuple(sorted(set(issues), key=_issue_sort_key))
    inventory = ChangeInventory(
        schema_version=1,
        requested_base_sha="0" * 40,
        comparison_base_sha="0" * 40,
        head_sha="0" * 40,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.LEGACY_PAIRWISE,
        declared_entry_count=len(entries),
        complete=True,
        entries=tuple(sorted(entries, key=lambda item: (item.path, item.status.value))),
    )
    return inventory, tuple(sorted(set(issues), key=_issue_sort_key))


def legacy_preflight_failure(
    inputs: _ScopeInputs, config: ReviewConfig
) -> ReviewPreflight | None:
    """Return only a legacy hard-bound failure; preserve the legacy ready path."""

    if not isinstance(config, ReviewConfig):
        raise ReviewError("config must be ReviewConfig")
    inventory, issues = _legacy_inventory(inputs)
    if inventory is not None and not issues:
        return None
    gate1_reasons = tuple(
        issue for issue in issues if issue.code.startswith("PREFLIGHT_G1_")
    )
    gate2_reasons = tuple(
        issue for issue in issues if issue.code.startswith("PREFLIGHT_G2_")
    )
    if gate1_reasons or inventory is None:
        gate1 = _gate_result(1, GateDisposition.FAIL, gate1_reasons, {})
        gates = (gate1, _upstream_gate(2), _upstream_gate(3))
    else:
        gate1 = _gate_result(
            1,
            GateDisposition.PASS_COMPLETE,
            (),
            {
                "change_count": len(inventory.entries),
                "inventory_complete": inventory.complete,
                "inventory_source": inventory.source.value,
            },
        )
        gate2 = _gate_result(2, GateDisposition.FAIL, gate2_reasons, {})
        gates = (gate1, gate2, _upstream_gate(3))
    return ReviewPreflight(
        schema_version=1,
        requested_base_sha="0" * 40,
        comparison_base_sha="0" * 40,
        head_sha="0" * 40,
        gates=gates,
        ready_for_provider=False,
        review_status_ceiling=ReviewStatus.UNAVAILABLE,
        scope=None,
    )


def _upstream_gate(gate: int) -> PreflightGateResult:
    return PreflightGateResult(
        gate=gate,
        disposition=GateDisposition.NOT_EVALUATED,
        reasons=(ScopeIssue(code="PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"),),
        metrics={},
    )


def _snapshot_root_issue(snapshot: object, code: str) -> ScopeIssue | None:
    if not isinstance(snapshot, SnapshotIdentity):
        return ScopeIssue(code=code)
    try:
        root = snapshot.root
        if (
            root.is_symlink()
            or root.resolve(strict=True) != root
            or not root.is_dir()
        ):
            return ScopeIssue(code=code)
    except (OSError, RuntimeError, TypeError, ValueError):
        return ScopeIssue(code=code)
    return None


def _inventory_shape_issues(inventory: object) -> tuple[ScopeIssue, ...]:
    issues: list[ScopeIssue] = []
    required = (
        "schema_version",
        "requested_base_sha",
        "comparison_base_sha",
        "head_sha",
        "comparison_basis",
        "source",
        "declared_entry_count",
        "complete",
        "entries",
    )
    if any(not hasattr(inventory, field) for field in required):
        return (ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"),)
    if getattr(inventory, "schema_version") != 1:
        issues.append(
            ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_SCHEMA_UNSUPPORTED")
        )
    entries = getattr(inventory, "entries")
    if not isinstance(entries, tuple):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"))
        entries = ()
    declared = getattr(inventory, "declared_entry_count")
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared < 0
    ):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"))
    elif declared != len(entries):
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G1_CHANGED_FILE_COUNT_MISMATCH",
                observed=len(entries),
                limit=declared,
            )
        )
    if len(entries) > 8_192:
        issues.append(
            ScopeIssue(
                code="PREFLIGHT_G1_CHANGESET_PATH_LIMIT",
                observed=len(entries),
                limit=8_192,
            )
        )
    if getattr(inventory, "complete") is not True:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGESET_INCOMPLETE"))
    paths: list[str] = []
    previous: tuple[str, str] | None = None
    for entry in entries:
        path = getattr(entry, "path", None)
        status = getattr(entry, "status", None)
        if not isinstance(path, str):
            issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"))
            continue
        try:
            validated = ChangeEntry(path, ChangeStatus.ADDED).path
        except (ReviewError, TypeError, ValueError):
            issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH"))
            continue
        paths.append(validated)
        if not isinstance(status, ChangeStatus):
            issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_STATUS_INVALID"))
            continue
        current = (validated, status.value)
        if previous is not None and current < previous:
            issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"))
        previous = current
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_DUPLICATE_PATH"))
    try:
        metadata_bytes = len(
            json.dumps(
                plain(inventory),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if metadata_bytes > MAX_CHANGESET_METADATA_BYTES:
            issues.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_CHANGESET_METADATA_BYTES_LIMIT",
                    observed=metadata_bytes,
                    limit=MAX_CHANGESET_METADATA_BYTES,
                )
            )
    except (OverflowError, TypeError, ValueError, RecursionError):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"))
    return tuple(sorted(set(issues), key=_issue_sort_key))


def _coordinate_sha(inputs: _ScopeInputs, field: str, inventory_field: str) -> str:
    coordinates = getattr(inputs, "coordinates", None)
    value = (
        getattr(coordinates, inventory_field, None)
        if isinstance(coordinates, DeclaredReviewCoordinates)
        else None
    )
    snapshot = getattr(inputs, field, None)
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", value
    ):
        value = getattr(snapshot, "sha", None)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        inventory = getattr(inputs, "inventory", None)
        value = getattr(inventory, inventory_field, None)
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)
        else "0" * 40
    )


def _coordinate_basis(inputs: _ScopeInputs) -> ComparisonBasis | None:
    coordinates = getattr(inputs, "coordinates", None)
    if isinstance(coordinates, DeclaredReviewCoordinates):
        return coordinates.comparison_basis
    inventory = getattr(inputs, "inventory", None)
    basis = getattr(inventory, "comparison_basis", None)
    return basis if isinstance(basis, ComparisonBasis) else None


def _gate1_precheck(inputs: _ScopeInputs) -> tuple[ScopeIssue, ...]:
    issues: list[ScopeIssue] = []
    requested = getattr(inputs, "requested_base", None)
    comparison = getattr(inputs, "comparison_base", None)
    head = getattr(inputs, "head", None)
    inventory = getattr(inputs, "inventory", None)
    coordinates = getattr(inputs, "coordinates", None)
    inventory_failure = getattr(inputs, "inventory_failure", None)
    root_issues: list[ScopeIssue] = []
    if isinstance(coordinates, DeclaredReviewCoordinates):
        root_codes = {
            SnapshotRole.REQUESTED_BASE: "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID",
            SnapshotRole.COMPARISON_BASE: "PREFLIGHT_G1_COMPARISON_BASE_ROOT_INVALID",
            SnapshotRole.HEAD: "PREFLIGHT_G1_HEAD_ROOT_INVALID",
        }
        root_issues.extend(
            ScopeIssue(code=root_codes[role])
            for role in coordinates.invalid_root_roles
        )
    else:
        for snapshot, code in (
            (head, "PREFLIGHT_G1_HEAD_ROOT_INVALID"),
            (requested, "PREFLIGHT_G1_REQUESTED_BASE_ROOT_INVALID"),
            (comparison, "PREFLIGHT_G1_COMPARISON_BASE_ROOT_INVALID"),
        ):
            if issue := _snapshot_root_issue(snapshot, code):
                root_issues.append(issue)
    if root_issues and isinstance(coordinates, DeclaredReviewCoordinates):
        return tuple(sorted(set(root_issues), key=_issue_sort_key))
    issues.extend(root_issues)
    if comparison is None and not isinstance(coordinates, DeclaredReviewCoordinates):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_UNDECLARED"))
    if isinstance(inventory_failure, ReviewInventoryFailure):
        return (ScopeIssue(code=inventory_failure.code),)
    if inventory_failure is not None:
        return (ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED"),)
    if inventory is None:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_UNAVAILABLE"))
        return tuple(sorted(set(issues), key=_issue_sort_key))
    issues.extend(_inventory_shape_issues(inventory))
    if not all(
        isinstance(snapshot, SnapshotIdentity)
        for snapshot in (requested, comparison, head)
    ):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"))
        return tuple(sorted(set(issues), key=_issue_sort_key))
    if (
        requested.role is not SnapshotRole.REQUESTED_BASE
        or comparison.role is not SnapshotRole.COMPARISON_BASE
        or head.role is not SnapshotRole.HEAD
    ):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"))
    root = Path(inputs.repository_root)
    if root != head.root:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_HEAD_ROOT_INVALID"))
    coordinates = (
        (requested.sha, getattr(inventory, "requested_base_sha", None)),
        (comparison.sha, getattr(inventory, "comparison_base_sha", None)),
        (head.sha, getattr(inventory, "head_sha", None)),
    )
    if any(actual != declared for actual, declared in coordinates):
        issues.append(ScopeIssue(code="PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH"))
    basis = getattr(inventory, "comparison_basis", None)
    declared_coordinates = getattr(inputs, "coordinates", None)
    if isinstance(declared_coordinates, DeclaredReviewCoordinates):
        if (
            requested.sha != declared_coordinates.requested_base_sha
            or comparison.sha != declared_coordinates.comparison_base_sha
            or head.sha != declared_coordinates.head_sha
        ):
            issues.append(ScopeIssue(code="PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH"))
        if basis is not declared_coordinates.comparison_basis:
            issues.append(ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_MISMATCH"))
    if basis is ComparisonBasis.DIRECT_BASE and comparison.sha != requested.sha:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_MISMATCH"))
    if basis not in {ComparisonBasis.DIRECT_BASE, ComparisonBasis.MERGE_BASE}:
        issues.append(ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_MISMATCH"))
    return tuple(sorted(set(issues), key=_issue_sort_key))


def _gate_result(
    gate: int,
    disposition: GateDisposition,
    reasons: tuple[ScopeIssue, ...],
    metrics: dict[str, int | str | bool],
) -> PreflightGateResult:
    return PreflightGateResult(
        gate=gate,
        disposition=disposition,
        reasons=tuple(sorted(set(reasons), key=_issue_sort_key)),
        metrics=metrics,
    )


def _gate2(scope: ReviewScope) -> PreflightGateResult:
    reasons = tuple(issue for issue in scope.issues if issue.code.startswith("PREFLIGHT_G2_"))
    references = _source_path_references(scope.sources)
    selected_kinds = {
        path: classify_review_material(path) for path in scope.selected_paths
    }
    routed_seed_count = sum(
        1
        for seed in scope.seeds
        if any(
            _exact_mentions(path, seed, references) or _kind_agrees(kind, seed)
            for path, kind in selected_kinds.items()
        )
    )
    routable = routed_seed_count > 0
    disposition = (
        GateDisposition.FAIL
        if not routable
        else (GateDisposition.PASS_PARTIAL if reasons else GateDisposition.PASS_COMPLETE)
    )
    return _gate_result(
        2,
        disposition,
        reasons,
        {
            "issue_count": len(reasons),
            "material_seed_count": len(scope.seeds),
            "routed_seed_count": routed_seed_count,
            "selected_path_count": len(scope.selected_paths),
        },
    )


def _routed_seeds(scope: ReviewScope) -> tuple[MaterialClaimSeed, ...]:
    references = _source_path_references(scope.sources)
    selected_kinds = {
        path: classify_review_material(path) for path in scope.selected_paths
    }
    return tuple(
        seed
        for seed in scope.seeds
        if any(
            _exact_mentions(path, seed, references) or _kind_agrees(kind, seed)
            for path, kind in selected_kinds.items()
        )
    )


def _scope_routes_required_seeds(
    scope: ReviewScope, required: tuple[MaterialClaimSeed, ...]
) -> bool:
    source_ids = {source.source_id for source in scope.sources}
    routed = set(_routed_seeds(scope))
    return bool(required) and all(
        seed.origin_source_id in source_ids and seed in routed for seed in required
    )


def _trim_scope_for_gate3(scope: ReviewScope, config: ReviewConfig) -> ReviewScope:
    """Drop lowest-ranked issued paths until both provider envelopes can fit."""

    limits = config.limits
    original_count = len(scope.selected_paths)
    current = scope
    required_seeds = _routed_seeds(scope)
    trimmed: list[str] = []
    while current.selected_paths:
        bundle = scope_source_bundle(current)
        extraction = build_extraction_request_parts(bundle, limits.max_claims)
        extraction_chars = logical_request_chars(
            extraction.task, extraction.payload, extraction.schema
        )
        synthesis_chars = worst_valid_synthesis_request_chars(current, limits)
        if (
            extraction_chars <= limits.max_context_chars
            and synthesis_chars <= limits.max_context_chars
        ):
            break
        removable = None
        for path in reversed(current.selected_paths):
            retained = tuple(item for item in current.selected_paths if item != path)
            retained_set = set(retained)
            candidate = replace(
                current,
                issued_paths=tuple(item for item in current.issued_paths if item in retained_set),
                issued_changed_paths=tuple(
                    item for item in current.issued_changed_paths if item in retained_set
                ),
                selected_paths=retained,
                sources=tuple(
                    source
                    for source in current.sources
                    if source.path is None or source.path in retained_set
                ),
                materialized_chars=current.materialized_chars
                - dict(current.materialized_path_chars).get(path, 0),
                materialized_path_chars=tuple(
                    item for item in current.materialized_path_chars if item[0] in retained_set
                ),
            )
            if _scope_routes_required_seeds(candidate, required_seeds):
                removable = (path, candidate)
                break
        if removable is None:
            break
        path, current = removable
        trimmed.append(path)
    if not trimmed:
        return current
    issues = (*current.issues,) + tuple(
        ScopeIssue(
            code="PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT",
            path=path,
            observed=original_count,
            limit=len(current.selected_paths),
        )
        for path in trimmed
    )
    return replace(
        current,
        complete=False,
        issues=tuple(sorted(set(issues), key=_issue_sort_key)),
    )


def _gate3(scope: ReviewScope, config: ReviewConfig) -> PreflightGateResult:
    limits = config.limits
    bundle = scope_source_bundle(scope)
    extraction = build_extraction_request_parts(bundle, limits.max_claims)
    extraction_chars = logical_request_chars(
        extraction.task, extraction.payload, extraction.schema
    )
    empty = SourceBundle()
    fixed_extraction = build_extraction_request_parts(empty, limits.max_claims)
    extraction_fixed_chars = logical_request_chars(
        fixed_extraction.task, fixed_extraction.payload, fixed_extraction.schema
    )
    fixed_synthesis = build_synthesis_request_parts((), (), {}, {}, (), ())
    synthesis_fixed_chars = logical_request_chars(
        fixed_synthesis.task, fixed_synthesis.payload, fixed_synthesis.schema
    )
    synthesis_reserved_chars = worst_valid_synthesis_request_chars(scope, limits)
    reasons: list[ScopeIssue] = []
    hard_failure = False
    reasons.extend(
        issue
        for issue in scope.issues
        if issue.code == "PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT"
    )
    if limits.max_calls < 2:
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_CALL_LIMIT_LT_TWO",
                observed=limits.max_calls,
                limit=2,
            )
        )
        hard_failure = True
    if extraction_fixed_chars > limits.max_context_chars:
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_EXTRACTION_FIXED_OVERHEAD_LIMIT",
                observed=extraction_fixed_chars,
                limit=limits.max_context_chars,
            )
        )
        hard_failure = True
    elif extraction_chars > limits.max_context_chars:
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_EXTRACTION_CONTEXT_LIMIT",
                observed=extraction_chars,
                limit=limits.max_context_chars,
            )
        )
        hard_failure = True
    if synthesis_fixed_chars > limits.max_context_chars:
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_SYNTHESIS_FIXED_OVERHEAD_LIMIT",
                observed=synthesis_fixed_chars,
                limit=limits.max_context_chars,
            )
        )
        hard_failure = True
    elif synthesis_reserved_chars > limits.max_context_chars:
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_SYNTHESIS_RESERVED_CONTEXT_LIMIT",
                observed=synthesis_reserved_chars,
                limit=limits.max_context_chars,
            )
        )
        hard_failure = True
    truncated = tuple(
        issue
        for issue in scope.issues
        if issue.code == "PREFLIGHT_G2_CANDIDATE_SELECTION_TRUNCATED"
    )
    if truncated:
        required = max(
            (issue.observed or 0 for issue in truncated),
            default=len(scope.selected_paths) + len({issue.path for issue in truncated}),
        )
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_SELECTED_FILE_LIMIT",
                observed=required,
                limit=limits.max_files,
            )
        )
    locality = tuple(
        issue
        for issue in scope.issues
        if issue.code == "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE"
    )
    if locality:
        observed = max(
            (issue.observed or limits.max_file_chars + 1) for issue in locality
        )
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_SELECTED_SOURCE_CHAR_LIMIT",
                observed=observed,
                limit=limits.max_file_chars,
            )
        )
    if not output_budget_ready(limits):
        reasons.append(
            ScopeIssue(
                code="PREFLIGHT_G3_OUTPUT_RESERVE_LIMIT",
                observed=limits.max_output_chars,
                limit=24_000,
            )
        )
        hard_failure = True
    disposition = (
        GateDisposition.FAIL
        if hard_failure
        else (GateDisposition.PASS_PARTIAL if reasons else GateDisposition.PASS_COMPLETE)
    )
    return _gate_result(
        3,
        disposition,
        tuple(reasons),
        {
            "extraction_context_chars": extraction_chars,
            "extraction_fixed_overhead_chars": extraction_fixed_chars,
            "max_context_chars": limits.max_context_chars,
            "selected_file_count": len(scope.selected_paths),
            "selected_source_chars": scope.materialized_chars,
            "synthesis_fixed_overhead_chars": synthesis_fixed_chars,
            "synthesis_reserved_context_chars": synthesis_reserved_chars,
        },
    )


def _evaluate_material_gates(
    scope: ReviewScope,
    config: ReviewConfig,
    *,
    requested_sha: str,
    comparison_sha: str,
    head_sha: str,
) -> ReviewPreflight:
    inventory = scope.inventory
    material_issues = tuple(
        issue for issue in scope.issues if issue.code.startswith("PREFLIGHT_G1_")
    )
    if material_issues:
        gate1 = _gate_result(1, GateDisposition.FAIL, material_issues, {})
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=scope,
            comparison_basis=inventory.comparison_basis,
        )
    gate1 = _gate_result(
        1,
        GateDisposition.PASS_COMPLETE,
        (),
        {
            "change_count": len(inventory.entries),
            "inventory_complete": inventory.complete,
            "inventory_source": inventory.source.value,
        },
    )
    gate2 = _gate2(scope)
    if gate2.disposition is GateDisposition.FAIL:
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, gate2, _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=scope,
            comparison_basis=inventory.comparison_basis,
        )
    scope = _trim_scope_for_gate3(scope, config)
    gate3 = _gate3(scope, config)
    gates = (gate1, gate2, gate3)
    ready = gate3.disposition in {
        GateDisposition.PASS_COMPLETE,
        GateDisposition.PASS_PARTIAL,
    }
    ceiling = (
        ReviewStatus.UNAVAILABLE
        if not ready
        else (
            ReviewStatus.PARTIAL
            if GateDisposition.PASS_PARTIAL
            in (gate2.disposition, gate3.disposition)
            else ReviewStatus.COMPLETE
        )
    )
    return ReviewPreflight(
        schema_version=1,
        requested_base_sha=requested_sha,
        comparison_base_sha=comparison_sha,
        head_sha=head_sha,
        gates=gates,
        ready_for_provider=ready,
        review_status_ceiling=ceiling,
        scope=scope,
        comparison_basis=inventory.comparison_basis,
    )


def preflight_review(inputs: _ScopeInputs, config: ReviewConfig) -> ReviewPreflight:
    """Evaluate all free gates before any provider module can be imported."""

    if not isinstance(config, ReviewConfig):
        raise ReviewError("config must be ReviewConfig")
    declared = any(
        getattr(inputs, field, None) is not None
        for field in (
            "requested_base",
            "comparison_base",
            "head",
            "inventory",
            "coordinates",
            "inventory_failure",
        )
    )
    if not declared:
        inventory, legacy_issues = _legacy_inventory(inputs)
        if inventory is None:
            gate1 = _gate_result(1, GateDisposition.FAIL, legacy_issues, {})
            return ReviewPreflight(
                schema_version=1,
                requested_base_sha="0" * 40,
                comparison_base_sha="0" * 40,
                head_sha="0" * 40,
                gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
                ready_for_provider=False,
                review_status_ceiling=ReviewStatus.UNAVAILABLE,
                scope=None,
            )
        scope = build_review_scope(inputs, config, inventory)
        if legacy_issues:
            scope = replace(
                scope,
                complete=False,
                issues=tuple(
                    sorted(set((*scope.issues, *legacy_issues)), key=_issue_sort_key)
                ),
            )
        return _evaluate_material_gates(
            scope,
            config,
            requested_sha="0" * 40,
            comparison_sha="0" * 40,
            head_sha="0" * 40,
        )
    requested_sha = _coordinate_sha(inputs, "requested_base", "requested_base_sha")
    comparison_sha = _coordinate_sha(inputs, "comparison_base", "comparison_base_sha")
    head_sha = _coordinate_sha(inputs, "head", "head_sha")
    comparison_basis = _coordinate_basis(inputs)
    gate1_reasons = _gate1_precheck(inputs)
    inventory = getattr(inputs, "inventory", None)
    if gate1_reasons or not isinstance(inventory, ChangeInventory):
        gate1 = _gate_result(1, GateDisposition.FAIL, gate1_reasons, {})
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=None,
            comparison_basis=comparison_basis,
        )
    requested = inputs.requested_base
    comparison = inputs.comparison_base
    head = inputs.head
    assert requested is not None and comparison is not None and head is not None
    try:
        verified = build_git_change_inventory(
            requested, comparison, head, inventory.comparison_basis
        )
    except InventoryVerificationError as exc:
        gate1 = _gate_result(
            1,
            GateDisposition.FAIL,
            (ScopeIssue(code=exc.code),),
            {},
        )
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=None,
            comparison_basis=comparison_basis,
        )
    except ReviewError:
        gate1 = _gate_result(
            1,
            GateDisposition.FAIL,
            (
                ScopeIssue(
                    code="PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"
                ),
            ),
            {},
        )
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=None,
            comparison_basis=comparison_basis,
        )
    if verified != inventory:
        reasons: list[ScopeIssue] = []
        if (
            verified.requested_base_sha != inventory.requested_base_sha
            or verified.head_sha != inventory.head_sha
        ):
            reasons.append(ScopeIssue(code="PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH"))
        if (
            verified.comparison_base_sha != inventory.comparison_base_sha
            or verified.comparison_basis is not inventory.comparison_basis
        ):
            reasons.append(
                ScopeIssue(code="PREFLIGHT_G1_COMPARISON_BASE_MISMATCH")
            )
        if len(verified.entries) != len(inventory.entries):
            reasons.append(
                ScopeIssue(
                    code="PREFLIGHT_G1_CHANGED_FILE_COUNT_MISMATCH",
                    observed=len(inventory.entries),
                    limit=len(verified.entries),
                )
            )
        elif verified.entries != inventory.entries:
            reasons.append(
                ScopeIssue(code="PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED")
            )
        gate1 = _gate_result(
            1,
            GateDisposition.FAIL,
            tuple(reasons),
            {},
        )
        return ReviewPreflight(
            schema_version=1,
            requested_base_sha=requested_sha,
            comparison_base_sha=comparison_sha,
            head_sha=head_sha,
            gates=(gate1, _upstream_gate(2), _upstream_gate(3)),
            ready_for_provider=False,
            review_status_ceiling=ReviewStatus.UNAVAILABLE,
            scope=None,
            comparison_basis=comparison_basis,
        )
    return _evaluate_material_gates(
        build_review_scope(inputs, config, inventory),
        config,
        requested_sha=requested_sha,
        comparison_sha=comparison_sha,
        head_sha=head_sha,
    )


__all__ = [
    "build_review_scope",
    "classify_review_material",
    "legacy_preflight_failure",
    "preflight_review",
    "scope_source_bundle",
]
