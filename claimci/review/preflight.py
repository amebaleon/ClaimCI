"""Provider-free deterministic planning for a bounded review material scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from claimci.passive_files import PassiveFileError, capture_confined_regular_file

from .models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    MaterialClaimSeed,
    ReviewConfig,
    ReviewError,
    ReviewMaterialKind,
    ReviewScope,
    ScopeIssue,
    SourceBundle,
    SourceKind,
    SourceRecord,
)
from .path_policy import classify_review_material
from .sources import MAX_SOURCE_FILE_BYTES, _record, source_bundle_from_scope


class _ScopeInputs(Protocol):
    repository_root: Path
    pr_title: str
    pr_description: str


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
                code = (
                    "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"
                    if exc.code == "too_large"
                    else "PREFLIGHT_G2_CANDIDATE_UNREADABLE"
                )
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
    )


def scope_source_bundle(scope: ReviewScope) -> SourceBundle:
    """Adapt a planned scope to the unchanged extraction SourceBundle contract."""

    return source_bundle_from_scope(scope)


__all__ = ["build_review_scope", "classify_review_material", "scope_source_bundle"]
