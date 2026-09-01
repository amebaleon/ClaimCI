"""Deterministic, path-confined evidence discovery for extracted claims."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

from claimci.passive_files import PassiveFileError, capture_confined_regular_file

from .models import ClaimType, ReviewError, ReviewLimits, ScientificClaim
from .path_policy import is_source_file, is_test_source_file


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


class EvidenceProvenance(str, Enum):
    """What a repository artifact can establish about a measurement."""

    SUPPORTING_ARTIFACT = "supporting_artifact"
    REPORTED_MEASUREMENT = "reported_measurement"
    EXECUTABLE_BENCHMARK_DEFINITION = "executable_benchmark_definition"
    EXECUTED_RESULT_ARTIFACT = "executed_result_artifact"


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
    provenance: EvidenceProvenance = EvidenceProvenance.SUPPORTING_ARTIFACT


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
    routing_incomplete: bool = False


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

_TABLE_GENERIC_TOKENS = {
    "benchmark",
    "measurement",
    "metric",
    "result",
    "value",
}


@dataclass(frozen=True)
class _MarkdownTable:
    start_line: int
    end_line: int
    excerpt: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


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
    if (
        lowered.startswith("tests/")
        or name.startswith("test_")
        or is_test_source_file(path)
    ):
        return EvidenceKind.TEST
    if is_source_file(path):
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


def _has_executed_result_shape(path: str, excerpt: str) -> bool:
    marker_keys = {
        "duration_ms",
        "elapsed_ms",
        "executed",
        "executed_at",
        "finished_at",
        "run_id",
        "started_at",
    }
    payload_keys = {"measurements", "observations", "queries", "records", "runs", "samples"}

    def marked(value: object) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).strip().casefold()
                if normalized == "executed" and (
                    item is True
                    or (isinstance(item, str) and item.casefold() == "true")
                ):
                    return True
                if normalized == "status" and isinstance(item, str) and item.casefold() in {
                    "complete",
                    "completed",
                    "success",
                    "succeeded",
                }:
                    return True
                if (
                    normalized in marker_keys - {"executed"}
                    and item is not None
                    and item != ""
                    and item is not False
                ):
                    return True
                if marked(item):
                    return True
        elif isinstance(value, list):
            return any(marked(item) for item in value)
        return False

    def has_raw_payload(value: object) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).strip().casefold()
                if (
                    normalized in payload_keys
                    and isinstance(item, list)
                    and bool(item)
                    and all(isinstance(record, Mapping) for record in item)
                ):
                    return True
                if has_raw_payload(item):
                    return True
        elif isinstance(value, list):
            return any(has_raw_payload(item) for item in value)
        return False

    suffix = PurePosixPath(path).suffix.casefold()
    if suffix == ".json":
        try:
            value = json.loads(excerpt)
            return marked(value) and has_raw_payload(value)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            return False
    if suffix == ".jsonl":
        try:
            values = [json.loads(line) for line in excerpt.splitlines() if line.strip()]
            metadata = marker_keys | {"status", "summary", "expected"}
            return marked(values) and any(
                isinstance(value, Mapping)
                and bool({str(key).casefold() for key in value} - metadata)
                for value in values
            )
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            return False
    if suffix in {".csv", ".tsv"}:
        lines = [line for line in excerpt.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        delimiter = "," if suffix == ".csv" else "\t"
        headers = [cell.strip().casefold() for cell in lines[0].split(delimiter)]
        rows = [
            dict(zip(headers, line.split(delimiter), strict=False))
            for line in lines[1:]
        ]
        metadata = marker_keys | {"status", "summary", "expected"}
        return any(marked(row) for row in rows) and bool(set(headers) - metadata)
    return False


def _provenance(
    path: str,
    kind: EvidenceKind,
    excerpt: str,
) -> EvidenceProvenance:
    """Classify regular artifacts without upgrading summaries into execution."""

    portable = path.casefold().replace("\\", "/")
    suffix = PurePosixPath(portable).suffix
    tokens = _tokens(portable)
    if (
        suffix == ".sql"
        and re.search(
            r"(?im)^\s*(?:alter|create|delete|insert|select|update|with)\b",
            excerpt,
        )
        and (_tokens(excerpt) | tokens)
        & {
            "benchmark",
            "benchmarks",
            "cost",
            "fold",
            "gold",
            "latency",
            "memory",
            "metric",
            "metrics",
            "profile",
            "rollup",
        }
    ):
        return EvidenceProvenance.EXECUTABLE_BENCHMARK_DEFINITION
    if (
        suffix in {".csv", ".json", ".jsonl", ".tsv"}
        and kind in {EvidenceKind.RESULTS, EvidenceKind.BENCHMARK}
        and tokens & {"result", "results", "metric", "metrics", "score", "scores"}
        and _has_executed_result_shape(path, excerpt)
    ):
        return EvidenceProvenance.EXECUTED_RESULT_ARTIFACT
    return EvidenceProvenance.SUPPORTING_ARTIFACT


def _is_markdown(path: str) -> bool:
    return PurePosixPath(path).suffix.casefold() in {".md", ".markdown"}


def _markdown_cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = tuple(cell.strip() for cell in stripped.split("|"))
    return cells if cells and all(cells) else None


def _markdown_tables(text: str) -> tuple[_MarkdownTable, ...]:
    """Return conventional pipe tables outside fenced code, with exact spans."""

    lines = text.splitlines(keepends=True)
    fence: tuple[str, int] | None = None
    eligible: list[bool] = []
    for line in lines:
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if fence is not None:
            eligible.append(False)
            if re.fullmatch(
                rf"{re.escape(fence[0])}{{{fence[1]},}}[ \t]*(?:\r?\n)?",
                stripped,
            ):
                fence = None
            continue
        if marker is not None:
            fence = (marker.group(0)[0], len(marker.group(0)))
            eligible.append(False)
            continue
        eligible.append(not line.startswith(("    ", "\t")))

    tables: list[_MarkdownTable] = []
    index = 0
    while index + 2 < len(lines):
        header = _markdown_cells(lines[index]) if eligible[index] else None
        separator = _markdown_cells(lines[index + 1]) if eligible[index + 1] else None
        if (
            header is None
            or separator is None
            or len(header) != len(separator)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        ):
            index += 1
            continue
        end = index + 2
        while end < len(lines) and eligible[end]:
            row = _markdown_cells(lines[end])
            if row is None or len(row) != len(header):
                break
            end += 1
        if end == index + 2:
            index += 1
            continue
        excerpt = "".join(lines[index:end])
        rows = tuple(
            row
            for row_index in range(index + 2, end)
            if (row := _markdown_cells(lines[row_index])) is not None
        )
        tables.append(
            _MarkdownTable(
                start_line=index + 1,
                end_line=end,
                excerpt=excerpt,
                header=header,
                rows=rows,
            )
        )
        index = end
    return tuple(tables)


def _semantic_tokens(value: str | None) -> set[str]:
    if not isinstance(value, str):
        return set()
    tokens: set[str] = set()
    for token in _tokens(value) - _LITERAL_STOPWORDS:
        if len(token) < 3 and not any(character.isdigit() for character in token):
            continue
        tokens.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return tokens


def _magnitude_cells(claim: ScientificClaim) -> set[str]:
    magnitude = claim.claimed_magnitude
    if magnitude is None:
        return set()
    candidates = {
        match.group(0).replace(" ", "").casefold()
        for match in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:x|%|×))?", magnitude.raw)
    }
    if magnitude.value is not None:
        value = float(magnitude.value)
        rendered = {str(magnitude.value), f"{value:g}"}
        if value.is_integer():
            rendered.add(str(int(value)))
        units = {"", (magnitude.unit or "").strip().casefold()}
        candidates.update(f"{number}{unit}" for number in rendered for unit in units)
    return {candidate for candidate in candidates if candidate}


def _table_supports_claim(
    table: _MarkdownTable,
    claim: ScientificClaim,
    *,
    path: str,
) -> bool:
    magnitude_cells = _magnitude_cells(claim)
    if not magnitude_cells:
        return False
    metric_tokens = _semantic_tokens(claim.metric) - _TABLE_GENERIC_TOKENS
    subject_tokens = _semantic_tokens(claim.subject) - _TABLE_GENERIC_TOKENS
    claim_tokens = metric_tokens | subject_tokens
    for row_offset, row in enumerate(table.rows, start=2):
        magnitude_indexes = {
            index
            for index, cell in enumerate(row)
            if cell.replace(" ", "").casefold() in magnitude_cells
        }
        if not magnitude_indexes:
            continue
        if (
            claim.source.path == path
            and claim.source.start_line <= table.start_line + row_offset
            and claim.source.end_line >= table.start_line + row_offset
        ):
            return True
        row_tokens = _semantic_tokens(" ".join(row)) - _TABLE_GENERIC_TOKENS
        header_tokens = _semantic_tokens(
            " ".join(table.header[index] for index in magnitude_indexes)
        ) - _TABLE_GENERIC_TOKENS
        if row_tokens & claim_tokens:
            return True
        if len(header_tokens & metric_tokens) >= 2:
            return True
        if (header_tokens & metric_tokens) and (row_tokens & subject_tokens):
            return True
        if (header_tokens & subject_tokens) and (row_tokens & metric_tokens):
            return True
    return False


def _matches(
    claim: ScientificClaim,
    path: str,
    changed_paths: set[str],
) -> bool:
    lowered = path.casefold().replace("\\", "/")
    name = PurePosixPath(lowered).name
    segments = tuple(part for part in PurePosixPath(lowered).parts if part not in {"/", ""})
    tokens = _tokens(lowered)
    kind = _kind(path)

    if (
        claim.claim_type in _CHANGED_SOURCE_CLAIMS
        and kind in {EvidenceKind.SOURCE, EvidenceKind.TEST}
        and path not in changed_paths
    ):
        return False

    if (
        claim.claim_type is ClaimType.IMPLEMENTATION_CLAIM
        and kind in {EvidenceKind.SOURCE, EvidenceKind.TEST}
    ):
        return True

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
        return None, "Evidence not available in the analyzed snapshot."
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
    priority_paths: Mapping[str, Sequence[str]] | None = None,
    selected_paths: Sequence[str] = (),
    changed_paths: Sequence[str] | None = None,
) -> EvidenceBundle:
    """Select bounded indexed paths within one audit-wide unique-file budget.

    ``selected_paths`` contains repository files already sent during claim
    extraction.  They may be reused without consuming another slot; every new
    evidence path consumes one of the same ``limits.max_files`` slots.
    ``priority_paths`` is trusted deterministic routing data, never a provider
    hint; those paths are issued before broader heuristic matches.
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
    # With a trusted base checkout, heuristic discovery is confined to
    # files changed by the pull request. Unchanged files remain reachable only
    # through trusted manifest priority or an exact, route-valid provider hint.
    # Without a base checkout, every indexed path is treated as changed.
    heuristic_index = (
        sorted_index
        if changed_paths is None
        else tuple(path for path in sorted_index if path in changed)
    )

    by_path: dict[str, set[str]] = defaultdict(set)
    priority_order: list[str] = []
    missing: list[MissingEvidence] = []
    unresolved_hint_claim_ids: set[str] = set()

    claims_by_id = {claim.claim_id: claim for claim in claims}
    claim_ids = set(claims_by_id)
    if priority_paths is not None:
        if not isinstance(priority_paths, Mapping):
            raise ReviewError("priority evidence paths must be a mapping")
        if any(claim_id not in claim_ids for claim_id in priority_paths):
            raise ReviewError("priority evidence paths cite an unknown claim")
        for claim in claims:
            raw_paths = priority_paths.get(claim.claim_id, ())
            if not isinstance(raw_paths, Sequence) or isinstance(
                raw_paths, (str, bytes)
            ):
                raise ReviewError("priority evidence paths must be sequences")
            for raw in raw_paths:
                normalized = _safe_relative(raw)
                if normalized is None or normalized not in normalized_index:
                    raise ReviewError(
                        "priority evidence path is unsafe or not indexed"
                    )
                by_path[normalized].add(claim.claim_id)
                if normalized not in priority_order:
                    priority_order.append(normalized)

    for claim in claims:
        for path in heuristic_index:
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
                unresolved_hint_claim_ids.add(claim.claim_id)
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="unsafe_path",
                        description="Suggested evidence path is not a confined relative path.",
                    )
                )
                continue
            if normalized not in normalized_index:
                unresolved_hint_claim_ids.add(claim.claim_id)
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="unresolved_provider_hint",
                        description=(
                            "Provider-suggested evidence hint was unresolved because "
                            "it did not exactly match an indexed repository path; no "
                            "repository file availability conclusion was made."
                        ),
                    )
                )
                continue
            if not _matches(claim, normalized, changed):
                unresolved_hint_claim_ids.add(claim.claim_id)
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
            if resolved is None:
                missing.append(
                    MissingEvidence(
                        claim_id=claim.claim_id,
                        requested_path=str(raw_hint),
                        reason="not_available",
                        description=(
                            reason
                            or "Suggested path was not available in ClaimCI's bounded analyzed snapshot."
                        ),
                    )
                )
                continue
            by_path[normalized].add(claim.claim_id)

    markdown_candidates: dict[str, set[str]] = {
        path: {
            claim.claim_id
            for claim in claims
            if claim.claimed_magnitude is not None
        }
        for path in heuristic_index
        if _is_markdown(path)
        and (
            path in selected
            or path in by_path
            or PurePosixPath(path).name.casefold().startswith("readme")
        )
    }
    markdown_candidates = {
        path: candidate_ids
        for path, candidate_ids in markdown_candidates.items()
        if candidate_ids
    }

    references: list[EvidenceReference] = []
    file_limit_counts: dict[str, int] = defaultdict(int)
    remaining = limits.max_context_chars
    priority_set = set(priority_order)
    routed_paths = set(by_path) | set(markdown_candidates)
    ordinarily_routed = set(by_path) - priority_set
    markdown_only = routed_paths - set(by_path) - priority_set
    ordered_paths = (
        *priority_order,
        *sorted(ordinarily_routed),
        *sorted(markdown_only),
    )
    for relative in ordered_paths:
        routed_claim_ids = set(by_path.get(relative, ())) | set(
            markdown_candidates.get(relative, ())
        )
        if remaining <= 0:
            break
        if relative not in selected and len(selected) >= limits.max_files:
            # Content-only Markdown candidates have not been shown to match;
            # do not report an evidence gap for a file ClaimCI never inspected.
            for claim_id in sorted(by_path.get(relative, ())):
                file_limit_counts[claim_id] += 1
            continue
        resolved, reason = _validate_regular(root, relative)
        if resolved is None:
            for claim_id in sorted(routed_claim_ids):
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
            for claim_id in sorted(routed_claim_ids):
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
            capture = capture_confined_regular_file(
                root,
                relative,
                max_bytes=MAX_EVIDENCE_FILE_BYTES,
            )
            text = capture.content.decode("utf-8")
            excerpt = text.replace("\r\n", "\n").replace("\r", "\n")[:char_limit]
            digest = capture.sha256
            size = capture.size
        except (PassiveFileError, UnicodeError, ValueError, RecursionError) as exc:
            for claim_id in sorted(routed_claim_ids):
                missing.append(
                    MissingEvidence(
                        claim_id=claim_id,
                        requested_path=relative,
                        reason="unreadable",
                        description=f"Evidence file could not be read: {type(exc).__name__}.",
                    )
                )
            continue
        selected.add(relative)
        if not excerpt:
            continue
        table_claim_ids: set[str] = set()
        table_chars = 0
        if relative in markdown_candidates:
            for table in _markdown_tables(excerpt):
                matching_ids = {
                    claim_id
                    for claim_id in markdown_candidates[relative]
                    if _table_supports_claim(
                        table,
                        claims_by_id[claim_id],
                        path=relative,
                    )
                }
                if not matching_ids or len(table.excerpt) > remaining:
                    continue
                provenance = EvidenceProvenance.REPORTED_MEASUREMENT
                evidence_id = "evidence-" + hashlib.sha256(
                    (
                        f"{relative}\0{digest}\0{table.start_line}:{table.end_line}"
                        f"\0{provenance.value}"
                    ).encode("utf-8")
                ).hexdigest()[:16]
                references.append(
                    EvidenceReference(
                        evidence_id=evidence_id,
                        claim_ids=tuple(sorted(matching_ids)),
                        kind=EvidenceKind.BENCHMARK,
                        path=relative,
                        start_line=table.start_line,
                        end_line=table.end_line,
                        sha256=digest,
                        size=size,
                        excerpt=table.excerpt,
                        provenance=provenance,
                    )
                )
                table_claim_ids.update(matching_ids)
                table_chars += len(table.excerpt)
                remaining -= len(table.excerpt)

        regular_claim_ids = set(by_path.get(relative, ())) - table_claim_ids
        regular_limit = min(
            max(0, limits.max_file_chars - table_chars),
            remaining,
        )
        regular_excerpt = excerpt[:regular_limit]
        if regular_claim_ids and regular_excerpt:
            evidence_id = "evidence-" + hashlib.sha256(
                f"{relative}\0{digest}".encode("utf-8")
            ).hexdigest()[:16]
            kind = _kind(relative)
            references.append(
                EvidenceReference(
                    evidence_id=evidence_id,
                    claim_ids=tuple(sorted(regular_claim_ids)),
                    kind=kind,
                    path=relative,
                    start_line=1,
                    end_line=max(
                        1,
                        regular_excerpt.count("\n")
                        + (0 if regular_excerpt.endswith("\n") else 1),
                    ),
                    sha256=digest,
                    size=size,
                    excerpt=regular_excerpt,
                    provenance=_provenance(relative, kind, regular_excerpt),
                )
            )
            remaining -= len(regular_excerpt)

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
    verification_needed_claims = {
        claim_id
        for reference in references
        if reference.provenance
        in {
            EvidenceProvenance.REPORTED_MEASUREMENT,
            EvidenceProvenance.EXECUTABLE_BENCHMARK_DEFINITION,
        }
        for claim_id in reference.claim_ids
    }
    executed_claims = {
        claim_id
        for reference in references
        if reference.provenance is EvidenceProvenance.EXECUTED_RESULT_ARTIFACT
        for claim_id in reference.claim_ids
    }
    for claim_id in sorted(verification_needed_claims - executed_claims):
        missing.append(
            MissingEvidence(
                claim_id=claim_id,
                reason="executed_result_not_available",
                description=(
                    "This claim could not be independently verified from the "
                    "available artifacts in the analyzed snapshot; no executed "
                    "result artifact was available to ClaimCI."
                ),
            )
        )
    for claim in claims:
        if claim.claim_id not in covered and not any(
            item.claim_id == claim.claim_id for item in missing
        ):
            missing.append(
                MissingEvidence(
                    claim_id=claim.claim_id,
                    reason="no_matching_evidence",
                    description=(
                        "Evidence not available in the analyzed snapshot; no matching "
                        "artifact was selected within ClaimCI's bounded review scope."
                    ),
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
        routing_incomplete=bool(unresolved_hint_claim_ids - covered),
    )
