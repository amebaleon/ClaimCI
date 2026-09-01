"""Passive, bounded source collection and trusted claim validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claimci.passive_files import PassiveFileError, capture_confined_regular_file

from .models import (
    ClaimDirection,
    ClaimMagnitude,
    ClaimType,
    MagnitudeKind,
    ReviewError,
    ReviewLimits,
    ScientificClaim,
    SourceBundle,
    SourceKind,
    SourceLocation,
    SourceRecord,
)


MAX_REPOSITORY_PATHS = 2_048
MAX_REPOSITORY_ENTRIES = 8_192
MAX_REPOSITORY_DEPTH = 64
MAX_CHANGE_COMPARISON_FILES = 512
MAX_CHANGE_COMPARISON_BYTES = 16 * 1024 * 1024
MAX_CHANGE_COMPARISON_FILE_BYTES = 1024 * 1024
MAX_CLAIMS = 16
MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024


def _root(path: Path, label: str) -> Path:
    try:
        resolved = Path(path).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"{label} is invalid: {exc}") from exc
    if not resolved.is_dir():
        raise ReviewError(f"{label} is not a directory: {resolved}")
    return resolved


def _iter_regular_paths(root: Path) -> tuple[str, ...]:
    paths: list[str] = []
    visited_entries = 0
    seen_directories: set[Path] = {root}

    def visit(directory: Path, depth: int) -> bool:
        nonlocal visited_entries
        if depth > MAX_REPOSITORY_DEPTH:
            raise ReviewError("repository nesting exceeds the review index limit")
        entries: list[os.DirEntry[str]] = []
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    visited_entries += 1
                    if visited_entries > MAX_REPOSITORY_ENTRIES:
                        raise ReviewError(
                            "repository entries exceed the review index limit"
                        )
                    entries.append(entry)
        except ReviewError:
            raise
        except OSError as exc:
            raise ReviewError(f"could not enumerate repository: {exc}") from exc

        for entry in sorted(
            entries,
            key=lambda item: (item.name.casefold(), item.name),
        ):
            candidate = Path(entry.path)
            try:
                if entry.name.casefold() in {".git", "__pycache__"}:
                    continue
                if entry.is_symlink() or candidate.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    resolved_directory = candidate.resolve()
                    resolved_directory.relative_to(root)
                    if resolved_directory in seen_directories:
                        continue
                    seen_directories.add(resolved_directory)
                    if visit(candidate, depth + 1):
                        return True
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                resolved = candidate.resolve()
                relative = resolved.relative_to(root).as_posix()
                if "\\" in relative:
                    # A literal POSIX backslash cannot be represented by the
                    # portable review-path model without changing identity.
                    continue
            except ReviewError:
                raise
            except (OSError, ValueError, RuntimeError):
                continue
            paths.append(relative)
            if len(paths) >= MAX_REPOSITORY_PATHS:
                return True
        return False

    visit(root, 0)
    return tuple(sorted(set(paths)))


def _confined_regular_file(root: Path, relative: str) -> Path | None:
    """Resolve one lexical path without following symlinked components."""

    candidate = root / Path(relative)
    cursor = root
    try:
        for part in Path(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        resolved = candidate.resolve()
        resolved.relative_to(root)
        return resolved if resolved.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _eligible_document(path: str) -> bool:
    relative = Path(path)
    name = relative.name.casefold()
    suffix = relative.suffix.casefold()
    return suffix in {".md", ".markdown"} or name == "paper.tex"


_CHANGE_EVIDENCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".go",
    ".sql",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
}


def _change_candidate(path: str) -> bool:
    """Return paths whose base/head status can affect review evidence."""

    relative = Path(path)
    lowered = relative.as_posix().casefold()
    name = relative.name.casefold()
    return (
        _eligible_document(path)
        or relative.suffix.casefold() in _CHANGE_EVIDENCE_SUFFIXES
        or lowered.startswith(("src/", "lib/", "claimci/", "tests/"))
        or name.startswith("test_")
    )


def _changed(
    path: str,
    head: Path,
    base: Path | None,
    *,
    comparison_budget: dict[str, int],
) -> bool:
    head_path = _confined_regular_file(head, path)
    if head_path is None:
        raise ReviewError(f"review source is not a confined regular file: {path}")
    try:
        head_size = head_path.stat().st_size
    except OSError as exc:
        raise ReviewError(f"could not inspect review source {head_path}: {exc}") from exc
    if head_size < 0 or head_size > MAX_SOURCE_FILE_BYTES:
        raise ReviewError(f"review source exceeds the change-inspection limit: {path}")
    if base is None:
        return True
    base_path = _confined_regular_file(base, path)
    if base_path is None:
        return True
    try:
        base_size = base_path.stat().st_size
    except OSError as exc:
        raise ReviewError(f"could not inspect base review source {base_path}: {exc}") from exc
    if base_size != head_size:
        return True
    if base_size > MAX_SOURCE_FILE_BYTES:
        raise ReviewError(f"base review source exceeds the change-inspection limit: {path}")
    required_bytes = head_size + base_size
    if (
        head_size > MAX_CHANGE_COMPARISON_FILE_BYTES
        or comparison_budget["files"] <= 0
        or required_bytes > comparison_budget["bytes"]
    ):
        raise ReviewError(
            "repository change-comparison budget was exhausted before review"
        )
    comparison_budget["files"] -= 1
    comparison_budget["bytes"] -= required_bytes
    try:
        head_capture = capture_confined_regular_file(
            head,
            path,
            max_bytes=head_size,
        )
        base_capture = capture_confined_regular_file(
            base,
            path,
            max_bytes=base_size,
        )
    except PassiveFileError as exc:
        raise ReviewError(f"review source identity changed during comparison: {path}") from exc
    if head_capture.size != head_size or base_capture.size != base_size:
        raise ReviewError(f"review source size changed during comparison: {path}")
    return head_capture.sha256 != base_capture.sha256


def _bounded_text(root: Path, relative: str, max_chars: int) -> str:
    try:
        capture = capture_confined_regular_file(
            root,
            relative,
            max_bytes=MAX_SOURCE_FILE_BYTES,
        )
        text = capture.content.decode("utf-8")
        # Universal-newline normalization keeps source locations and exact
        # quotes stable across Windows and POSIX checkouts.
        return text.replace("\r\n", "\n").replace("\r", "\n")[:max_chars]
    except (PassiveFileError, UnicodeError, ValueError, RecursionError) as exc:
        raise ReviewError(
            f"could not read confined review source {relative}: {exc}"
        ) from exc


def _source_id(kind: SourceKind, path: str | None, text: str) -> str:
    material = json.dumps(
        {"kind": kind.value, "path": path, "text": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "source-" + hashlib.sha256(material).hexdigest()[:16]


def _record(kind: SourceKind, path: str | None, text: str) -> SourceRecord:
    return SourceRecord(
        source_id=_source_id(kind, path, text),
        kind=kind,
        path=path,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def collect_review_sources(
    head_root: Path,
    *,
    base_root: Path | None = None,
    pr_title: str = "",
    pr_description: str = "",
    limits: ReviewLimits = ReviewLimits(),
) -> SourceBundle:
    """Collect bounded PR metadata and changed research documents."""

    if not isinstance(limits, ReviewLimits):
        raise ReviewError("limits must be ReviewLimits")
    head = _root(head_root, "head repository root")
    base = None if base_root is None else _root(base_root, "base repository root")
    repository_paths = _iter_regular_paths(head)
    comparison_budget = {
        "files": MAX_CHANGE_COMPARISON_FILES,
        "bytes": MAX_CHANGE_COMPARISON_BYTES,
    }
    changed_paths = tuple(
        relative
        for relative in repository_paths
        if _change_candidate(relative)
        if _changed(
            relative,
            head,
            base,
            comparison_budget=comparison_budget,
        )
    )
    changed_set = set(changed_paths)
    records: list[SourceRecord] = []
    remaining = limits.max_context_chars

    for kind, value in (
        (SourceKind.PULL_REQUEST_TITLE, pr_title),
        (SourceKind.PULL_REQUEST_DESCRIPTION, pr_description),
    ):
        if not isinstance(value, str):
            raise ReviewError(f"{kind.value} must be a string")
        if value and remaining > 0:
            text = value.replace("\r\n", "\n").replace("\r", "\n")[:remaining]
            records.append(_record(kind, None, text))
            remaining -= len(text)

    selected = 0
    for relative in repository_paths:
        if selected >= limits.max_files or remaining <= 0:
            break
        if not _eligible_document(relative) or relative not in changed_set:
            continue
        text = _bounded_text(
            head,
            relative,
            min(limits.max_file_chars, remaining),
        )
        records.append(_record(SourceKind.REPOSITORY_FILE, relative, text))
        remaining -= len(text)
        selected += 1

    return SourceBundle(
        sources=tuple(records),
        repository_paths=repository_paths,
        changed_paths=changed_paths,
        total_chars=sum(len(record.text) for record in records),
    )


def _strict_fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewError(f"{label} must be an object")
    fields = set(value)
    if fields != expected:
        missing = expected - fields
        extra = fields - expected
        raise ReviewError(f"{label} fields are invalid; missing={sorted(missing)}, extra={sorted(map(str, extra))}")
    return value


def _string_list(value: object, label: str, *, max_items: int = 32) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ReviewError(f"{label} must be a bounded list")
    if not all(isinstance(item, str) and item.strip() and len(item) <= 4_096 for item in value):
        raise ReviewError(f"{label} must contain bounded non-empty strings")
    return tuple(value)


def _magnitude(value: object) -> ClaimMagnitude | None:
    if value is None:
        return None
    mapping = _strict_fields(value, {"raw", "value", "unit", "kind"}, "claimed_magnitude")
    return ClaimMagnitude(
        raw=mapping["raw"],
        value=mapping["value"],
        unit=mapping["unit"],
        kind=MagnitudeKind(mapping["kind"]),
    )


def _quote(record: SourceRecord, start_line: int, end_line: int) -> str:
    lines = record.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise ReviewError("claim source line span is outside the selected source")
    return "\n".join(lines[start_line - 1 : end_line])


@dataclass(frozen=True)
class ClaimCandidateValidation:
    """Trusted claims retained from one untrusted extraction response."""

    claims: tuple[ScientificClaim, ...]
    rejection_reasons: tuple[str, ...] = ()

    @property
    def rejected_count(self) -> int:
        return len(self.rejection_reasons)


def _recover_exact_quote_location(
    candidate_value: object,
    records: Mapping[str, SourceRecord],
) -> object:
    """Repair only a uniquely recoverable exact full-line quote location.

    Recovery cannot bless invented text. It succeeds only when the provider's
    exact quote occurs once as complete consecutive lines inside the already
    issued source. Paraphrases and repeated ambiguous text remain invalid.
    """

    if not isinstance(candidate_value, Mapping):
        return candidate_value
    source_value = candidate_value.get("source")
    source_text = candidate_value.get("source_text")
    if not isinstance(source_value, Mapping) or not isinstance(source_text, str):
        return candidate_value
    source_id = source_value.get("source_id")
    if not isinstance(source_id, str) or source_id not in records or not source_text:
        return candidate_value

    record = records[source_id]
    start_line = source_value.get("start_line")
    end_line = source_value.get("end_line")
    if (
        isinstance(start_line, int)
        and not isinstance(start_line, bool)
        and isinstance(end_line, int)
        and not isinstance(end_line, bool)
    ):
        try:
            if source_text == _quote(record, start_line, end_line):
                return candidate_value
        except ReviewError:
            pass

    source_lines = record.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    quote_lines = source_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not quote_lines or any(line == "" for line in quote_lines):
        return candidate_value
    width = len(quote_lines)
    matches = [
        index
        for index in range(0, len(source_lines) - width + 1)
        if source_lines[index : index + width] == quote_lines
    ]
    if len(matches) != 1:
        return candidate_value

    recovered = dict(candidate_value)
    recovered_source = dict(source_value)
    recovered_source["start_line"] = matches[0] + 1
    recovered_source["end_line"] = matches[0] + width
    recovered["source"] = recovered_source
    return recovered


def _claim_rejection_reason(exc: ReviewError) -> str:
    message = str(exc)
    if "source quote does not match" in message:
        return "source_quote_mismatch"
    if (
        "outside the selected source" in message
        or "cites an unknown source" in message
        or "claim source" in message
    ):
        return "source_location_invalid"
    if "duplicates an earlier claim" in message:
        return "duplicate_claim"
    return "invalid_claim_candidate"


def validate_claim_candidates_best_effort(
    payload: object,
    sources: SourceBundle,
    *,
    max_claims: int = MAX_CLAIMS,
) -> ClaimCandidateValidation:
    """Retain valid claims while excluding isolated malformed candidates.

    Top-level response corruption still raises and fails closed. Each candidate
    is then checked by the unchanged strict validator. This permits a mixed
    response to retain good advisory claims without weakening source identity,
    field, authority, or citation boundaries.
    """

    if not isinstance(sources, SourceBundle):
        raise ReviewError("sources must be a SourceBundle")
    if (
        isinstance(max_claims, bool)
        or not isinstance(max_claims, int)
        or not 1 <= max_claims <= MAX_CLAIMS
    ):
        raise ReviewError("max_claims must be an integer from 1 through 16")
    root = _strict_fields(payload, {"claims"}, "claim extraction response")
    candidates = root["claims"]
    if not isinstance(candidates, list) or len(candidates) > max_claims:
        raise ReviewError("claims must be a bounded list")

    records = {source.source_id: source for source in sources.sources}
    accepted: list[ScientificClaim] = []
    accepted_ids: set[str] = set()
    rejection_reasons: list[str] = []
    for candidate in candidates:
        recovered = _recover_exact_quote_location(candidate, records)
        try:
            validated = validate_claim_candidates(
                {"claims": [recovered]},
                sources,
                max_claims=1,
            )
        except ReviewError as exc:
            rejection_reasons.append(_claim_rejection_reason(exc))
            continue
        claim = validated[0]
        if claim.claim_id in accepted_ids:
            rejection_reasons.append("duplicate_claim")
            continue
        accepted.append(claim)
        accepted_ids.add(claim.claim_id)

    return ClaimCandidateValidation(
        claims=tuple(accepted),
        rejection_reasons=tuple(rejection_reasons),
    )


def validate_claim_candidates(
    payload: object,
    sources: SourceBundle,
    *,
    max_claims: int = MAX_CLAIMS,
) -> tuple[ScientificClaim, ...]:
    """Validate extraction output and assign trusted deterministic claim IDs."""

    if not isinstance(sources, SourceBundle):
        raise ReviewError("sources must be a SourceBundle")
    if (
        isinstance(max_claims, bool)
        or not isinstance(max_claims, int)
        or not 1 <= max_claims <= MAX_CLAIMS
    ):
        raise ReviewError("max_claims must be an integer from 1 through 16")
    root = _strict_fields(payload, {"claims"}, "claim extraction response")
    candidates = root["claims"]
    if not isinstance(candidates, list) or len(candidates) > max_claims:
        raise ReviewError("claims must be a bounded list")
    records = {source.source_id: source for source in sources.sources}
    expected = {
        "source_text",
        "claim_type",
        "subject",
        "metric",
        "direction",
        "claimed_magnitude",
        "qualifiers",
        "source",
        "confidence",
        "evidence_hints",
    }
    accepted: list[ScientificClaim] = []
    for index, candidate_value in enumerate(candidates):
        candidate = _strict_fields(candidate_value, expected, f"claim {index}")
        source_value = _strict_fields(
            candidate["source"],
            {"source_id", "start_line", "end_line"},
            f"claim {index} source",
        )
        source_id = source_value["source_id"]
        if not isinstance(source_id, str) or source_id not in records:
            raise ReviewError(f"claim {index} cites an unknown source")
        start_line = source_value["start_line"]
        end_line = source_value["end_line"]
        if isinstance(start_line, bool) or not isinstance(start_line, int):
            raise ReviewError("claim source start_line must be an integer")
        if isinstance(end_line, bool) or not isinstance(end_line, int):
            raise ReviewError("claim source end_line must be an integer")
        record = records[source_id]
        source_text = candidate["source_text"]
        if not isinstance(source_text, str) or source_text != _quote(record, start_line, end_line):
            raise ReviewError(f"claim {index} source quote does not match the trusted source")
        confidence = candidate["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise ReviewError("claim confidence must be finite")
        metric = candidate["metric"]
        if metric is not None and not isinstance(metric, str):
            raise ReviewError("claim metric must be a string or null")

        canonical = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        claim_id = "claim-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        if any(existing.claim_id == claim_id for existing in accepted):
            raise ReviewError(f"claim {index} duplicates an earlier claim")
        try:
            claim = ScientificClaim(
                claim_id=claim_id,
                source_text=source_text,
                claim_type=ClaimType(candidate["claim_type"]),
                subject=candidate["subject"],
                source=SourceLocation(
                    source_id=source_id,
                    kind=record.kind,
                    path=record.path,
                    start_line=start_line,
                    end_line=end_line,
                ),
                metric=metric,
                direction=ClaimDirection(candidate["direction"]),
                claimed_magnitude=_magnitude(candidate["claimed_magnitude"]),
                qualifiers=_string_list(candidate["qualifiers"], "claim qualifiers"),
                confidence=float(confidence),
                evidence_hints=_string_list(candidate["evidence_hints"], "evidence hints"),
            )
        except (TypeError, ValueError) as exc:
            raise ReviewError(f"claim {index} is invalid: {exc}") from exc
        accepted.append(claim)
    return tuple(accepted)
