"""Shared deterministic parsing and security boundaries for v0.3 adapters.

Every helper in this module operates on already-confined repository paths and
immutable passive bytes.  The helpers never import customer code, execute
expressions, or resolve selectors outside the decoded artifact value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactCandidate,
    ArtifactKind,
    ArtifactOccurrence,
    EvidenceSelector,
    FieldMapping,
    FieldProvenance,
    PassiveArtifact,
    ProvenanceKind,
    SelectorKind,
)
from claimci.parsing import unique_json_object

from ..evidence_identity import canonical_selector_material


MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_COLUMNS = 256
MAX_DEPTH = 64
MAX_NODES = 100_000
MAX_LOGICAL_RECORD_BYTES = 1024 * 1024

_DOTTED_SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")
_CANONICAL_ARRAY_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class _ArtifactEnvelope(Protocol):
    occurrence: ArtifactOccurrence
    candidate: ArtifactCandidate


class AdapterError(AnalysisContractError):
    """A controlled failure isolated to one passive artifact."""


class AdapterParseError(AdapterError):
    """The artifact is recognized but cannot be parsed safely."""


class AdapterLimitError(AdapterError):
    """The artifact hits a declared resource boundary."""


class AdapterSelectorError(AdapterError):
    """A selector or externally supplied mapping is invalid."""


class AdapterIntegrityError(AdapterError):
    """Passive bytes no longer match discovery-time metadata."""


def _verify_integrity(artifact: PassiveArtifact) -> None:
    """Re-check size and SHA immediately before probing or extraction."""

    if not isinstance(artifact, PassiveArtifact):
        raise TypeError("artifact must be PassiveArtifact")
    content = artifact.content
    if not isinstance(content, bytes):
        raise AdapterIntegrityError("passive artifact content is not immutable bytes")
    actual_size = len(content)
    if actual_size > MAX_ARTIFACT_BYTES:
        raise AdapterLimitError(
            f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte (8 MiB) limit"
        )
    if actual_size != artifact.candidate.size:
        raise AdapterIntegrityError("artifact integrity size mismatch")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != artifact.candidate.sha256:
        raise AdapterIntegrityError("artifact integrity sha256 mismatch")


def _decode_utf8(content: bytes) -> str:
    """Decode one artifact as strict UTF-8 without replacement characters."""

    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AdapterParseError("artifact must contain valid UTF-8") from exc


def _reject_json_constant(token: str) -> object:
    raise AdapterParseError(f"JSON constant {token!r} is not finite")


def _parse_json(content: bytes) -> object:
    """Parse one bounded JSON document with duplicate and finite checks."""

    text = _decode_utf8(content)
    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except AdapterError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AdapterParseError(f"malformed JSON artifact: {exc}") from exc
    _validate_graph(value)
    return value


def _validate_graph(value: object) -> None:
    """Validate a decoded data tree against depth, node, and alias limits.

    Containers and scalar values each count as one node.  Mapping keys are
    counted as nodes too.  The root scalar has depth zero; a root container is
    depth one, so exactly ``MAX_DEPTH`` nested containers is accepted.
    """

    nodes = 0
    active: set[int] = set()
    completed: set[int] = set()
    root_depth = 1 if isinstance(value, (Mapping, list, tuple)) else 0
    stack: list[tuple[bool, object, int]] = [(False, value, root_depth)]

    while stack:
        exiting, current, depth = stack.pop()
        if exiting:
            identity = id(current)
            active.remove(identity)
            completed.add(identity)
            continue

        nodes += 1
        if nodes > MAX_NODES:
            raise AdapterLimitError(
                f"artifact exceeds the {MAX_NODES}-nodes limit"
            )

        if isinstance(current, float) and not math.isfinite(current):
            raise AdapterParseError("artifact numbers must be finite")
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        if depth > MAX_DEPTH:
            raise AdapterLimitError(
                f"artifact exceeds the maximum supported depth {MAX_DEPTH}"
            )

        identity = id(current)
        if identity in active:
            raise AdapterParseError("cyclic container graph is not supported")
        if identity in completed:
            raise AdapterParseError("repeated container aliases are not supported")
        active.add(identity)
        stack.append((True, current, depth))
        if isinstance(current, Mapping):
            keys = sorted(current, key=lambda item: str(item))
            if any(not isinstance(key, str) for key in keys):
                raise AdapterParseError("artifact mappings require string keys")
            children = tuple(
                child for key in keys for child in (key, current[key])
            )
        else:
            children = tuple(current)
        for child in reversed(children):
            stack.append((False, child, depth + 1))


def _decode_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            result.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise AdapterSelectorError("JSON pointer contains an invalid escape")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _resolve_json_pointer(value: object, expression: str) -> object:
    """Resolve a strict RFC 6901 pointer inside one decoded JSON value."""

    if not isinstance(expression, str) or not expression.startswith("/"):
        raise AdapterSelectorError("JSON pointer must start with '/'")
    current = value
    for encoded in expression[1:].split("/"):
        token = _decode_pointer_token(encoded)
        if isinstance(current, Mapping):
            if token not in current:
                raise AdapterSelectorError(f"JSON pointer token {token!r} is missing")
            current = current[token]
        elif isinstance(current, list):
            if not _CANONICAL_ARRAY_INDEX.fullmatch(token):
                raise AdapterSelectorError(
                    "JSON pointer array indices must be canonical non-negative integers"
                )
            index = int(token)
            if index >= len(current):
                raise AdapterSelectorError("JSON pointer array index is out of bounds")
            current = current[index]
        else:
            raise AdapterSelectorError("JSON pointer traverses a scalar value")
    return current


def _resolve_dotted_path(value: object, expression: str) -> object:
    """Resolve a structural dotted path without expression evaluation."""

    if not isinstance(expression, str):
        raise AdapterSelectorError("dotted selector must be text")
    segments = expression.split(".")
    if not segments or any(not _DOTTED_SEGMENT.fullmatch(item) for item in segments):
        raise AdapterSelectorError(
            "dotted selector segments must match [A-Za-z0-9_-]+"
        )
    current = value
    for segment in segments:
        if not isinstance(current, Mapping):
            raise AdapterSelectorError("dotted selector traverses a non-mapping value")
        if segment not in current:
            raise AdapterSelectorError(f"dotted selector segment {segment!r} is missing")
        current = current[segment]
    return current


def _scalar_leaves(value: object) -> tuple[tuple[tuple[str, object], ...], tuple[str, ...]]:
    """Return representable dotted scalar leaves and unsupported paths."""

    leaves: list[tuple[str, object]] = []
    unsupported: list[str] = []

    def walk(current: object, parts: tuple[str, ...]) -> None:
        label = ".".join(parts)
        if isinstance(current, Mapping):
            for key in sorted(current, key=lambda item: str(item)):
                key_label = str(key)
                child_parts = (*parts, key_label)
                child_label = ".".join(child_parts)
                if not isinstance(key, str) or not _DOTTED_SEGMENT.fullmatch(key):
                    unsupported.append(child_label)
                    continue
                walk(current[key], child_parts)
            return
        if isinstance(current, list):
            unsupported.append(label)
            return
        if not parts:
            unsupported.append("<root>")
            return
        leaves.append((label, current))

    walk(value, ())
    return tuple(leaves), tuple(sorted(unsupported))


def _finite_number(value: object, *, label: str) -> float:
    """Normalize an integer/float while rejecting booleans and non-finites."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterParseError(f"{label} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise AdapterParseError(f"{label} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise AdapterParseError(f"{label} must be a finite number")
    return normalized


def _adapter_provenance(
    artifact: _ArtifactEnvelope,
    *,
    adapter_id: str,
    selector: str,
    inferred: bool,
    kind: ProvenanceKind = ProvenanceKind.ADAPTER_EXTRACTION,
) -> FieldProvenance:
    """Build field-level provenance with path, selector, and content hash."""

    detail = (
        f"adapter={adapter_id}; selector={selector}; "
        f"sha256={artifact.candidate.sha256}; inferred={str(inferred).lower()}"
    )
    return FieldProvenance(
        kind=kind,
        detail=detail,
        source_path=artifact.candidate.path,
        source_id=f"{adapter_id}:{str(artifact.candidate.sha256)[:16]}",
    )


def _mapping(
    artifact: _ArtifactEnvelope,
    *,
    adapter_id: str,
    target_field: str,
    selector_kind: SelectorKind,
    selector: str,
    inferred: bool,
    provenance_kind: ProvenanceKind = ProvenanceKind.ADAPTER_EXTRACTION,
) -> FieldMapping:
    provenance = _adapter_provenance(
        artifact,
        adapter_id=adapter_id,
        selector=selector,
        inferred=inferred,
        kind=provenance_kind,
    )
    return FieldMapping(
        target_field=target_field,
        selector=EvidenceSelector(selector_kind, selector, provenance),
        provenance=provenance,
    )


def _config_target(selector: str) -> str:
    digest = hashlib.sha256(selector.encode("utf-8")).hexdigest()[:16]
    return f"config.k{digest}"


def _supports(
    artifact: _ArtifactEnvelope,
    *,
    kinds: frozenset[ArtifactKind],
    suffixes: frozenset[str],
) -> bool:
    suffix = PurePosixPath(str(artifact.candidate.path)).suffix.lower()
    return artifact.candidate.kind in kinds and suffix in suffixes


def _validate_match(
    artifact: _ArtifactEnvelope,
    match: AdapterMatch,
    *,
    adapter_id: str,
    selector_kind: SelectorKind,
    allowed_targets: frozenset[str],
) -> dict[str, FieldMapping]:
    """Validate external mappings before deterministic selector resolution."""

    if not isinstance(match, AdapterMatch):
        raise AdapterSelectorError("adapter match must be AdapterMatch")
    if match.adapter_id != adapter_id:
        raise AdapterSelectorError("adapter match uses the wrong adapter identifier")
    if match.path != artifact.candidate.path:
        raise AdapterSelectorError("adapter match path does not match artifact path")

    validated: dict[str, FieldMapping] = {}
    expected_sha256 = str(artifact.candidate.sha256)
    for provenance in match.match_evidence:
        if provenance.source_path != artifact.candidate.path:
            raise AdapterSelectorError(
                "adapter match provenance path does not match artifact path"
            )
        if expected_sha256 not in provenance.detail:
            raise AdapterSelectorError(
                "adapter match provenance does not bind the current sha256"
            )
    for mapping in match.mappings:
        if mapping.target_field not in allowed_targets:
            raise AdapterSelectorError(
                f"mapping target {mapping.target_field!r} is not supported"
            )
        if mapping.selector.kind is not selector_kind:
            raise AdapterSelectorError("mapping selector kind is not supported")
        for provenance in (mapping.provenance, mapping.selector.provenance):
            if provenance.source_path != artifact.candidate.path:
                raise AdapterSelectorError(
                    "mapping selector provenance path does not match artifact path"
                )
            if expected_sha256 not in provenance.detail:
                raise AdapterSelectorError(
                    "mapping selector provenance does not bind the current sha256"
                )
        validated[mapping.target_field] = mapping
    return validated


def _legacy_tabular_evidence_id(
    adapter_id: str,
    artifact: _ArtifactEnvelope,
    *,
    adapter_semantic_version: str,
    selector_identity: object,
) -> str:
    """Preserve the selector-scoped-v1 table identity until Task 4."""

    if not isinstance(adapter_semantic_version, str) or not adapter_semantic_version:
        raise TypeError("selector-scoped evidence requires an adapter semantic version")
    material = json.dumps(
        {
            "schema": "claimci-selector-scoped-evidence-v1",
            "artifact": {
                "path": str(artifact.candidate.path),
                "kind": artifact.candidate.kind.value,
                "sha256": str(artifact.candidate.sha256),
                "size": artifact.candidate.size,
            },
            "adapter_id": adapter_id,
            "adapter_semantic_version": adapter_semantic_version,
            "selector": selector_identity,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return (
        f"evidence-{adapter_id}-table-"
        + hashlib.sha256(material).hexdigest()[:16]
    )


def _evidence_id(
    adapter_id: str,
    adapter_semantic_version: str,
    artifact: _ArtifactEnvelope,
    mappings: tuple[FieldMapping, ...],
) -> str:
    """Return the exact v2 identity for one trusted occurrence and selector."""

    if not isinstance(adapter_id, str) or not adapter_id:
        raise TypeError("adapter ID must be a non-empty string")
    if not isinstance(adapter_semantic_version, str) or not adapter_semantic_version:
        raise TypeError("adapter semantic version must be a non-empty string")
    occurrence = getattr(artifact, "occurrence", None)
    if type(occurrence) is not ArtifactOccurrence:
        raise TypeError(
            "evidence v2 identity requires a concrete ArtifactOccurrence"
        )
    if getattr(artifact, "candidate", None) is not occurrence.candidate:
        raise TypeError(
            "evidence v2 identity requires the candidate bound to the occurrence"
        )
    candidate = occurrence.candidate
    material = {
        "schema_version": 2,
        "repository": {
            "owner": occurrence.repository.owner,
            "name": occurrence.repository.name,
        },
        "snapshot": {
            "role": occurrence.snapshot_role.value,
            "commit": str(occurrence.commit),
        },
        "artifact": {
            "path": str(candidate.path),
            "kind": candidate.kind.value,
            "sha256": str(candidate.sha256),
            "size": candidate.size,
        },
        "adapter": {"id": adapter_id, "version": adapter_semantic_version},
        "selector": list(canonical_selector_material(mappings)),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "evidence-v2-" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MAX_ARTIFACT_BYTES",
    "MAX_RECORDS",
    "MAX_COLUMNS",
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_LOGICAL_RECORD_BYTES",
    "AdapterError",
    "AdapterParseError",
    "AdapterLimitError",
    "AdapterSelectorError",
    "AdapterIntegrityError",
]
