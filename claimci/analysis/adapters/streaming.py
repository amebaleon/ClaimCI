"""Bounded fixed-registry scanners for descriptor-backed artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from claimci.parsing import unique_json_object

from ..artifact_source import (
    ArtifactScan,
    ArtifactSource,
    ScanCompleteness,
    ScanPurpose,
    ScanReason,
    ScanState,
    StreamingLimits,
)
from ..confidence import Confidence
from ..contracts import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactKind,
    ExperimentRole,
    NormalizedEvidence,
    NormalizedObservation,
    SelectorKind,
)
from .core import (
    AdapterError,
    AdapterSelectorError,
    _adapter_provenance,
    _evidence_id,
    _supports,
    _validate_match,
)
from .dataset import _dataset_evidence_id
from .structured import (
    _GENERIC_JSONL_KINDS,
    _RESULT_TARGETS,
    _is_finite_number,
    _match,
    _observation,
    _pointer_leaves,
    _result_mappings,
)


_JSONL_ADAPTER_ID = "claimci-jsonl-v1"
_DATASET_ADAPTER_ID = "claimci-jsonl-dataset-v1"


@dataclass(frozen=True, slots=True)
class StreamingSchemaScan:
    """A bounded common-schema result that never retains decoded records."""

    match: AdapterMatch | None
    common_leaves: tuple[tuple[str, object], ...]
    completeness: ScanCompleteness
    retained_raw_records: int = 0

    def __post_init__(self) -> None:
        if type(self.completeness) is not ScanCompleteness:
            raise TypeError("streaming schema completeness is invalid")
        if self.retained_raw_records != 0:
            raise AnalysisContractError(
                "streaming schema scans cannot retain raw records"
            )
        if self.completeness.complete:
            if self.match is None:
                raise AnalysisContractError(
                    "complete streaming schema requires an adapter match"
                )
        elif self.match is not None or self.common_leaves:
            raise AnalysisContractError(
                "incomplete streaming schema cannot expose provisional fields"
            )


@dataclass(frozen=True, slots=True)
class StreamingExtraction:
    """Normalized evidence eligible only after a complete verified scan."""

    evidence: NormalizedEvidence | None
    completeness: ScanCompleteness
    retained_raw_records: int = 0

    def __post_init__(self) -> None:
        if type(self.completeness) is not ScanCompleteness:
            raise TypeError("streaming extraction completeness is invalid")
        if self.retained_raw_records != 0:
            raise AnalysisContractError(
                "streaming extraction cannot retain raw records"
            )
        if self.completeness.complete:
            if not self.completeness.integrity_verified or self.evidence is None:
                raise AnalysisContractError(
                    "complete streaming extraction requires verified evidence"
                )
        elif self.evidence is not None:
            raise AnalysisContractError(
                "incomplete streaming extraction cannot expose evidence"
            )


@dataclass(slots=True)
class _LeafState:
    first: object
    all_numeric: bool
    all_identifier: bool


class _ScanStop(Exception):
    def __init__(self, state: ScanState, reason: ScanReason) -> None:
        super().__init__(reason.value)
        self.state = state
        self.reason = reason


def _json_constant(_token: str) -> object:
    raise ValueError("non-finite JSON constant")


def _record_value(raw: bytes, limits: StreamingLimits) -> Mapping[str, object]:
    if not raw.strip():
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _ScanStop(ScanState.FAILED, ScanReason.INVALID_UTF8) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED) from error
    if not isinstance(value, Mapping):
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
    _validate_record_graph(value, limits)
    return value


def _validate_record_graph(value: object, limits: StreamingLimits) -> None:
    nodes = 0
    root_depth = 1 if isinstance(value, (Mapping, list, tuple)) else 0
    stack: list[tuple[object, int]] = [(value, root_depth)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_nodes_per_record:
            raise _ScanStop(ScanState.INCOMPLETE_LIMIT, ScanReason.NODE_LIMIT)
        if isinstance(current, float) and not math.isfinite(current):
            raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        if depth > limits.max_depth:
            raise _ScanStop(ScanState.INCOMPLETE_LIMIT, ScanReason.DEPTH_LIMIT)
        if isinstance(current, Mapping):
            keys = sorted(current)
            children = tuple(
                child for key in keys for child in (key, current[key])
            )
        else:
            children = tuple(current)
        for child in reversed(children):
            stack.append((child, depth + 1))


def _binary_lines(
    scan: ArtifactScan,
    *,
    maximum: int,
) -> Iterator[tuple[bytes, int]]:
    line = bytearray()
    physical_bytes = 0
    while True:
        block = scan.read_chunk()
        if not block:
            if line:
                if len(line) > maximum:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_LIMIT,
                        ScanReason.LOGICAL_RECORD_LIMIT,
                    )
                yield bytes(line), physical_bytes
            return
        cursor = 0
        while cursor < len(block):
            newline = block.find(b"\n", cursor)
            endpoint = len(block) if newline < 0 else newline
            segment = block[cursor:endpoint]
            if len(line) + len(segment) > maximum + 1:
                scan.note_semantic_bytes(0, buffered=min(len(line), maximum))
                raise _ScanStop(
                    ScanState.INCOMPLETE_LIMIT,
                    ScanReason.LOGICAL_RECORD_LIMIT,
                )
            line.extend(segment)
            physical_bytes += len(segment)
            scan.note_semantic_bytes(
                0,
                buffered=len(line) - 1 if line.endswith(b"\r") else len(line),
            )
            if newline < 0:
                break
            raw = bytes(line[:-1] if line.endswith(b"\r") else line)
            if len(raw) > maximum:
                raise _ScanStop(
                    ScanState.INCOMPLETE_LIMIT,
                    ScanReason.LOGICAL_RECORD_LIMIT,
                )
            yield raw, physical_bytes + 1
            line.clear()
            physical_bytes = 0
            cursor = newline + 1


def _stopped_report(
    scan: ArtifactScan,
    *,
    records_scanned: int,
    stop: _ScanStop,
) -> ScanCompleteness:
    return scan.finish(
        records_scanned=records_scanned,
        semantic_complete=False,
        state=stop.state,
        reason=stop.reason,
    )


def _finish_empty_failure(
    scan: ArtifactScan,
    *,
    records_scanned: int,
) -> ScanCompleteness:
    return scan.finish(
        records_scanned=records_scanned,
        semantic_complete=False,
        state=ScanState.FAILED,
        reason=ScanReason.MALFORMED,
    )


def _supports_result_jsonl(source: ArtifactSource) -> bool:
    return _supports(
        source,
        kinds=_GENERIC_JSONL_KINDS,
        suffixes=frozenset({".jsonl"}),
    )


def scan_jsonl_schema(
    source: ArtifactSource,
    *,
    limits: StreamingLimits = StreamingLimits(),
) -> StreamingSchemaScan:
    """Scan a JSONL common schema without retaining decoded records."""

    if type(source) is not ArtifactSource:
        raise TypeError("JSONL schema scan requires ArtifactSource")
    if type(limits) is not StreamingLimits:
        raise TypeError("JSONL schema limits must be StreamingLimits")
    if not _supports_result_jsonl(source):
        raise AdapterSelectorError(
            "streaming JSONL schema requires a fixed result-compatible source"
        )

    states: dict[str, _LeafState] = {}
    records = 0
    with source.open_scan(ScanPurpose.SCHEMA, limits) as scan:
        try:
            for raw, physical_bytes in _binary_lines(
                scan,
                maximum=limits.max_logical_record_bytes,
            ):
                if records >= limits.max_semantic_records:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.RECORD_BUDGET,
                    )
                if scan.semantic_bytes + physical_bytes > limits.max_semantic_bytes:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.SEMANTIC_BYTE_BUDGET,
                    )
                scan.note_semantic_bytes(physical_bytes, buffered=len(raw))
                value = _record_value(raw, limits)
                leaves = dict(_pointer_leaves(value))
                if records == 0:
                    states = {
                        pointer: _LeafState(
                            first=item,
                            all_numeric=_is_finite_number(item),
                            all_identifier=isinstance(item, (str, int))
                            and not isinstance(item, bool),
                        )
                        for pointer, item in leaves.items()
                    }
                else:
                    for pointer in tuple(states):
                        if pointer not in leaves:
                            del states[pointer]
                            continue
                        item = leaves[pointer]
                        state = states[pointer]
                        state.all_numeric = state.all_numeric and _is_finite_number(item)
                        state.all_identifier = state.all_identifier and isinstance(
                            item, (str, int)
                        ) and not isinstance(item, bool)
                        if not state.all_numeric and not state.all_identifier:
                            del states[pointer]
                records += 1
        except _ScanStop as stop:
            report = _stopped_report(
                scan,
                records_scanned=records,
                stop=stop,
            )
            return StreamingSchemaScan(None, (), report)
        except (AdapterError, AnalysisContractError, TypeError, ValueError):
            report = _finish_empty_failure(scan, records_scanned=records)
            return StreamingSchemaScan(None, (), report)

        if records == 0:
            report = _finish_empty_failure(scan, records_scanned=0)
            return StreamingSchemaScan(None, (), report)
        common = tuple(
            (pointer, state.first)
            for pointer, state in sorted(states.items())
            if state.all_numeric or state.all_identifier
        )
        try:
            mappings = _result_mappings(
                source,
                adapter_id=_JSONL_ADAPTER_ID,
                leaves=common,
            )
            confidence = (
                0.95
                if any(item.target_field == "metric_value" for item in mappings)
                else 0.45
            )
            match = _match(
                source,
                adapter_id=_JSONL_ADAPTER_ID,
                mappings=mappings,
                confidence=confidence,
            )
        except (AnalysisContractError, TypeError, ValueError):
            report = _finish_empty_failure(scan, records_scanned=records)
            return StreamingSchemaScan(None, (), report)
        report = scan.finish(records_scanned=records, semantic_complete=True)
        return StreamingSchemaScan(match, common, report)


def scan_jsonl_observations(
    source: ArtifactSource,
    match: AdapterMatch,
    *,
    limits: StreamingLimits = StreamingLimits(),
) -> StreamingExtraction:
    """Extract only selected JSONL observations after a complete scan."""

    if type(source) is not ArtifactSource:
        raise TypeError("JSONL observation scan requires ArtifactSource")
    if type(limits) is not StreamingLimits:
        raise TypeError("JSONL observation limits must be StreamingLimits")
    if not _supports_result_jsonl(source):
        raise AdapterSelectorError(
            "streaming JSONL observations require a fixed result-compatible source"
        )
    mappings = _validate_match(
        source,
        match,
        adapter_id=_JSONL_ADAPTER_ID,
        selector_kind=SelectorKind.JSON_POINTER,
        allowed_targets=_RESULT_TARGETS,
    )
    metric_mapping = mappings.get("metric_value")
    if metric_mapping is None:
        raise AdapterSelectorError(
            "ambiguous JSONL evidence requires an external metric_value mapping"
        )

    observations: list[NormalizedObservation] = []
    records = 0
    with source.open_scan(ScanPurpose.SELECTED_OBSERVATIONS, limits) as scan:
        try:
            for raw, physical_bytes in _binary_lines(
                scan,
                maximum=limits.max_logical_record_bytes,
            ):
                if records >= limits.max_semantic_records:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.RECORD_BUDGET,
                    )
                if len(observations) >= limits.max_observations:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.OBSERVATION_BUDGET,
                    )
                if scan.semantic_bytes + physical_bytes > limits.max_semantic_bytes:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.SEMANTIC_BYTE_BUDGET,
                    )
                scan.note_semantic_bytes(physical_bytes, buffered=len(raw))
                value = _record_value(raw, limits)
                observations.append(
                    _observation(
                        value,
                        mappings,
                        provenance=_adapter_provenance(
                            source,
                            adapter_id=_JSONL_ADAPTER_ID,
                            selector=(
                                f"record[{records}]"
                                f"{metric_mapping.selector.expression}"
                            ),
                            inferred=(
                                "inferred=true"
                                in metric_mapping.provenance.detail
                            ),
                        ),
                    )
                )
                records += 1
        except _ScanStop as stop:
            report = _stopped_report(
                scan,
                records_scanned=records,
                stop=stop,
            )
            return StreamingExtraction(None, report)
        except (AdapterError, AnalysisContractError, TypeError, ValueError):
            report = _finish_empty_failure(scan, records_scanned=records)
            return StreamingExtraction(None, report)

        if not observations:
            report = _finish_empty_failure(scan, records_scanned=records)
            return StreamingExtraction(None, report)
        report = scan.finish(records_scanned=records, semantic_complete=True)
        evidence = NormalizedEvidence(
            evidence_id=_evidence_id(_JSONL_ADAPTER_ID, source),
            artifact=source.candidate,
            adapter_match=match,
            observations=tuple(observations),
            scan_completeness=report,
        )
        return StreamingExtraction(evidence, report)


def _extract_dataset_identity(
    source: ArtifactSource,
    limits: StreamingLimits,
) -> StreamingExtraction:
    provenance = _adapter_provenance(
        source,
        adapter_id=_DATASET_ADAPTER_ID,
        selector="<whole passive dataset artifact>",
        inferred=False,
    )
    match = AdapterMatch(
        adapter_id=_DATASET_ADAPTER_ID,
        path=source.candidate.path,
        confidence=Confidence(0.99),
        mappings=(),
        match_evidence=(provenance,),
    )
    with source.open_scan(ScanPurpose.INTEGRITY_ONLY, limits) as scan:
        report = scan.finish(records_scanned=0, semantic_complete=True)
    from ..contracts import DatasetReference

    evidence = NormalizedEvidence(
        evidence_id=_dataset_evidence_id(_DATASET_ADAPTER_ID, source),
        artifact=source.candidate,
        adapter_match=match,
        observations=(
            NormalizedObservation(
                provenance=provenance,
                experiment_role=ExperimentRole.UNSPECIFIED,
                dataset_references=(
                    DatasetReference(
                        path=source.candidate.path,
                        split=None,
                        provenance=provenance,
                    ),
                ),
            ),
        ),
        scan_completeness=report,
    )
    return StreamingExtraction(evidence, report)


def extract_registered_source(
    source: ArtifactSource,
    *,
    limits: StreamingLimits = StreamingLimits(),
) -> StreamingExtraction | None:
    """Extract through the fixed source-capable registry only."""

    if type(source) is not ArtifactSource:
        raise TypeError("registered source extraction requires ArtifactSource")
    if type(limits) is not StreamingLimits:
        raise TypeError("registered source limits must be StreamingLimits")
    suffix = PurePosixPath(str(source.candidate.path)).suffix.casefold()
    if suffix != ".jsonl":
        return None
    if source.candidate.kind is ArtifactKind.DATASET:
        return _extract_dataset_identity(source, limits)
    if source.candidate.kind not in _GENERIC_JSONL_KINDS:
        return None
    schema = scan_jsonl_schema(source, limits=limits)
    if not schema.completeness.complete:
        return StreamingExtraction(None, schema.completeness)
    assert schema.match is not None
    return scan_jsonl_observations(source, schema.match, limits=limits)


__all__ = [
    "StreamingExtraction",
    "StreamingSchemaScan",
    "extract_registered_source",
    "scan_jsonl_observations",
    "scan_jsonl_schema",
]
