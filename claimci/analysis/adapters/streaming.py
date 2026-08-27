"""Bounded fixed-registry scanners for descriptor-backed artifacts."""

from __future__ import annotations

import csv
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
    ConfigValue,
    ExperimentRole,
    NormalizedEvidence,
    NormalizedObservation,
    SelectorKind,
    TableSelector,
    selector_identity,
)
from .core import (
    AdapterError,
    AdapterSelectorError,
    _adapter_provenance,
    _evidence_id,
    _legacy_tabular_evidence_id,
    _supports,
    _validate_match,
)
from .dataset import _dataset_evidence_id
from .structured import (
    _GENERIC_JSONL_KINDS,
    _RESULT_TARGETS,
    _is_finite_number,
    JsonLinesAdapter,
    _match,
    _observation,
    _pointer_leaves,
    _result_mappings,
)
from .tabular import (
    _CSV_KINDS,
    _HEADER,
    _INTEGER,
    _TARGETS,
    _canonical_table_match,
    _finite_csv_number,
    _generated_mappings_from_numeric,
    _row_value,
    _table_cell_matches,
)
from ..profiles import _BENCHMARK_PROFILE_FIELDS


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


@dataclass(frozen=True, slots=True)
class StreamingDelimitedSchemaScan:
    """Header and numeric eligibility from a complete delimited scan."""

    match: AdapterMatch | None
    header: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    completeness: ScanCompleteness
    retained_raw_rows: int = 0

    def __post_init__(self) -> None:
        if type(self.completeness) is not ScanCompleteness:
            raise TypeError("streaming delimited completeness is invalid")
        if self.retained_raw_rows != 0:
            raise AnalysisContractError(
                "streaming delimited schema cannot retain rows"
            )
        if self.completeness.complete:
            if self.match is None or not self.header:
                raise AnalysisContractError(
                    "complete delimited schema requires a header and match"
                )
        elif self.match is not None or self.header or self.numeric_columns:
            raise AnalysisContractError(
                "incomplete delimited schema cannot expose provisional fields"
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


class _DelimitedPhysicalLines:
    """Strict text iterator whose logical-byte counter is reset per CSV row."""

    __slots__ = ("_lines", "_limits", "_scan", "logical_bytes")

    def __init__(self, scan: ArtifactScan, limits: StreamingLimits) -> None:
        self._scan = scan
        self._limits = limits
        self._lines = _binary_lines(
            scan,
            maximum=limits.max_logical_record_bytes,
        )
        self.logical_bytes = 0

    def __iter__(self) -> _DelimitedPhysicalLines:
        return self

    def __next__(self) -> str:
        raw, physical_bytes = next(self._lines)
        next_logical = self.logical_bytes + physical_bytes
        if next_logical > self._limits.max_logical_record_bytes:
            raise _ScanStop(
                ScanState.INCOMPLETE_LIMIT,
                ScanReason.LOGICAL_RECORD_LIMIT,
            )
        if (
            self._scan.semantic_bytes + physical_bytes
            > self._limits.max_semantic_bytes
        ):
            raise _ScanStop(
                ScanState.INCOMPLETE_BUDGET,
                ScanReason.SEMANTIC_BYTE_BUDGET,
            )
        if physical_bytes == len(raw) + 2:
            encoded = raw + b"\r\n"
        elif physical_bytes == len(raw) + 1:
            encoded = raw + b"\n"
        else:
            encoded = raw
        try:
            text = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise _ScanStop(ScanState.FAILED, ScanReason.INVALID_UTF8) from error
        if "\x00" in text:
            raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
        self._scan.note_semantic_bytes(
            physical_bytes,
            buffered=next_logical,
        )
        self.logical_bytes = next_logical
        return text

    def reset_logical_record(self) -> None:
        self.logical_bytes = 0


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
            evidence_id=_evidence_id(
                _JSONL_ADAPTER_ID,
                JsonLinesAdapter.semantic_version,
                source,
                match.mappings,
            ),
            artifact=source.candidate,
            adapter_match=match,
            observations=tuple(observations),
            scan_completeness=report,
            source_trace_sha256=scan.trace_sha256,
        )
        return StreamingExtraction(evidence, report)


def _delimited_format(source: ArtifactSource) -> tuple[str, str, str]:
    suffix = PurePosixPath(str(source.candidate.path)).suffix.casefold()
    if suffix == ".csv":
        return "excel", "CSV", "claimci-csv-v1"
    if suffix == ".tsv":
        return "excel-tab", "TSV", "claimci-tsv-v1"
    raise AdapterSelectorError("streaming delimited source must be CSV or TSV")


def _validated_header(
    row: list[str],
    *,
    limits: StreamingLimits,
) -> tuple[str, ...]:
    if len(row) > limits.max_columns:
        raise _ScanStop(ScanState.INCOMPLETE_LIMIT, ScanReason.COLUMN_LIMIT)
    if not row or any(not item for item in row):
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
    if len(set(row)) != len(row):
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
    if any(not _HEADER.fullmatch(item) for item in row):
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
    return tuple(row)


def _reader_row(
    reader: Iterator[list[str]],
    physical: _DelimitedPhysicalLines,
) -> list[str]:
    try:
        row = next(reader)
    except csv.Error as error:
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED) from error
    physical.reset_logical_record()
    return row


def scan_delimited_schema(
    source: ArtifactSource,
    *,
    limits: StreamingLimits = StreamingLimits(),
) -> StreamingDelimitedSchemaScan:
    """Scan CSV/TSV header and numeric columns without retaining data rows."""

    if type(source) is not ArtifactSource:
        raise TypeError("delimited schema scan requires ArtifactSource")
    if type(limits) is not StreamingLimits:
        raise TypeError("delimited schema limits must be StreamingLimits")
    if not _supports(
        source,
        kinds=_CSV_KINDS,
        suffixes=frozenset({".csv", ".tsv"}),
    ):
        raise AdapterSelectorError(
            "streaming delimited schema requires a fixed result-compatible source"
        )
    dialect, label, adapter_id = _delimited_format(source)
    records = 0
    with source.open_scan(ScanPurpose.SCHEMA, limits) as scan:
        physical = _DelimitedPhysicalLines(scan, limits)
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(limits.max_logical_record_bytes)
        try:
            reader = csv.reader(physical, dialect=dialect, strict=True)
            try:
                header_row = _reader_row(reader, physical)
            except StopIteration:
                report = _finish_empty_failure(scan, records_scanned=0)
                return StreamingDelimitedSchemaScan(None, (), (), report)
            header = _validated_header(header_row, limits=limits)
            numeric = [True] * len(header)
            while True:
                if records >= limits.max_semantic_records:
                    if scan.semantic_bytes < source.candidate.size:
                        raise _ScanStop(
                            ScanState.INCOMPLETE_BUDGET,
                            ScanReason.RECORD_BUDGET,
                        )
                    break
                try:
                    row = _reader_row(reader, physical)
                except StopIteration:
                    break
                if len(row) != len(header):
                    raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
                for index, cell in enumerate(row):
                    if not numeric[index]:
                        continue
                    try:
                        _finite_csv_number(cell, label=f"{label} value")
                    except AdapterError:
                        numeric[index] = False
                records += 1
            if records == 0:
                report = _finish_empty_failure(scan, records_scanned=0)
                return StreamingDelimitedSchemaScan(None, (), (), report)
            numeric_state = tuple(numeric)
            mappings = _generated_mappings_from_numeric(
                source,
                header,
                numeric_state,
                adapter_id=adapter_id,
            )
            unambiguous = any(
                item.target_field == "metric_value" for item in mappings
            )
            provenance = _adapter_provenance(
                source,
                adapter_id=adapter_id,
                selector=f"<{label} header>",
                inferred=not unambiguous,
            )
            match = AdapterMatch(
                adapter_id=adapter_id,
                path=source.candidate.path,
                confidence=Confidence(0.95 if unambiguous else 0.45),
                mappings=mappings,
                match_evidence=(provenance,),
            )
            report = scan.finish(
                records_scanned=records,
                semantic_complete=True,
            )
            numeric_columns = tuple(
                column
                for column, eligible in zip(header, numeric_state, strict=True)
                if eligible
            )
            return StreamingDelimitedSchemaScan(
                match,
                header,
                numeric_columns,
                report,
            )
        except _ScanStop as stop:
            report = _stopped_report(
                scan,
                records_scanned=records,
                stop=stop,
            )
            return StreamingDelimitedSchemaScan(None, (), (), report)
        except (AdapterError, AnalysisContractError, TypeError, ValueError):
            report = _finish_empty_failure(scan, records_scanned=records)
            return StreamingDelimitedSchemaScan(None, (), (), report)
        finally:
            csv.field_size_limit(previous_limit)


def _delimited_observation(
    source: ArtifactSource,
    *,
    adapter_id: str,
    label: str,
    row: tuple[str, ...],
    row_index: int,
    indices: dict[str, int],
    mappings: dict[str, object],
) -> NormalizedObservation:
    metric_mapping = mappings["metric_value"]
    metric_text = _row_value(
        row,
        indices,
        mappings,
        "metric_value",
        label=label,
    )
    assert metric_text is not None
    run_text = _row_value(
        row,
        indices,
        mappings,
        "run_id",
        label=label,
    )
    seed_text = _row_value(
        row,
        indices,
        mappings,
        "seed",
        label=label,
    )
    if run_text is not None:
        run_text = run_text.strip()
        if not run_text:
            raise AdapterSelectorError("run_id column contains an empty value")
    seed: str | int | None = None
    if seed_text is not None:
        seed_text = seed_text.strip()
        if not seed_text:
            raise AdapterSelectorError("seed column contains an empty value")
        seed = int(seed_text) if _INTEGER.fullmatch(seed_text) else seed_text
    provenance = _adapter_provenance(
        source,
        adapter_id=adapter_id,
        selector=f"row[{row_index}].{metric_mapping.selector.expression}",
        inferred="inferred=true" in metric_mapping.provenance.detail,
    )
    config_values: list[ConfigValue] = []
    for target, mapping in sorted(mappings.items()):
        if target not in _BENCHMARK_PROFILE_FIELDS:
            continue
        assert type(mapping.selector) is TableSelector
        cell = row[indices[mapping.selector.column]]
        if not cell:
            raise AdapterSelectorError(
                f"selected profile cell {mapping.selector.column!r} is empty"
            )
        config_values.append(ConfigValue(target, cell, provenance))
    try:
        return NormalizedObservation(
            provenance=provenance,
            metric_name=metric_mapping.selector.expression,
            metric_value=_finite_csv_number(
                metric_text,
                label=f"{label} metric value",
            ),
            run_id=run_text,
            seed=seed,
            experiment_role=ExperimentRole.UNSPECIFIED,
            config_values=tuple(config_values),
        )
    except AdapterError:
        raise
    except AnalysisContractError as error:
        raise AdapterSelectorError(
            f"selected {label} identifier is invalid: {error}"
        ) from error


def scan_delimited_observations(
    source: ArtifactSource,
    match: AdapterMatch,
    *,
    limits: StreamingLimits = StreamingLimits(),
) -> StreamingExtraction:
    """Extract exact selected CSV/TSV cells only after complete scanning."""

    if type(source) is not ArtifactSource:
        raise TypeError("delimited observation scan requires ArtifactSource")
    if type(limits) is not StreamingLimits:
        raise TypeError("delimited observation limits must be StreamingLimits")
    if not _supports(
        source,
        kinds=_CSV_KINDS,
        suffixes=frozenset({".csv", ".tsv"}),
    ):
        raise AdapterSelectorError(
            "streaming delimited observations require a fixed source"
        )
    dialect, label, adapter_id = _delimited_format(source)
    mappings = _validate_match(
        source,
        match,
        adapter_id=adapter_id,
        selector_kind=SelectorKind.COLUMN,
        allowed_targets=_TARGETS,
    )
    metric_mapping = mappings.get("metric_value")
    if metric_mapping is None:
        raise AdapterSelectorError(
            f"ambiguous {label} evidence requires an external metric_value mapping"
        )
    if any(
        type(mapping.selector) is TableSelector
        for target, mapping in mappings.items()
        if target in {"run_id", "seed"}
    ):
        raise AdapterSelectorError("run_id and seed cannot use table selectors")
    table_selector = (
        metric_mapping.selector
        if type(metric_mapping.selector) is TableSelector
        else None
    )
    profile_mappings = tuple(
        mapping
        for target, mapping in mappings.items()
        if target in _BENCHMARK_PROFILE_FIELDS
    )
    if profile_mappings and table_selector is None:
        raise AdapterSelectorError(
            "profile cells require an exact metric table row selector"
        )
    if any(
        type(mapping.selector) is not TableSelector
        for mapping in profile_mappings
    ):
        raise AdapterSelectorError("profile cells require exact table selectors")
    if table_selector is not None and any(
        mapping.selector.predicates != table_selector.predicates
        or mapping.selector.expected_cardinality
        != table_selector.expected_cardinality
        for mapping in profile_mappings
    ):
        raise AdapterSelectorError(
            "profile cells and metric must select the same exact row"
        )

    records = 0
    selected_count = 0
    observations: list[NormalizedObservation] = []
    with source.open_scan(ScanPurpose.SELECTED_OBSERVATIONS, limits) as scan:
        physical = _DelimitedPhysicalLines(scan, limits)
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(limits.max_logical_record_bytes)
        try:
            reader = csv.reader(physical, dialect=dialect, strict=True)
            try:
                header_row = _reader_row(reader, physical)
            except StopIteration:
                report = _finish_empty_failure(scan, records_scanned=0)
                return StreamingExtraction(None, report)
            header = _validated_header(header_row, limits=limits)
            indices = {name: index for index, name in enumerate(header)}
            for mapping in mappings.values():
                if mapping.selector.expression not in indices:
                    raise AdapterSelectorError(
                        f"mapped {label} column "
                        f"{mapping.selector.expression!r} is not in the header"
                    )
            if table_selector is not None:
                for predicate in table_selector.predicates:
                    if predicate.column not in indices:
                        raise AdapterSelectorError(
                            f"table predicate column {predicate.column!r} "
                            "is not in the header"
                        )
            canonical_match = _canonical_table_match(source, match)
            canonical_mappings = {
                item.target_field: item for item in canonical_match.mappings
            }
            canonical_metric = canonical_mappings["metric_value"]
            canonical_table = (
                canonical_metric.selector
                if type(canonical_metric.selector) is TableSelector
                else None
            )
            while True:
                if records >= limits.max_semantic_records:
                    if scan.semantic_bytes < source.candidate.size:
                        raise _ScanStop(
                            ScanState.INCOMPLETE_BUDGET,
                            ScanReason.RECORD_BUDGET,
                        )
                    break
                try:
                    row_value = _reader_row(reader, physical)
                except StopIteration:
                    break
                if len(row_value) != len(header):
                    raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
                row = tuple(row_value)
                selected = canonical_table is None or all(
                    _table_cell_matches(row[indices[item.column]], item)
                    for item in canonical_table.predicates
                )
                if selected:
                    selected_count += 1
                    if len(observations) >= limits.max_observations:
                        raise _ScanStop(
                            ScanState.INCOMPLETE_BUDGET,
                            ScanReason.OBSERVATION_BUDGET,
                        )
                    observations.append(
                        _delimited_observation(
                            source,
                            adapter_id=adapter_id,
                            label=label,
                            row=row,
                            row_index=records,
                            indices=indices,
                            mappings=canonical_mappings,
                        )
                    )
                records += 1
            if records == 0:
                report = _finish_empty_failure(scan, records_scanned=0)
                return StreamingExtraction(None, report)
            if canonical_table is not None and (
                selected_count != canonical_table.expected_cardinality
            ):
                report = scan.finish(
                    records_scanned=records,
                    semantic_complete=False,
                    state=ScanState.FAILED,
                    reason=ScanReason.SELECTOR_MISMATCH,
                )
                return StreamingExtraction(None, report)
            if not observations:
                report = _finish_empty_failure(scan, records_scanned=records)
                return StreamingExtraction(None, report)
            report = scan.finish(
                records_scanned=records,
                semantic_complete=True,
            )
            scoped = tuple(
                (
                    mapping.target_field,
                    selector_identity(mapping.selector),
                )
                for mapping in canonical_match.mappings
                if type(mapping.selector) is TableSelector
            )
            evidence_id = (
                _legacy_tabular_evidence_id(
                    adapter_id,
                    source,
                    adapter_semantic_version="1",
                    selector_identity=scoped,
                )
                if scoped
                else f"evidence-{adapter_id}-{str(source.candidate.sha256)[:16]}"
            )
            evidence = NormalizedEvidence(
                evidence_id=evidence_id,
                artifact=source.candidate,
                adapter_match=canonical_match,
                observations=tuple(observations),
                scan_completeness=report,
                source_trace_sha256=scan.trace_sha256,
            )
            return StreamingExtraction(evidence, report)
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
        finally:
            csv.field_size_limit(previous_limit)


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
        source_trace_sha256=scan.trace_sha256,
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
    if suffix in {".csv", ".tsv"}:
        if source.candidate.kind not in _CSV_KINDS:
            return None
        schema = scan_delimited_schema(source, limits=limits)
        if not schema.completeness.complete:
            return StreamingExtraction(None, schema.completeness)
        assert schema.match is not None
        return scan_delimited_observations(source, schema.match, limits=limits)
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
    "StreamingDelimitedSchemaScan",
    "StreamingExtraction",
    "StreamingSchemaScan",
    "extract_registered_source",
    "scan_delimited_observations",
    "scan_delimited_schema",
    "scan_jsonl_observations",
    "scan_jsonl_schema",
]
