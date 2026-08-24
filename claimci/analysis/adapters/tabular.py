"""Deterministic bounded comma-separated-value adapter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from decimal import Decimal, InvalidOperation

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactKind,
    Confidence,
    ConfigValue,
    ExperimentRole,
    FieldMapping,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    SelectorKind,
    TablePredicate,
    TableScalarType,
    TableSelector,
    selector_identity,
)
from claimci.analysis.profiles import _BENCHMARK_PROFILE_FIELDS

from .core import (
    MAX_COLUMNS,
    MAX_LOGICAL_RECORD_BYTES,
    MAX_RECORDS,
    AdapterError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _ArtifactEnvelope,
    _adapter_provenance,
    _evidence_id,
    _mapping,
    _supports,
    _validate_match,
    _verify_integrity,
)


_CSV_KINDS = frozenset(
    {ArtifactKind.RESULTS, ArtifactKind.BENCHMARK, ArtifactKind.DOCUMENT}
)
_RUN_NAMES = frozenset({"run", "run_id", "trial"})
_SEED_NAMES = frozenset({"seed", "random_seed"})
_TARGETS = frozenset({"metric_value", "run_id", "seed"}) | _BENCHMARK_PROFILE_FIELDS
_HEADER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_. -]{0,255}\Z")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_TABLE_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)


def _logical_record_size(row: list[str]) -> int:
    return sum(len(item.encode("utf-8")) for item in row) + max(0, len(row) - 1)


def _parse_delimited(
    content: bytes,
    *,
    dialect: str,
    label: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AdapterParseError(f"{label} artifact must contain valid UTF-8") from exc
    if "\x00" in text:
        raise AdapterParseError(f"{label} artifact contains NUL")

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(MAX_LOGICAL_RECORD_BYTES)
    try:
        reader = csv.reader(
            io.StringIO(text, newline=""),
            dialect=dialect,
            strict=True,
        )
        try:
            header_row = next(reader)
        except StopIteration as exc:
            raise AdapterParseError(f"{label} artifact must contain a header") from exc
        if _logical_record_size(header_row) > MAX_LOGICAL_RECORD_BYTES:
            raise AdapterLimitError(
                f"{label} header exceeds the 1 MiB logical record limit"
            )
        if len(header_row) > MAX_COLUMNS:
            raise AdapterLimitError(
                f"{label} artifact exceeds the {MAX_COLUMNS}-columns limit"
            )
        if not header_row or any(not item for item in header_row):
            raise AdapterParseError(f"{label} artifact contains an empty header")
        if len(set(header_row)) != len(header_row):
            raise AdapterParseError(f"{label} artifact contains a duplicate header")
        if any(not _HEADER.fullmatch(item) for item in header_row):
            raise AdapterParseError(f"{label} artifact contains an unsupported header")

        rows: list[tuple[str, ...]] = []
        for row in reader:
            if len(rows) >= MAX_RECORDS:
                raise AdapterLimitError(
                    f"{label} artifact exceeds the {MAX_RECORDS}-records limit"
                )
            if _logical_record_size(row) > MAX_LOGICAL_RECORD_BYTES:
                raise AdapterLimitError(
                    f"{label} logical record exceeds the 1 MiB limit"
                )
            if len(row) != len(header_row):
                raise AdapterParseError(
                    f"{label} row field count does not match header columns"
                )
            rows.append(tuple(row))
    except csv.Error as exc:
        raise AdapterParseError(f"malformed {label} artifact: {exc}") from exc
    finally:
        csv.field_size_limit(previous_limit)

    if not rows:
        raise AdapterParseError(
            f"{label} artifact must contain at least one data record"
        )
    return tuple(header_row), tuple(rows)


def _parse_csv(content: bytes) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    return _parse_delimited(content, dialect="excel", label="CSV")


def _parse_tsv(content: bytes) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    return _parse_delimited(content, dialect="excel-tab", label="TSV")


def _finite_csv_number(value: str, *, label: str) -> float:
    text = value.strip()
    if not text:
        raise AdapterParseError(f"{label} must be a finite number")
    try:
        number = float(text)
    except ValueError as exc:
        raise AdapterParseError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise AdapterParseError(f"{label} must be a finite number")
    return number


def _numeric_column(rows: tuple[tuple[str, ...], ...], index: int) -> bool:
    try:
        for row in rows:
            _finite_csv_number(row[index], label="CSV value")
    except AdapterParseError:
        return False
    return True


def _unique_named_header(header: tuple[str, ...], names: frozenset[str]) -> str | None:
    matches = tuple(item for item in header if item.lower() in names)
    return matches[0] if len(matches) == 1 else None


def _generated_mappings(
    artifact: _ArtifactEnvelope,
    header: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    *,
    adapter_id: str,
) -> tuple[FieldMapping, ...]:
    numeric = tuple(
        _numeric_column(rows, index) for index in range(len(header))
    )
    return _generated_mappings_from_numeric(
        artifact,
        header,
        numeric,
        adapter_id=adapter_id,
    )


def _generated_mappings_from_numeric(
    artifact: _ArtifactEnvelope,
    header: tuple[str, ...],
    numeric: tuple[bool, ...],
    *,
    adapter_id: str,
) -> tuple[FieldMapping, ...]:
    if len(numeric) != len(header):
        raise AnalysisContractError(
            "numeric column state must match the delimited header"
        )
    run_header = _unique_named_header(header, _RUN_NAMES)
    seed_header = _unique_named_header(header, _SEED_NAMES)
    excluded = {item for item in (run_header, seed_header) if item is not None}
    metric_headers = tuple(
        item
        for index, item in enumerate(header)
        if item not in excluded and numeric[index]
    )

    selected: list[tuple[str, str]] = []
    if len(metric_headers) == 1:
        selected.append(("metric_value", metric_headers[0]))
    if run_header is not None:
        selected.append(("run_id", run_header))
    if seed_header is not None:
        selected.append(("seed", seed_header))
    return tuple(
        _mapping(
            artifact,
            adapter_id=adapter_id,
            target_field=target,
            selector_kind=SelectorKind.COLUMN,
            selector=column,
            inferred=True,
        )
        for target, column in sorted(selected)
    )


def _row_value(
    row: tuple[str, ...],
    indices: dict[str, int],
    mappings: dict[str, FieldMapping],
    target: str,
    *,
    label: str,
) -> str | None:
    mapping = mappings.get(target)
    if mapping is None:
        return None
    column = mapping.selector.expression
    if column not in indices:
        raise AdapterSelectorError(
            f"mapped {label} column {column!r} is not in the header"
        )
    return row[indices[column]]


def _table_cell_matches(cell: str, predicate: TablePredicate) -> bool:
    if predicate.scalar_type is TableScalarType.STRING:
        return cell == predicate.value
    if predicate.scalar_type is TableScalarType.BOOLEAN:
        if cell not in {"true", "false"}:
            raise AdapterSelectorError(
                f"table predicate column {predicate.column!r} is not a strict boolean"
            )
        return cell == ("true" if predicate.value else "false")
    if predicate.scalar_type is TableScalarType.NULL:
        raise AdapterSelectorError(
            "CSV/TSV tables have no schema-backed null representation"
        )
    if not _TABLE_NUMBER.fullmatch(cell):
        raise AdapterSelectorError(
            f"table predicate column {predicate.column!r} is not a strict number"
        )
    try:
        parsed = Decimal(cell)
        expected = Decimal(predicate.canonical_value)
    except InvalidOperation as error:
        raise AdapterSelectorError("table predicate number is malformed") from error
    if not parsed.is_finite():
        raise AdapterSelectorError("table predicate number must be finite")
    return parsed == expected


def _selected_row_indices(
    rows: tuple[tuple[str, ...], ...],
    indices: dict[str, int],
    selector: TableSelector,
) -> tuple[int, ...]:
    if selector.column not in indices:
        raise AdapterSelectorError(
            f"table target column {selector.column!r} is not in the header"
        )
    for predicate in selector.predicates:
        if predicate.column not in indices:
            raise AdapterSelectorError(
                f"table predicate column {predicate.column!r} is not in the header"
            )
    selected = tuple(
        row_index
        for row_index, row in enumerate(rows)
        if all(
            _table_cell_matches(row[indices[item.column]], item)
            for item in selector.predicates
        )
    )
    if len(selected) != selector.expected_cardinality:
        raise AdapterSelectorError(
            "table selector matched "
            f"{len(selected)} rows; expected exactly {selector.expected_cardinality}"
        )
    return selected


def _canonical_table_match(
    artifact: _ArtifactEnvelope,
    match: AdapterMatch,
) -> AdapterMatch:
    table_mappings = tuple(
        item for item in match.mappings if type(item.selector) is TableSelector
    )
    if not table_mappings:
        return match
    canonical: list[FieldMapping] = []
    for mapping in match.mappings:
        if type(mapping.selector) is not TableSelector:
            canonical.append(mapping)
            continue
        material = json.dumps(
            selector_identity(mapping.selector),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        provenance = _adapter_provenance(
            artifact,
            adapter_id=match.adapter_id,
            selector="table:" + hashlib.sha256(material).hexdigest()[:16],
            inferred=False,
        )
        canonical.append(
            FieldMapping(
                target_field=mapping.target_field,
                selector=TableSelector(
                    column=mapping.selector.column,
                    predicates=mapping.selector.predicates,
                    expected_cardinality=mapping.selector.expected_cardinality,
                    provenance=provenance,
                ),
                provenance=provenance,
            )
        )
    evidence = _adapter_provenance(
        artifact,
        adapter_id=match.adapter_id,
        selector="<validated table selection>",
        inferred=False,
    )
    return AdapterMatch(
        adapter_id=match.adapter_id,
        path=match.path,
        confidence=match.confidence,
        mappings=tuple(canonical),
        match_evidence=(evidence,),
    )


class CsvAdapter:
    """Generic CSV observations with exact column or table-row selectors."""

    __slots__ = ()
    adapter_id = "claimci-csv-v1"
    semantic_version = "1"
    _suffixes = frozenset({".csv"})
    _parse = staticmethod(_parse_csv)
    _label = "CSV"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=_CSV_KINDS,
            suffixes=self._suffixes,
        ):
            return None
        _verify_integrity(artifact)
        header, rows = self._parse(artifact.content)
        mappings = _generated_mappings(
            artifact,
            header,
            rows,
            adapter_id=self.adapter_id,
        )
        unambiguous = any(item.target_field == "metric_value" for item in mappings)
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector=f"<{self._label} header>",
            inferred=not unambiguous,
        )
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(0.95 if unambiguous else 0.45),
            mappings=mappings,
            match_evidence=(provenance,),
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        header, rows = self._parse(artifact.content)
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.COLUMN,
            allowed_targets=_TARGETS,
        )
        metric_mapping = mappings.get("metric_value")
        if metric_mapping is None:
            raise AdapterSelectorError(
                f"ambiguous {self._label} evidence requires an external metric_value mapping"
            )
        indices = {name: index for index, name in enumerate(header)}
        for mapping in mappings.values():
            if (
                type(mapping.selector) is not TableSelector
                and mapping.selector.expression not in indices
            ):
                raise AdapterSelectorError(
                    f"mapped {self._label} column {mapping.selector.expression!r} is not in the header"
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
        if any(type(mapping.selector) is not TableSelector for mapping in profile_mappings):
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
        selected_rows = (
            tuple(range(len(rows)))
            if table_selector is None
            else _selected_row_indices(rows, indices, table_selector)
        )
        canonical_match = _canonical_table_match(artifact, match)
        canonical_mappings = {
            item.target_field: item for item in canonical_match.mappings
        }
        metric_mapping = canonical_mappings["metric_value"]

        observations: list[NormalizedObservation] = []
        for row_index in selected_rows:
            row = rows[row_index]
            metric_text = _row_value(
                row,
                indices,
                canonical_mappings,
                "metric_value",
                label=self._label,
            )
            assert metric_text is not None
            run_text = _row_value(
                row,
                indices,
                canonical_mappings,
                "run_id",
                label=self._label,
            )
            seed_text = _row_value(
                row,
                indices,
                canonical_mappings,
                "seed",
                label=self._label,
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
                artifact,
                adapter_id=self.adapter_id,
                selector=f"row[{row_index}].{metric_mapping.selector.expression}",
                inferred="inferred=true" in metric_mapping.provenance.detail,
            )
            config_values: list[ConfigValue] = []
            for target, mapping in sorted(canonical_mappings.items()):
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
                observations.append(
                    NormalizedObservation(
                        provenance=provenance,
                        metric_name=metric_mapping.selector.expression,
                        metric_value=_finite_csv_number(
                            metric_text,
                            label=f"{self._label} metric value",
                        ),
                        run_id=run_text,
                        seed=seed,
                        experiment_role=ExperimentRole.UNSPECIFIED,
                        config_values=tuple(config_values),
                    )
                )
            except AdapterError:
                raise
            except AnalysisContractError as exc:
                raise AdapterSelectorError(
                    f"selected {self._label} identifier is invalid: {exc}"
                ) from exc
        scoped_selector_identity = tuple(
            (
                mapping.target_field,
                selector_identity(mapping.selector),
            )
            for mapping in canonical_match.mappings
            if type(mapping.selector) is TableSelector
        )
        return NormalizedEvidence(
            evidence_id=_evidence_id(
                self.adapter_id,
                artifact,
                adapter_semantic_version=(
                    self.semantic_version if scoped_selector_identity else None
                ),
                selector_identity=(
                    scoped_selector_identity if scoped_selector_identity else None
                ),
            ),
            artifact=artifact.candidate,
            adapter_match=canonical_match,
            observations=tuple(observations),
        )


class TsvAdapter(CsvAdapter):
    """Generic TSV observations with exact column or table-row selectors."""

    __slots__ = ()
    adapter_id = "claimci-tsv-v1"
    semantic_version = "1"
    _suffixes = frozenset({".tsv"})
    _parse = staticmethod(_parse_tsv)
    _label = "TSV"


__all__ = ["CsvAdapter", "TsvAdapter"]
