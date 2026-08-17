"""Deterministic bounded comma-separated-value adapter."""

from __future__ import annotations

import csv
import io
import math
import re

from claimci.analysis import (
    AdapterMatch,
    ArtifactKind,
    Confidence,
    ExperimentRole,
    FieldMapping,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    SelectorKind,
)

from .core import (
    MAX_COLUMNS,
    MAX_LOGICAL_RECORD_BYTES,
    MAX_RECORDS,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
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
_TARGETS = frozenset({"metric_value", "run_id", "seed"})
_HEADER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_. -]{0,255}\Z")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


def _logical_record_size(row: list[str]) -> int:
    return sum(len(item.encode("utf-8")) for item in row) + max(0, len(row) - 1)


def _parse_csv(content: bytes) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AdapterParseError("CSV artifact must contain valid UTF-8") from exc
    if "\x00" in text:
        raise AdapterParseError("CSV artifact contains NUL")

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(MAX_LOGICAL_RECORD_BYTES)
    try:
        reader = csv.reader(
            io.StringIO(text, newline=""),
            dialect="excel",
            strict=True,
        )
        try:
            header_row = next(reader)
        except StopIteration as exc:
            raise AdapterParseError("CSV artifact must contain a header") from exc
        if _logical_record_size(header_row) > MAX_LOGICAL_RECORD_BYTES:
            raise AdapterLimitError("CSV header exceeds the 1 MiB logical record limit")
        if len(header_row) > MAX_COLUMNS:
            raise AdapterLimitError(
                f"CSV artifact exceeds the {MAX_COLUMNS}-columns limit"
            )
        if not header_row or any(not item for item in header_row):
            raise AdapterParseError("CSV artifact contains an empty header")
        if len(set(header_row)) != len(header_row):
            raise AdapterParseError("CSV artifact contains a duplicate header")
        if any(not _HEADER.fullmatch(item) for item in header_row):
            raise AdapterParseError("CSV artifact contains an unsupported header")

        rows: list[tuple[str, ...]] = []
        for row in reader:
            if len(rows) >= MAX_RECORDS:
                raise AdapterLimitError(
                    f"CSV artifact exceeds the {MAX_RECORDS}-records limit"
                )
            if _logical_record_size(row) > MAX_LOGICAL_RECORD_BYTES:
                raise AdapterLimitError("CSV logical record exceeds the 1 MiB limit")
            if len(row) != len(header_row):
                raise AdapterParseError(
                    "CSV row field count does not match header columns"
                )
            rows.append(tuple(row))
    except csv.Error as exc:
        raise AdapterParseError(f"malformed CSV artifact: {exc}") from exc
    finally:
        csv.field_size_limit(previous_limit)

    if not rows:
        raise AdapterParseError("CSV artifact must contain at least one data record")
    return tuple(header_row), tuple(rows)


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
    artifact: PassiveArtifact,
    header: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    *,
    adapter_id: str,
) -> tuple[FieldMapping, ...]:
    run_header = _unique_named_header(header, _RUN_NAMES)
    seed_header = _unique_named_header(header, _SEED_NAMES)
    excluded = {item for item in (run_header, seed_header) if item is not None}
    metric_headers = tuple(
        item
        for index, item in enumerate(header)
        if item not in excluded and _numeric_column(rows, index)
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
) -> str | None:
    mapping = mappings.get(target)
    if mapping is None:
        return None
    column = mapping.selector.expression
    if column not in indices:
        raise AdapterSelectorError(f"mapped CSV column {column!r} is not in the header")
    return row[indices[column]]


class CsvAdapter:
    """Generic CSV observations with exact-column selectors only."""

    __slots__ = ()
    adapter_id = "claimci-csv-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=_CSV_KINDS,
            suffixes=frozenset({".csv"}),
        ):
            return None
        _verify_integrity(artifact)
        header, rows = _parse_csv(artifact.content)
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
            selector="<CSV header>",
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
        header, rows = _parse_csv(artifact.content)
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
                "ambiguous CSV evidence requires an external metric_value mapping"
            )
        indices = {name: index for index, name in enumerate(header)}
        for mapping in mappings.values():
            if mapping.selector.expression not in indices:
                raise AdapterSelectorError(
                    f"mapped CSV column {mapping.selector.expression!r} is not in the header"
                )

        observations: list[NormalizedObservation] = []
        for row_index, row in enumerate(rows):
            metric_text = _row_value(
                row, indices, mappings, "metric_value"
            )
            assert metric_text is not None
            run_text = _row_value(row, indices, mappings, "run_id")
            seed_text = _row_value(row, indices, mappings, "seed")
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
            observations.append(
                NormalizedObservation(
                    provenance=provenance,
                    metric_name=metric_mapping.selector.expression,
                    metric_value=_finite_csv_number(
                        metric_text,
                        label="CSV metric value",
                    ),
                    run_id=run_text,
                    seed=seed,
                    experiment_role=ExperimentRole.UNSPECIFIED,
                )
            )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=tuple(observations),
        )


__all__ = ["CsvAdapter"]
