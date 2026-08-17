"""Bounded generic JSON and per-record JSONL adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping

from claimci.analysis import (
    AdapterMatch,
    ArtifactKind,
    Confidence,
    ConfigValue,
    ExperimentRole,
    FieldMapping,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    SelectorKind,
)

from .core import (
    MAX_LOGICAL_RECORD_BYTES,
    MAX_NODES,
    MAX_RECORDS,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _adapter_provenance,
    _config_target,
    _evidence_id,
    _finite_number,
    _mapping,
    _parse_json,
    _resolve_json_pointer,
    _supports,
    _validate_match,
    _verify_integrity,
)


_RUN_NAMES = frozenset({"run", "run_id", "trial"})
_SEED_NAMES = frozenset({"seed", "random_seed"})
_GENERIC_JSON_KINDS = frozenset(
    {
        ArtifactKind.RESULTS,
        ArtifactKind.BENCHMARK,
        ArtifactKind.DOCUMENT,
        ArtifactKind.CONFIG,
    }
)
_GENERIC_JSONL_KINDS = frozenset(
    {ArtifactKind.RESULTS, ArtifactKind.BENCHMARK, ArtifactKind.DOCUMENT}
)
_RESULT_TARGETS = frozenset({"metric_name", "metric_value", "run_id", "seed"})


def _encode_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_leaves(value: object) -> tuple[tuple[str, object], ...]:
    leaves: list[tuple[str, object]] = []

    def walk(current: object, tokens: tuple[str, ...]) -> None:
        if isinstance(current, Mapping):
            for key in sorted(current):
                walk(current[key], (*tokens, _encode_pointer_token(key)))
            return
        if isinstance(current, list):
            for index, item in enumerate(current):
                walk(item, (*tokens, str(index)))
            return
        pointer = "/" + "/".join(tokens)
        leaves.append((pointer, current))

    walk(value, ())
    return tuple(leaves)


def _pointer_terminal(pointer: str) -> str:
    token = pointer.rsplit("/", 1)[-1]
    return token.replace("~1", "/").replace("~0", "~")


def _dotted_from_pointer(pointer: str) -> str:
    tokens = pointer[1:].split("/")
    return ".".join(token.replace("~1", "/").replace("~0", "~") for token in tokens)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _unique_named_pointer(
    leaves: tuple[tuple[str, object], ...], names: frozenset[str]
) -> str | None:
    matches = tuple(
        pointer
        for pointer, _value in leaves
        if _pointer_terminal(pointer).lower() in names
    )
    return matches[0] if len(matches) == 1 else None


def _result_mappings(
    artifact: PassiveArtifact,
    *,
    adapter_id: str,
    leaves: tuple[tuple[str, object], ...],
) -> tuple[FieldMapping, ...]:
    run_pointer = _unique_named_pointer(leaves, _RUN_NAMES)
    seed_pointer = _unique_named_pointer(leaves, _SEED_NAMES)
    excluded = {item for item in (run_pointer, seed_pointer) if item is not None}
    metrics = tuple(
        pointer
        for pointer, value in leaves
        if pointer not in excluded and _is_finite_number(value)
    )

    mappings: list[FieldMapping] = []
    if len(metrics) == 1:
        mappings.append(
            _mapping(
                artifact,
                adapter_id=adapter_id,
                target_field="metric_value",
                selector_kind=SelectorKind.JSON_POINTER,
                selector=metrics[0],
                inferred=True,
            )
        )
    if run_pointer is not None:
        mappings.append(
            _mapping(
                artifact,
                adapter_id=adapter_id,
                target_field="run_id",
                selector_kind=SelectorKind.JSON_POINTER,
                selector=run_pointer,
                inferred=True,
            )
        )
    if seed_pointer is not None:
        mappings.append(
            _mapping(
                artifact,
                adapter_id=adapter_id,
                target_field="seed",
                selector_kind=SelectorKind.JSON_POINTER,
                selector=seed_pointer,
                inferred=True,
            )
        )
    return tuple(sorted(mappings, key=lambda item: (item.target_field, item.selector.expression)))


def _match(
    artifact: PassiveArtifact,
    *,
    adapter_id: str,
    mappings: tuple[FieldMapping, ...],
    confidence: float,
) -> AdapterMatch:
    evidence = _adapter_provenance(
        artifact,
        adapter_id=adapter_id,
        selector="<artifact>",
        inferred=confidence < 0.8,
    )
    return AdapterMatch(
        adapter_id=adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(confidence),
        mappings=mappings,
        match_evidence=(evidence,),
    )


def _metric_name(value: object, mapping: dict[str, FieldMapping]) -> str:
    explicit = mapping.get("metric_name")
    if explicit is not None:
        selected = _resolve_json_pointer(value, explicit.selector.expression)
        if not isinstance(selected, str) or not selected.strip():
            raise AdapterSelectorError("metric_name selector must resolve to text")
        return selected
    inferred = _pointer_terminal(mapping["metric_value"].selector.expression)
    if not inferred:
        raise AdapterSelectorError("metric name cannot be derived from the selector")
    return inferred


def _run_id(value: object, mapping: dict[str, FieldMapping]) -> str | None:
    selected_mapping = mapping.get("run_id")
    if selected_mapping is None:
        return None
    selected = _resolve_json_pointer(value, selected_mapping.selector.expression)
    if isinstance(selected, bool) or not isinstance(selected, (str, int)):
        raise AdapterSelectorError("run_id selector must resolve to text or an integer")
    return str(selected)


def _seed(value: object, mapping: dict[str, FieldMapping]) -> str | int | None:
    selected_mapping = mapping.get("seed")
    if selected_mapping is None:
        return None
    selected = _resolve_json_pointer(value, selected_mapping.selector.expression)
    if isinstance(selected, bool) or not isinstance(selected, (str, int)):
        raise AdapterSelectorError("seed selector must resolve to text or an integer")
    return selected


def _observation(
    value: object,
    mappings: dict[str, FieldMapping],
    *,
    provenance,
) -> NormalizedObservation:
    metric_mapping = mappings.get("metric_value")
    if metric_mapping is None:
        raise AdapterSelectorError(
            "ambiguous structured evidence requires an external metric_value mapping"
        )
    metric_value = _finite_number(
        _resolve_json_pointer(value, metric_mapping.selector.expression),
        label="metric value",
    )
    return NormalizedObservation(
        provenance=provenance,
        metric_name=_metric_name(value, mappings),
        metric_value=metric_value,
        run_id=_run_id(value, mappings),
        seed=_seed(value, mappings),
        experiment_role=ExperimentRole.UNSPECIFIED,
    )


def _jsonl_records(content: bytes) -> tuple[Mapping[str, object], ...]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AdapterParseError("artifact must contain valid UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise AdapterParseError("JSONL artifact must contain at least one record")
    if len(lines) > MAX_RECORDS:
        raise AdapterLimitError(f"JSONL artifact exceeds the {MAX_RECORDS}-records limit")

    records: list[Mapping[str, object]] = []
    total_nodes = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AdapterParseError(f"JSONL logical record {line_number} is blank")
        if len(line.encode("utf-8")) > MAX_LOGICAL_RECORD_BYTES:
            raise AdapterLimitError(
                f"JSONL logical record {line_number} exceeds the 1 MiB limit"
            )
        parsed = _parse_json(line.encode("utf-8"))
        if not isinstance(parsed, Mapping):
            raise AdapterParseError(
                f"JSONL logical record {line_number} must be a JSON object"
            )
        records.append(parsed)
        total_nodes += _node_count(parsed)
        if total_nodes > MAX_NODES:
            raise AdapterLimitError(
                f"JSONL artifact exceeds the {MAX_NODES}-nodes limit"
            )
    return tuple(records)


def _node_count(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + sum(1 + _node_count(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_node_count(item) for item in value)
    return 1


def _common_leaves(
    records: tuple[Mapping[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    first = dict(_pointer_leaves(records[0]))
    others = tuple(dict(_pointer_leaves(item)) for item in records[1:])
    common: list[tuple[str, object]] = []
    for pointer in sorted(first):
        if all(pointer in item for item in others):
            values = (first[pointer], *(item[pointer] for item in others))
            if all(_is_finite_number(item) for item in values):
                common.append((pointer, values[0]))
            elif all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in values):
                common.append((pointer, values[0]))
    return tuple(common)


class JsonAdapter:
    """Generic finite JSON evidence adapter."""

    __slots__ = ()
    adapter_id = "claimci-json-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=_GENERIC_JSON_KINDS,
            suffixes=frozenset({".json"}),
        ):
            return None
        _verify_integrity(artifact)
        value = _parse_json(artifact.content)
        leaves = _pointer_leaves(value)
        if artifact.candidate.kind is ArtifactKind.CONFIG:
            mappings = tuple(
                _mapping(
                    artifact,
                    adapter_id=self.adapter_id,
                    target_field=_config_target(_dotted_from_pointer(pointer)),
                    selector_kind=SelectorKind.JSON_POINTER,
                    selector=pointer,
                    inferred=True,
                )
                for pointer, _item in leaves
            )
            if not mappings:
                raise AdapterParseError("JSON config contains no scalar evidence")
            return _match(
                artifact,
                adapter_id=self.adapter_id,
                mappings=mappings,
                confidence=0.9,
            )

        mappings = _result_mappings(
            artifact,
            adapter_id=self.adapter_id,
            leaves=leaves,
        )
        confidence = 0.95 if any(item.target_field == "metric_value" for item in mappings) else 0.45
        return _match(
            artifact,
            adapter_id=self.adapter_id,
            mappings=mappings,
            confidence=confidence,
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        value = _parse_json(artifact.content)
        if artifact.candidate.kind is ArtifactKind.CONFIG:
            leaves = _pointer_leaves(value)
            allowed = frozenset(
                _config_target(_dotted_from_pointer(pointer)) for pointer, _item in leaves
            )
            mappings = _validate_match(
                artifact,
                match,
                adapter_id=self.adapter_id,
                selector_kind=SelectorKind.JSON_POINTER,
                allowed_targets=allowed,
            )
            config_values: list[ConfigValue] = []
            for mapping in sorted(mappings.values(), key=lambda item: item.selector.expression):
                dotted = _dotted_from_pointer(mapping.selector.expression)
                if mapping.target_field != _config_target(dotted):
                    raise AdapterSelectorError("config mapping target does not match selector")
                selected = _resolve_json_pointer(value, mapping.selector.expression)
                if isinstance(selected, (Mapping, list)):
                    raise AdapterSelectorError("config selector must resolve to a scalar")
                config_values.append(ConfigValue(dotted, selected, mapping.provenance))
            if not config_values:
                raise AdapterSelectorError("config extraction requires at least one mapping")
            provenance = _adapter_provenance(
                artifact,
                adapter_id=self.adapter_id,
                selector="<config mappings>",
                inferred=False,
            )
            observations = (
                NormalizedObservation(
                    provenance=provenance,
                    experiment_role=ExperimentRole.UNSPECIFIED,
                    config_values=tuple(config_values),
                ),
            )
        else:
            mappings = _validate_match(
                artifact,
                match,
                adapter_id=self.adapter_id,
                selector_kind=SelectorKind.JSON_POINTER,
                allowed_targets=_RESULT_TARGETS,
            )
            metric_mapping = mappings.get("metric_value")
            provenance = (
                metric_mapping.provenance
                if metric_mapping is not None
                else _adapter_provenance(
                    artifact,
                    adapter_id=self.adapter_id,
                    selector="<missing metric mapping>",
                    inferred=True,
                )
            )
            observations = (_observation(value, mappings, provenance=provenance),)
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=observations,
        )


class JsonLinesAdapter:
    """Generic JSONL adapter whose selectors apply to each record."""

    __slots__ = ()
    adapter_id = "claimci-jsonl-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=_GENERIC_JSONL_KINDS,
            suffixes=frozenset({".jsonl"}),
        ):
            return None
        _verify_integrity(artifact)
        records = _jsonl_records(artifact.content)
        mappings = _result_mappings(
            artifact,
            adapter_id=self.adapter_id,
            leaves=_common_leaves(records),
        )
        confidence = 0.95 if any(item.target_field == "metric_value" for item in mappings) else 0.45
        return _match(
            artifact,
            adapter_id=self.adapter_id,
            mappings=mappings,
            confidence=confidence,
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        records = _jsonl_records(artifact.content)
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.JSON_POINTER,
            allowed_targets=_RESULT_TARGETS,
        )
        metric_mapping = mappings.get("metric_value")
        if metric_mapping is None:
            raise AdapterSelectorError(
                "ambiguous JSONL evidence requires an external metric_value mapping"
            )
        observations = tuple(
            _observation(
                record,
                mappings,
                provenance=_adapter_provenance(
                    artifact,
                    adapter_id=self.adapter_id,
                    selector=(
                        f"record[{index}]{metric_mapping.selector.expression}"
                    ),
                    inferred="inferred=true" in metric_mapping.provenance.detail,
                ),
            )
            for index, record in enumerate(records)
        )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=observations,
        )


__all__ = ["JsonAdapter", "JsonLinesAdapter"]
