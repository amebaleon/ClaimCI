"""Canonical runtime identity helpers for selector-scoped passive evidence."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import (
    ArtifactBinding,
    EvidenceSelector,
    FieldMapping,
    NormalizedEvidence,
    TableSelector,
    field_mapping_identity,
)


def field_mapping_projection(
    mappings: tuple[FieldMapping, ...],
) -> tuple[tuple[object, ...], ...]:
    """Project selectors without provenance while preserving legacy identity."""

    return tuple(sorted(field_mapping_identity(item) for item in mappings))


def evidence_matches_binding(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
) -> bool:
    """Match one binding to one independently extracted evidence variant."""

    if (
        evidence.artifact.path != binding.path
        or evidence.artifact.kind is not binding.kind
    ):
        return False
    if (
        binding.adapter_id is not None
        and evidence.adapter_match.adapter_id != binding.adapter_id
    ):
        return False
    return not binding.mappings or field_mapping_projection(
        evidence.adapter_match.mappings
    ) == field_mapping_projection(binding.mappings)


def evidence_for_binding(
    binding: ArtifactBinding,
    evidence_values: Iterable[NormalizedEvidence],
) -> NormalizedEvidence | None:
    """Resolve only a unique exact runtime match; ambiguity always fails closed."""

    matches = tuple(
        item for item in evidence_values if evidence_matches_binding(item, binding)
    )
    return matches[0] if len(matches) == 1 else None


def field_mapping_material(mapping: FieldMapping) -> dict[str, object]:
    """Return JSON-safe selector material while preserving legacy projections."""

    if type(mapping) is not FieldMapping:
        raise TypeError("field mapping material requires FieldMapping")
    material: dict[str, object] = {
        "target": mapping.target_field,
        "kind": mapping.selector.kind.value,
        "expression": mapping.selector.expression,
    }
    if type(mapping.selector) is TableSelector:
        material.update(
            {
                "selector_type": "table",
                "expected_cardinality": mapping.selector.expected_cardinality,
                "predicates": [
                    {
                        "column": item.column,
                        "scalar_type": item.scalar_type.value,
                        "value": item.canonical_value,
                    }
                    for item in mapping.selector.predicates
                ],
            }
        )
    elif type(mapping.selector) is not EvidenceSelector:
        raise TypeError("field mapping contains an unsupported selector")
    return material


def canonical_selector_material(
    mappings: tuple[FieldMapping, ...],
) -> tuple[dict[str, object], ...]:
    """Return sorted selector material without provenance or scientific role."""

    if not isinstance(mappings, tuple) or not all(
        type(item) is FieldMapping for item in mappings
    ):
        raise TypeError("selector material requires a tuple of FieldMapping values")
    return tuple(
        field_mapping_material(item)
        for item in sorted(mappings, key=field_mapping_identity)
    )


__all__ = [
    "evidence_for_binding",
    "evidence_matches_binding",
    "canonical_selector_material",
    "field_mapping_material",
    "field_mapping_projection",
]
