"""Deterministic mapping proposals and one minimal clarification question."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from claimci.analysis.confidence import Confidence
from claimci.analysis.contracts import (
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    MappingCandidate,
    MappingChoice,
    MappingQuestion,
    MappingTrust,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
)

from .models import DiscoveredClaim, DiscoveryError, DiscoveryLimits


@dataclass(frozen=True, slots=True)
class MappingResolution:
    candidates: tuple[MappingCandidate, ...]
    approved_mapping: RepoMapping | None
    question: MappingQuestion | None


@dataclass(frozen=True, slots=True)
class _Ambiguity:
    kind: ArtifactKind
    role: ExperimentRole
    artifacts: tuple[ArtifactCandidate, ...]
    affected_claim_ids: tuple[str, ...]
    unlock_count: int


_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_BASELINE_TOKENS = frozenset({"baseline", "base", "control", "reference"})
_CANDIDATE_TOKENS = frozenset({"candidate", "proposed", "new"})
_MAPPED_KINDS = (ArtifactKind.RESULTS, ArtifactKind.CONFIG)
_SLOT_PRIORITY = {
    (ArtifactKind.RESULTS, ExperimentRole.CANDIDATE): 40,
    (ArtifactKind.RESULTS, ExperimentRole.BASELINE): 30,
    (ArtifactKind.CONFIG, ExperimentRole.CANDIDATE): 20,
    (ArtifactKind.CONFIG, ExperimentRole.BASELINE): 10,
}


def _tokens(path: RepositoryPath) -> set[str]:
    return set(_TOKEN.findall(str(path).casefold()))


def _role(path: RepositoryPath) -> ExperimentRole:
    tokens = _tokens(path)
    baseline = bool(tokens & _BASELINE_TOKENS)
    candidate = bool(tokens & _CANDIDATE_TOKENS)
    if baseline == candidate:
        return ExperimentRole.UNSPECIFIED
    return ExperimentRole.BASELINE if baseline else ExperimentRole.CANDIDATE


def _binding(
    artifact: ArtifactCandidate,
    role: ExperimentRole,
    provenance: FieldProvenance,
) -> ArtifactBinding:
    return ArtifactBinding(
        path=artifact.path,
        kind=artifact.kind,
        role=role,
        adapter_id=None,
        mappings=(),
        provenance=provenance,
    )


def _mapping_id(prefix: str, bindings: tuple[ArtifactBinding, ...]) -> str:
    material = json.dumps(
        [
            (str(binding.path), binding.kind.value, binding.role.value)
            for binding in bindings
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"mapping-{prefix}-" + hashlib.sha256(material).hexdigest()[:16]


def _inferred_mapping(
    artifacts: tuple[ArtifactCandidate, ...],
) -> tuple[MappingCandidate | None, tuple[_Ambiguity, ...]]:
    grouped: dict[tuple[ArtifactKind, ExperimentRole], list[ArtifactCandidate]] = (
        defaultdict(list)
    )
    for artifact in artifacts:
        role = _role(artifact.path)
        if artifact.kind in _MAPPED_KINDS and role in {
            ExperimentRole.BASELINE,
            ExperimentRole.CANDIDATE,
        }:
            grouped[(artifact.kind, role)].append(artifact)

    selected: list[tuple[ArtifactCandidate, ExperimentRole]] = []
    ambiguities: list[_Ambiguity] = []
    for kind in _MAPPED_KINDS:
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
            candidates = sorted(
                grouped.get((kind, role), ()),
                key=lambda item: (
                    -item.confidence.value,
                    str(item.path).casefold(),
                    str(item.path),
                ),
            )
            if not candidates:
                continue
            if (
                len(candidates) > 1
                and candidates[0].confidence.value
                - candidates[1].confidence.value
                < 0.10
            ):
                affected = tuple(
                    sorted(
                        {
                            claim_id
                            for artifact in candidates
                            for claim_id in artifact.relevant_claim_ids
                        }
                    )
                )
                ambiguities.append(
                    _Ambiguity(
                        kind=kind,
                        role=role,
                        artifacts=tuple(candidates),
                        affected_claim_ids=affected,
                        unlock_count=1,
                    )
                )
                continue
            selected.append((candidates[0], role))

    if not selected:
        return None, tuple(ambiguities)
    selected.sort(key=lambda item: (item[0].kind.value, item[1].value, str(item[0].path)))
    provenance = FieldProvenance(
        kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
        detail="experiment roles inferred from complete repository path tokens",
    )
    bindings = tuple(
        _binding(artifact, role, provenance) for artifact, role in selected
    )
    confidence = Confidence(
        min(0.98, sum(item.confidence.value for item, _ in selected) / len(selected))
    )
    return (
        MappingCandidate(
            mapping_id=_mapping_id("inferred", bindings),
            bindings=bindings,
            confidence=confidence,
            trust=MappingTrust.INFERRED,
            provenance=provenance,
        ),
        tuple(ambiguities),
    )


def _strict_fields(value: object, expected: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DiscoveryError(f"provider {label} has invalid fields")
    return value


def _provider_mappings(
    payload: object,
    artifacts: tuple[ArtifactCandidate, ...],
    limits: DiscoveryLimits,
) -> tuple[MappingCandidate, ...]:
    try:
        root = _strict_fields(payload, {"mappings"}, "mapping response")
        values = root["mappings"]
        if (
            not isinstance(values, list)
            or len(values) > limits.max_provider_mappings
        ):
            raise DiscoveryError("provider mapping list exceeds its bound")
        by_path = {artifact.path: artifact for artifact in artifacts}
        accepted: list[MappingCandidate] = []
        for index, raw_mapping in enumerate(values):
            mapping = _strict_fields(
                raw_mapping,
                {"confidence", "bindings"},
                f"mapping {index}",
            )
            raw_bindings = mapping["bindings"]
            if not isinstance(raw_bindings, list) or not 1 <= len(raw_bindings) <= 16:
                raise DiscoveryError("provider mapping bindings are invalid")
            bindings: list[ArtifactBinding] = []
            for binding_index, raw_binding in enumerate(raw_bindings):
                binding = _strict_fields(
                    raw_binding,
                    {"path", "kind", "role", "adapter_id", "mappings"},
                    f"mapping {index} binding {binding_index}",
                )
                path = RepositoryPath(binding["path"])
                artifact = by_path.get(path)
                kind = ArtifactKind(binding["kind"])
                role = ExperimentRole(binding["role"])
                if artifact is None or artifact.kind is not kind:
                    raise DiscoveryError(
                        "provider mapping path is not a matching artifact candidate"
                    )
                raw_fields = binding["mappings"]
                if not isinstance(raw_fields, list) or len(raw_fields) > 32:
                    raise DiscoveryError("provider field mappings exceed their bound")
                provenance = FieldProvenance(
                    kind=ProvenanceKind.PROVIDER_PROPOSAL,
                    detail="validated provider mapping proposal",
                    source_path=path,
                )
                field_mappings: list[FieldMapping] = []
                for field_index, raw_field in enumerate(raw_fields):
                    field = _strict_fields(
                        raw_field,
                        {"target_field", "selector"},
                        f"mapping {index} binding {binding_index} field {field_index}",
                    )
                    selector_value = _strict_fields(
                        field["selector"],
                        {"kind", "expression"},
                        "selector",
                    )
                    selector = EvidenceSelector(
                        kind=SelectorKind(selector_value["kind"]),
                        expression=selector_value["expression"],
                        provenance=provenance,
                    )
                    field_mappings.append(
                        FieldMapping(
                            target_field=field["target_field"],
                            selector=selector,
                            provenance=provenance,
                        )
                    )
                bindings.append(
                    ArtifactBinding(
                        path=path,
                        kind=kind,
                        role=role,
                        adapter_id=binding["adapter_id"],
                        mappings=tuple(field_mappings),
                        provenance=provenance,
                    )
                )
            binding_tuple = tuple(bindings)
            candidate_provenance = FieldProvenance(
                kind=ProvenanceKind.PROVIDER_PROPOSAL,
                detail="strictly validated provider mapping proposal",
                source_id=_mapping_id("provider-source", binding_tuple),
            )
            accepted.append(
                MappingCandidate(
                    mapping_id=_mapping_id("provider", binding_tuple),
                    bindings=binding_tuple,
                    confidence=Confidence(mapping["confidence"]),
                    trust=MappingTrust.INFERRED,
                    provenance=candidate_provenance,
                )
            )
        return tuple(accepted)
    except DiscoveryError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise DiscoveryError("provider mapping proposal is invalid") from error


def _question_prompt(kind: ArtifactKind, role: ExperimentRole) -> str:
    role_text = role.value
    kind_text = kind.value
    return f"Which file contains the {role_text} {kind_text}?"


def _mapping_question(
    ambiguities: tuple[_Ambiguity, ...],
    *,
    selected_slot_count: int,
) -> MappingQuestion | None:
    if not ambiguities:
        return None
    ranked = sorted(
        ambiguities,
        key=lambda item: (
            -len(item.affected_claim_ids),
            -(item.unlock_count + selected_slot_count),
            -_SLOT_PRIORITY.get((item.kind, item.role), 0),
            item.kind.value,
            item.role.value,
        ),
    )
    ambiguity = ranked[0]
    choices: list[MappingChoice] = []
    for artifact in ambiguity.artifacts[:8]:
        provenance = FieldProvenance(
            kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
            detail=f"clarification choice for {ambiguity.role.value} {ambiguity.kind.value}",
            source_path=artifact.path,
        )
        binding = _binding(artifact, ambiguity.role, provenance)
        choice_id = "choice-" + hashlib.sha256(
            f"{ambiguity.kind.value}:{ambiguity.role.value}:{artifact.path}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        choices.append(
            MappingChoice(
                choice_id=choice_id,
                label=str(artifact.path),
                bindings=(binding,),
            )
        )
    material = "|".join(choice.choice_id for choice in choices)
    question_id = "question-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    relevant_claim_id = (
        ambiguity.affected_claim_ids[0]
        if len(ambiguity.affected_claim_ids) == 1
        else None
    )
    return MappingQuestion(
        question_id=question_id,
        prompt=_question_prompt(ambiguity.kind, ambiguity.role),
        choices=tuple(choices),
        relevant_claim_id=relevant_claim_id,
    )


def resolve_mappings(
    artifacts: tuple[ArtifactCandidate, ...],
    claims: tuple[DiscoveredClaim, ...],
    manifest_mappings: tuple[MappingCandidate, ...],
    *,
    repository: RepositoryIdentity,
    approved_mapping: RepoMapping | None = None,
    provider_payload: object | None = None,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> MappingResolution:
    """Return stable mapping proposals and at most one blocking clarification."""

    if not isinstance(artifacts, tuple) or not all(
        isinstance(item, ArtifactCandidate) for item in artifacts
    ):
        raise TypeError("artifacts must be a tuple of ArtifactCandidate values")
    if not isinstance(claims, tuple) or not all(
        isinstance(item, DiscoveredClaim) for item in claims
    ):
        raise TypeError("claims must be a tuple of DiscoveredClaim values")
    if not isinstance(manifest_mappings, tuple) or not all(
        isinstance(item, MappingCandidate)
        and item.trust is MappingTrust.MANIFEST_HINT
        for item in manifest_mappings
    ):
        raise TypeError("manifest_mappings must contain manifest-hint candidates")
    if not isinstance(repository, RepositoryIdentity):
        raise TypeError("repository must be RepositoryIdentity")
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("limits must be DiscoveryLimits")
    if approved_mapping is not None:
        if not isinstance(approved_mapping, RepoMapping):
            raise TypeError("approved_mapping must be RepoMapping or null")
        if approved_mapping.repository != repository:
            raise DiscoveryError("approved mapping repository does not match discovery repository")

    inferred, ambiguities = _inferred_mapping(artifacts)
    candidates = list(manifest_mappings)
    if inferred is not None:
        candidates.append(inferred)
    if provider_payload is not None:
        candidates.extend(_provider_mappings(provider_payload, artifacts, limits))
    unique: dict[str, MappingCandidate] = {}
    for candidate in candidates:
        if candidate.mapping_id in unique:
            raise DiscoveryError("mapping proposal identifiers must be unique")
        unique[candidate.mapping_id] = candidate
    tier = {MappingTrust.MANIFEST_HINT: 0, MappingTrust.INFERRED: 1}
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                tier[item.trust],
                -item.confidence.value,
                item.mapping_id,
            ),
        )[: limits.max_mapping_candidates]
    )
    question = None
    if approved_mapping is None and not manifest_mappings:
        selected_slots = 0 if inferred is None else len(inferred.bindings)
        question = _mapping_question(
            ambiguities,
            selected_slot_count=selected_slots,
        )
    return MappingResolution(
        candidates=ordered,
        approved_mapping=approved_mapping,
        question=question,
    )


__all__ = ["MappingResolution", "resolve_mappings"]
