"""Deterministic metadata for comparing represented verification inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from .contracts import ArtifactKind, DatasetSplit, ExperimentRole, RepositoryPath, Sha256Digest
from .replay import ReplayRecipe
from .verification import (
    VerificationArtifactIdentity,
    VerificationInputSnapshot,
    VerificationSnapshotCapability,
)


STALE_EVIDENCE_VERSION = "claimci.stale-evidence.v1"


class StalenessState(str, Enum):
    FRESH = "fresh"
    BYTE_ONLY_CHANGE = "byte_only_change"
    IDENTITY_RELOCATION = "identity_relocation"
    MATERIAL_CHANGE = "material_change"
    INDETERMINATE = "indeterminate"
    LEGACY_UNAVAILABLE = "legacy_unavailable"


class ReverificationDecision(str, Enum):
    NO_REVERIFY_TRIGGER_DETECTED = "no_reverify_trigger_detected"
    REVERIFY_REQUIRED = "reverify_required"
    UNDETERMINED = "undetermined"


class RequiredAction(str, Enum):
    NONE = "none"
    REVERIFY = "reverify"
    REMAP = "remap"
    SUPPLY_PROVENANCE = "supply_provenance"


class StalenessReason(str, Enum):
    LEGACY_REPLAY_UNAVAILABLE = "legacy_replay_unavailable"
    SNAPSHOT_UNAVAILABLE = "snapshot_unavailable"
    SNAPSHOT_DETAILS_UNAVAILABLE = "snapshot_details_unavailable"
    REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE = (
        "required_semantic_coverage_incomplete"
    )
    REPOSITORY_IDENTITY_CHANGED = "repository_identity_changed"
    CLAIM_STATEMENT_CHANGED = "claim_statement_changed"
    AUDIT_CLAIM_SPEC_CHANGED = "audit_claim_spec_changed"
    EVIDENCE_PROFILE_CHANGED = "evidence_profile_changed"
    EVIDENCE_OBLIGATIONS_CHANGED = "evidence_obligations_changed"
    MEASUREMENT_SEMANTIC_PROTOCOL_CHANGED = (
        "measurement_semantic_protocol_changed"
    )
    MEASUREMENT_SOURCE_SNAPSHOT_CHANGED = "measurement_source_snapshot_changed"
    AUDIT_SEMANTICS_COMPATIBILITY_CHANGED = (
        "audit_semantics_compatibility_changed"
    )
    SOURCE_BYTES_CHANGED = "source_bytes_changed"
    EXTRACTION_IDENTITY_CHANGED = "extraction_identity_changed"
    ARTIFACT_AUDIT_SEMANTICS_CHANGED = "artifact_audit_semantics_changed"
    SELECTOR_IDENTITY_CHANGED = "selector_identity_changed"
    ADAPTER_SEMANTICS_CHANGED = "adapter_semantics_changed"
    EVIDENCE_PROJECTOR_CHANGED = "evidence_projector_changed"
    MAPPING_BINDING_CHANGED = "mapping_binding_changed"
    MAPPING_PROVENANCE_CHANGED = "mapping_provenance_changed"
    ARTIFACT_RELOCATED = "artifact_relocated"
    RELOCATION_AMBIGUOUS = "relocation_ambiguous"
    RELOCATION_NOT_FOUND = "relocation_not_found"
    HEAD_ONLY_CHANGE = "head_only_change"


@dataclass(frozen=True, slots=True, init=False)
class ArtifactRelocation:
    previous_evidence_id: str
    current_evidence_id: str
    previous_path: RepositoryPath
    current_path: RepositoryPath
    kind: ArtifactKind
    role: ExperimentRole
    dataset_split: DatasetSplit | None

    def __init__(self) -> None:
        raise TypeError("ArtifactRelocation is created only by stale-evidence comparison")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ArtifactRelocation is final")


@dataclass(frozen=True, slots=True, init=False)
class StaleEvidenceAssessment:
    state: StalenessState
    decision: ReverificationDecision
    required_action: RequiredAction
    reason_codes: tuple[StalenessReason, ...]
    relocations: tuple[ArtifactRelocation, ...]
    previous_input_snapshot_sha256: Sha256Digest | None
    current_input_snapshot_sha256: Sha256Digest
    comparison_sha256: Sha256Digest
    version: str

    scheduled_audit_continues: ClassVar[bool] = True
    authorizes_audit_skip: ClassVar[bool] = False

    def __init__(self) -> None:
        raise TypeError("StaleEvidenceAssessment must be created by comparison")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StaleEvidenceAssessment is final")


def _assessment(
    *,
    state: StalenessState,
    decision: ReverificationDecision,
    required_action: RequiredAction,
    reasons: set[StalenessReason],
    relocations: tuple[ArtifactRelocation, ...],
    previous_snapshot: VerificationInputSnapshot | None,
    current_snapshot: VerificationInputSnapshot,
) -> StaleEvidenceAssessment:
    ordered_reasons = tuple(sorted(reasons, key=lambda item: item.value))
    material = {
        "version": STALE_EVIDENCE_VERSION,
        "state": state.value,
        "decision": decision.value,
        "required_action": required_action.value,
        "reason_codes": tuple(item.value for item in ordered_reasons),
        "relocations": tuple(
            {
                "previous_evidence_id": item.previous_evidence_id,
                "current_evidence_id": item.current_evidence_id,
                "previous_path": str(item.previous_path),
                "current_path": str(item.current_path),
                "kind": item.kind.value,
                "role": item.role.value,
                "dataset_split": (
                    None if item.dataset_split is None else item.dataset_split.value
                ),
            }
            for item in relocations
        ),
        "previous_input_snapshot_sha256": (
            None
            if previous_snapshot is None
            else str(previous_snapshot.input_snapshot_sha256)
        ),
        "current_input_snapshot_sha256": str(
            current_snapshot.input_snapshot_sha256
        ),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    instance = object.__new__(StaleEvidenceAssessment)
    object.__setattr__(instance, "state", state)
    object.__setattr__(instance, "decision", decision)
    object.__setattr__(instance, "required_action", required_action)
    object.__setattr__(instance, "reason_codes", ordered_reasons)
    object.__setattr__(instance, "relocations", relocations)
    object.__setattr__(
        instance,
        "previous_input_snapshot_sha256",
        None if previous_snapshot is None else previous_snapshot.input_snapshot_sha256,
    )
    object.__setattr__(
        instance,
        "current_input_snapshot_sha256",
        current_snapshot.input_snapshot_sha256,
    )
    object.__setattr__(
        instance,
        "comparison_sha256",
        Sha256Digest(hashlib.sha256(b"claimci.stale-evidence.v1\0" + encoded).hexdigest()),
    )
    object.__setattr__(instance, "version", STALE_EVIDENCE_VERSION)
    return instance


def _indeterminate(
    *,
    reason: StalenessReason,
    previous_snapshot: VerificationInputSnapshot,
    current_snapshot: VerificationInputSnapshot,
) -> StaleEvidenceAssessment:
    return _assessment(
        state=StalenessState.INDETERMINATE,
        decision=ReverificationDecision.UNDETERMINED,
        required_action=RequiredAction.SUPPLY_PROVENANCE,
        reasons={reason},
        relocations=(),
        previous_snapshot=previous_snapshot,
        current_snapshot=current_snapshot,
    )


def _slot(
    artifact: VerificationArtifactIdentity,
) -> tuple[RepositoryPath, ArtifactKind, ExperimentRole, DatasetSplit | None]:
    return artifact.path, artifact.kind, artifact.role, artifact.dataset_split


def _relocation_key(
    artifact: VerificationArtifactIdentity,
) -> tuple[ArtifactKind, ExperimentRole, DatasetSplit | None, Sha256Digest]:
    return (
        artifact.kind,
        artifact.role,
        artifact.dataset_split,
        artifact.audit_semantics_sha256,
    )


def _relocation(
    previous: VerificationArtifactIdentity,
    current: VerificationArtifactIdentity,
) -> ArtifactRelocation:
    instance = object.__new__(ArtifactRelocation)
    object.__setattr__(instance, "previous_evidence_id", previous.verification_evidence_id)
    object.__setattr__(instance, "current_evidence_id", current.verification_evidence_id)
    object.__setattr__(instance, "previous_path", previous.path)
    object.__setattr__(instance, "current_path", current.path)
    object.__setattr__(instance, "kind", previous.kind)
    object.__setattr__(instance, "role", previous.role)
    object.__setattr__(instance, "dataset_split", previous.dataset_split)
    return instance


def _compare_artifact_pair(
    previous: VerificationArtifactIdentity,
    current: VerificationArtifactIdentity,
    *,
    material_reasons: set[StalenessReason],
    indeterminate_reasons: set[StalenessReason],
) -> bool:
    """Compare one paired binding and return whether exact bytes changed."""

    if (
        previous.kind is not current.kind
        or previous.role is not current.role
        or previous.dataset_split is not current.dataset_split
    ):
        material_reasons.add(StalenessReason.MAPPING_BINDING_CHANGED)
    if previous.selectors != current.selectors:
        material_reasons.add(StalenessReason.SELECTOR_IDENTITY_CHANGED)
    if previous.audit_semantics_sha256 != current.audit_semantics_sha256:
        material_reasons.add(StalenessReason.ARTIFACT_AUDIT_SEMANTICS_CHANGED)
    if (
        previous.adapter_id != current.adapter_id
        or previous.adapter_semantic_version != current.adapter_semantic_version
    ):
        if (
            previous.adapter_semantic_version is None
            or current.adapter_semantic_version is None
        ):
            indeterminate_reasons.add(
                StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE
            )
        else:
            material_reasons.add(StalenessReason.ADAPTER_SEMANTICS_CHANGED)
    if previous.evidence_projector_version != current.evidence_projector_version:
        material_reasons.add(StalenessReason.EVIDENCE_PROJECTOR_CHANGED)
    if previous.extraction_sha256 != current.extraction_sha256:
        indeterminate_reasons.add(StalenessReason.EXTRACTION_IDENTITY_CHANGED)
    return previous.source_sha256 != current.source_sha256


def assess_stale_evidence(
    previous_recipe: ReplayRecipe | None,
    current_snapshot: VerificationInputSnapshot,
    *,
    issued_candidates: tuple[VerificationArtifactIdentity, ...] = (),
) -> StaleEvidenceAssessment:
    """Compare represented inputs without authorizing an Audit skip.

    ``NO_REVERIFY_TRIGGER_DETECTED`` is bounded to the represented, versioned
    captured scope. It does not prove global experiment stability or
    reproducibility, and an already scheduled Audit always continues. Optional
    relocation candidates must come from trusted passive issuance for the
    current head; provider proposals cannot construct the accepted identity
    type.
    """

    if previous_recipe is not None and type(previous_recipe) is not ReplayRecipe:
        raise TypeError("stale evidence requires an exact ReplayRecipe")
    if type(current_snapshot) is not VerificationInputSnapshot:
        raise TypeError("stale evidence requires an exact VerificationInputSnapshot")
    if (
        not isinstance(issued_candidates, tuple)
        or len(issued_candidates) > 32
        or not all(type(item) is VerificationArtifactIdentity for item in issued_candidates)
        or len({item.verification_evidence_id for item in issued_candidates})
        != len(issued_candidates)
    ):
        raise TypeError("issued relocation candidates must be unique exact identities")
    if previous_recipe is None:
        return _assessment(
            state=StalenessState.LEGACY_UNAVAILABLE,
            decision=ReverificationDecision.UNDETERMINED,
            required_action=RequiredAction.SUPPLY_PROVENANCE,
            reasons={StalenessReason.LEGACY_REPLAY_UNAVAILABLE},
            relocations=(),
            previous_snapshot=None,
            current_snapshot=current_snapshot,
        )
    previous = previous_recipe.input_snapshot
    if type(previous) is not VerificationInputSnapshot:
        raise TypeError("Replay recipe contains an invalid verification snapshot")
    if (
        previous.capability is VerificationSnapshotCapability.UNAVAILABLE
        or current_snapshot.capability is VerificationSnapshotCapability.UNAVAILABLE
    ):
        return _indeterminate(
            reason=StalenessReason.SNAPSHOT_UNAVAILABLE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if previous.repository != current_snapshot.repository:
        return _indeterminate(
            reason=StalenessReason.REPOSITORY_IDENTITY_CHANGED,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if (
        previous.capability is not VerificationSnapshotCapability.COMPLETE
        or current_snapshot.capability is not VerificationSnapshotCapability.COMPLETE
    ):
        return _indeterminate(
            reason=StalenessReason.SNAPSHOT_DETAILS_UNAVAILABLE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    required_roots = (
        previous.claim_identity,
        current_snapshot.claim_identity,
        previous.mapping_identity,
        current_snapshot.mapping_identity,
        previous.profile_identity,
        current_snapshot.profile_identity,
        previous.obligation_identity,
        current_snapshot.obligation_identity,
        previous.baseline_measurement,
        current_snapshot.baseline_measurement,
        previous.candidate_measurement,
        current_snapshot.candidate_measurement,
        previous.audit_compatibility,
        current_snapshot.audit_compatibility,
        previous.captured_audit_input_sha256,
        current_snapshot.captured_audit_input_sha256,
        previous.comparison_frame_sha256,
        current_snapshot.comparison_frame_sha256,
    )
    if any(item is None for item in required_roots):
        return _indeterminate(
            reason=StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if not previous.profile_identity.complete or not current_snapshot.profile_identity.complete:
        return _indeterminate(
            reason=StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if not previous.artifacts or not current_snapshot.artifacts:
        return _indeterminate(
            reason=StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if any(
        item.adapter_semantic_version is None
        for item in (*previous.artifacts, *current_snapshot.artifacts)
    ):
        return _indeterminate(
            reason=StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if previous.input_snapshot_sha256 == current_snapshot.input_snapshot_sha256:
        return _assessment(
            state=StalenessState.FRESH,
            decision=ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED,
            required_action=RequiredAction.NONE,
            reasons=set(),
            relocations=(),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )

    reasons: set[StalenessReason] = set()
    material = False
    if (
        previous.claim_identity.canonical_claim_sha256
        != current_snapshot.claim_identity.canonical_claim_sha256
    ):
        reasons.add(StalenessReason.CLAIM_STATEMENT_CHANGED)
        material = True
    if (
        previous.claim_identity.claim_id != current_snapshot.claim_identity.claim_id
        or previous.claim_identity.audit_spec_sha256
        != current_snapshot.claim_identity.audit_spec_sha256
    ):
        reasons.add(StalenessReason.AUDIT_CLAIM_SPEC_CHANGED)
        material = True
    if (
        previous.profile_identity.profile_semantics_sha256
        != current_snapshot.profile_identity.profile_semantics_sha256
    ):
        reasons.add(StalenessReason.EVIDENCE_PROFILE_CHANGED)
        material = True
    if (
        previous.obligation_identity.obligation_semantics_sha256
        != current_snapshot.obligation_identity.obligation_semantics_sha256
        or previous.obligation_identity.policy_id
        != current_snapshot.obligation_identity.policy_id
    ):
        reasons.add(StalenessReason.EVIDENCE_OBLIGATIONS_CHANGED)
        material = True
    measurement_source_changed = False
    for before, after in (
        (previous.baseline_measurement, current_snapshot.baseline_measurement),
        (previous.candidate_measurement, current_snapshot.candidate_measurement),
    ):
        if before.semantic_protocol_id != after.semantic_protocol_id:
            reasons.add(StalenessReason.MEASUREMENT_SEMANTIC_PROTOCOL_CHANGED)
            material = True
        if before.source_snapshot_id != after.source_snapshot_id:
            measurement_source_changed = True
    if (
        previous.audit_compatibility.audit_semantics_sha256
        != current_snapshot.audit_compatibility.audit_semantics_sha256
    ):
        reasons.add(StalenessReason.AUDIT_SEMANTICS_COMPATIBILITY_CHANGED)
        material = True

    material_reasons: set[StalenessReason] = set(reasons if material else ())
    indeterminate_reasons: set[StalenessReason] = set()
    source_changed = False
    relocations: list[ArtifactRelocation] = []
    ambiguous_relocation = False

    previous_slots = {_slot(item): item for item in previous.artifacts}
    current_slots = {_slot(item): item for item in current_snapshot.artifacts}
    if (
        len(previous_slots) != len(previous.artifacts)
        or len(current_slots) != len(current_snapshot.artifacts)
    ):
        indeterminate_reasons.add(
            StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE
        )

    candidate_pool: dict[str, VerificationArtifactIdentity] = {
        item.verification_evidence_id: item
        for item in (*current_snapshot.artifacts, *issued_candidates)
    }
    consumed_current_ids: set[str] = set()
    for previous_slot, before in previous_slots.items():
        after = current_slots.get(previous_slot)
        if after is not None:
            consumed_current_ids.add(after.verification_evidence_id)
            source_changed = (
                _compare_artifact_pair(
                    before,
                    after,
                    material_reasons=material_reasons,
                    indeterminate_reasons=indeterminate_reasons,
                )
                or source_changed
            )
            continue
        matches = tuple(
            item
            for item in candidate_pool.values()
            if item.path != before.path and _relocation_key(item) == _relocation_key(before)
        )
        if len(matches) == 1:
            after = matches[0]
            consumed_current_ids.add(after.verification_evidence_id)
            source_changed = (
                _compare_artifact_pair(
                    before,
                    after,
                    material_reasons=material_reasons,
                    indeterminate_reasons=indeterminate_reasons,
                )
                or source_changed
            )
            relocations.append(_relocation(before, after))
            reasons.add(StalenessReason.ARTIFACT_RELOCATED)
        elif len(matches) > 1:
            ambiguous_relocation = True
            reasons.add(StalenessReason.RELOCATION_AMBIGUOUS)
        else:
            alternatives = tuple(
                item
                for item in candidate_pool.values()
                if item.kind is before.kind
                and item.role is before.role
                and item.dataset_split is before.dataset_split
            )
            reasons.add(StalenessReason.RELOCATION_NOT_FOUND)
            if alternatives:
                material_reasons.add(StalenessReason.RELOCATION_NOT_FOUND)
            else:
                indeterminate_reasons.add(StalenessReason.RELOCATION_NOT_FOUND)

    unmatched_current = tuple(
        item
        for item in current_snapshot.artifacts
        if item.verification_evidence_id not in consumed_current_ids
    )
    if unmatched_current and not ambiguous_relocation:
        material_reasons.add(StalenessReason.MAPPING_BINDING_CHANGED)
    if source_changed:
        reasons.add(StalenessReason.SOURCE_BYTES_CHANGED)
    if measurement_source_changed:
        reasons.add(StalenessReason.MEASUREMENT_SOURCE_SNAPSHOT_CHANGED)
    # ``bindings_sha256`` commits full evidence identities, so a direct
    # inequality would collapse harmless byte changes and relocations into a
    # generic mapping change. Binding semantics are compared field-by-field
    # above; these fields record only the mapping trust transition/provenance.
    if (
        previous.mapping_identity.source_mapping_id
        != current_snapshot.mapping_identity.source_mapping_id
        or previous.mapping_identity.trust is not current_snapshot.mapping_identity.trust
        or previous.mapping_identity.approval_sha256
        != current_snapshot.mapping_identity.approval_sha256
    ):
        reasons.add(StalenessReason.MAPPING_PROVENANCE_CHANGED)

    reasons.update(material_reasons)
    reasons.update(indeterminate_reasons)
    if material_reasons:
        return _assessment(
            state=StalenessState.MATERIAL_CHANGE,
            decision=ReverificationDecision.REVERIFY_REQUIRED,
            required_action=RequiredAction.REVERIFY,
            reasons=reasons,
            relocations=tuple(relocations),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if ambiguous_relocation:
        return _assessment(
            state=StalenessState.INDETERMINATE,
            decision=ReverificationDecision.UNDETERMINED,
            required_action=RequiredAction.REMAP,
            reasons=reasons,
            relocations=(),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if indeterminate_reasons:
        return _assessment(
            state=StalenessState.INDETERMINATE,
            decision=ReverificationDecision.UNDETERMINED,
            required_action=RequiredAction.SUPPLY_PROVENANCE,
            reasons=reasons,
            relocations=tuple(relocations),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if (
        previous.captured_audit_input_sha256
        != current_snapshot.captured_audit_input_sha256
        or previous.comparison_frame_sha256
        != current_snapshot.comparison_frame_sha256
    ):
        return _indeterminate(
            reason=StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE,
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if relocations:
        return _assessment(
            state=StalenessState.IDENTITY_RELOCATION,
            decision=ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED,
            required_action=RequiredAction.REMAP,
            reasons=reasons,
            relocations=tuple(relocations),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if previous.head_sha != current_snapshot.head_sha:
        if not source_changed and not measurement_source_changed and not reasons:
            reasons.add(StalenessReason.HEAD_ONLY_CHANGE)
        return _assessment(
            state=(
                StalenessState.BYTE_ONLY_CHANGE
                if source_changed or measurement_source_changed
                else StalenessState.FRESH
            ),
            decision=ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED,
            required_action=RequiredAction.NONE,
            reasons=reasons,
            relocations=(),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    if source_changed or measurement_source_changed:
        return _assessment(
            state=StalenessState.BYTE_ONLY_CHANGE,
            decision=ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED,
            required_action=RequiredAction.NONE,
            reasons=reasons,
            relocations=(),
            previous_snapshot=previous,
            current_snapshot=current_snapshot,
        )
    return _assessment(
        state=StalenessState.FRESH,
        decision=ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED,
        required_action=RequiredAction.NONE,
        reasons=reasons,
        relocations=(),
        previous_snapshot=previous,
        current_snapshot=current_snapshot,
    )


__all__ = [
    "ArtifactRelocation",
    "RequiredAction",
    "ReverificationDecision",
    "STALE_EVIDENCE_VERSION",
    "StaleEvidenceAssessment",
    "StalenessReason",
    "StalenessState",
    "assess_stale_evidence",
]
