"""Non-executable post-Audit replay commitments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

from claimci import __version__
from claimci.models import AuditResult, Verdict

from .contracts import AnalysisAuthority, GitCommitSha, Sha256Digest, to_jsonable
from .trace import (
    DeterministicAuditTrace,
    EvidenceTraceBundle,
    TraceCompleteness,
    trace_json_bytes,
)
from .verification import VerificationInputSnapshot, VerificationSnapshotCapability


REPLAY_RECIPE_VERSION = "claimci.replay-recipe.v1"


def _digest(domain: str, value: object) -> Sha256Digest:
    material = domain.encode("ascii") + b"\0" + json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(material).hexdigest())


@dataclass(frozen=True, slots=True, init=False)
class ReplayEngineProvenance:
    package_version: str
    source_revision: GitCommitSha | None
    distribution_sha256: Sha256Digest | None

    def __init__(self) -> None:
        raise TypeError("ReplayEngineProvenance must be created through current")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReplayEngineProvenance is final")

    @classmethod
    def current(
        cls,
        *,
        source_revision: str | GitCommitSha | None = None,
        distribution_sha256: str | Sha256Digest | None = None,
    ) -> "ReplayEngineProvenance":
        revision = None if source_revision is None else GitCommitSha(source_revision)
        distribution = (
            None
            if distribution_sha256 is None
            else Sha256Digest(distribution_sha256)
        )
        instance = object.__new__(ReplayEngineProvenance)
        object.__setattr__(instance, "package_version", __version__)
        object.__setattr__(instance, "source_revision", revision)
        object.__setattr__(instance, "distribution_sha256", distribution)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class ReplayAuditCommitment:
    verdict: Verdict
    stable_audit_sha256: Sha256Digest
    applied_rule_ids: tuple[str, ...]
    authority: AnalysisAuthority

    def __init__(self) -> None:
        raise TypeError("ReplayAuditCommitment must be created through from_audit_result")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReplayAuditCommitment is final")

    @classmethod
    def from_audit_result(cls, result: AuditResult) -> "ReplayAuditCommitment":
        if type(result) is not AuditResult:
            raise TypeError("Replay Audit commitment requires an actual AuditResult")
        if type(result.verdict) is not Verdict:
            raise TypeError("Replay Audit commitment requires an actual Verdict")
        trace = DeterministicAuditTrace.from_audit_result(result)
        instance = object.__new__(ReplayAuditCommitment)
        object.__setattr__(instance, "verdict", trace.verdict)
        object.__setattr__(instance, "stable_audit_sha256", trace.audit_sha256)
        object.__setattr__(instance, "applied_rule_ids", trace.rule_ids)
        object.__setattr__(instance, "authority", trace.authority)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class ReplayTraceReference:
    trace_sha256: Sha256Digest
    trace_version: str
    completeness: TraceCompleteness
    head_sha: GitCommitSha
    stable_audit_sha256: Sha256Digest
    omitted_entry_count: int
    omitted_entries_sha256: Sha256Digest | None
    reason_code: str | None

    def __init__(self) -> None:
        raise TypeError("ReplayTraceReference must be created from EvidenceTraceBundle")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReplayTraceReference is final")

    @classmethod
    def from_trace(
        cls,
        trace: EvidenceTraceBundle,
        commitment: ReplayAuditCommitment,
    ) -> "ReplayTraceReference":
        if type(trace) is not EvidenceTraceBundle:
            raise TypeError("Replay trace reference requires EvidenceTraceBundle")
        if type(commitment) is not ReplayAuditCommitment:
            raise TypeError("Replay trace reference requires ReplayAuditCommitment")
        authority = trace.deterministic_authority
        if (
            authority.verdict is not commitment.verdict
            or authority.audit_sha256 != commitment.stable_audit_sha256
            or authority.rule_ids != commitment.applied_rule_ids
        ):
            raise ValueError("Replay trace authority does not match its Audit commitment")
        encoded = trace_json_bytes(trace)
        instance = object.__new__(ReplayTraceReference)
        object.__setattr__(
            instance,
            "trace_sha256",
            Sha256Digest(hashlib.sha256(encoded).hexdigest()),
        )
        object.__setattr__(instance, "trace_version", trace.version)
        object.__setattr__(instance, "completeness", trace.completeness)
        object.__setattr__(instance, "head_sha", trace.head_sha)
        object.__setattr__(instance, "stable_audit_sha256", authority.audit_sha256)
        object.__setattr__(instance, "omitted_entry_count", trace.omitted_entry_count)
        object.__setattr__(instance, "omitted_entries_sha256", trace.omitted_entries_sha256)
        object.__setattr__(instance, "reason_code", trace.reason_code)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class ReplayRecipe:
    """A non-executable companion to one exact deterministic Audit execution."""

    input_snapshot: VerificationInputSnapshot
    engine_provenance: ReplayEngineProvenance
    audit_commitment: ReplayAuditCommitment
    trace_reference: ReplayTraceReference
    recipe_sha256: Sha256Digest
    version: str

    experiment_replay_supported: ClassVar[bool] = False

    def __init__(self) -> None:
        raise TypeError("ReplayRecipe must be created through from_execution")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReplayRecipe is final")

    @classmethod
    def from_execution(
        cls,
        *,
        input_snapshot: VerificationInputSnapshot,
        engine_provenance: ReplayEngineProvenance,
        audit_commitment: ReplayAuditCommitment,
        trace_reference: ReplayTraceReference,
    ) -> "ReplayRecipe":
        if type(input_snapshot) is not VerificationInputSnapshot:
            raise TypeError("Replay recipe requires VerificationInputSnapshot")
        if input_snapshot.capability is VerificationSnapshotCapability.UNAVAILABLE:
            raise ValueError("unavailable verification snapshot cannot create a Replay recipe")
        if type(engine_provenance) is not ReplayEngineProvenance:
            raise TypeError("Replay recipe requires ReplayEngineProvenance")
        if type(audit_commitment) is not ReplayAuditCommitment:
            raise TypeError("Replay recipe requires ReplayAuditCommitment")
        if type(trace_reference) is not ReplayTraceReference:
            raise TypeError("Replay recipe requires ReplayTraceReference")
        if trace_reference.head_sha != input_snapshot.head_sha:
            raise ValueError("Replay trace head does not match the verification snapshot")
        if trace_reference.stable_audit_sha256 != audit_commitment.stable_audit_sha256:
            raise ValueError("Replay trace does not match the Audit commitment")
        material = {
            "version": REPLAY_RECIPE_VERSION,
            "input_snapshot_sha256": str(input_snapshot.input_snapshot_sha256),
            "engine_provenance": {
                "package_version": engine_provenance.package_version,
                "source_revision": None
                if engine_provenance.source_revision is None
                else str(engine_provenance.source_revision),
                "distribution_sha256": None
                if engine_provenance.distribution_sha256 is None
                else str(engine_provenance.distribution_sha256),
            },
            "audit_commitment": {
                "verdict": audit_commitment.verdict.value,
                "stable_audit_sha256": str(audit_commitment.stable_audit_sha256),
                "applied_rule_ids": audit_commitment.applied_rule_ids,
                "authority": audit_commitment.authority.value,
            },
            "trace_reference": {
                "trace_sha256": str(trace_reference.trace_sha256),
                "trace_version": trace_reference.trace_version,
                "completeness": trace_reference.completeness.value,
                "head_sha": str(trace_reference.head_sha),
                "stable_audit_sha256": str(trace_reference.stable_audit_sha256),
                "omitted_entry_count": trace_reference.omitted_entry_count,
                "omitted_entries_sha256": None
                if trace_reference.omitted_entries_sha256 is None
                else str(trace_reference.omitted_entries_sha256),
                "reason_code": trace_reference.reason_code,
            },
            "experiment_replay_supported": False,
        }
        instance = object.__new__(ReplayRecipe)
        object.__setattr__(instance, "input_snapshot", input_snapshot)
        object.__setattr__(instance, "engine_provenance", engine_provenance)
        object.__setattr__(instance, "audit_commitment", audit_commitment)
        object.__setattr__(instance, "trace_reference", trace_reference)
        object.__setattr__(
            instance,
            "recipe_sha256",
            _digest("claimci.replay-recipe.v1", material),
        )
        object.__setattr__(instance, "version", REPLAY_RECIPE_VERSION)
        return instance


def replay_recipe_json_bytes(recipe: ReplayRecipe) -> bytes:
    """Serialize one recipe companion without defining a transport allocation."""

    if type(recipe) is not ReplayRecipe:
        raise TypeError("Replay serialization requires ReplayRecipe")
    return json.dumps(
        to_jsonable(recipe),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "REPLAY_RECIPE_VERSION",
    "ReplayAuditCommitment",
    "ReplayEngineProvenance",
    "ReplayRecipe",
    "ReplayTraceReference",
    "replay_recipe_json_bytes",
]
