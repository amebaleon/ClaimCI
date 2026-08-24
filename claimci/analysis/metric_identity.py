"""Exact, bounded metric identities for passive result artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

from .artifact_source import (
    ArtifactSource,
    ScanCompleteness,
    ScanReason,
    ScanState,
    StreamingLimits,
)
from .contracts import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    EvidenceSelector,
    ExperimentRole,
    FieldProvenance,
    FieldMapping,
    GitCommitSha,
    MappingCandidate,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    RepositoryIdentity,
    RepoMapping,
    SelectorKind,
    Sha256Digest,
    TableSelector,
    NormalizedEvidence,
    selector_identity,
)


_ADAPTER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_CANDIDATE_ID = re.compile(r"metric-candidate-[0-9a-f]{24}\Z")
_IDENTITY_ID = re.compile(r"metric-identity-[0-9a-f]{24}\Z")
_CANONICAL_METRIC = re.compile(r"[a-z][a-z0-9_.-]{0,255}\Z")
_SUPPORTED_KINDS = frozenset({ArtifactKind.RESULTS, ArtifactKind.BENCHMARK})
_LEXICAL_ALIASES = MappingProxyType({"accuracy": frozenset({"acc"})})


class MetricCandidateLimitError(AnalysisContractError):
    """Raised when one passive artifact exceeds the metric-candidate bound."""


@dataclass(frozen=True, slots=True)
class MetricCandidateLimits:
    max_candidates_per_artifact: int = 32

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_candidates_per_artifact, bool)
            or not isinstance(self.max_candidates_per_artifact, int)
            or not 1 <= self.max_candidates_per_artifact <= 64
        ):
            raise AnalysisContractError(
                "max_candidates_per_artifact must be between one and 64"
            )


@dataclass(frozen=True, slots=True)
class MetricCandidateScan:
    """Candidate set issued only by one complete verified streaming schema."""

    candidates: tuple[MetricCandidate, ...]
    match: AdapterMatch | None
    completeness: ScanCompleteness
    retained_raw_records: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not all(
            type(item) is MetricCandidate for item in self.candidates
        ):
            raise TypeError("streaming metric candidates are invalid")
        if type(self.completeness) is not ScanCompleteness:
            raise TypeError("streaming metric completeness is invalid")
        if self.retained_raw_records != 0:
            raise AnalysisContractError(
                "streaming metric discovery cannot retain raw records"
            )
        if self.completeness.complete:
            if self.match is None or not self.completeness.integrity_verified:
                raise AnalysisContractError(
                    "complete streaming metrics require a verified adapter match"
                )
        elif self.candidates or self.match is not None:
            raise AnalysisContractError(
                "incomplete streaming metrics cannot expose provisional candidates"
            )


def _candidate_id(
    *,
    role: ExperimentRole,
    path: RepositoryPath,
    sha256: Sha256Digest,
    adapter_id: str,
    selector: EvidenceSelector | TableSelector,
    raw_metric_name: str,
) -> str:
    material = json.dumps(
        {
            "role": role.value,
            "path": str(path),
            "sha256": str(sha256),
            "adapter_id": adapter_id,
            "selector": selector_identity(selector),
            "raw_metric_name": raw_metric_name,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "metric-candidate-" + hashlib.sha256(material).hexdigest()[:24]


@dataclass(frozen=True, slots=True, init=False)
class MetricCandidate:
    """One fixed-adapter metric selector bound to exact passive bytes."""

    candidate_id: str
    experiment_role: ExperimentRole
    artifact_path: RepositoryPath
    artifact_sha256: Sha256Digest
    adapter_id: str
    selector: EvidenceSelector | TableSelector
    raw_metric_name: str
    provenance: FieldProvenance

    def __init__(self) -> None:
        raise TypeError("MetricCandidate values are created by fixed adapter extraction")

    @classmethod
    def _from_extraction(
        cls,
        *,
        role: ExperimentRole,
        path: RepositoryPath,
        sha256: Sha256Digest,
        adapter_id: str,
        selector: EvidenceSelector | TableSelector,
        raw_metric_name: str,
        provenance: FieldProvenance,
    ) -> MetricCandidate:
        if role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
            raise AnalysisContractError(
                "metric candidate role must be baseline or candidate"
            )
        if not isinstance(path, RepositoryPath):
            path = RepositoryPath(path)
        if not isinstance(sha256, Sha256Digest):
            sha256 = Sha256Digest(sha256)
        if not isinstance(adapter_id, str) or not _ADAPTER_ID.fullmatch(adapter_id):
            raise AnalysisContractError("metric candidate adapter ID is invalid")
        if type(selector) not in {EvidenceSelector, TableSelector}:
            raise TypeError("metric candidate selector is invalid")
        if (
            not isinstance(raw_metric_name, str)
            or not raw_metric_name
            or raw_metric_name != raw_metric_name.strip()
            or len(raw_metric_name) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_metric_name)
        ):
            raise AnalysisContractError("metric candidate raw name is invalid")
        if not isinstance(provenance, FieldProvenance):
            raise TypeError("metric candidate provenance is invalid")
        if provenance.kind is not ProvenanceKind.ADAPTER_EXTRACTION:
            raise AnalysisContractError(
                "metric candidates require fixed adapter extraction provenance"
            )
        candidate_id = _candidate_id(
            role=role,
            path=path,
            sha256=sha256,
            adapter_id=adapter_id,
            selector=selector,
            raw_metric_name=raw_metric_name,
        )
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise AssertionError("derived metric candidate ID is not canonical")
        instance = object.__new__(MetricCandidate)
        object.__setattr__(instance, "candidate_id", candidate_id)
        object.__setattr__(instance, "experiment_role", role)
        object.__setattr__(instance, "artifact_path", path)
        object.__setattr__(instance, "artifact_sha256", sha256)
        object.__setattr__(instance, "adapter_id", adapter_id)
        object.__setattr__(instance, "selector", selector)
        object.__setattr__(instance, "raw_metric_name", raw_metric_name)
        object.__setattr__(instance, "provenance", provenance)
        return instance


class MetricIdentityKind(str, Enum):
    EXACT = "exact"
    LEXICAL_ALIAS = "lexical_alias"


@dataclass(frozen=True, slots=True, init=False)
class MetricIdentity:
    """One non-authoritative lexical comparison for one exact candidate."""

    identity_id: str
    candidate: MetricCandidate
    canonical_metric: str
    kind: MetricIdentityKind

    def __init__(self) -> None:
        raise TypeError("MetricIdentity values are created by the identify factory")

    @classmethod
    def _identified(
        cls,
        candidate: MetricCandidate,
        canonical_metric: str,
        kind: MetricIdentityKind,
    ) -> MetricIdentity:
        material = json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "canonical_metric": canonical_metric,
                "kind": kind.value,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        instance = object.__new__(MetricIdentity)
        object.__setattr__(
            instance,
            "identity_id",
            "metric-identity-" + hashlib.sha256(material).hexdigest()[:24],
        )
        object.__setattr__(instance, "candidate", candidate)
        object.__setattr__(instance, "canonical_metric", canonical_metric)
        object.__setattr__(instance, "kind", kind)
        if not _IDENTITY_ID.fullmatch(instance.identity_id):
            raise AssertionError("derived metric identity ID is not canonical")
        return instance


def identify_metric_candidate(
    candidate: MetricCandidate,
    *,
    canonical_metric: str,
) -> MetricIdentity | None:
    """Recognize only an exact name or one fixed conservative lexical alias."""

    if type(candidate) is not MetricCandidate:
        raise TypeError("metric identity requires an extracted MetricCandidate")
    if (
        not isinstance(canonical_metric, str)
        or not _CANONICAL_METRIC.fullmatch(canonical_metric)
    ):
        raise AnalysisContractError("canonical metric name is invalid")
    raw = candidate.raw_metric_name.casefold()
    if raw == canonical_metric:
        kind = MetricIdentityKind.EXACT
    elif raw in _LEXICAL_ALIASES.get(canonical_metric, frozenset()):
        kind = MetricIdentityKind.LEXICAL_ALIAS
    else:
        return None
    return MetricIdentity._identified(candidate, canonical_metric, kind)


class MetricBindingAuthority(str, Enum):
    DETERMINISTIC_EXACT = "deterministic_exact"
    USER_APPROVED = "user_approved"


class MetricBindingState(str, Enum):
    BOUND = "bound"
    APPROVAL_REQUIRED = "approval_required"
    UNRESOLVED = "unresolved"


def _bounded_identifier(value: object, label: str, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AnalysisContractError(f"{label} is invalid")
    return value


def _pair_material(
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    canonical_metric: str,
    baseline_identity: MetricIdentity,
    candidate_identity: MetricIdentity,
) -> dict[str, object]:
    return {
        "repository": repository.full_name,
        "head_sha": str(head_sha),
        "canonical_metric": canonical_metric,
        "baseline_identity_id": baseline_identity.identity_id,
        "candidate_identity_id": candidate_identity.identity_id,
    }


def _digest_id(prefix: str, material: object) -> str:
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(canonical).hexdigest()[:24]


def _validate_pair(
    baseline_identity: MetricIdentity,
    candidate_identity: MetricIdentity,
    canonical_metric: str,
) -> None:
    if type(baseline_identity) is not MetricIdentity or type(
        candidate_identity
    ) is not MetricIdentity:
        raise TypeError("metric binding pair requires MetricIdentity values")
    if (
        baseline_identity.candidate.experiment_role is not ExperimentRole.BASELINE
        or candidate_identity.candidate.experiment_role
        is not ExperimentRole.CANDIDATE
    ):
        raise AnalysisContractError(
            "metric binding pair requires one baseline and one candidate"
        )
    if (
        baseline_identity.canonical_metric != canonical_metric
        or candidate_identity.canonical_metric != canonical_metric
    ):
        raise AnalysisContractError(
            "metric binding pair identities must share the canonical metric"
        )


@dataclass(frozen=True, slots=True, init=False)
class MetricBindingProposal:
    proposal_id: str
    label: str
    baseline_identity: MetricIdentity
    candidate_identity: MetricIdentity

    def __init__(self) -> None:
        raise TypeError("MetricBindingProposal values are created by resolution")

    @classmethod
    def _from_pair(
        cls,
        baseline_identity: MetricIdentity,
        candidate_identity: MetricIdentity,
    ) -> MetricBindingProposal:
        canonical_metric = baseline_identity.canonical_metric
        _validate_pair(baseline_identity, candidate_identity, canonical_metric)
        baseline_raw = baseline_identity.candidate.raw_metric_name
        candidate_raw = candidate_identity.candidate.raw_metric_name
        label = (
            f"[{baseline_raw}]"
            if baseline_raw == candidate_raw
            else f"[baseline: {baseline_raw}; candidate: {candidate_raw}]"
        )
        material = {
            "canonical_metric": canonical_metric,
            "baseline_identity_id": baseline_identity.identity_id,
            "candidate_identity_id": candidate_identity.identity_id,
        }
        instance = object.__new__(MetricBindingProposal)
        object.__setattr__(instance, "proposal_id", _digest_id("metric-pair-", material))
        object.__setattr__(instance, "label", label)
        object.__setattr__(instance, "baseline_identity", baseline_identity)
        object.__setattr__(instance, "candidate_identity", candidate_identity)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class MetricBindingQuestion:
    question_id: str
    prompt: str
    repository: RepositoryIdentity
    head_sha: GitCommitSha
    canonical_metric: str
    proposals: tuple[MetricBindingProposal, ...]
    relevant_claim_id: str | None

    def __init__(self) -> None:
        raise TypeError("MetricBindingQuestion values are created by resolution")

    @classmethod
    def _from_proposals(
        cls,
        *,
        repository: RepositoryIdentity,
        head_sha: GitCommitSha,
        canonical_metric: str,
        proposals: tuple[MetricBindingProposal, ...],
        relevant_claim_id: str | None,
        stale: bool,
    ) -> MetricBindingQuestion:
        if not 1 <= len(proposals) <= 8:
            raise AnalysisContractError(
                "metric binding question requires one through eight exact pairs"
            )
        if len({item.proposal_id for item in proposals}) != len(proposals):
            raise AnalysisContractError("metric binding proposals must be unique")
        if relevant_claim_id is not None:
            _bounded_identifier(relevant_claim_id, "relevant metric claim ID", 128)
        material = {
            "repository": repository.full_name,
            "head_sha": str(head_sha),
            "canonical_metric": canonical_metric,
            "proposal_ids": tuple(item.proposal_id for item in proposals),
            "relevant_claim_id": relevant_claim_id,
            "stale": stale,
        }
        prompt = (
            f"Re-approve the exact baseline and candidate pair for {canonical_metric}."
            if stale
            else f"Approve the exact baseline and candidate pair for {canonical_metric}."
        )
        instance = object.__new__(MetricBindingQuestion)
        object.__setattr__(instance, "question_id", _digest_id("metric-question-", material))
        object.__setattr__(instance, "prompt", prompt)
        object.__setattr__(instance, "repository", repository)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "canonical_metric", canonical_metric)
        object.__setattr__(instance, "proposals", proposals)
        object.__setattr__(instance, "relevant_claim_id", relevant_claim_id)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class MetricBinding:
    """One indivisible baseline/candidate metric-pair authority."""

    binding_id: str
    repository: RepositoryIdentity
    head_sha: GitCommitSha
    canonical_metric: str
    baseline_identity: MetricIdentity
    candidate_identity: MetricIdentity
    authority: MetricBindingAuthority
    approved_by: str | None
    approval_provenance: FieldProvenance | None

    def __init__(self) -> None:
        raise TypeError("MetricBinding values require exact-pair factory approval")

    @classmethod
    def _from_pair(
        cls,
        *,
        repository: RepositoryIdentity,
        head_sha: GitCommitSha,
        canonical_metric: str,
        baseline_identity: MetricIdentity,
        candidate_identity: MetricIdentity,
        authority: MetricBindingAuthority,
        approved_by: str | None,
    ) -> MetricBinding:
        _validate_pair(baseline_identity, candidate_identity, canonical_metric)
        if authority is MetricBindingAuthority.DETERMINISTIC_EXACT:
            if approved_by is not None or any(
                identity.kind is not MetricIdentityKind.EXACT
                for identity in (baseline_identity, candidate_identity)
            ):
                raise AnalysisContractError(
                    "deterministic metric binding requires an exact/exact pair"
                )
            provenance = None
        elif authority is MetricBindingAuthority.USER_APPROVED:
            approver = _bounded_identifier(approved_by, "metric binding approver")
            approved_by = approver
            provenance = FieldProvenance(
                ProvenanceKind.USER_APPROVED,
                f"exact metric candidate pair explicitly approved by {approver}",
                source_id=approver,
            )
        else:
            raise TypeError("metric binding authority is invalid")
        material = {
            **_pair_material(
                repository=repository,
                head_sha=head_sha,
                canonical_metric=canonical_metric,
                baseline_identity=baseline_identity,
                candidate_identity=candidate_identity,
            ),
            "authority": authority.value,
            "approved_by": approved_by,
        }
        instance = object.__new__(MetricBinding)
        object.__setattr__(instance, "binding_id", _digest_id("metric-binding-", material))
        object.__setattr__(instance, "repository", repository)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "canonical_metric", canonical_metric)
        object.__setattr__(instance, "baseline_identity", baseline_identity)
        object.__setattr__(instance, "candidate_identity", candidate_identity)
        object.__setattr__(instance, "authority", authority)
        object.__setattr__(instance, "approved_by", approved_by)
        object.__setattr__(instance, "approval_provenance", provenance)
        return instance

    @property
    def baseline_candidate(self) -> MetricCandidate:
        return self.baseline_identity.candidate

    @property
    def candidate_candidate(self) -> MetricCandidate:
        return self.candidate_identity.candidate


@dataclass(frozen=True, slots=True, init=False)
class MetricIdentityAuditContext:
    """Trusted pair evidence created only after exact-head revalidation."""

    binding: MetricBinding

    def __init__(self) -> None:
        raise TypeError(
            "MetricIdentityAuditContext requires exact-head materialization"
        )

    @classmethod
    def _from_revalidation(
        cls,
        binding: MetricBinding,
    ) -> MetricIdentityAuditContext:
        if type(binding) is not MetricBinding:
            raise TypeError("metric identity Audit context requires MetricBinding")
        instance = object.__new__(MetricIdentityAuditContext)
        object.__setattr__(instance, "binding", binding)
        return instance


@dataclass(frozen=True, slots=True)
class MetricBindingResolution:
    state: MetricBindingState
    binding: MetricBinding | None = None
    question: MetricBindingQuestion | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, MetricBindingState):
            raise TypeError("metric binding resolution state is invalid")
        if self.state is MetricBindingState.BOUND:
            if self.binding is None or self.question is not None or self.reason is not None:
                raise AnalysisContractError("bound metric resolution requires only a binding")
        elif self.state is MetricBindingState.APPROVAL_REQUIRED:
            if self.binding is not None or self.question is None or self.reason is not None:
                raise AnalysisContractError(
                    "approval-required metric resolution requires only a question"
                )
        elif self.binding is not None or self.question is not None or self.reason is None:
            raise AnalysisContractError(
                "unresolved metric resolution requires only a reason"
            )


def _binding_is_current(
    binding: MetricBinding,
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    canonical_metric: str,
    proposals: tuple[MetricBindingProposal, ...],
) -> bool:
    if (
        binding.repository != repository
        or binding.head_sha != head_sha
        or binding.canonical_metric != canonical_metric
    ):
        return False
    return any(
        proposal.baseline_identity == binding.baseline_identity
        and proposal.candidate_identity == binding.candidate_identity
        for proposal in proposals
    )


def resolve_metric_binding(
    candidates: tuple[MetricCandidate, ...],
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    canonical_metric: str,
    relevant_claim_id: str | None = None,
    approved_binding: MetricBinding | None = None,
) -> MetricBindingResolution:
    """Resolve one exact pair, without turning lexical recognition into authority."""

    if not isinstance(candidates, tuple) or not all(
        type(item) is MetricCandidate for item in candidates
    ):
        raise TypeError("metric binding resolution requires MetricCandidate values")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise AnalysisContractError("metric candidate identifiers must be unique")
    if type(repository) is not RepositoryIdentity:
        raise TypeError("metric binding repository must be RepositoryIdentity")
    if not isinstance(head_sha, GitCommitSha):
        head_sha = GitCommitSha(head_sha)
    if not _CANONICAL_METRIC.fullmatch(canonical_metric):
        raise AnalysisContractError("canonical metric name is invalid")
    if approved_binding is not None and type(approved_binding) is not MetricBinding:
        raise TypeError("approved metric binding must be MetricBinding or null")

    identities = tuple(
        identity
        for candidate in candidates
        if (
            identity := identify_metric_candidate(
                candidate,
                canonical_metric=canonical_metric,
            )
        )
        is not None
    )
    baseline = tuple(
        item
        for item in identities
        if item.candidate.experiment_role is ExperimentRole.BASELINE
    )
    candidate = tuple(
        item
        for item in identities
        if item.candidate.experiment_role is ExperimentRole.CANDIDATE
    )
    if not baseline or not candidate:
        reason = (
            "approved_metric_binding_stale_reapproval_required"
            if approved_binding is not None
            else "matching_baseline_and_candidate_metric_candidates_not_recovered"
        )
        return MetricBindingResolution(MetricBindingState.UNRESOLVED, reason=reason)
    proposals = tuple(
        sorted(
            (
                MetricBindingProposal._from_pair(baseline_item, candidate_item)
                for baseline_item in baseline
                for candidate_item in candidate
            ),
            key=lambda item: item.proposal_id,
        )
    )
    if len(proposals) > 8:
        return MetricBindingResolution(
            MetricBindingState.UNRESOLVED,
            reason="metric_candidate_pair_ambiguity_exceeds_bound",
        )
    if approved_binding is not None and _binding_is_current(
        approved_binding,
        repository=repository,
        head_sha=head_sha,
        canonical_metric=canonical_metric,
        proposals=proposals,
    ):
        return MetricBindingResolution(
            MetricBindingState.BOUND,
            binding=approved_binding,
        )
    if approved_binding is None and len(proposals) == 1 and all(
        identity.kind is MetricIdentityKind.EXACT
        for identity in (
            proposals[0].baseline_identity,
            proposals[0].candidate_identity,
        )
    ):
        proposal = proposals[0]
        return MetricBindingResolution(
            MetricBindingState.BOUND,
            binding=MetricBinding._from_pair(
                repository=repository,
                head_sha=head_sha,
                canonical_metric=canonical_metric,
                baseline_identity=proposal.baseline_identity,
                candidate_identity=proposal.candidate_identity,
                authority=MetricBindingAuthority.DETERMINISTIC_EXACT,
                approved_by=None,
            ),
        )
    question = MetricBindingQuestion._from_proposals(
        repository=repository,
        head_sha=head_sha,
        canonical_metric=canonical_metric,
        proposals=proposals,
        relevant_claim_id=relevant_claim_id,
        stale=approved_binding is not None,
    )
    return MetricBindingResolution(
        MetricBindingState.APPROVAL_REQUIRED,
        question=question,
    )


def approve_metric_binding(
    question: MetricBindingQuestion,
    *,
    proposal_id: str,
    approved_by: str,
) -> MetricBinding:
    """Approve one explicitly enumerated baseline/candidate proposal atomically."""

    if type(question) is not MetricBindingQuestion:
        raise TypeError("metric binding approval requires MetricBindingQuestion")
    proposal = tuple(
        item for item in question.proposals if item.proposal_id == proposal_id
    )
    if len(proposal) != 1:
        raise AnalysisContractError(
            "metric binding approval must select one issued exact pair"
        )
    selected = proposal[0]
    return MetricBinding._from_pair(
        repository=question.repository,
        head_sha=question.head_sha,
        canonical_metric=question.canonical_metric,
        baseline_identity=selected.baseline_identity,
        candidate_identity=selected.candidate_identity,
        authority=MetricBindingAuthority.USER_APPROVED,
        approved_by=approved_by,
    )


def _endpoint(binding: MetricBinding, role: ExperimentRole) -> MetricCandidate:
    if type(binding) is not MetricBinding:
        raise TypeError("bound metric evidence requires MetricBinding")
    if role is ExperimentRole.BASELINE:
        return binding.baseline_candidate
    if role is ExperimentRole.CANDIDATE:
        return binding.candidate_candidate
    raise AnalysisContractError("bound metric evidence role must be baseline or candidate")


def _metric_field(candidate: MetricCandidate) -> FieldMapping:
    return FieldMapping(
        target_field="metric_value",
        selector=candidate.selector,
        provenance=candidate.provenance,
    )


def integrate_metric_binding(
    mapping: MappingCandidate | RepoMapping,
    binding: MetricBinding,
) -> MappingCandidate | RepoMapping:
    """Attach both exact metric selectors to one mapping without partial success."""

    if type(mapping) not in {MappingCandidate, RepoMapping}:
        raise TypeError(
            "metric binding integration requires MappingCandidate or RepoMapping"
        )
    if type(binding) is not MetricBinding:
        raise TypeError("metric binding integration requires MetricBinding")
    endpoints = {
        ExperimentRole.BASELINE: binding.baseline_candidate,
        ExperimentRole.CANDIDATE: binding.candidate_candidate,
    }
    integrated: list[ArtifactBinding] = []
    matched: dict[ExperimentRole, int] = {
        ExperimentRole.BASELINE: 0,
        ExperimentRole.CANDIDATE: 0,
    }
    for artifact_binding in mapping.bindings:
        endpoint = endpoints.get(artifact_binding.role)
        if endpoint is None or artifact_binding.path != endpoint.artifact_path:
            integrated.append(artifact_binding)
            continue
        if artifact_binding.kind not in _SUPPORTED_KINDS:
            raise AnalysisContractError(
                "metric candidate is not bound to a result or benchmark artifact"
            )
        if artifact_binding.adapter_id not in {None, endpoint.adapter_id}:
            raise AnalysisContractError(
                "metric candidate adapter conflicts with the selected mapping"
            )
        fields = {
            item.target_field: item for item in artifact_binding.mappings
        }
        existing = fields.get("metric_value")
        metric_field = _metric_field(endpoint)
        if existing is not None and selector_identity(existing.selector) != selector_identity(
            metric_field.selector
        ):
            raise AnalysisContractError(
                "metric candidate selector conflicts with the selected mapping"
            )
        fields["metric_value"] = metric_field
        integrated.append(
            replace(
                artifact_binding,
                adapter_id=endpoint.adapter_id,
                mappings=tuple(
                    sorted(
                        fields.values(),
                        key=lambda item: (
                            item.target_field,
                            selector_identity(item.selector),
                        ),
                    )
                ),
            )
        )
        matched[artifact_binding.role] += 1
    if set(matched.values()) != {1}:
        raise AnalysisContractError(
            "selected mapping does not contain the entire metric candidate pair"
        )
    if type(mapping) is RepoMapping:
        return mapping.scope_to_runtime_bindings(tuple(integrated))
    mapping_id = _digest_id(
        "mapping-metric-",
        {
            "source_mapping_id": mapping.mapping_id,
            "metric_binding_id": binding.binding_id,
        },
    )
    return MappingCandidate(
        mapping_id=mapping_id,
        bindings=tuple(integrated),
        confidence=mapping.confidence,
        trust=mapping.trust,
        provenance=mapping.provenance,
    )


def extract_bound_metric_evidence(
    passive: PassiveArtifact | ArtifactSource,
    binding: MetricBinding,
    *,
    role: ExperimentRole,
) -> NormalizedEvidence:
    """Rerun a fixed adapter and project one exact raw metric to its claim name."""

    if type(passive) not in {PassiveArtifact, ArtifactSource}:
        raise TypeError(
            "bound metric extraction requires PassiveArtifact or ArtifactSource"
        )
    endpoint = _endpoint(binding, role)
    if (
        passive.candidate.path != endpoint.artifact_path
        or passive.candidate.sha256 != endpoint.artifact_sha256
    ):
        raise AnalysisContractError(
            "artifact is not the exact bound metric candidate"
        )
    if type(passive) is ArtifactSource:
        scan = extract_metric_candidate_scan(passive, role=role)
        if not scan.completeness.complete or endpoint not in scan.candidates:
            raise AnalysisContractError(
                "exact bound metric candidate is stale and requires pair re-approval"
            )
        match = scan.match
    else:
        match = _probe(passive)
    if match is None or match.adapter_id != endpoint.adapter_id:
        raise AnalysisContractError("bound metric adapter identity changed")
    mappings = {
        item.target_field: item
        for item in match.mappings
        if item.target_field != "metric_value"
    }
    mappings["metric_value"] = _metric_field(endpoint)
    selected_match = AdapterMatch(
        adapter_id=match.adapter_id,
        path=match.path,
        confidence=match.confidence,
        mappings=tuple(
            sorted(
                mappings.values(),
                key=lambda item: (
                    item.target_field,
                    selector_identity(item.selector),
                ),
            )
        ),
        match_evidence=tuple(
            dict.fromkeys((*match.match_evidence, endpoint.provenance))
        ),
    )
    if type(passive) is ArtifactSource:
        from .adapters import scan_jsonl_observations

        outcome = scan_jsonl_observations(passive, selected_match)
        if not outcome.completeness.complete or outcome.evidence is None:
            raise AnalysisContractError(
                "exact bound metric source did not produce complete evidence"
            )
        evidence = outcome.evidence
    else:
        current = extract_metric_candidates(
            passive,
            role=role,
            adapter_match=selected_match,
        )
        if endpoint not in current:
            raise AnalysisContractError(
                "exact bound metric candidate is stale and requires pair re-approval"
            )
        from .adapters import get_adapter

        evidence = get_adapter(endpoint.adapter_id).extract(passive, selected_match)
    metric_observations = tuple(
        item for item in evidence.observations if item.metric_name is not None
    )
    if not metric_observations or any(
        item.metric_name != endpoint.raw_metric_name for item in metric_observations
    ):
        raise AnalysisContractError(
            "fixed adapter raw metric identity does not match the bound candidate"
        )
    canonical = tuple(
        replace(item, metric_name=binding.canonical_metric)
        if item.metric_name is not None
        else item
        for item in evidence.observations
    )
    return replace(evidence, observations=canonical)


def metric_candidate_material(candidate: MetricCandidate) -> dict[str, object]:
    """Return the complete provenance-free commitment for one exact candidate."""

    if type(candidate) is not MetricCandidate:
        raise TypeError("metric candidate material requires MetricCandidate")
    return {
        "candidate_id": candidate.candidate_id,
        "experiment_role": candidate.experiment_role.value,
        "artifact_path": str(candidate.artifact_path),
        "artifact_sha256": str(candidate.artifact_sha256),
        "adapter_id": candidate.adapter_id,
        "selector": selector_identity(candidate.selector),
        "raw_metric_name": candidate.raw_metric_name,
    }


def metric_binding_material(binding: MetricBinding) -> dict[str, object]:
    """Return the complete atomic pair commitment used by plans and Audit."""

    if type(binding) is not MetricBinding:
        raise TypeError("metric binding material requires MetricBinding")
    return {
        "binding_id": binding.binding_id,
        "repository": binding.repository.full_name,
        "head_sha": str(binding.head_sha),
        "canonical_metric": binding.canonical_metric,
        "authority": binding.authority.value,
        "approved_by": binding.approved_by,
        "baseline": metric_candidate_material(binding.baseline_candidate),
        "candidate": metric_candidate_material(binding.candidate_candidate),
    }


def validate_metric_binding_inputs(
    binding: MetricBinding,
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    canonical_metric: str,
    artifacts: tuple[ArtifactCandidate, ...],
    normalized_evidence: tuple[NormalizedEvidence, ...],
    selected_mapping: MappingCandidate | RepoMapping | None = None,
) -> None:
    """Validate that an entire pair is represented by issued canonical evidence."""

    if type(binding) is not MetricBinding:
        raise TypeError("metric binding input validation requires MetricBinding")
    if (
        binding.repository != repository
        or binding.head_sha != head_sha
        or binding.canonical_metric != canonical_metric
    ):
        raise AnalysisContractError(
            "metric binding repository, head, or canonical metric is stale"
        )
    if not isinstance(artifacts, tuple) or not all(
        type(item) is ArtifactCandidate for item in artifacts
    ):
        raise TypeError("metric binding artifacts must contain ArtifactCandidate")
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("metric binding evidence must contain NormalizedEvidence")
    if selected_mapping is not None and type(selected_mapping) not in {
        MappingCandidate,
        RepoMapping,
    }:
        raise TypeError("metric binding selected mapping is invalid")
    for endpoint in (binding.baseline_candidate, binding.candidate_candidate):
        issued = tuple(
            item
            for item in artifacts
            if item.path == endpoint.artifact_path
            and item.sha256 == endpoint.artifact_sha256
            and item.kind in _SUPPORTED_KINDS
        )
        if len(issued) != 1:
            raise AnalysisContractError(
                "metric binding candidate is not one exact issued artifact"
            )
        matching_evidence = tuple(
            item
            for item in normalized_evidence
            if item.artifact == issued[0]
            and item.adapter_match.adapter_id == endpoint.adapter_id
            and any(
                mapping.target_field == "metric_value"
                and selector_identity(mapping.selector)
                == selector_identity(endpoint.selector)
                for mapping in item.adapter_match.mappings
            )
        )
        if len(matching_evidence) != 1:
            raise AnalysisContractError(
                "metric binding candidate has no unique exact normalized evidence"
            )
        observations = tuple(
            item
            for item in matching_evidence[0].observations
            if item.metric_name is not None
        )
        if not observations or any(
            item.metric_name != canonical_metric for item in observations
        ):
            raise AnalysisContractError(
                "metric binding evidence is not canonically projected"
            )
        if selected_mapping is not None:
            matching_bindings = tuple(
                item
                for item in selected_mapping.bindings
                if item.path == endpoint.artifact_path
                and item.role is endpoint.experiment_role
                and item.adapter_id == endpoint.adapter_id
                and any(
                    mapping.target_field == "metric_value"
                    and selector_identity(mapping.selector)
                    == selector_identity(endpoint.selector)
                    for mapping in item.mappings
                )
            )
            if len(matching_bindings) != 1:
                raise AnalysisContractError(
                    "selected mapping does not preserve the entire metric pair"
                )


def _probe(passive: PassiveArtifact) -> AdapterMatch | None:
    from .adapters import ADAPTERS

    for adapter in ADAPTERS:
        match = adapter.probe(passive)
        if match is not None:
            return match
    return None


def _selected_match_candidates(
    passive: PassiveArtifact,
    match: AdapterMatch,
    role: ExperimentRole,
) -> tuple[MetricCandidate, ...] | None:
    metric_fields = tuple(
        item for item in match.mappings if item.target_field == "metric_value"
    )
    if not metric_fields:
        return None
    if len(metric_fields) != 1:
        raise AnalysisContractError(
            "selected metric adapter match must contain one metric_value selector"
        )
    fixed_provenance = (
        *match.match_evidence,
        *(item.provenance for item in match.mappings),
        *(item.selector.provenance for item in match.mappings),
    )
    if any(
        item.kind is not ProvenanceKind.ADAPTER_EXTRACTION
        for item in fixed_provenance
    ):
        raise AnalysisContractError(
            "metric candidates require fixed adapter extraction provenance"
        )
    from .adapters import get_adapter

    evidence = get_adapter(match.adapter_id).extract(passive, match)
    names = {
        item.metric_name
        for item in evidence.observations
        if item.metric_name is not None
    }
    if len(names) != 1:
        return ()
    canonical_fields = tuple(
        item
        for item in evidence.adapter_match.mappings
        if item.target_field == "metric_value"
    )
    if len(canonical_fields) != 1:
        raise AnalysisContractError(
            "fixed adapter did not preserve one selected metric selector"
        )
    field = canonical_fields[0]
    return (
        MetricCandidate._from_extraction(
            role=role,
            path=passive.candidate.path,
            sha256=passive.candidate.sha256,
            adapter_id=evidence.adapter_match.adapter_id,
            selector=field.selector,
            raw_metric_name=next(iter(names)),
            provenance=field.provenance,
        ),
    )


def _structured_candidates_from_leaves(
    passive: PassiveArtifact | ArtifactSource,
    match: AdapterMatch,
    role: ExperimentRole,
    leaves: tuple[tuple[str, object], ...],
) -> tuple[MetricCandidate, ...]:
    from .adapters.core import _adapter_provenance
    from .adapters.structured import (
        _is_finite_number,
        _pointer_terminal,
        _unique_named_pointer,
        _METRIC_NAME_NAMES,
        _RUN_NAMES,
        _SEED_NAMES,
    )

    excluded = {
        pointer
        for pointer in (
            _unique_named_pointer(leaves, _RUN_NAMES),
            _unique_named_pointer(leaves, _SEED_NAMES),
        )
        if pointer is not None
    }
    numeric = tuple(
        (pointer, raw_value)
        for pointer, raw_value in leaves
        if pointer not in excluded and _is_finite_number(raw_value)
    )
    metric_name_pointer = _unique_named_pointer(leaves, _METRIC_NAME_NAMES)
    explicit_name = next(
        (
            value
            for pointer, value in leaves
            if pointer == metric_name_pointer
            and isinstance(value, str)
            and bool(value.strip())
            and value == value.strip()
        ),
        None,
    )
    candidates: list[MetricCandidate] = []
    for pointer, _raw_value in numeric:
        raw_name = (
            explicit_name
            if len(numeric) == 1 and explicit_name is not None
            else _pointer_terminal(pointer)
        )
        provenance = _adapter_provenance(
            passive,
            adapter_id=match.adapter_id,
            selector=pointer,
            inferred=False,
        )
        selector = EvidenceSelector(
            SelectorKind.JSON_POINTER,
            pointer,
            provenance,
        )
        candidates.append(
            MetricCandidate._from_extraction(
                role=role,
                path=passive.candidate.path,
                sha256=passive.candidate.sha256,
                adapter_id=match.adapter_id,
                selector=selector,
                raw_metric_name=raw_name,
                provenance=provenance,
            )
        )
    return tuple(candidates)


def _incomplete_candidate_report(
    completeness: ScanCompleteness,
    reason: ScanReason,
) -> ScanCompleteness:
    state = (
        ScanState.FAILED
        if reason is ScanReason.MALFORMED
        else ScanState.INCOMPLETE_LIMIT
    )
    return replace(completeness, state=state, reason=reason)


def extract_metric_candidate_scan(
    source: ArtifactSource,
    *,
    role: ExperimentRole,
    limits: StreamingLimits = StreamingLimits(),
) -> MetricCandidateScan:
    """Discover exact metric candidates from one complete streaming schema."""

    if type(source) is not ArtifactSource:
        raise TypeError("streaming metric candidate extraction requires ArtifactSource")
    if role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
        raise AnalysisContractError(
            "metric candidate extraction role must be baseline or candidate"
        )
    if type(limits) is not StreamingLimits:
        raise TypeError("streaming metric candidate limits must be StreamingLimits")
    if source.candidate.kind not in _SUPPORTED_KINDS:
        raise AnalysisContractError(
            "streaming metric candidates require a results or benchmark artifact"
        )

    from .adapters.streaming import scan_jsonl_schema

    schema = scan_jsonl_schema(source, limits=limits)
    if not schema.completeness.complete:
        return MetricCandidateScan((), None, schema.completeness)
    assert schema.match is not None
    try:
        candidates = _structured_candidates_from_leaves(
            source,
            schema.match,
            role,
            schema.common_leaves,
        )
    except (AnalysisContractError, TypeError, ValueError):
        return MetricCandidateScan(
            (),
            None,
            _incomplete_candidate_report(
                schema.completeness,
                ScanReason.MALFORMED,
            ),
        )
    if len(candidates) > limits.max_metric_candidates:
        return MetricCandidateScan(
            (),
            None,
            _incomplete_candidate_report(
                schema.completeness,
                ScanReason.CANDIDATE_LIMIT,
            ),
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.raw_metric_name.casefold(),
                item.raw_metric_name,
                selector_identity(item.selector),
            ),
        )
    )
    return MetricCandidateScan(ordered, schema.match, schema.completeness)


def _json_candidates(
    passive: PassiveArtifact,
    match: AdapterMatch,
    role: ExperimentRole,
) -> tuple[MetricCandidate, ...]:
    from .adapters.core import _parse_json
    from .adapters.structured import _pointer_leaves

    return _structured_candidates_from_leaves(
        passive,
        match,
        role,
        _pointer_leaves(_parse_json(passive.content)),
    )


def _record_candidates(
    passive: PassiveArtifact,
    match: AdapterMatch,
    role: ExperimentRole,
) -> tuple[MetricCandidate, ...]:
    from .adapters.native import _parse_native_results
    from .adapters.structured import _common_leaves, _jsonl_records

    if match.adapter_id == "claimci-jsonl-v1":
        records = _jsonl_records(passive.content)
    else:
        parsed = _parse_native_results(passive.content)
        if parsed is None:
            return ()
        records, _summary = parsed
    return _structured_candidates_from_leaves(
        passive,
        match,
        role,
        _common_leaves(records),
    )


def _table_candidates(
    passive: PassiveArtifact,
    match: AdapterMatch,
    role: ExperimentRole,
) -> tuple[MetricCandidate, ...]:
    from .adapters.core import _adapter_provenance
    from .adapters.tabular import (
        _RUN_NAMES,
        _SEED_NAMES,
        _numeric_column,
        _parse_csv,
        _parse_tsv,
        _unique_named_header,
    )

    parser = _parse_csv if match.adapter_id == "claimci-csv-v1" else _parse_tsv
    header, rows = parser(passive.content)
    excluded = {
        name
        for name in (
            _unique_named_header(header, _RUN_NAMES),
            _unique_named_header(header, _SEED_NAMES),
        )
        if name is not None
    }
    candidates: list[MetricCandidate] = []
    for index, column in enumerate(header):
        if column in excluded or not _numeric_column(rows, index):
            continue
        provenance = _adapter_provenance(
            passive,
            adapter_id=match.adapter_id,
            selector=column,
            inferred=False,
        )
        selector = EvidenceSelector(SelectorKind.COLUMN, column, provenance)
        candidates.append(
            MetricCandidate._from_extraction(
                role=role,
                path=passive.candidate.path,
                sha256=passive.candidate.sha256,
                adapter_id=match.adapter_id,
                selector=selector,
                raw_metric_name=column,
                provenance=provenance,
            )
        )
    return tuple(candidates)


def extract_metric_candidates(
    passive: PassiveArtifact,
    *,
    role: ExperimentRole,
    limits: MetricCandidateLimits = MetricCandidateLimits(),
    adapter_match: AdapterMatch | None = None,
) -> tuple[MetricCandidate, ...]:
    """Extract stable metric selector candidates through the fixed registry."""

    if type(passive) is not PassiveArtifact:
        raise TypeError("metric candidate extraction requires PassiveArtifact")
    if role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
        raise AnalysisContractError(
            "metric candidate extraction role must be baseline or candidate"
        )
    if type(limits) is not MetricCandidateLimits:
        raise TypeError("metric candidate limits must be MetricCandidateLimits")
    if passive.candidate.kind not in _SUPPORTED_KINDS:
        return ()
    probed = _probe(passive)
    if probed is None:
        return ()
    if adapter_match is not None:
        if type(adapter_match) is not AdapterMatch:
            raise TypeError(
                "metric candidate adapter_match must be AdapterMatch or null"
            )
        if (
            adapter_match.path != passive.candidate.path
            or adapter_match.adapter_id != probed.adapter_id
        ):
            raise AnalysisContractError(
                "metric candidate adapter match conflicts with the fixed registry"
            )
        match = adapter_match
    else:
        match = probed
    selected = _selected_match_candidates(passive, match, role)
    if selected is not None:
        candidates = selected
    elif match.adapter_id == "claimci-json-v1":
        candidates = _json_candidates(passive, match, role)
    elif match.adapter_id in {
        "claimci-jsonl-v1",
        "claimci-native-results-v1",
    }:
        candidates = _record_candidates(passive, match, role)
    elif match.adapter_id in {"claimci-csv-v1", "claimci-tsv-v1"}:
        candidates = _table_candidates(passive, match, role)
    else:
        return ()
    if len(candidates) > limits.max_candidates_per_artifact:
        raise MetricCandidateLimitError(
            "passive artifact exceeds the metric-candidate bound"
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.raw_metric_name.casefold(),
                item.raw_metric_name,
                selector_identity(item.selector),
            ),
        )
    )


__all__ = [
    "MetricCandidate",
    "MetricCandidateScan",
    "MetricCandidateLimitError",
    "MetricCandidateLimits",
    "MetricBinding",
    "MetricBindingAuthority",
    "MetricBindingProposal",
    "MetricBindingQuestion",
    "MetricBindingResolution",
    "MetricBindingState",
    "MetricIdentity",
    "MetricIdentityAuditContext",
    "MetricIdentityKind",
    "extract_metric_candidates",
    "extract_metric_candidate_scan",
    "extract_bound_metric_evidence",
    "approve_metric_binding",
    "identify_metric_candidate",
    "integrate_metric_binding",
    "metric_binding_material",
    "metric_candidate_material",
    "validate_metric_binding_inputs",
    "resolve_metric_binding",
]
