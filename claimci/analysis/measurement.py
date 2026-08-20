"""Source-bound measurement-policy recovery for pre-Audit obligations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping

from claimci.measurement import (
    MeasurementAuditContext,
    MeasurementSourceBinding,
    MeasurementSourceSelector,
    MeasurementSourceSnapshot,
    UpstreamAggregationProcedure,
)

from .claim_types import CanonicalScientificClaim, recover_scientific_claim
from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ClaimReference,
    EphemeralAuditPlan,
    ExperimentRole,
    NormalizedEvidence,
    PassiveArtifact,
)


_SPACE = re.compile(r"\s+")
_PROCEDURE_PATTERNS = (
    (
        UpstreamAggregationProcedure.ARITHMETIC_MEAN_V1,
        re.compile(
            r"\b(?:using|via|with|aggregated\s+(?:using|with)|reduced\s+(?:using|with|by))\s+"
            r"(?:the\s+)?(?:arithmetic\s+mean|average)(?:\s+across\s+(?:supplied\s+)?runs)?\b"
        ),
    ),
    (
        UpstreamAggregationProcedure.WEIGHTED_MEAN,
        re.compile(
            r"\b(?:using|via|with|aggregated\s+(?:using|with)|reduced\s+(?:using|with|by))\s+"
            r"(?:the\s+)?weighted\s+(?:mean|average)\b"
        ),
    ),
    (
        UpstreamAggregationProcedure.MEDIAN,
        re.compile(
            r"\b(?:using|via|with|aggregated\s+(?:using|with)|reduced\s+(?:using|with|by))\s+"
            r"(?:the\s+)?median\b"
        ),
    ),
    (
        UpstreamAggregationProcedure.BEST_OF_N,
        re.compile(
            r"\b(?:using|via|with|aggregated\s+(?:using|with)|reduced\s+(?:using|with|by))\s+"
            r"(?:the\s+)?best(?:-|\s+)of(?:-|\s+)(?:n|[1-9][0-9]{0,2})\b"
        ),
    ),
    (
        UpstreamAggregationProcedure.RETRY_FILTERING,
        re.compile(
            r"\b(?:using|via|with)\s+(?:the\s+)?(?:retry\s+filtering|filtered\s+retries)\b"
        ),
    ),
    (
        UpstreamAggregationProcedure.ADJUDICATION,
        re.compile(r"\b(?:using|via|with)\s+(?:the\s+)?adjudication\b"),
    ),
)
_EXPLICIT_UNSUPPORTED_PROCEDURE = re.compile(
    r"\b(?:using|via|with|aggregated\s+(?:using|with)|"
    r"reduced\s+(?:using|with|by))\s+(?:the\s+)?"
    r"(?:[a-z][a-z0-9-]*\s+){0,4}"
    r"(?:mean|average|aggregation|reduction|filtering|adjudication)"
    r"(?:\s+across\s+(?:supplied\s+)?runs)?\b"
)


@dataclass(frozen=True, slots=True)
class UpstreamProcedureRequirement:
    procedure: UpstreamAggregationProcedure
    source_text_sha256: str

    def __post_init__(self) -> None:
        if type(self.procedure) is not UpstreamAggregationProcedure:
            raise TypeError("upstream procedure must use its canonical enum")
        if (
            not isinstance(self.source_text_sha256, str)
            or len(self.source_text_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_text_sha256)
        ):
            raise ValueError("upstream procedure source commitment is invalid")

    @property
    def deterministically_supported(self) -> bool:
        return self.procedure is UpstreamAggregationProcedure.ARITHMETIC_MEAN_V1


def recover_upstream_procedure_requirement(
    reference: ClaimReference,
) -> UpstreamProcedureRequirement | None:
    """Recover only an explicit procedure phrase from exact claim source text."""

    if type(reference) is not ClaimReference:
        raise TypeError("upstream procedure recovery requires ClaimReference")
    recovered = recover_scientific_claim(reference)
    if type(recovered) is not CanonicalScientificClaim:
        return None
    text = _SPACE.sub(" ", reference.text.strip()).casefold()
    matched = {
        procedure
        for procedure, pattern in _PROCEDURE_PATTERNS
        if pattern.search(text) is not None
    }
    if not matched:
        if _EXPLICIT_UNSUPPORTED_PROCEDURE.search(text) is None:
            return None
        matched = {UpstreamAggregationProcedure.UNSUPPORTED}
    procedure = (
        next(iter(matched))
        if len(matched) == 1
        else UpstreamAggregationProcedure.AMBIGUOUS
    )
    return UpstreamProcedureRequirement(
        procedure=procedure,
        source_text_sha256=hashlib.sha256(reference.text.encode("utf-8")).hexdigest(),
    )


_ADAPTER_VERSION = re.compile(r"(?:^|[-_.])(v[0-9]+)\Z")


def measurement_audit_context_from_materialization(
    plan: EphemeralAuditPlan,
    bound: tuple[tuple[NormalizedEvidence, ArtifactBinding], ...],
    captured: Mapping[str, PassiveArtifact],
) -> MeasurementAuditContext:
    """Commit exact passive bindings after runtime revalidation, before Audit."""

    if type(plan) is not EphemeralAuditPlan:
        raise TypeError("measurement source context requires EphemeralAuditPlan")
    if plan.scientific_claim is None or plan.claim_policy is None:
        raise AnalysisContractError(
            "measurement source context requires canonical scientific claim policy"
        )
    if (
        not isinstance(bound, tuple)
        or not bound
        or len(bound) > 16
        or not all(
            type(evidence) is NormalizedEvidence and type(binding) is ArtifactBinding
            for evidence, binding in bound
        )
    ):
        raise AnalysisContractError("measurement bound evidence exceeds its limit")
    if not isinstance(captured, Mapping):
        raise TypeError("measurement source context requires captured passive artifacts")
    by_role: dict[ExperimentRole, list[MeasurementSourceBinding]] = {
        ExperimentRole.BASELINE: [],
        ExperimentRole.CANDIDATE: [],
    }
    for evidence, binding in bound:
        passive = captured.get(str(binding.path))
        if type(passive) is not PassiveArtifact or passive.candidate != evidence.artifact:
            raise AnalysisContractError(
                "measurement source binding is not the exact captured artifact"
            )
        if binding.role not in by_role:
            raise AnalysisContractError("measurement source role is not executable")
        matched = _ADAPTER_VERSION.search(binding.adapter_id)
        selectors = tuple(
            MeasurementSourceSelector(
                target_field=item.target_field,
                kind=item.selector.kind.value,
                expression=item.selector.expression,
            )
            for item in binding.mappings
        )
        by_role[binding.role].append(
            MeasurementSourceBinding(
                path=str(binding.path),
                kind=binding.kind.value,
                sha256=str(evidence.artifact.sha256),
                size=evidence.artifact.size,
                adapter_id=binding.adapter_id,
                adapter_version=None if matched is None else matched.group(1),
                selectors=selectors,
                role=binding.role.value,
                split=(
                    None
                    if binding.dataset_split is None
                    else binding.dataset_split.value
                ),
            )
        )
    if not all(by_role.values()):
        raise AnalysisContractError(
            "measurement source context requires both experiment roles"
        )
    repository_id = f"{plan.repository.owner}/{plan.repository.name}"
    return MeasurementAuditContext(
        baseline_source=MeasurementSourceSnapshot.from_bindings(
            repository_id=repository_id,
            head_sha=str(plan.head_sha),
            bindings=tuple(by_role[ExperimentRole.BASELINE]),
        ),
        candidate_source=MeasurementSourceSnapshot.from_bindings(
            repository_id=repository_id,
            head_sha=str(plan.head_sha),
            bindings=tuple(by_role[ExperimentRole.CANDIDATE]),
        ),
    )


__all__ = [
    "UpstreamProcedureRequirement",
    "measurement_audit_context_from_materialization",
    "recover_upstream_procedure_requirement",
]
