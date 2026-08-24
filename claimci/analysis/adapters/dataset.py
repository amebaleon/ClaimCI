"""Identity-only adapter for bounded passive JSON Lines dataset bytes."""

from __future__ import annotations

import hashlib

from claimci.analysis import (
    AdapterMatch,
    ArtifactKind,
    Confidence,
    DatasetReference,
    ExperimentRole,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    SelectorKind,
)

from .core import (
    _ArtifactEnvelope,
    _adapter_provenance,
    _supports,
    _validate_match,
    _verify_integrity,
)


def _dataset_evidence_id(adapter_id: str, artifact: _ArtifactEnvelope) -> str:
    material = (
        f"{adapter_id}\0{artifact.candidate.path}\0{artifact.candidate.sha256}"
    ).encode("utf-8")
    return f"evidence-{adapter_id}-{hashlib.sha256(material).hexdigest()[:24]}"


class PassiveJsonLinesDatasetAdapter:
    """Bind exact JSONL bytes without interpreting customer dataset rows."""

    __slots__ = ()
    adapter_id = "claimci-jsonl-dataset-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=frozenset({ArtifactKind.DATASET}),
            suffixes=frozenset({".jsonl"}),
        ):
            return None
        _verify_integrity(artifact)
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="<whole passive dataset artifact>",
            inferred=False,
        )
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(0.99),
            mappings=(),
            match_evidence=(provenance,),
        )

    def extract(
        self,
        artifact: PassiveArtifact,
        match: AdapterMatch,
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.DOTTED_PATH,
            allowed_targets=frozenset(),
        )
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="<whole passive dataset artifact>",
            inferred=False,
        )
        return NormalizedEvidence(
            evidence_id=_dataset_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=(
                NormalizedObservation(
                    provenance=provenance,
                    experiment_role=ExperimentRole.UNSPECIFIED,
                    dataset_references=(
                        DatasetReference(
                            path=artifact.candidate.path,
                            split=None,
                            provenance=provenance,
                        ),
                    ),
                ),
            ),
        )


__all__ = ["PassiveJsonLinesDatasetAdapter"]
