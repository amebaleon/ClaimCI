"""Production passive dataset identity and split-mapping regressions."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from claimci.analysis import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    DatasetSplit,
    ExperimentRole,
    FieldProvenance,
    MappingCandidate,
    MappingTrust,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    Sha256Digest,
    to_jsonable,
)


PROVENANCE = FieldProvenance(
    ProvenanceKind.DETERMINISTIC_DISCOVERY,
    "dataset identity fixture",
    RepositoryPath("data/baseline-train.jsonl"),
    "dataset-fixture",
)
REPOSITORY = RepositoryIdentity("example", "research")


def _dataset_binding(
    *,
    path: str = "data/baseline-train.jsonl",
    role: ExperimentRole = ExperimentRole.BASELINE,
    split: DatasetSplit | None = DatasetSplit.TRAIN,
) -> ArtifactBinding:
    return ArtifactBinding(
        path=RepositoryPath(path),
        kind=ArtifactKind.DATASET,
        role=role,
        adapter_id="claimci-jsonl-dataset-v1",
        mappings=(),
        provenance=PROVENANCE,
        dataset_split=split,
    )


def _mapping(binding: ArtifactBinding) -> MappingCandidate:
    return MappingCandidate(
        mapping_id="mapping-dataset-fixture",
        bindings=(binding,),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=PROVENANCE,
    )


def test_dataset_split_is_typed_immutable_and_json_safe() -> None:
    binding = _dataset_binding()

    assert DatasetSplit.TRAIN.value == "train"
    assert DatasetSplit.EVAL.value == "eval"
    assert binding.dataset_split is DatasetSplit.TRAIN
    assert to_jsonable(binding)["dataset_split"] == "train"
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.dataset_split = DatasetSplit.EVAL  # type: ignore[misc]


def test_only_dataset_bindings_may_carry_a_typed_split() -> None:
    with pytest.raises(TypeError, match="DatasetSplit"):
        _dataset_binding(split="train")  # type: ignore[arg-type]

    with pytest.raises(AnalysisContractError, match="dataset split"):
        ArtifactBinding(
            path=RepositoryPath("results.csv"),
            kind=ArtifactKind.RESULTS,
            role=ExperimentRole.BASELINE,
            adapter_id="claimci-csv-v1",
            mappings=(),
            provenance=PROVENANCE,
            dataset_split=DatasetSplit.TRAIN,
        )


def test_explicit_approval_preserves_split_and_rejects_runtime_split_drift() -> None:
    binding = _dataset_binding()
    approved = RepoMapping.approve(
        REPOSITORY,
        _mapping(binding),
        approved_by="repository-owner",
    )

    assert approved.bindings[0].dataset_split is DatasetSplit.TRAIN
    assert approved.scope_to_runtime_bindings((binding,)) is approved
    with pytest.raises(AnalysisContractError, match="approved|split|binding"):
        approved.scope_to_runtime_bindings(
            (dataclasses.replace(binding, dataset_split=DatasetSplit.EVAL),)
        )


def test_artifact_candidate_still_binds_exact_dataset_bytes() -> None:
    content = b'{"id":"one"}\n'
    candidate = ArtifactCandidate(
        path=RepositoryPath("data/baseline-train.jsonl"),
        kind=ArtifactKind.DATASET,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.9),
        discovery_reason="dataset fixture",
        relevant_claim_ids=("claim-1",),
        provenance=PROVENANCE,
    )

    assert candidate.size == len(content)

