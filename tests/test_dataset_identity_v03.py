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
    NormalizedEvidence,
    PassiveArtifact,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    Sha256Digest,
    to_jsonable,
)
from claimci.analysis.adapters import (
    MAX_ARTIFACT_BYTES,
    AdapterLimitError,
    PassiveJsonLinesDatasetAdapter,
    extract_registered_artifact,
    get_adapter,
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


def test_one_path_may_explicitly_fill_both_dataset_splits_for_audit_detection() -> None:
    train = _dataset_binding()
    evaluation = dataclasses.replace(train, dataset_split=DatasetSplit.EVAL)

    mapping = MappingCandidate(
        mapping_id="mapping-shared-dataset-fixture",
        bindings=(train, evaluation),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=PROVENANCE,
    )

    assert tuple(item.dataset_split for item in mapping.bindings) == (
        DatasetSplit.TRAIN,
        DatasetSplit.EVAL,
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


def _passive_dataset(
    content: bytes = b'{"id":"one"}\n',
    *,
    path: str = "data/baseline-train.jsonl",
    kind: ArtifactKind = ArtifactKind.DATASET,
) -> PassiveArtifact:
    return PassiveArtifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=kind,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.9),
            discovery_reason="dataset fixture",
            relevant_claim_ids=("claim-1",),
            provenance=PROVENANCE,
        ),
        content,
    )


def test_passive_jsonl_dataset_adapter_is_registered_identity_only_evidence() -> None:
    artifact = _passive_dataset()
    adapter = get_adapter("claimci-jsonl-dataset-v1")

    assert type(adapter) is PassiveJsonLinesDatasetAdapter
    match = adapter.probe(artifact)
    assert match is not None
    assert match.adapter_id == "claimci-jsonl-dataset-v1"
    assert match.path == artifact.candidate.path
    assert match.mappings == ()
    evidence = adapter.extract(artifact, match)
    assert type(evidence) is NormalizedEvidence
    assert evidence.adapter_match == match
    assert len(evidence.observations) == 1
    observation = evidence.observations[0]
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED
    assert len(observation.dataset_references) == 1
    reference = observation.dataset_references[0]
    assert reference.path == artifact.candidate.path
    assert reference.split is None


def test_dataset_adapter_preserves_arbitrary_bytes_without_row_interpretation() -> None:
    artifact = _passive_dataset(b"not parsed as JSONL by the identity adapter\xff")
    adapter = PassiveJsonLinesDatasetAdapter()

    match = adapter.probe(artifact)
    assert match is not None
    evidence = adapter.extract(artifact, match)
    assert evidence.artifact.sha256 == artifact.candidate.sha256
    assert evidence.observations[0].dataset_references[0].path == artifact.candidate.path


def test_identical_dataset_bytes_at_distinct_paths_have_distinct_evidence_ids() -> None:
    adapter = PassiveJsonLinesDatasetAdapter()
    first = _passive_dataset(path="data/baseline-eval.jsonl")
    second = _passive_dataset(path="data/candidate-eval.jsonl")

    first_match = adapter.probe(first)
    second_match = adapter.probe(second)
    assert first_match is not None
    assert second_match is not None

    assert adapter.extract(first, first_match).evidence_id != adapter.extract(
        second,
        second_match,
    ).evidence_id


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("data/train.json", ArtifactKind.DATASET),
        ("data/train.jsonl", ArtifactKind.RESULTS),
    ],
)
def test_dataset_adapter_requires_exact_jsonl_dataset_kind(
    path: str,
    kind: ArtifactKind,
) -> None:
    assert PassiveJsonLinesDatasetAdapter().probe(
        _passive_dataset(path=path, kind=kind)
    ) is None


def test_dataset_adapter_enforces_existing_passive_byte_limit() -> None:
    artifact = _passive_dataset(b"x" * (MAX_ARTIFACT_BYTES + 1))

    with pytest.raises(AdapterLimitError, match="8 MiB|bytes"):
        PassiveJsonLinesDatasetAdapter().probe(artifact)


def test_fixed_registry_helper_extracts_dataset_without_extension_fallback() -> None:
    artifact = _passive_dataset()

    evidence = extract_registered_artifact(artifact)

    assert evidence is not None
    assert evidence.adapter_match.adapter_id == "claimci-jsonl-dataset-v1"
