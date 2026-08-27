"""Deterministic bounded CSV adapter behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ExperimentRole,
    FieldProvenance,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.adapters.core import (
    MAX_COLUMNS,
    MAX_LOGICAL_RECORD_BYTES,
    MAX_RECORDS,
    AdapterIntegrityError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _mapping,
)
from claimci.analysis.adapters.tabular import CsvAdapter


FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def _passive(content: bytes, *, path: str = "results/runs.csv") -> PassiveArtifact:
    return passive_artifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=ArtifactKind.RESULTS,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.9),
            discovery_reason="CSV fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "CSV fixture bytes",
                RepositoryPath(path),
                "fixture",
            ),
        ),
        content,
    )


def _fixture(name: str) -> PassiveArtifact:
    return _passive((FIXTURES / name).read_bytes(), path=f"results/{name}")


def _targets(match: AdapterMatch) -> dict[str, str]:
    return {
        mapping.target_field: mapping.selector.expression
        for mapping in match.mappings
    }


def test_csv_probe_and_extract_unambiguous_seed_runs_in_source_order() -> None:
    artifact = _fixture("seed_runs.csv")
    adapter = CsvAdapter()
    match = adapter.probe(artifact)

    assert match is not None
    assert match.adapter_id == "claimci-csv-v1"
    assert _targets(match) == {
        "metric_value": "accuracy",
        "run_id": "run_id",
        "seed": "seed",
    }
    observations = adapter.extract(artifact, match).observations
    assert [item.metric_name for item in observations] == ["accuracy"] * 3
    assert [item.metric_value for item in observations] == [0.88, 0.90, 0.92]
    assert [item.run_id for item in observations] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert [item.seed for item in observations] == [11, 12, 13]
    assert all(item.experiment_role is ExperimentRole.UNSPECIFIED for item in observations)
    assert all(item.provenance.source_path == artifact.candidate.path for item in observations)
    assert all(artifact.candidate.sha256 in item.provenance.detail for item in observations)


def test_ambiguous_csv_requires_exact_external_column_mapping() -> None:
    artifact = _fixture("ambiguous_metrics.csv")
    adapter = CsvAdapter()
    ambiguous = adapter.probe(artifact)

    assert ambiguous is not None
    assert "metric_value" not in _targets(ambiguous)
    with pytest.raises(AdapterSelectorError, match="metric_value|ambiguous|mapping"):
        adapter.extract(artifact, ambiguous)

    mappings = tuple(
        _mapping(
            artifact,
            adapter_id=adapter.adapter_id,
            target_field=target,
            selector_kind=SelectorKind.COLUMN,
            selector=column,
            inferred=False,
        )
        for target, column in (
            ("metric_value", "accuracy"),
            ("run_id", "run_id"),
            ("seed", "seed"),
        )
    )
    explicit = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=mappings,
        match_evidence=tuple(item.provenance for item in mappings),
    )
    observations = adapter.extract(artifact, explicit).observations
    assert [item.metric_value for item in observations] == [0.88, 0.90, 0.92]
    assert all(item.experiment_role is ExperimentRole.UNSPECIFIED for item in observations)


@pytest.mark.parametrize(
    "content, message",
    [
        (b"run_id,,accuracy\na,1,0.9\n", "empty header"),
        (b"run_id,seed,seed\na,1,1\n", "duplicate header"),
        (b"run_id,seed,accuracy\na,1\n", "columns|field count"),
        (b"run_id,seed,accuracy\na,1,0.9,extra\n", "columns|field count"),
        (b'run_id,seed,accuracy\n"unterminated,1,0.9\n', "malformed CSV"),
        (b"run_id,seed,accuracy\n\xff,1,0.9\n", "UTF-8"),
    ],
)
def test_csv_rejects_malformed_headers_rows_quotes_and_encoding(
    content: bytes, message: str
) -> None:
    with pytest.raises(AdapterParseError, match=message):
        CsvAdapter().probe(_passive(content))


def test_csv_column_record_and_logical_record_limits() -> None:
    headers = ",".join(f"c{index}" for index in range(MAX_COLUMNS + 1))
    row = ",".join("1" for _ in range(MAX_COLUMNS + 1))
    with pytest.raises(AdapterLimitError, match="columns"):
        CsvAdapter().probe(_passive(f"{headers}\n{row}\n".encode()))

    too_many = b"metric\n" + (b"0.9\n" * (MAX_RECORDS + 1))
    with pytest.raises(AdapterLimitError, match="records"):
        CsvAdapter().probe(_passive(too_many))

    oversized = b"metric,description\n0.9," + b"x" * MAX_LOGICAL_RECORD_BYTES + b"\n"
    with pytest.raises(AdapterLimitError, match="logical record|1 MiB"):
        CsvAdapter().probe(_passive(oversized))


def test_csv_nonfinite_and_locale_numbers_are_not_silently_interpreted() -> None:
    for token in ("NaN", "Infinity", "-Infinity"):
        artifact = _passive(f"accuracy\n{token}\n".encode())
        match = CsvAdapter().probe(artifact)
        assert match is not None
        assert "metric_value" not in _targets(match)

    locale = _passive(b'accuracy\n"0,90"\n')
    ambiguous = CsvAdapter().probe(locale)
    assert ambiguous is not None
    assert "metric_value" not in _targets(ambiguous)
    external = _mapping(
        locale,
        adapter_id="claimci-csv-v1",
        target_field="metric_value",
        selector_kind=SelectorKind.COLUMN,
        selector="accuracy",
        inferred=False,
    )
    match = AdapterMatch(
        adapter_id="claimci-csv-v1",
        path=locale.candidate.path,
        confidence=Confidence(1),
        mappings=(external,),
        match_evidence=(external.provenance,),
    )
    with pytest.raises(AdapterParseError, match="finite|number"):
        CsvAdapter().extract(locale, match)


def test_csv_requires_exact_header_and_column_selector_kind() -> None:
    artifact = _fixture("seed_runs.csv")
    adapter = CsvAdapter()

    missing = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field="metric_value",
        selector_kind=SelectorKind.COLUMN,
        selector="Accuracy",
        inferred=False,
    )
    missing_match = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(missing,),
        match_evidence=(missing.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="header|column"):
        adapter.extract(artifact, missing_match)

    pointer = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/accuracy",
        inferred=False,
    )
    pointer_match = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(pointer,),
        match_evidence=(pointer.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="selector"):
        adapter.extract(artifact, pointer_match)


def test_csv_extract_rechecks_sha_and_never_infers_role_from_filename() -> None:
    artifact = _passive(
        b"run_id,seed,accuracy\nr1,1,0.9\n",
        path="baseline/candidate.csv",
    )
    adapter = CsvAdapter()
    match = adapter.probe(artifact)
    assert match is not None
    assert adapter.extract(artifact, match).observations[0].experiment_role is ExperimentRole.UNSPECIFIED

    object.__setattr__(artifact, "content", b"run_id,seed,accuracy\nr1,1,0.1\n")
    with pytest.raises(AdapterIntegrityError):
        adapter.extract(artifact, match)


def test_csv_rejects_unsupported_artifact_kind_without_parsing() -> None:
    content = b"accuracy\n0.9\n"
    candidate = ArtifactCandidate(
        path=RepositoryPath("data/eval.csv"),
        kind=ArtifactKind.DATASET,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.5),
        discovery_reason="dataset",
        relevant_claim_ids=(),
        provenance=FieldProvenance(ProvenanceKind.DETERMINISTIC_DISCOVERY, "dataset"),
    )
    assert CsvAdapter().probe(passive_artifact(candidate, content)) is None
