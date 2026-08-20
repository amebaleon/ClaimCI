"""Passive benchmark-table selector and adapter regressions."""

from __future__ import annotations

import hashlib

import pytest

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    MappingCandidate,
    MappingTrust,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    TablePredicate,
    TableScalarType,
    TableSelector,
    selector_identity,
)
from claimci.analysis.adapters import (
    AdapterParseError,
    AdapterSelectorError,
    CsvAdapter,
    TsvAdapter,
    get_adapter,
)
from claimci.analysis.discovery.mappings import resolve_mappings
from claimci.analysis.discovery.artifacts import _classify_path
from claimci.analysis.discovery.models import DiscoveryLimits


def _provenance() -> FieldProvenance:
    return FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "issued table selector proposal",
    )


def _artifact(content: bytes, path: str = "benchmarks/vessl.csv") -> PassiveArtifact:
    return PassiveArtifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=ArtifactKind.BENCHMARK,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.8),
            discovery_reason="benchmark table fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "benchmark table fixture",
                RepositoryPath(path),
            ),
        ),
        content,
    )


def _table_match(
    artifact: PassiveArtifact,
    *,
    adapter_id: str,
    target: str,
    predicates: tuple[TablePredicate, ...],
    expected_cardinality: int = 1,
) -> AdapterMatch:
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        f"issued table selector; sha256={artifact.candidate.sha256}",
        artifact.candidate.path,
    )
    mapping = FieldMapping(
        "metric_value",
        TableSelector(target, predicates, expected_cardinality, provenance),
        provenance,
    )
    return AdapterMatch(
        adapter_id=adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(0.7),
        mappings=(mapping,),
        match_evidence=(provenance,),
    )


def test_table_selector_canonicalizes_typed_predicates_for_stable_identity() -> None:
    selector = TableSelector(
        column="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "B200"),
            TablePredicate("concurrency", TableScalarType.NUMBER, 1024),
        ),
        expected_cardinality=1,
        provenance=_provenance(),
    )

    assert selector.kind is SelectorKind.COLUMN
    assert selector.expression == "throughput"
    assert tuple(item.column for item in selector.predicates) == (
        "concurrency",
        "hardware",
    )
    assert selector_identity(selector) == (
        "table",
        "throughput",
        1,
        (
            ("concurrency", "number", "1024"),
            ("hardware", "string", "B200"),
        ),
    )


def test_csv_table_selector_extracts_only_the_exact_vessl_row() -> None:
    artifact = _artifact(
        b"hardware,concurrency,throughput\n"
        b"A100,1024,100\n"
        b"B200,1024,150\n"
    )
    adapter = CsvAdapter()
    match = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "B200"),
            TablePredicate("concurrency", TableScalarType.NUMBER, 1024),
        ),
    )

    evidence = adapter.extract(artifact, match)

    assert tuple(item.metric_value for item in evidence.observations) == (150.0,)
    assert evidence.observations[0].metric_name == "throughput"
    assert evidence.adapter_match.mappings[0].selector.provenance.kind is (
        ProvenanceKind.ADAPTER_EXTRACTION
    )


def test_selector_scoped_evidence_identity_preserves_multi_match_sequence() -> None:
    artifact = _artifact(
        b"hardware,concurrency,throughput\n"
        b"B200,512,120\n"
        b"B200,1024,150\n"
    )
    adapter = CsvAdapter()
    two_rows = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "B200"),
        ),
        expected_cardinality=2,
    )
    one_row = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "B200"),
            TablePredicate("concurrency", TableScalarType.NUMBER, 1024),
        ),
    )

    selected = adapter.extract(artifact, two_rows)
    narrower = adapter.extract(artifact, one_row)

    assert tuple(item.metric_value for item in selected.observations) == (120.0, 150.0)
    assert selected.evidence_id == adapter.extract(artifact, two_rows).evidence_id
    assert selected.evidence_id != narrower.evidence_id

    class CsvAdapterSemanticV2(CsvAdapter):
        semantic_version = "2"

    assert (
        CsvAdapterSemanticV2().extract(artifact, two_rows).evidence_id
        != selected.evidence_id
    )

    changed_artifact = _artifact(
        b"hardware,concurrency,throughput\n"
        b"B200,512,121\n"
        b"B200,1024,151\n"
    )
    changed_match = _table_match(
        changed_artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("hardware", TableScalarType.STRING, "B200"),),
        expected_cardinality=2,
    )
    assert (
        adapter.extract(changed_artifact, changed_match).evidence_id
        != selected.evidence_id
    )


def test_selector_scoped_evidence_identity_commits_companion_field_selectors() -> None:
    artifact = _artifact(
        b"commit,throughput,run_a,run_b\n"
        b"candidate,150,run-a,run-b\n"
    )
    adapter = CsvAdapter()
    table_match = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("commit", TableScalarType.STRING, "candidate"),
        ),
    )
    provenance = table_match.mappings[0].provenance

    def with_run_selector(column: str) -> AdapterMatch:
        return AdapterMatch(
            adapter_id=adapter.adapter_id,
            path=artifact.candidate.path,
            confidence=table_match.confidence,
            mappings=(
                *table_match.mappings,
                FieldMapping(
                    "run_id",
                    EvidenceSelector(SelectorKind.COLUMN, column, provenance),
                    provenance,
                ),
            ),
            match_evidence=table_match.match_evidence,
        )

    run_a = adapter.extract(artifact, with_run_selector("run_a"))
    run_b = adapter.extract(artifact, with_run_selector("run_b"))

    assert run_a.observations[0].run_id == "run-a"
    assert run_b.observations[0].run_id == "run-b"
    assert run_a.evidence_id != run_b.evidence_id


def test_tsv_adapter_selects_autoresearch_baseline_and_candidate_independently() -> None:
    artifact = _artifact(
        b"commit\tval_bpb\tmemory\tstatus\n"
        b"baseline\t1.010748\t80\tcomplete\n"
        b"candidate\t0.985552\t80\tcomplete\n",
        path="benchmarks/autoresearch.tsv",
    )
    adapter = TsvAdapter()
    baseline = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="val_bpb",
        predicates=(
            TablePredicate("commit", TableScalarType.STRING, "baseline"),
        ),
    )
    candidate = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="val_bpb",
        predicates=(
            TablePredicate("commit", TableScalarType.STRING, "candidate"),
        ),
    )

    baseline_evidence = adapter.extract(artifact, baseline)
    candidate_evidence = adapter.extract(artifact, candidate)

    assert tuple(item.metric_value for item in baseline_evidence.observations) == (
        1.010748,
    )
    assert tuple(item.metric_value for item in candidate_evidence.observations) == (
        0.985552,
    )
    assert baseline_evidence.evidence_id != candidate_evidence.evidence_id


def test_tsv_adapter_is_available_only_through_the_fixed_registry() -> None:
    adapter = get_adapter("claimci-tsv-v1")

    assert type(adapter) is TsvAdapter


@pytest.mark.parametrize("path", ["artifacts/alpha.csv", "reports/run-17.tsv"])
def test_structured_table_suffix_is_discoverable_without_filename_role_tokens(
    path: str,
) -> None:
    assert _classify_path(RepositoryPath(path)) is ArtifactKind.BENCHMARK


def test_exact_row_selector_rejects_zero_and_ambiguous_matches() -> None:
    artifact = _artifact(
        b"hardware,concurrency,throughput\n"
        b"B200,1024,150\n"
        b"B200,1024,151\n"
    )
    adapter = CsvAdapter()
    ambiguous = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "B200"),
            TablePredicate("concurrency", TableScalarType.NUMBER, 1024),
        ),
    )
    absent = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("hardware", TableScalarType.STRING, "A100"),
        ),
    )

    with pytest.raises(AdapterSelectorError, match="matched 2 rows; expected exactly 1"):
        adapter.extract(artifact, ambiguous)
    with pytest.raises(AdapterSelectorError, match="matched 0 rows; expected exactly 1"):
        adapter.extract(artifact, absent)


def test_declared_predicate_types_are_strict_and_never_inferred() -> None:
    artifact = _artifact(
        b"key,enabled,throughput\n"
        b"01024,true,150\n"
    )
    adapter = CsvAdapter()
    numeric = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("key", TableScalarType.NUMBER, 1024),),
    )
    textual = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("key", TableScalarType.STRING, "01024"),),
    )
    boolean = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("enabled", TableScalarType.BOOLEAN, True),),
    )
    null = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("key", TableScalarType.NULL, None),),
    )

    with pytest.raises(AdapterSelectorError, match="not a strict number"):
        adapter.extract(artifact, numeric)
    assert adapter.extract(artifact, textual).observations[0].metric_value == 150.0
    assert adapter.extract(artifact, boolean).observations[0].metric_value == 150.0
    with pytest.raises(AdapterSelectorError, match="no schema-backed null"):
        adapter.extract(artifact, null)
    with pytest.raises(AnalysisContractError, match="string predicate value"):
        TablePredicate("key", TableScalarType.STRING, "B200\u202e")

    noncanonical_boolean = _artifact(
        b"key,enabled,throughput\n1024,TRUE,150\n"
    )
    invalid_boolean_match = _table_match(
        noncanonical_boolean,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(
            TablePredicate("enabled", TableScalarType.BOOLEAN, True),
        ),
    )
    with pytest.raises(AdapterSelectorError, match="not a strict boolean"):
        adapter.extract(noncanonical_boolean, invalid_boolean_match)


def test_table_selector_rejects_missing_columns_and_malformed_tsv() -> None:
    artifact = _artifact(
        b"hardware,throughput\nB200,150\n",
    )
    adapter = CsvAdapter()
    missing_target = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="latency",
        predicates=(TablePredicate("hardware", TableScalarType.STRING, "B200"),),
    )
    missing_key = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="throughput",
        predicates=(TablePredicate("mode", TableScalarType.STRING, "fp8"),),
    )

    with pytest.raises(AdapterSelectorError, match="target column 'latency'"):
        adapter.extract(artifact, missing_target)
    with pytest.raises(AdapterSelectorError, match="predicate column 'mode'"):
        adapter.extract(artifact, missing_key)

    malformed = _artifact(
        b'commit\tval_bpb\n"candidate\t0.9\n',
        path="benchmarks/autoresearch.tsv",
    )
    with pytest.raises(AdapterParseError, match="malformed TSV"):
        TsvAdapter().probe(malformed)


def test_provider_table_selector_is_typed_but_nonexistent_row_is_rejected_on_reread() -> None:
    artifact = _artifact(
        b"hardware,concurrency,throughput\nB200,1024,150\n"
    ).candidate
    payload = {
        "mappings": [
            {
                "confidence": 0.99,
                "bindings": [
                    {
                        "path": str(artifact.path),
                        "kind": artifact.kind.value,
                        "role": "candidate",
                        "adapter_id": "claimci-csv-v1",
                        "mappings": [
                            {
                                "target_field": "metric_value",
                                "selector": {
                                    "kind": "table",
                                    "column": "throughput",
                                    "predicates": [
                                        {
                                            "column": "hardware",
                                            "scalar_type": "string",
                                            "value": "H100",
                                        }
                                    ],
                                    "expected_cardinality": 1,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    resolution = resolve_mappings(
        (artifact,),
        (),
        (),
        repository=RepositoryIdentity("amebaleon", "ClaimCI"),
        provider_payload=payload,
        limits=DiscoveryLimits(),
    )
    proposal = resolution.candidates[0]
    assert proposal.trust is MappingTrust.INFERRED
    assert proposal.provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL
    binding = proposal.bindings[0]
    assert type(binding.mappings[0].selector) is TableSelector

    passive = _artifact(b"hardware,concurrency,throughput\nB200,1024,150\n")
    match = AdapterMatch(
        adapter_id=binding.adapter_id,
        path=binding.path,
        confidence=proposal.confidence,
        mappings=binding.mappings,
        match_evidence=(binding.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="matched 0 rows"):
        CsvAdapter().extract(passive, match)


def test_one_physical_table_may_bind_roles_only_through_distinct_exact_selectors() -> None:
    artifact = _artifact(
        b"commit,val_bpb\nbaseline,1.010748\ncandidate,0.985552\n"
    )
    adapter = CsvAdapter()
    baseline_match = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="val_bpb",
        predicates=(TablePredicate("commit", TableScalarType.STRING, "baseline"),),
    )
    candidate_match = _table_match(
        artifact,
        adapter_id=adapter.adapter_id,
        target="val_bpb",
        predicates=(TablePredicate("commit", TableScalarType.STRING, "candidate"),),
    )
    provenance = _provenance()

    mapping = MappingCandidate(
        mapping_id="mapping-shared-benchmark-table",
        bindings=(
            ArtifactBinding(
                artifact.candidate.path,
                artifact.candidate.kind,
                ExperimentRole.BASELINE,
                adapter.adapter_id,
                baseline_match.mappings,
                provenance,
            ),
            ArtifactBinding(
                artifact.candidate.path,
                artifact.candidate.kind,
                ExperimentRole.CANDIDATE,
                adapter.adapter_id,
                candidate_match.mappings,
                provenance,
            ),
        ),
        confidence=Confidence(0.8),
        trust=MappingTrust.INFERRED,
        provenance=provenance,
    )

    assert len(mapping.bindings) == 2
    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        MappingCandidate(
            mapping_id="mapping-shared-legacy-column",
            bindings=tuple(
                ArtifactBinding(
                    artifact.candidate.path,
                    artifact.candidate.kind,
                    role,
                    adapter.adapter_id,
                    adapter.probe(artifact).mappings,
                    provenance,
                )
                for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
            ),
            confidence=Confidence(0.8),
            trust=MappingTrust.INFERRED,
            provenance=provenance,
        )
