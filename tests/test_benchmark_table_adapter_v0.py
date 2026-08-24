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
    ConfigValue,
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


def test_csv_table_selector_projects_only_fixed_profile_cells_from_same_exact_row() -> None:
    artifact = _artifact(
        b"role,latency,workload,timing,hardware,ignored\n"
        b"baseline,100,decode-v1,wall_clock,A100,secret-row-data\n"
        b"candidate,80,decode-v1,wall_clock,A100,secret-row-data\n"
    )
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        f"issued table selector; sha256={artifact.candidate.sha256}",
        artifact.candidate.path,
    )
    predicates = (TablePredicate("role", TableScalarType.STRING, "baseline"),)
    mappings = tuple(
        FieldMapping(
            target,
            TableSelector(column, predicates, 1, provenance),
            provenance,
        )
        for target, column in (
            ("metric_value", "latency"),
            ("benchmark.workload.id", "workload"),
            ("benchmark.measurement_config.timing_boundary", "timing"),
            ("benchmark.environment.hardware", "hardware"),
        )
    )
    match = AdapterMatch(
        CsvAdapter.adapter_id,
        artifact.candidate.path,
        Confidence(0.7),
        mappings,
        (provenance,),
    )

    evidence = CsvAdapter().extract(artifact, match)

    assert tuple(evidence.observations[0].config_values) == (
        ConfigValue(
            "benchmark.environment.hardware",
            "A100",
            evidence.observations[0].config_values[0].provenance,
        ),
        ConfigValue(
            "benchmark.measurement_config.timing_boundary",
            "wall_clock",
            evidence.observations[0].config_values[1].provenance,
        ),
        ConfigValue(
            "benchmark.workload.id",
            "decode-v1",
            evidence.observations[0].config_values[2].provenance,
        ),
    )
    assert "ignored" not in repr(evidence.observations)
    assert "secret-row-data" not in repr(evidence.observations)
    assert all(
        item.provenance.kind is ProvenanceKind.ADAPTER_EXTRACTION
        for item in evidence.observations[0].config_values
    )


def test_profile_cell_selectors_must_share_the_metric_row_identity() -> None:
    artifact = _artifact(
        b"role,latency,workload\n"
        b"baseline,100,decode-v1\n"
        b"candidate,80,decode-v2\n"
    )
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        f"issued table selector; sha256={artifact.candidate.sha256}",
        artifact.candidate.path,
    )
    match = AdapterMatch(
        CsvAdapter.adapter_id,
        artifact.candidate.path,
        Confidence(0.7),
        (
            FieldMapping(
                "metric_value",
                TableSelector(
                    "latency",
                    (TablePredicate("role", TableScalarType.STRING, "baseline"),),
                    1,
                    provenance,
                ),
                provenance,
            ),
            FieldMapping(
                "benchmark.workload.id",
                TableSelector(
                    "workload",
                    (TablePredicate("role", TableScalarType.STRING, "candidate"),),
                    1,
                    provenance,
                ),
                provenance,
            ),
        ),
        (provenance,),
    )

    with pytest.raises(AdapterSelectorError, match="same exact row"):
        CsvAdapter().extract(artifact, match)


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


def _shared_table_mapping_candidate(
    artifact: PassiveArtifact,
    baseline_mappings: tuple[FieldMapping, ...],
    candidate_mappings: tuple[FieldMapping, ...],
) -> MappingCandidate:
    provenance = _provenance()
    return MappingCandidate(
        mapping_id="mapping-shared-table-selector-regression",
        bindings=(
            ArtifactBinding(
                artifact.candidate.path,
                artifact.candidate.kind,
                ExperimentRole.BASELINE,
                CsvAdapter.adapter_id,
                baseline_mappings,
                provenance,
            ),
            ArtifactBinding(
                artifact.candidate.path,
                artifact.candidate.kind,
                ExperimentRole.CANDIDATE,
                CsvAdapter.adapter_id,
                candidate_mappings,
                provenance,
            ),
        ),
        confidence=Confidence(0.8),
        trust=MappingTrust.INFERRED,
        provenance=provenance,
    )


def _table_mapping(
    artifact: PassiveArtifact,
    *,
    target_field: str,
    column: str,
    predicates: tuple[TablePredicate, ...],
    expected_cardinality: int = 1,
) -> FieldMapping:
    provenance = _provenance()
    return FieldMapping(
        target_field,
        TableSelector(column, predicates, expected_cardinality, provenance),
        provenance,
    )


def test_shared_table_selector_rejects_subset_and_superset_row_predicates() -> None:
    artifact = _artifact(b"commit,status,val_bpb\nbaseline,ok,1.0\n")
    baseline = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(TablePredicate("commit", TableScalarType.STRING, "baseline"),),
    )
    candidate = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(
            TablePredicate("commit", TableScalarType.STRING, "baseline"),
            TablePredicate("status", TableScalarType.STRING, "ok"),
        ),
    )

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _shared_table_mapping_candidate(artifact, (baseline,), (candidate,))


def test_shared_table_selector_rejects_same_metric_with_extra_profile_mapping() -> None:
    artifact = _artifact(b"commit,workload,val_bpb\nbaseline,decode-v1,1.0\n")
    predicates = (TablePredicate("commit", TableScalarType.STRING, "baseline"),)
    baseline = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=predicates,
    )
    candidate = (
        _table_mapping(
            artifact,
            target_field="metric_value",
            column="val_bpb",
            predicates=predicates,
        ),
        _table_mapping(
            artifact,
            target_field="benchmark.workload.id",
            column="workload",
            predicates=predicates,
        ),
    )

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _shared_table_mapping_candidate(artifact, (baseline,), candidate)


@pytest.mark.parametrize(
    ("baseline_predicate", "candidate_predicate"),
    [
        (
            TablePredicate("status", TableScalarType.STRING, "true"),
            TablePredicate("status", TableScalarType.BOOLEAN, True),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 0),
            TablePredicate("version", TableScalarType.STRING, "-0"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1e0"),
        ),
        (
            TablePredicate("version", TableScalarType.NUMBER, 1),
            TablePredicate("version", TableScalarType.STRING, "1.0"),
        ),
    ],
)
def test_shared_table_selector_rejects_typed_predicates_that_can_match_same_cell(
    baseline_predicate: TablePredicate,
    candidate_predicate: TablePredicate,
) -> None:
    artifact = _artifact(b"status,version,val_bpb\ntrue,1,1.0\n")
    baseline = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(baseline_predicate,),
    )
    candidate = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(candidate_predicate,),
    )

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _shared_table_mapping_candidate(artifact, (baseline,), (candidate,))


def test_shared_table_selector_rejects_overlap_without_shared_predicate_columns() -> None:
    artifact = _artifact(b"commit,status,val_bpb\nbaseline,ok,1.0\n")
    baseline = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(TablePredicate("commit", TableScalarType.STRING, "baseline"),),
    )
    candidate = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=(TablePredicate("status", TableScalarType.STRING, "ok"),),
    )

    with pytest.raises(AnalysisContractError, match="conflicting baseline"):
        _shared_table_mapping_candidate(artifact, (baseline,), (candidate,))


def test_shared_table_selector_allows_disjoint_multirow_profile_mappings() -> None:
    artifact = _artifact(
        b"commit,workload,val_bpb\n"
        b"baseline,decode-v1,1.0\n"
        b"baseline,decode-v1,1.1\n"
        b"candidate,decode-v1,0.9\n"
        b"candidate,decode-v1,0.8\n"
    )
    baseline_predicate = (
        TablePredicate("commit", TableScalarType.STRING, "baseline"),
    )
    candidate_predicate = (
        TablePredicate("commit", TableScalarType.STRING, "candidate"),
    )
    baseline = tuple(
        _table_mapping(
            artifact,
            target_field=target_field,
            column=column,
            predicates=baseline_predicate,
            expected_cardinality=2,
        )
        for target_field, column in (
            ("metric_value", "val_bpb"),
            ("benchmark.workload.id", "workload"),
        )
    )
    candidate = tuple(
        _table_mapping(
            artifact,
            target_field=target_field,
            column=column,
            predicates=candidate_predicate,
            expected_cardinality=2,
        )
        for target_field, column in (
            ("metric_value", "val_bpb"),
            ("benchmark.workload.id", "workload"),
        )
    )

    mapping = _shared_table_mapping_candidate(artifact, baseline, candidate)

    assert len(mapping.bindings) == 2


def test_shared_table_selector_allows_distinct_target_columns_with_overlapping_rows() -> None:
    artifact = _artifact(b"commit,val_bpb,latency\nbaseline,1.0,10\n")
    predicates = (TablePredicate("commit", TableScalarType.STRING, "baseline"),)
    baseline = _table_mapping(
        artifact,
        target_field="metric_value",
        column="val_bpb",
        predicates=predicates,
    )
    candidate = _table_mapping(
        artifact,
        target_field="metric_value",
        column="latency",
        predicates=predicates,
    )

    mapping = _shared_table_mapping_candidate(artifact, (baseline,), (candidate,))

    assert len(mapping.bindings) == 2
