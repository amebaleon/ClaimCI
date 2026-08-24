"""Cross-format hostile and trust-boundary regression tests."""

from __future__ import annotations

import hashlib

import pytest

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.adapters import (
    ADAPTERS,
    MAX_ARTIFACT_BYTES,
    MAX_NODES,
    MAX_RECORDS,
    AdapterError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    CsvAdapter,
    JsonAdapter,
    JsonLinesAdapter,
    NativeConfigAdapter,
    NativeManifestAdapter,
    NativeResultsAdapter,
    TomlConfigAdapter,
    YamlConfigAdapter,
)
from claimci.analysis.adapters.core import _mapping


def _passive(
    content: bytes,
    *,
    path: str,
    kind: ArtifactKind,
) -> PassiveArtifact:
    return PassiveArtifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=kind,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.8),
            discovery_reason="hostile fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "hostile fixture bytes",
                RepositoryPath(path),
                "fixture",
            ),
        ),
        content,
    )


@pytest.mark.parametrize(
    "adapter, path, kind",
    [
        (JsonAdapter(), "data.json", ArtifactKind.RESULTS),
        (JsonLinesAdapter(), "data.jsonl", ArtifactKind.RESULTS),
        (CsvAdapter(), "data.csv", ArtifactKind.RESULTS),
        (YamlConfigAdapter(), "config.yaml", ArtifactKind.CONFIG),
        (TomlConfigAdapter(), "config.toml", ArtifactKind.CONFIG),
        (NativeManifestAdapter(), "research.yaml", ArtifactKind.MANIFEST),
    ],
)
def test_every_recognized_adapter_rejects_artifact_over_8_mib_before_parsing(
    adapter: object, path: str, kind: ArtifactKind
) -> None:
    artifact = _passive(
        b"x" * (MAX_ARTIFACT_BYTES + 1),
        path=path,
        kind=kind,
    )
    with pytest.raises(AdapterLimitError, match="8 MiB|bytes"):
        adapter.probe(artifact)  # type: ignore[attr-defined]


def test_exact_jsonl_and_csv_record_limits_are_accepted_without_truncation() -> None:
    jsonl = b"\n".join(b'{"accuracy":0.9}' for _ in range(MAX_RECORDS))
    jsonl_match = JsonLinesAdapter().probe(
        _passive(jsonl, path="runs.jsonl", kind=ArtifactKind.RESULTS)
    )
    assert jsonl_match is not None

    csv = b"accuracy\n" + b"0.9\n" * MAX_RECORDS
    csv_match = CsvAdapter().probe(
        _passive(csv, path="runs.csv", kind=ArtifactKind.RESULTS)
    )
    assert csv_match is not None


def test_json_node_limit_is_a_whole_artifact_failure() -> None:
    exact = b"[" + b",".join(b"null" for _ in range(MAX_NODES - 1)) + b"]"
    assert JsonAdapter().probe(
        _passive(exact, path="large.json", kind=ArtifactKind.DOCUMENT)
    ) is not None

    over = b"[" + b",".join(b"null" for _ in range(MAX_NODES)) + b"]"
    with pytest.raises(AdapterLimitError, match="nodes"):
        JsonAdapter().probe(
            _passive(over, path="large.json", kind=ArtifactKind.DOCUMENT)
        )


@pytest.mark.parametrize(
    "adapter, content, path, kind",
    [
        (JsonAdapter(), b'{"x":1,"x":2}', "data.json", ArtifactKind.RESULTS),
        (JsonLinesAdapter(), b'{"x":1}\n{"x":1,"x":2}', "data.jsonl", ArtifactKind.RESULTS),
        (CsvAdapter(), b"x,x\n1,2\n", "data.csv", ArtifactKind.RESULTS),
        (YamlConfigAdapter(), b"x: 1\nx: 2\n", "data.yaml", ArtifactKind.CONFIG),
        (TomlConfigAdapter(), b"x=1\nx=2\n", "data.toml", ArtifactKind.CONFIG),
    ],
)
def test_duplicate_keys_or_headers_are_controlled_artifact_failures(
    adapter: object, content: bytes, path: str, kind: ArtifactKind
) -> None:
    with pytest.raises(AdapterError, match="duplicate|overwrite"):
        adapter.probe(_passive(content, path=path, kind=kind))  # type: ignore[attr-defined]


def test_malformed_late_record_never_returns_partial_jsonl_or_csv_match() -> None:
    for adapter, content, path in (
        (JsonLinesAdapter(), b'{"accuracy":0.9}\n{broken}', "runs.jsonl"),
        (CsvAdapter(), b"accuracy\n0.9\n0.8,extra\n", "runs.csv"),
    ):
        with pytest.raises(AdapterError):
            adapter.probe(  # type: ignore[attr-defined]
                _passive(content, path=path, kind=ArtifactKind.RESULTS)
            )


def test_mapping_provenance_must_bind_selector_path_and_current_artifact_hash() -> None:
    artifact = _passive(
        b'{"accuracy":0.9}', path="results.json", kind=ArtifactKind.RESULTS
    )
    adapter = JsonAdapter()
    missing_binding = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "provider proposed a pointer without artifact binding",
    )
    selector = EvidenceSelector(
        SelectorKind.JSON_POINTER,
        "/accuracy",
        missing_binding,
    )
    mapping = FieldMapping("metric_value", selector, missing_binding)
    match = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(0.7),
        mappings=(mapping,),
        match_evidence=(missing_binding,),
    )
    with pytest.raises(AdapterSelectorError, match="provenance|path|sha256"):
        adapter.extract(artifact, match)

    stale = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        f"selector=/accuracy; sha256={'0' * 64}",
        artifact.candidate.path,
        "stale",
    )
    stale_mapping = FieldMapping(
        "metric_value",
        EvidenceSelector(SelectorKind.JSON_POINTER, "/accuracy", stale),
        stale,
    )
    stale_match = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(0.7),
        mappings=(stale_mapping,),
        match_evidence=(stale,),
    )
    with pytest.raises(AdapterSelectorError, match="sha256|provenance"):
        adapter.extract(artifact, stale_match)


def test_hostile_run_identifiers_fail_as_controlled_adapter_errors() -> None:
    artifact = _passive(
        b'{"run_id":"bad\\nid","accuracy":0.9}',
        path="results.json",
        kind=ArtifactKind.RESULTS,
    )
    adapter = JsonAdapter()
    match = adapter.probe(artifact)
    assert match is not None
    with pytest.raises(AdapterError, match="run_id|identifier|control"):
        adapter.extract(artifact, match)


def test_unrepresentable_json_selector_and_huge_integer_are_controlled() -> None:
    hostile_key = _passive(
        b'{"bad\\nkey":0.9}', path="results.json", kind=ArtifactKind.RESULTS
    )
    with pytest.raises(AdapterParseError, match="selector|control|represent"):
        JsonAdapter().probe(hostile_key)

    huge = _passive(
        ('{"value":' + ('9' * 4000) + '}').encode(),
        path="huge.json",
        kind=ArtifactKind.RESULTS,
    )
    match = JsonAdapter().probe(huge)
    assert match is not None
    assert "metric_value" not in {item.target_field for item in match.mappings}
    selected = _mapping(
        huge,
        adapter_id="claimci-json-v1",
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/value",
        inferred=False,
    )
    explicit = AdapterMatch(
        adapter_id="claimci-json-v1",
        path=huge.candidate.path,
        confidence=Confidence(1),
        mappings=(selected,),
        match_evidence=(selected.provenance,),
    )
    with pytest.raises(AdapterParseError, match="finite"):
        JsonAdapter().extract(huge, explicit)


@pytest.mark.parametrize(
    "content",
    [
        b'{"":1}',
        b'{"bad\\nkey":1}',
        b'{"a.b":1,"a":{"b":2}}',
    ],
)
def test_json_config_unrepresentable_or_colliding_keys_fail_controlled(
    content: bytes,
) -> None:
    artifact = _passive(
        content,
        path="config.json",
        kind=ArtifactKind.CONFIG,
    )
    with pytest.raises(AdapterParseError, match="config|represent|selector|collision"):
        JsonAdapter().probe(artifact)


def test_native_manifest_rejects_negative_threshold_as_non_native() -> None:
    content = b"""claim:\n  metric: accuracy\n  minimum_improvement: -0.1\nbaseline:\n  config: base.yaml\n  results: base.json\n  train_dataset: base-train.jsonl\n  eval_dataset: base-eval.jsonl\ncandidate:\n  config: candidate.yaml\n  results: candidate.json\n  train_dataset: candidate-train.jsonl\n  eval_dataset: candidate-eval.jsonl\n"""
    artifact = _passive(
        content,
        path="research.yaml",
        kind=ArtifactKind.MANIFEST,
    )
    with pytest.raises(AdapterParseError, match="non-negative|minimum"):
        NativeManifestAdapter().probe(artifact)


def test_native_config_requires_the_complete_declared_compute_input_shape() -> None:
    partial = _passive(
        b"epochs: 4\nmodel: tiny\n",
        path="config.yaml",
        kind=ArtifactKind.CONFIG,
    )
    assert NativeConfigAdapter().probe(partial) is None


def test_native_results_summary_with_unrepresentable_key_fails_controlled() -> None:
    artifact = _passive(
        b'{"runs":[{"seed":1,"accuracy":0.9}],"summary":{"bad\\nkey":0.9}}',
        path="results.json",
        kind=ArtifactKind.RESULTS,
    )
    adapter = NativeResultsAdapter()
    match = adapter.probe(artifact)
    assert match is not None
    with pytest.raises(AdapterError, match="summary|config|identifier|control"):
        adapter.extract(artifact, match)


def test_generic_roles_remain_unspecified_even_with_role_words_in_every_surface() -> None:
    fixtures = (
        (
            JsonAdapter(),
            _passive(
                b'{"baseline":0.9}',
                path="candidate/baseline.json",
                kind=ArtifactKind.RESULTS,
            ),
        ),
        (
            JsonLinesAdapter(),
            _passive(
                b'{"candidate":0.9}',
                path="baseline/candidate.jsonl",
                kind=ArtifactKind.RESULTS,
            ),
        ),
        (
            CsvAdapter(),
            _passive(
                b"candidate\n0.9\n",
                path="baseline/candidate.csv",
                kind=ArtifactKind.RESULTS,
            ),
        ),
    )
    for adapter, artifact in fixtures:
        match = adapter.probe(artifact)
        assert match is not None
        evidence = adapter.extract(artifact, match)
        assert all(
            item.experiment_role is ExperimentRole.UNSPECIFIED
            for item in evidence.observations
        )


def test_generated_ordering_and_full_field_provenance_are_stable() -> None:
    artifact = _passive(
        b"z: 1\na:\n  c: 3\n  b: 2\n",
        path="config.yaml",
        kind=ArtifactKind.CONFIG,
    )
    adapter = YamlConfigAdapter()
    first = adapter.probe(artifact)
    second = adapter.probe(artifact)
    assert first == second
    assert first is not None
    assert tuple(item.selector.expression for item in first.mappings) == (
        "a.b",
        "a.c",
        "z",
    )
    values = adapter.extract(artifact, first).observations[0].config_values
    for value in values:
        assert value.provenance.source_path == artifact.candidate.path
        assert value.key in value.provenance.detail
        assert artifact.candidate.sha256 in value.provenance.detail


def test_repository_discovery_can_isolate_one_adapter_error_and_continue() -> None:
    artifacts = (
        _passive(
            b'{"accuracy":NaN}',
            path="bad.json",
            kind=ArtifactKind.RESULTS,
        ),
        _passive(
            b'{"accuracy":0.9}',
            path="good.json",
            kind=ArtifactKind.RESULTS,
        ),
    )
    outcomes: list[str] = []
    adapter = JsonAdapter()
    for artifact in artifacts:
        try:
            outcomes.append("match" if adapter.probe(artifact) is not None else "none")
        except AdapterError:
            outcomes.append("error")
    assert outcomes == ["error", "match"]


def test_fixed_registry_contains_one_identity_only_dataset_adapter_and_no_execution_adapter() -> None:
    assert len(ADAPTERS) == 10
    assert tuple(
        item.adapter_id for item in ADAPTERS if "dataset" in item.adapter_id
    ) == ("claimci-jsonl-dataset-v1",)
    assert tuple(
        item.adapter_id for item in ADAPTERS if "tsv" in item.adapter_id
    ) == ("claimci-tsv-v1",)
    assert all("pickle" not in item.adapter_id for item in ADAPTERS)
    assert all("plugin" not in item.adapter_id for item in ADAPTERS)
