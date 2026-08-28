"""Security and bounded-parsing contracts shared by v0.3 adapters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    FieldMapping,
    FieldProvenance,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.adapters.core import (
    MAX_ARTIFACT_BYTES,
    MAX_DEPTH,
    MAX_NODES,
    AdapterIntegrityError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _adapter_provenance,
    _config_target,
    _decode_utf8,
    _finite_number,
    _mapping,
    _parse_json,
    _resolve_dotted_path,
    _resolve_json_pointer,
    _scalar_leaves,
    _supports,
    _validate_graph,
    _validate_match,
    _verify_integrity,
)


def _passive(
    content: bytes,
    *,
    path: str = "results/metrics.json",
    kind: ArtifactKind = ArtifactKind.RESULTS,
) -> PassiveArtifact:
    candidate = ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.9),
        discovery_reason="test fixture",
        relevant_claim_ids=(),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "test fixture bytes",
            RepositoryPath(path),
            "fixture",
        ),
    )
    return passive_artifact(candidate, content)


def test_integrity_is_recomputed_immediately_before_extraction() -> None:
    artifact = _passive(b'{"accuracy":0.9}')
    object.__setattr__(artifact, "content", b'{"accuracy":0.1}')

    with pytest.raises(AdapterIntegrityError, match="sha256|integrity"):
        _verify_integrity(artifact)


def test_artifact_size_limit_accepts_exact_boundary_and_rejects_one_byte_over() -> None:
    exact = _passive(b"x" * MAX_ARTIFACT_BYTES)
    assert _verify_integrity(exact) is None

    oversized = _passive(b"x" * (MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(AdapterLimitError, match="8 MiB|bytes|size"):
        _verify_integrity(oversized)


def test_strict_utf8_decode_rejects_invalid_bytes() -> None:
    assert _decode_utf8(b"metric=accuracy") == "metric=accuracy"
    with pytest.raises(AdapterParseError, match="UTF-8"):
        _decode_utf8(b"\xff")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_json_parser_rejects_nonfinite_tokens(token: str) -> None:
    with pytest.raises(AdapterParseError, match="finite|constant"):
        _parse_json(f'{{"value":{token}}}'.encode())


def test_json_parser_rejects_duplicate_keys_at_nested_depth() -> None:
    with pytest.raises(AdapterParseError, match="duplicate"):
        _parse_json(b'{"outer":{"value":1,"value":2}}')


def test_graph_limits_accept_exact_depth_and_reject_deeper_values() -> None:
    exact: object = 1
    for index in range(MAX_DEPTH):
        exact = {f"k{index}": exact}
    assert _validate_graph(exact) is None

    too_deep: object = exact
    too_deep = {"overflow": too_deep}
    with pytest.raises(AdapterLimitError, match="depth"):
        _validate_graph(too_deep)


def test_graph_limits_accept_exact_node_count_and_reject_one_more() -> None:
    exact = [None] * (MAX_NODES - 1)
    assert _validate_graph(exact) is None

    with pytest.raises(AdapterLimitError, match="nodes"):
        _validate_graph([None] * MAX_NODES)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_graph_rejects_nested_nonfinite_values(value: float) -> None:
    with pytest.raises(AdapterParseError, match="finite"):
        _validate_graph({"nested": [value]})


def test_graph_rejects_non_string_mapping_keys_and_repeated_containers() -> None:
    with pytest.raises(AdapterParseError, match="string keys"):
        _validate_graph({1: "value"})

    shared: list[object] = []
    with pytest.raises(AdapterParseError, match="repeated|alias"):
        _validate_graph({"a": shared, "b": shared})


def test_json_pointer_decodes_rfc6901_tokens_and_canonical_array_indices() -> None:
    value = {"a/b": {"~metric": [0.9, 0.8]}}

    assert _resolve_json_pointer(value, "/a~1b/~0metric/0") == 0.9
    assert _resolve_json_pointer(value, "/a~1b/~0metric/1") == 0.8
    for selector in ("a/b", "/a~2b", "/a~", "/a~1b/~0metric/01", "/a~1b/~0metric/-"):
        with pytest.raises(AdapterSelectorError):
            _resolve_json_pointer(value, selector)


def test_json_pointer_rejects_missing_tokens_wrong_containers_and_bounds() -> None:
    value = {"runs": [{"accuracy": 0.9}]}

    for selector in ("/missing", "/runs/2", "/runs/accuracy", "/runs/0/accuracy/0"):
        with pytest.raises(AdapterSelectorError):
            _resolve_json_pointer(value, selector)


def test_dotted_path_is_structural_and_never_reinterprets_separator_keys() -> None:
    value = {"model": {"name": "tiny"}, "model.name": "literal"}

    assert _resolve_dotted_path(value, "model.name") == "tiny"
    for selector in (".model", "model.", "model..name", "model[name]", "model.*"):
        with pytest.raises(AdapterSelectorError):
            _resolve_dotted_path(value, selector)


def test_scalar_leaves_are_stable_and_report_unrepresentable_values() -> None:
    leaves, unsupported = _scalar_leaves(
        {"z": 1, "a": {"b": True}, "bad.key": 2, "list": [1, 2]}
    )

    assert leaves == (("a.b", True), ("z", 1))
    assert unsupported == ("bad.key", "list")


@pytest.mark.parametrize("value", [True, "1.0", math.nan, math.inf])
def test_finite_number_rejects_boolean_text_and_nonfinite_values(value: object) -> None:
    with pytest.raises(AdapterParseError):
        _finite_number(value, label="metric")


def test_provenance_mapping_and_config_target_are_stable_and_complete() -> None:
    artifact = _passive(b'{"metrics":{"accuracy":0.9}}')
    provenance = _adapter_provenance(
        artifact,
        adapter_id="claimci-json-v1",
        selector="/metrics/accuracy",
        inferred=True,
    )
    mapping = _mapping(
        artifact,
        adapter_id="claimci-json-v1",
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/metrics/accuracy",
        inferred=True,
    )

    assert provenance.kind is ProvenanceKind.ADAPTER_EXTRACTION
    assert provenance.source_path == artifact.candidate.path
    assert artifact.candidate.sha256 in provenance.detail
    assert "/metrics/accuracy" in provenance.detail
    assert mapping.selector.expression == "/metrics/accuracy"
    assert _config_target("training.batch_size") == "config.k7ea1ae97f8053097"
    with pytest.raises(FrozenInstanceError):
        mapping.target_field = "changed"  # type: ignore[misc]


def test_supported_kind_and_suffix_pairs_are_exact_and_case_insensitive() -> None:
    artifact = _passive(b"{}", path="RESULTS.JSON")

    assert _supports(
        artifact,
        kinds=frozenset({ArtifactKind.RESULTS}),
        suffixes=frozenset({".json"}),
    )
    assert not _supports(
        artifact,
        kinds=frozenset({ArtifactKind.CONFIG}),
        suffixes=frozenset({".json"}),
    )


def test_match_validation_rejects_adapter_path_kind_and_unused_target_mismatch() -> None:
    artifact = _passive(b'{"accuracy":0.9}')
    mapping = _mapping(
        artifact,
        adapter_id="claimci-json-v1",
        target_field="metric_value",
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/accuracy",
        inferred=False,
    )
    match = AdapterMatch(
        adapter_id="claimci-json-v1",
        path=artifact.candidate.path,
        confidence=Confidence(0.9),
        mappings=(mapping,),
        match_evidence=(mapping.provenance,),
    )

    validated = _validate_match(
        artifact,
        match,
        adapter_id="claimci-json-v1",
        selector_kind=SelectorKind.JSON_POINTER,
        allowed_targets=frozenset({"metric_value"}),
    )
    assert validated == {"metric_value": mapping}

    wrong_id = AdapterMatch(
        adapter_id="claimci-jsonl-v1",
        path=match.path,
        confidence=match.confidence,
        mappings=match.mappings,
        match_evidence=match.match_evidence,
    )
    with pytest.raises(AdapterSelectorError, match="adapter"):
        _validate_match(
            artifact,
            wrong_id,
            adapter_id="claimci-json-v1",
            selector_kind=SelectorKind.JSON_POINTER,
            allowed_targets=frozenset({"metric_value"}),
        )

    wrong_path_mapping = FieldMapping(
        target_field="metric_value",
        selector=EvidenceSelector(
            SelectorKind.JSON_POINTER,
            "/accuracy",
            FieldProvenance(
                ProvenanceKind.ADAPTER_EXTRACTION,
                "external mapping",
                RepositoryPath("other.json"),
                "external",
            ),
        ),
        provenance=mapping.provenance,
    )
    wrong_path = AdapterMatch(
        adapter_id=match.adapter_id,
        path=match.path,
        confidence=match.confidence,
        mappings=(wrong_path_mapping,),
        match_evidence=match.match_evidence,
    )
    with pytest.raises(AdapterSelectorError, match="path|provenance"):
        _validate_match(
            artifact,
            wrong_path,
            adapter_id="claimci-json-v1",
            selector_kind=SelectorKind.JSON_POINTER,
            allowed_targets=frozenset({"metric_value"}),
        )


def test_config_target_is_collision_stable_for_distinct_selectors() -> None:
    targets = {
        _config_target("model.name"),
        _config_target("model_name"),
        _config_target("Model.name"),
    }
    assert len(targets) == 3
