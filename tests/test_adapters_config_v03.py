"""Safe YAML and TOML configuration adapter behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

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
from claimci.analysis.adapters.config import TomlConfigAdapter, YamlConfigAdapter
from claimci.analysis.adapters.core import (
    MAX_DEPTH,
    AdapterIntegrityError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _config_target,
    _mapping,
)


FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def _passive(
    content: bytes,
    *,
    path: str,
    kind: ArtifactKind = ArtifactKind.CONFIG,
) -> PassiveArtifact:
    return PassiveArtifact(
        ArtifactCandidate(
            path=RepositoryPath(path),
            kind=kind,
            sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
            size=len(content),
            confidence=Confidence(0.9),
            discovery_reason="config fixture",
            relevant_claim_ids=(),
            provenance=FieldProvenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                "config fixture bytes",
                RepositoryPath(path),
                "fixture",
            ),
        ),
        content,
    )


def _fixture(name: str) -> PassiveArtifact:
    return _passive((FIXTURES / name).read_bytes(), path=f"configs/{name}")


@pytest.mark.parametrize(
    "adapter, name, expected",
    [
        (
            YamlConfigAdapter(),
            "config.yaml",
            (
                ("evaluation.metric", "accuracy"),
                ("evaluation.threshold", 0.8),
                ("model.name", "tiny-transformer"),
                ("training.batch_size", 16),
                ("training.deterministic", True),
                ("training.note", None),
            ),
        ),
        (
            TomlConfigAdapter(),
            "config.toml",
            (
                ("evaluation.metric", "accuracy"),
                ("evaluation.threshold", 0.8),
                ("model.name", "tiny-transformer"),
                ("training.batch_size", 16),
                ("training.deterministic", True),
            ),
        ),
    ],
)
def test_config_adapters_emit_sorted_structural_values(
    adapter: object, name: str, expected: tuple[tuple[str, object], ...]
) -> None:
    artifact = _fixture(name)
    match = adapter.probe(artifact)  # type: ignore[attr-defined]
    assert match is not None
    selectors = tuple(item.selector.expression for item in match.mappings)
    assert selectors == tuple(key for key, _value in expected)
    assert tuple(item.target_field for item in match.mappings) == tuple(
        _config_target(key) for key, _value in expected
    )

    observation = adapter.extract(artifact, match).observations[0]  # type: ignore[attr-defined]
    assert observation.experiment_role is ExperimentRole.UNSPECIFIED
    assert tuple((item.key, item.value) for item in observation.config_values) == expected
    assert all(item.provenance.source_path == artifact.candidate.path for item in observation.config_values)
    assert all(artifact.candidate.sha256 in item.provenance.detail for item in observation.config_values)


def test_config_adapter_can_extract_validated_external_subset_only() -> None:
    artifact = _fixture("config.yaml")
    adapter = YamlConfigAdapter()
    selected = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field=_config_target("training.batch_size"),
        selector_kind=SelectorKind.DOTTED_PATH,
        selector="training.batch_size",
        inferred=False,
    )
    match = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(selected,),
        match_evidence=(selected.provenance,),
    )

    values = adapter.extract(artifact, match).observations[0].config_values
    assert tuple((item.key, item.value) for item in values) == (
        ("training.batch_size", 16),
    )


@pytest.mark.parametrize(
    "adapter, content, path, message",
    [
        (YamlConfigAdapter(), b"model:\n  name: a\n  name: b\n", "config.yaml", "duplicate"),
        (YamlConfigAdapter(), b"value: !!python/object:builtins.object {}\n", "config.yaml", "tag|constructor|YAML"),
        (YamlConfigAdapter(), b"loop: &loop [*loop]\n", "config.yaml", "alias|recursive|sequence"),
        (YamlConfigAdapter(), b"items: [1, 2]\n", "config.yaml", "sequence|array"),
        (YamlConfigAdapter(), b"1: value\n", "config.yaml", "string keys"),
        (YamlConfigAdapter(), b"value: .nan\n", "config.yaml", "finite"),
        (YamlConfigAdapter(), b"value: .inf\n", "config.yaml", "finite"),
        (YamlConfigAdapter(), b"broken: [\n", "config.yaml", "YAML|malformed"),
        (TomlConfigAdapter(), b"value=1\nvalue=2\n", "config.toml", "duplicate|TOML"),
        (TomlConfigAdapter(), b"values=[1,2]\n", "config.toml", "sequence|array"),
        (TomlConfigAdapter(), b"value=nan\n", "config.toml", "finite"),
        (TomlConfigAdapter(), b"value=inf\n", "config.toml", "finite"),
        (TomlConfigAdapter(), b"broken=[\n", "config.toml", "TOML|malformed"),
    ],
)
def test_config_adapters_reject_unsafe_malformed_or_unsupported_values(
    adapter: object, content: bytes, path: str, message: str
) -> None:
    with pytest.raises((AdapterParseError, AdapterLimitError), match=message):
        adapter.probe(_passive(content, path=path))  # type: ignore[attr-defined]


def test_dotted_keys_are_not_reinterpreted_or_silently_renamed() -> None:
    artifact = _passive(b'"model.name": "literal"\nmodel:\n  name: "nested"\n', path="config.yaml")
    adapter = YamlConfigAdapter()
    match = adapter.probe(artifact)
    assert match is not None
    assert tuple(item.selector.expression for item in match.mappings) == ("model.name",)
    assert float(match.confidence) < 0.9
    assert adapter.extract(artifact, match).observations[0].config_values[0].value == "nested"

    literal_only = _passive(b'"model.name": "literal"\n', path="literal.yaml")
    with pytest.raises(AdapterParseError, match="representable|mapping"):
        adapter.probe(literal_only)


def test_config_depth_limit_and_strict_utf8_are_controlled() -> None:
    value = "leaf: 1\n"
    for index in reversed(range(MAX_DEPTH + 1)):
        indented = "".join(f"  {line}\n" for line in value.splitlines())
        value = f"k{index}:\n{indented}"
    with pytest.raises(AdapterLimitError, match="depth"):
        YamlConfigAdapter().probe(_passive(value.encode(), path="deep.yaml"))

    with pytest.raises(AdapterParseError, match="UTF-8"):
        TomlConfigAdapter().probe(_passive(b"name=\xff", path="config.toml"))


def test_config_match_validation_rejects_selector_target_path_and_sha_substitution() -> None:
    artifact = _fixture("config.toml")
    adapter = TomlConfigAdapter()
    match = adapter.probe(artifact)
    assert match is not None

    pointer = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field=_config_target("model.name"),
        selector_kind=SelectorKind.JSON_POINTER,
        selector="/model/name",
        inferred=False,
    )
    wrong_kind = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(pointer,),
        match_evidence=(pointer.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="selector"):
        adapter.extract(artifact, wrong_kind)

    wrong_target = _mapping(
        artifact,
        adapter_id=adapter.adapter_id,
        target_field=_config_target("model.name"),
        selector_kind=SelectorKind.DOTTED_PATH,
        selector="training.batch_size",
        inferred=False,
    )
    mismatch = AdapterMatch(
        adapter_id=adapter.adapter_id,
        path=artifact.candidate.path,
        confidence=Confidence(1),
        mappings=(wrong_target,),
        match_evidence=(wrong_target.provenance,),
    )
    with pytest.raises(AdapterSelectorError, match="target|selector"):
        adapter.extract(artifact, mismatch)

    object.__setattr__(artifact, "content", artifact.content + b"\n")
    with pytest.raises(AdapterIntegrityError):
        adapter.extract(artifact, match)


def test_config_adapters_accept_only_config_kind_and_matching_suffix() -> None:
    yaml = _passive(b"value: 1\n", path="config.yaml", kind=ArtifactKind.RESULTS)
    toml = _passive(b"value=1\n", path="config.txt")
    assert YamlConfigAdapter().probe(yaml) is None
    assert TomlConfigAdapter().probe(toml) is None
