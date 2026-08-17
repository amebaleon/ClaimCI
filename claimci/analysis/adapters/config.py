"""Safe structural YAML and TOML configuration adapters."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping

import yaml

from claimci.analysis import (
    AdapterMatch,
    ArtifactKind,
    Confidence,
    ConfigValue,
    ExperimentRole,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    SelectorKind,
)
from claimci.parsing import load_unique_yaml

from .core import (
    AdapterError,
    AdapterParseError,
    AdapterSelectorError,
    _adapter_provenance,
    _config_target,
    _decode_utf8,
    _evidence_id,
    _mapping,
    _resolve_dotted_path,
    _scalar_leaves,
    _supports,
    _validate_graph,
    _validate_match,
    _verify_integrity,
)


_SCALAR_TYPES = (str, bool, int, float, type(None))


def _validate_config_values(value: object, *, path: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = str(key) if path == "<root>" else f"{path}.{key}"
            _validate_config_values(item, path=child)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        raise AdapterParseError(
            f"config sequence or array at {path!r} is not supported in v1"
        )
    if not isinstance(value, _SCALAR_TYPES):
        raise AdapterParseError(
            f"config value at {path!r} is not a supported JSON scalar"
        )


def _parse_yaml(content: bytes) -> Mapping[str, object]:
    text = _decode_utf8(content)
    try:
        value = load_unique_yaml(text)
    except yaml.YAMLError as exc:
        raise AdapterParseError(f"malformed or unsafe YAML artifact: {exc}") from exc
    _validate_graph(value)
    if not isinstance(value, Mapping):
        raise AdapterParseError("YAML config root must be a mapping")
    _validate_config_values(value)
    return value


def _parse_toml(content: bytes) -> Mapping[str, object]:
    text = _decode_utf8(content)
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AdapterParseError(f"malformed TOML artifact: {exc}") from exc
    _validate_graph(value)
    _validate_config_values(value)
    return value


class _ConfigAdapter:
    adapter_id: str
    suffixes: frozenset[str]

    def _parse(self, content: bytes) -> Mapping[str, object]:
        raise NotImplementedError

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=frozenset({ArtifactKind.CONFIG}),
            suffixes=self.suffixes,
        ):
            return None
        _verify_integrity(artifact)
        value = self._parse(artifact.content)
        leaves, unsupported = _scalar_leaves(value)
        if not leaves:
            raise AdapterParseError(
                "config has no safely representable dotted-path scalar mapping"
            )
        mappings = tuple(
            _mapping(
                artifact,
                adapter_id=self.adapter_id,
                target_field=_config_target(selector),
                selector_kind=SelectorKind.DOTTED_PATH,
                selector=selector,
                inferred=True,
            )
            for selector, _item in leaves
        )
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="<config structure>",
            inferred=bool(unsupported),
        )
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(0.7 if unsupported else 0.95),
            mappings=mappings,
            match_evidence=(provenance,),
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        value = self._parse(artifact.content)
        leaves, _unsupported = _scalar_leaves(value)
        allowed = frozenset(_config_target(selector) for selector, _item in leaves)
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.DOTTED_PATH,
            allowed_targets=allowed,
        )
        config_values: list[ConfigValue] = []
        for mapping in sorted(
            mappings.values(), key=lambda item: item.selector.expression
        ):
            selector = mapping.selector.expression
            if mapping.target_field != _config_target(selector):
                raise AdapterSelectorError(
                    "config mapping target does not match its dotted selector"
                )
            selected = _resolve_dotted_path(value, selector)
            if not isinstance(selected, _SCALAR_TYPES):
                raise AdapterSelectorError("config selector must resolve to a scalar")
            config_values.append(
                ConfigValue(selector, selected, mapping.provenance)
            )
        if not config_values:
            raise AdapterSelectorError("config extraction requires a validated mapping")
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="<selected config mappings>",
            inferred=False,
        )
        observation = NormalizedObservation(
            provenance=provenance,
            experiment_role=ExperimentRole.UNSPECIFIED,
            config_values=tuple(config_values),
        )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=(observation,),
        )


class YamlConfigAdapter(_ConfigAdapter):
    """Safe duplicate-rejecting YAML config adapter."""

    adapter_id = "claimci-yaml-config-v1"
    suffixes = frozenset({".yaml", ".yml"})

    def _parse(self, content: bytes) -> Mapping[str, object]:
        try:
            return _parse_yaml(content)
        except AdapterError:
            raise
        except (RecursionError, ValueError, TypeError) as exc:
            raise AdapterParseError(f"malformed YAML config: {exc}") from exc


class TomlConfigAdapter(_ConfigAdapter):
    """Standard-library TOML config adapter."""

    adapter_id = "claimci-toml-config-v1"
    suffixes = frozenset({".toml"})

    def _parse(self, content: bytes) -> Mapping[str, object]:
        try:
            return _parse_toml(content)
        except AdapterError:
            raise
        except (RecursionError, ValueError, TypeError) as exc:
            raise AdapterParseError(f"malformed TOML config: {exc}") from exc


__all__ = ["YamlConfigAdapter", "TomlConfigAdapter"]
