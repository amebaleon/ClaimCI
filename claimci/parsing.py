"""Strict deterministic parsers shared by ClaimCI artifact loaders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TextIO

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, found {node.id}",
            node.start_mark,
        )

    # Resolve YAML merge keys first. An override that would make the final
    # artifact ambiguous is deliberately treated like any other duplicate.
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


# A finite bound keeps malformed artifacts from consuming an unbounded Python
# call stack (or CPU) before the audit layer can turn the input into a concise
# ClaimCI error.  Ordinary ClaimCI manifests/configs are shallow; this limit
# is deliberately generous while still rejecting adversarial nesting.
MAX_NESTING_DEPTH = 256


def _validate_yaml_graph(
    value: object,
    *,
    depth: int = 0,
    active: set[int] | None = None,
    completed: set[int] | None = None,
) -> None:
    """Reject recursive YAML aliases and excessively deep container graphs.

    ``SafeLoader`` intentionally preserves YAML aliases, which means a value
    such as ``extra: &loop [*loop]`` otherwise becomes a self-referential
    Python list. Repeated container aliases are also rejected: although an
    acyclic alias DAG can be parsed, later comparison or rendering may expand
    it exponentially. ClaimCI artifacts are intentionally simple data trees.
    """

    if depth > MAX_NESTING_DEPTH:
        raise yaml.YAMLError(
            f"YAML nesting exceeds maximum supported depth {MAX_NESTING_DEPTH}"
        )
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return

    if active is None:
        active = set()
    if completed is None:
        completed = set()
    identity = id(value)
    if identity in active:
        raise yaml.YAMLError("recursive or cyclic YAML alias graph")
    if identity in completed:
        raise yaml.YAMLError("repeated YAML container aliases are not supported")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            children = value.items()
            for key, item in children:
                _validate_yaml_graph(
                    key, depth=depth + 1, active=active, completed=completed
                )
                _validate_yaml_graph(
                    item, depth=depth + 1, active=active, completed=completed
                )
        else:
            for item in value:
                _validate_yaml_graph(
                    item, depth=depth + 1, active=active, completed=completed
                )
        completed.add(identity)
    finally:
        active.remove(identity)


def validate_json_graph(value: object, *, label: str = "JSON") -> None:
    """Reject excessively deep decoded JSON values.

    The standard-library JSON decoder has changed its recursion behavior
    across Python versions; validating the decoded graph gives ClaimCI a
    stable, version-independent boundary.  ``ValueError`` is intentional so
    callers can wrap it with their artifact-specific :class:`ClaimCIError`.
    """

    def walk(current: object, *, depth: int, active: set[int]) -> None:
        if depth > MAX_NESTING_DEPTH:
            raise ValueError(
                f"{label} nesting exceeds maximum supported depth {MAX_NESTING_DEPTH}"
            )
        if not isinstance(current, (Mapping, list, tuple, set, frozenset)):
            return
        identity = id(current)
        if identity in active:
            raise ValueError(f"recursive or cyclic {label} value")
        active.add(identity)
        try:
            if isinstance(current, Mapping):
                for key, item in current.items():
                    walk(key, depth=depth + 1, active=active)
                    walk(item, depth=depth + 1, active=active)
            else:
                for item in current:
                    walk(item, depth=depth + 1, active=active)
        finally:
            active.remove(identity)

    walk(value, depth=0, active=set())


def load_unique_yaml(source: str | TextIO) -> object:
    """Parse safe YAML while rejecting duplicate keys at every depth."""

    try:
        parsed = yaml.load(source, Loader=UniqueKeySafeLoader)
        _validate_yaml_graph(parsed)
        return parsed
    except RecursionError as exc:
        # PyYAML may recurse while composing/constructing a deeply nested
        # document before the post-load graph validator runs.  Expose the
        # same stable YAML error type either way.
        raise yaml.YAMLError(
            f"YAML nesting exceeds maximum supported depth {MAX_NESTING_DEPTH}"
        ) from exc


def unique_json_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """JSON ``object_pairs_hook`` that rejects last-key-wins ambiguity."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
