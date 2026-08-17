"""Bounded, fixed-registry adapters for passive repository artifacts."""

from .config import TomlConfigAdapter, YamlConfigAdapter
from .core import (
    MAX_ARTIFACT_BYTES,
    MAX_COLUMNS,
    MAX_DEPTH,
    MAX_LOGICAL_RECORD_BYTES,
    MAX_NODES,
    MAX_RECORDS,
    AdapterError,
    AdapterIntegrityError,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
)
from .native import NativeConfigAdapter, NativeManifestAdapter, NativeResultsAdapter
from .registry import ADAPTERS, get_adapter
from .structured import JsonAdapter, JsonLinesAdapter
from .tabular import CsvAdapter

__all__ = [
    "ADAPTERS",
    "MAX_ARTIFACT_BYTES",
    "MAX_COLUMNS",
    "MAX_DEPTH",
    "MAX_LOGICAL_RECORD_BYTES",
    "MAX_NODES",
    "MAX_RECORDS",
    "AdapterError",
    "AdapterIntegrityError",
    "AdapterLimitError",
    "AdapterParseError",
    "AdapterSelectorError",
    "CsvAdapter",
    "JsonAdapter",
    "JsonLinesAdapter",
    "NativeConfigAdapter",
    "NativeManifestAdapter",
    "NativeResultsAdapter",
    "TomlConfigAdapter",
    "YamlConfigAdapter",
    "get_adapter",
]
