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
from .dataset import PassiveJsonLinesDatasetAdapter
from .native import NativeConfigAdapter, NativeManifestAdapter, NativeResultsAdapter
from .registry import ADAPTERS, extract_registered_artifact, get_adapter
from .structured import JsonAdapter, JsonLinesAdapter
from .streaming import (
    StreamingExtraction,
    StreamingSchemaScan,
    extract_registered_source,
    scan_jsonl_observations,
    scan_jsonl_schema,
)
from .tabular import CsvAdapter, TsvAdapter

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
    "PassiveJsonLinesDatasetAdapter",
    "StreamingExtraction",
    "StreamingSchemaScan",
    "TomlConfigAdapter",
    "TsvAdapter",
    "YamlConfigAdapter",
    "extract_registered_artifact",
    "extract_registered_source",
    "get_adapter",
    "scan_jsonl_observations",
    "scan_jsonl_schema",
]
