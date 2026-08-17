"""Fixed trusted registry for built-in ClaimCI adapters."""

from __future__ import annotations

import re

from claimci.analysis import Adapter

from .config import TomlConfigAdapter, YamlConfigAdapter
from .core import AdapterSelectorError
from .native import NativeConfigAdapter, NativeManifestAdapter, NativeResultsAdapter
from .structured import JsonAdapter, JsonLinesAdapter
from .tabular import CsvAdapter


_REGISTRY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")

ADAPTERS = (
    NativeManifestAdapter(),
    NativeResultsAdapter(),
    NativeConfigAdapter(),
    JsonAdapter(),
    JsonLinesAdapter(),
    CsvAdapter(),
    YamlConfigAdapter(),
    TomlConfigAdapter(),
)


def get_adapter(adapter_id: str) -> Adapter:
    """Return one built-in adapter by exact data identifier."""

    if not isinstance(adapter_id, str) or not _REGISTRY_ID.fullmatch(adapter_id):
        raise AdapterSelectorError("adapter identifier is not canonical registry data")
    for adapter in ADAPTERS:
        if adapter.adapter_id == adapter_id:
            return adapter
    raise KeyError(adapter_id)


__all__ = ["ADAPTERS", "get_adapter"]
