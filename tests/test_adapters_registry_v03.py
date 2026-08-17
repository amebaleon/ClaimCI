"""The adapter registry is fixed, non-executable data."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from claimci.analysis.adapters import (
    ADAPTERS,
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
    CsvAdapter,
    JsonAdapter,
    JsonLinesAdapter,
    NativeConfigAdapter,
    NativeManifestAdapter,
    NativeResultsAdapter,
    TomlConfigAdapter,
    YamlConfigAdapter,
    get_adapter,
)


EXPECTED_IDS = (
    "claimci-native-manifest-v1",
    "claimci-native-results-v1",
    "claimci-native-config-v1",
    "claimci-json-v1",
    "claimci-jsonl-v1",
    "claimci-csv-v1",
    "claimci-yaml-config-v1",
    "claimci-toml-config-v1",
)


def test_registry_order_ids_instances_and_limits_are_exact() -> None:
    assert isinstance(ADAPTERS, tuple)
    assert tuple(item.adapter_id for item in ADAPTERS) == EXPECTED_IDS
    assert tuple(type(item) for item in ADAPTERS) == (
        NativeManifestAdapter,
        NativeResultsAdapter,
        NativeConfigAdapter,
        JsonAdapter,
        JsonLinesAdapter,
        CsvAdapter,
        YamlConfigAdapter,
        TomlConfigAdapter,
    )
    assert MAX_ARTIFACT_BYTES == 8 * 1024 * 1024
    assert MAX_RECORDS == 10_000
    assert MAX_COLUMNS == 256
    assert MAX_DEPTH == 64
    assert MAX_NODES == 100_000
    assert MAX_LOGICAL_RECORD_BYTES == 1024 * 1024
    assert issubclass(AdapterIntegrityError, AdapterError)
    assert issubclass(AdapterLimitError, AdapterError)
    assert issubclass(AdapterParseError, AdapterError)
    assert issubclass(AdapterSelectorError, AdapterError)


def test_registry_lookup_is_exact_and_has_no_extension_api() -> None:
    for adapter_id, adapter in zip(EXPECTED_IDS, ADAPTERS, strict=True):
        assert get_adapter(adapter_id) is adapter
    with pytest.raises(KeyError):
        get_adapter("claimci-json-v2")
    for invalid in (None, "", " claimci-json-v1", "claimci-json-v1\n", "module:Class"):
        with pytest.raises(AdapterSelectorError):
            get_adapter(invalid)  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        ADAPTERS.append(JsonAdapter())  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        ADAPTERS[0].adapter_id = "claimci-replaced-v1"  # type: ignore[misc]

    import claimci.analysis.adapters.registry as registry

    assert not hasattr(registry, "register")
    assert not hasattr(registry, "load_plugin")


def test_adapter_modules_have_no_execution_authority_or_dynamic_loading() -> None:
    package = Path(__file__).parents[1] / "claimci" / "analysis" / "adapters"
    forbidden_modules = {
        "subprocess",
        "socket",
        "pickle",
        "importlib",
        "requests",
        "httpx",
        "urllib",
        "claimci.audit",
        "claimci.review",
        "claimci.workflow",
    }
    forbidden_calls = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "system",
        "popen",
    }
    forbidden_authority = {"Verdict", "AuditResult", "RepoMapping", "UnifiedAnalysisResult"}

    for source_path in sorted(package.glob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not any(alias.name == item or alias.name.startswith(f"{item}.") for item in forbidden_modules)
                    for alias in node.names
                ), source_path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module == item or node.module.startswith(f"{item}.")
                    for item in forbidden_modules
                ), source_path
                assert not any(alias.name in forbidden_authority for alias in node.names), source_path
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    assert function.id not in forbidden_calls, source_path
                if isinstance(function, ast.Attribute):
                    assert function.attr not in {"system", "popen"}, source_path
                assert not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "load"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "yaml"
                ), source_path


def test_public_adapter_classes_implement_only_probe_and_extract_contract() -> None:
    for adapter in ADAPTERS:
        assert callable(adapter.probe)
        assert callable(adapter.extract)
        assert not hasattr(adapter, "audit")
        assert not hasattr(adapter, "review")
        assert not hasattr(adapter, "execute")
        assert not hasattr(adapter, "approve")
