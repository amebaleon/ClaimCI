"""Strict, default-disabled repository configuration for research review."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from claimci.parsing import load_unique_yaml

from .models import ReviewConfig, ReviewError, ReviewLimits


DEFAULT_CONFIG_PATH = Path(".claimci/review.yaml")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "enabled",
    "policy",
    "provider",
    "model",
    "limits",
}
_LIMIT_FIELDS = {
    "max_calls",
    "max_context_chars",
    "max_output_chars",
    "max_files",
    "max_file_chars",
    "max_output_tokens_per_call",
    "timeout_seconds",
}


def _resolve_config_path(repository_root: Path, config_path: Path | None) -> Path:
    try:
        root = Path(repository_root).resolve()
        candidate = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        resolved.relative_to(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"review configuration path is invalid or outside repository: {exc}") from exc
    return resolved


def load_review_config(
    repository_root: Path,
    config_path: Path | None = None,
) -> ReviewConfig:
    """Load trusted review configuration; absence is always disabled."""

    path = _resolve_config_path(repository_root, config_path)
    if not path.exists():
        return ReviewConfig()
    try:
        payload = load_unique_yaml(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, RecursionError) as exc:
        raise ReviewError(f"could not load review configuration {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReviewError("review configuration root must be a mapping")
    unknown = set(payload) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ReviewError(f"review configuration has unknown fields: {sorted(map(str, unknown))}")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ReviewError("review configuration schema_version must be 1")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise ReviewError("review configuration enabled must be a boolean")

    if enabled:
        missing = _TOP_LEVEL_FIELDS - set(payload)
        if missing:
            raise ReviewError(f"enabled review configuration is missing fields: {sorted(missing)}")

    policy = payload.get("policy", "advisory")
    provider = payload.get("provider", "openai")
    configured_model = payload.get("model", "gpt-5.6-terra")

    raw_limits = payload.get("limits", {})
    if not isinstance(raw_limits, Mapping):
        raise ReviewError("review configuration limits must be a mapping")
    unknown_limits = set(raw_limits) - _LIMIT_FIELDS
    if unknown_limits:
        raise ReviewError(f"review limits have unknown fields: {sorted(map(str, unknown_limits))}")
    if enabled:
        missing_limits = _LIMIT_FIELDS - set(raw_limits)
        if missing_limits:
            raise ReviewError(f"enabled review limits are missing fields: {sorted(missing_limits)}")
    try:
        limits = ReviewLimits(**dict(raw_limits))
        configured = ReviewConfig(
            schema_version=payload["schema_version"],
            enabled=enabled,
            policy=policy,
            provider=provider,
            model=configured_model,
            limits=limits,
        )
        if enabled and configured.provider == "openai":
            override = os.environ.get("CLAIMCI_OPENAI_MODEL")
            if override is not None:
                return ReviewConfig(
                    schema_version=configured.schema_version,
                    enabled=configured.enabled,
                    policy=configured.policy,
                    provider=configured.provider,
                    model=override,
                    limits=configured.limits,
                )
        return configured
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewError(f"invalid review configuration: {exc}") from exc
