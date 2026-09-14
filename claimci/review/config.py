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
_COMMON_LIMIT_FIELDS = {
    "max_calls",
    "max_context_chars",
    "max_output_chars",
    "max_files",
    "max_file_chars",
    "timeout_seconds",
}
_TASK_OUTPUT_LIMIT_FIELDS = {
    "max_claims",
    "extraction_max_output_tokens",
    "synthesis_max_output_tokens",
}
_LEGACY_OUTPUT_LIMIT_FIELD = "max_output_tokens_per_call"
_LIMIT_FIELDS = (
    _COMMON_LIMIT_FIELDS
    | _TASK_OUTPUT_LIMIT_FIELDS
    | {_LEGACY_OUTPUT_LIMIT_FIELD}
)


def _resolve_config_path(repository_root: Path, config_path: Path | None) -> Path:
    try:
        root = Path(repository_root).resolve()
        candidate = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        resolved.relative_to(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"review configuration path is invalid or outside repository: {exc}") from exc
    return resolved


def _normalize_limits(
    raw_limits: Mapping[object, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    fields = set(raw_limits)
    unknown_limits = fields - _LIMIT_FIELDS
    if unknown_limits:
        raise ReviewError(
            f"review limits have unknown fields: {sorted(map(str, unknown_limits))}"
        )

    uses_legacy = _LEGACY_OUTPUT_LIMIT_FIELD in fields
    task_fields = fields & _TASK_OUTPUT_LIMIT_FIELDS
    if uses_legacy and task_fields:
        raise ReviewError(
            "legacy max_output_tokens_per_call cannot be combined with "
            "task-specific output limits"
        )

    if enabled:
        required = _COMMON_LIMIT_FIELDS | (
            {_LEGACY_OUTPUT_LIMIT_FIELD}
            if uses_legacy
            else _TASK_OUTPUT_LIMIT_FIELDS
        )
        missing_limits = required - fields
        if missing_limits:
            raise ReviewError(
                f"enabled review limits are missing fields: {sorted(missing_limits)}"
            )

    normalized = dict(raw_limits)
    if uses_legacy:
        legacy_value = normalized.pop(_LEGACY_OUTPUT_LIMIT_FIELD)
        normalized.setdefault("max_claims", 16)
        normalized["extraction_max_output_tokens"] = legacy_value
        normalized["synthesis_max_output_tokens"] = legacy_value
    return normalized


def load_review_config(
    repository_root: Path,
    config_path: Path | None = None,
    *,
    content: bytes | None = None,
) -> ReviewConfig:
    """Load trusted review configuration; absence is always disabled.

    ``content`` lets a caller parse descriptor-captured bytes so effective values
    and provenance can be bound to one immutable snapshot instead of rereading a
    pathname.
    """

    path = _resolve_config_path(repository_root, config_path)
    if content is None and not path.exists():
        return ReviewConfig()
    try:
        if content is None:
            text = path.read_text(encoding="utf-8")
        else:
            if not isinstance(content, bytes):
                raise TypeError("captured review configuration must be bytes")
            text = content.decode("utf-8", errors="strict")
        payload = load_unique_yaml(text)
    except (
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
        RecursionError,
    ) as exc:
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
    normalized_limits = _normalize_limits(raw_limits, enabled=enabled)
    try:
        limits = ReviewLimits(**normalized_limits)
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
