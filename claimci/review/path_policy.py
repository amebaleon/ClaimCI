"""Explicit source-path policy shared by review collection and routing."""

from __future__ import annotations

from pathlib import PurePosixPath

from .models import ReviewMaterialKind


SOURCE_SUFFIXES = frozenset({".py", ".js", ".ts", ".rs", ".go", ".java"})
_TEST_SEGMENTS = frozenset({"test", "tests"})
_MIGRATION_SEGMENTS = frozenset({"migration", "migrations"})
_SUBMISSION_SHELL_TOKENS = frozenset(
    {"submit", "submission", "job", "launch", "eval", "evaluation", "benchmark"}
)
_CONFIG_SUFFIXES = frozenset({".cfg", ".conf", ".ini", ".toml", ".yaml", ".yml"})
_RESULT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".tsv"})
_GENERIC_EVIDENCE_SUFFIXES = frozenset(
    {".sql", *_CONFIG_SUFFIXES, *_RESULT_SUFFIXES}
)
_BENCHMARK_SUFFIXES = frozenset(
    {".txt", *_GENERIC_EVIDENCE_SUFFIXES, *SOURCE_SUFFIXES}
)
_MANIFEST_SUFFIXES = frozenset(
    {".in", ".lock", ".txt", *_CONFIG_SUFFIXES, *_RESULT_SUFFIXES}
)
_CONFIG_MATERIAL_SUFFIXES = frozenset({".json", *_CONFIG_SUFFIXES})


def is_source_file(path: str) -> bool:
    """Return whether *path* has one of the review's supported source suffixes."""

    portable = path.replace("\\", "/")
    return PurePosixPath(portable).suffix.casefold() in SOURCE_SUFFIXES


def is_test_source_file(path: str) -> bool:
    """Return whether a supported source file is located in a test path."""

    portable = path.replace("\\", "/").casefold()
    relative = PurePosixPath(portable)
    return is_source_file(portable) and (
        bool(set(relative.parts) & _TEST_SEGMENTS)
        or relative.name.startswith("test_")
    )


def _name_tokens(path: PurePosixPath) -> frozenset[str]:
    tokenized = "".join(
        character if character.isalnum() else " "
        for character in path.stem.casefold()
    )
    return frozenset(tokenized.split())


def classify_review_material(path: str) -> ReviewMaterialKind:
    """Classify one repository path without consulting or reading its content."""

    portable = path.replace("\\", "/")
    relative = PurePosixPath(portable)
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    name = relative.name.casefold()
    suffix = relative.suffix.casefold()
    tokens = _name_tokens(relative)

    if name in {"research.yaml", "research.yml"}:
        return ReviewMaterialKind.MANIFEST
    if suffix in {".sh", ".bash"}:
        if tokens & _SUBMISSION_SHELL_TOKENS:
            return ReviewMaterialKind.SUBMISSION_CONFIG
        return ReviewMaterialKind.OTHER
    if suffix in {".md", ".markdown"} or name == "paper.tex":
        return ReviewMaterialKind.DOCUMENT
    if is_test_source_file(portable):
        return ReviewMaterialKind.TEST
    if (
        any(part in {"benchmark", "benchmarks", "bench"} for part in lowered_parts)
        or tokens & {"benchmark", "bench"}
    ) and suffix in _BENCHMARK_SUFFIXES:
        return ReviewMaterialKind.BENCHMARK
    if (
        (
            any(
                part in {"result", "results", "output", "outputs", "metrics"}
                for part in lowered_parts
            )
            or tokens & {"result", "results", "metric", "metrics", "score", "scores"}
        )
        and suffix in _RESULT_SUFFIXES
    ):
        return ReviewMaterialKind.RESULT
    if (
        (
            any(part in {"manifest", "manifests"} for part in lowered_parts)
            or tokens & {"manifest", "requirements", "lockfile"}
        )
        and suffix in _MANIFEST_SUFFIXES
    ) or suffix == ".lock":
        return ReviewMaterialKind.MANIFEST
    if suffix in _CONFIG_SUFFIXES or (
        (
            any(part in {"config", "configs", "configuration"} for part in lowered_parts)
            or tokens & {"config", "configs", "configuration"}
        )
        and suffix in _CONFIG_MATERIAL_SUFFIXES
    ):
        return ReviewMaterialKind.CONFIG
    if suffix == ".sql" and bool(set(lowered_parts) & _MIGRATION_SEGMENTS):
        return ReviewMaterialKind.SOURCE
    if is_source_file(portable):
        return ReviewMaterialKind.SOURCE
    return ReviewMaterialKind.OTHER


def is_supported_review_evidence_path(path: str) -> bool:
    """Return whether a path has an approved passive review representation."""

    if classify_review_material(path) is not ReviewMaterialKind.OTHER:
        return True
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.casefold()
    return suffix in _GENERIC_EVIDENCE_SUFFIXES
