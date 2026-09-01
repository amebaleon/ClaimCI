"""Explicit source-path policy shared by review collection and routing."""

from __future__ import annotations

from pathlib import PurePosixPath


SOURCE_SUFFIXES = frozenset({".py", ".js", ".ts", ".rs", ".go", ".java"})
_TEST_SEGMENTS = frozenset({"test", "tests"})


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
