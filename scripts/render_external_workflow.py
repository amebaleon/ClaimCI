#!/usr/bin/env python3
"""Render the external workflow with an immutable ClaimCI action SHA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


MARKER = "__CLAIMCI_INSTALLER_SHA__"
SHA_RE = re.compile(r"[0-9a-f]{40}")
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "claimci-external.yml.tmpl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def render(installer_sha: str, output: Path) -> None:
    if SHA_RE.fullmatch(installer_sha) is None or not installer_sha.strip("0"):
        raise ValueError("installer-sha must be a non-zero, lowercase 40-hex commit SHA")

    template = TEMPLATE.read_text(encoding="utf-8")
    marker_count = template.count(MARKER)
    if marker_count < 1:
        raise ValueError(f"template must contain at least one {MARKER} marker")

    rendered = template.replace(MARKER, installer_sha)
    if MARKER in rendered:
        raise ValueError("failed to replace every installer marker")
    yaml.safe_load(rendered)

    output.write_text(rendered, encoding="utf-8", newline="\n")


def main() -> int:
    args = _parser().parse_args()
    try:
        render(args.installer_sha, args.output)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
