#!/usr/bin/env python3
"""Render the external workflow with an immutable ClaimCI action SHA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


MARKER = "__CLAIMCI_INSTALLER_SHA__"
SHA_RE = re.compile(r"[0-9a-f]{40}")
CLAIMCI_REF_RE = re.compile(r"amebaleon/ClaimCI@([^\s\"']+)")
ANY_CLAIMCI_REF_RE = re.compile(r"amebaleon/ClaimCI[^@\s\"']*@([^\s\"']+)")
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
    try:
        template_document = yaml.safe_load(template)
    except yaml.YAMLError as exc:
        raise ValueError(f"template is invalid YAML: {exc}") from exc
    if not isinstance(template_document, dict):
        raise ValueError("template must be a YAML mapping")
    refs = CLAIMCI_REF_RE.findall(template)
    all_refs = ANY_CLAIMCI_REF_RE.findall(template)
    marker_lines = [line for line in template.splitlines() if MARKER in line]
    if (
        not refs
        or all_refs != refs
        or any(ref != MARKER for ref in refs)
        or template.count(MARKER) != len(refs)
        or len(marker_lines) != len(refs)
        or any("uses:" not in line for line in marker_lines)
    ):
        raise ValueError("every ClaimCI action reference must use exactly the installer marker")

    rendered = template.replace(MARKER, installer_sha)
    if MARKER in rendered:
        raise ValueError("failed to replace every installer marker")
    try:
        rendered_document = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise ValueError(f"rendered workflow is invalid YAML: {exc}") from exc
    if not isinstance(rendered_document, dict):
        raise ValueError("rendered workflow must be a YAML mapping")
    rendered_refs = CLAIMCI_REF_RE.findall(rendered)
    rendered_all_refs = ANY_CLAIMCI_REF_RE.findall(rendered)
    if (
        not rendered_refs
        or rendered_all_refs != rendered_refs
        or any(ref != installer_sha for ref in rendered_refs)
    ):
        raise ValueError("every rendered ClaimCI action reference must use the supplied installer SHA")

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
