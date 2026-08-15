#!/usr/bin/env python3
"""Materialize the self-contained ClaimCI external-install demo repository."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePath

import yaml


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "examples" / "external_install"
OUTPUT_MARKER = "__CLAIMCI_WORKFLOW_SHA__"
SHA_RE = re.compile(r"[0-9a-f]{40}")
CUSTOMER_REF_RE = re.compile(
    r"amebaleon/ClaimCI/\.github/workflows/claimci-external\.yml@([^\s\"']+)"
)
ANY_CLAIMCI_REF_RE = re.compile(r"amebaleon/ClaimCI[^@\s\"']*@([^\s\"']+)")
FIXTURE_FILES = (
    "research.yaml",
    "baseline-config.yaml",
    "baseline-results.json",
    "baseline-train.jsonl",
    "baseline-eval.jsonl",
    "candidate-config.yaml",
    "candidate-results.json",
    "candidate-train.jsonl",
    "candidate-eval.jsonl",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enable-review", action="store_true")
    return parser


def _validate_output(path: Path) -> Path:
    raw = str(path)
    if "\x00" in raw:
        raise ValueError("output path must not contain NUL")
    if any(part == ".." for part in PurePath(raw.replace("\\", "/")).parts):
        raise ValueError("output path must not contain parent traversal")
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("output path must be outside the ClaimCI repository")
    if path.exists():
        if not path.is_dir():
            raise ValueError("output must be an absent or empty directory")
        if any(path.iterdir()):
            raise ValueError("output directory must be empty")
    elif not path.parent.exists():
        raise ValueError("output parent directory must exist")
    return path


def _validate_sha(value: str) -> None:
    if SHA_RE.fullmatch(value) is None or not value.strip("0"):
        raise ValueError("workflow-sha must be a non-zero, lowercase 40-hex commit SHA")


def _render_customer_workflow(template: str, workflow_sha: str) -> str:
    if template.count(OUTPUT_MARKER) != 1:
        raise ValueError("customer workflow template must contain exactly one SHA marker")
    try:
        parsed = yaml.safe_load(template)
    except yaml.YAMLError as exc:
        raise ValueError(f"customer workflow template is invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("customer workflow template must be a YAML mapping")
    refs = CUSTOMER_REF_RE.findall(template)
    all_refs = ANY_CLAIMCI_REF_RE.findall(template)
    marker_lines = [line for line in template.splitlines() if OUTPUT_MARKER in line]
    if refs != [OUTPUT_MARKER] or all_refs != [OUTPUT_MARKER] or marker_lines != [
        line for line in marker_lines if "uses:" in line
    ]:
        raise ValueError("customer workflow template must contain exactly one marked ClaimCI reusable-workflow ref")
    rendered = template.replace(OUTPUT_MARKER, workflow_sha)
    if OUTPUT_MARKER in rendered:
        raise ValueError("failed to replace customer workflow SHA marker")
    try:
        rendered_document = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise ValueError(f"rendered customer workflow is invalid YAML: {exc}") from exc
    if not isinstance(rendered_document, dict):
        raise ValueError("rendered customer workflow must be a YAML mapping")
    rendered_refs = CUSTOMER_REF_RE.findall(rendered)
    rendered_all_refs = ANY_CLAIMCI_REF_RE.findall(rendered)
    if rendered_refs != [workflow_sha] or rendered_all_refs != [workflow_sha]:
        raise ValueError("rendered customer workflow must contain exactly one supplied ClaimCI SHA")
    return rendered


def _require_regular_fixture_source(source: Path) -> None:
    try:
        mode = source.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"missing demo fixture source: {source}") from exc
    if source.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"demo fixture source must be a regular non-symlink file: {source}")


def materialize(workflow_sha: str, output: Path, *, enable_review: bool = False) -> None:
    _validate_sha(workflow_sha)
    output = _validate_output(output)
    template_path = ROOT / "examples" / "external_install" / "customer-workflow.yml.template"
    template = template_path.read_text(encoding="utf-8")
    rendered_workflow = _render_customer_workflow(template, workflow_sha)

    stage = Path(tempfile.mkdtemp(prefix="claimci-demo-", dir=str(output.parent)))
    try:
        for name in FIXTURE_FILES:
            source = FIXTURE_ROOT / name
            _require_regular_fixture_source(source)
            destination = stage / name
            shutil.copyfile(source, destination)

        workflow_path = stage / ".github" / "workflows" / "claimci.yml"
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_path.write_text(rendered_workflow, encoding="utf-8", newline="\n")

        if enable_review:
            review_path = stage / ".claimci" / "review.yaml"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_source = ROOT / "examples" / "external_install" / "review-disabled.yaml"
            _require_regular_fixture_source(review_source)
            review_path.write_text(
                review_source.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )

        if output.exists():
            output.rmdir()
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    try:
        materialize(args.workflow_sha, args.output, enable_review=args.enable_review)
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
