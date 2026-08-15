"""Focused contracts for the external-install v0 carrier and renderer."""

from __future__ import annotations

import subprocess
import sys
import contextlib
import io
import runpy
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "action.yml"
TEMPLATE = ROOT / "templates" / "claimci-external.yml.tmpl"
RENDERER = ROOT / "scripts" / "render_external_workflow.py"


def _run_renderer(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the renderer's real CLI entrypoint without Windows handle races."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(RENDERER), *args]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(str(RENDERER), run_name="__main__")
            except SystemExit as exc:
                code = exc.code
                returncode = code if isinstance(code, int) else 0
            else:
                returncode = 0
    finally:
        sys.argv = old_argv
    return subprocess.CompletedProcess(
        [sys.executable, str(RENDERER), *args],
        returncode,
        stdout.getvalue(),
        stderr.getvalue(),
    )


def _action_script() -> str:
    document = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    return "\n".join(
        str(step.get("run", ""))
        for step in document.get("runs", {}).get("steps", [])
        if isinstance(step, dict)
    )


def test_root_action_is_composite_and_installs_only_from_action_path():
    document = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert document["runs"]["using"] == "composite"
    assert document["inputs"]["install-review-dependencies"]["default"] == "false"

    script = _action_script()
    assert 'CLAIMCI_ACTION_PATH: ${{ github.action_path }}' in ACTION.read_text(encoding="utf-8")
    assert '"$CLAIMCI_ACTION_PATH"' in script
    assert '"$CLAIMCI_ACTION_PATH[llm]"' in script
    install_lines = [
        line.strip()
        for line in script.splitlines()
        if line.strip().startswith("python -m pip install")
    ]
    assert install_lines == [
        'python -m pip install --disable-pip-version-check "$CLAIMCI_ACTION_PATH"',
        'python -m pip install --disable-pip-version-check "$CLAIMCI_ACTION_PATH[llm]"',
    ]
    assert "pip install" in script
    assert "pip install ." not in script
    for forbidden in ("consumer-base", "pull-request"):
        assert forbidden not in script


def test_root_action_strictly_validates_dependency_flag_and_supports_llm_extra():
    document = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert set(document["inputs"]) == {"install-review-dependencies"}
    script = _action_script()
    assert "install-review-dependencies" in script
    assert "true" in script and "false" in script
    assert "[llm]" in script
    assert "must be exactly true or false" in script.lower()


@pytest.mark.parametrize(
    "sha",
    [
        "",
        "abc",
        "A" * 40,
        "g" * 40,
        "0" * 40,
        "1" * 39,
        "1" * 41,
    ],
)
def test_renderer_rejects_malformed_or_zero_sha_without_writing(tmp_path: Path, sha: str):
    output = tmp_path / "rendered.yml"
    result = _run_renderer("--installer-sha", sha, "--output", str(output))
    assert result.returncode != 0
    assert not output.exists()


def test_renderer_requires_output_argument_and_reports_controlled_error():
    result = _run_renderer("--installer-sha", "1" * 40)
    assert result.returncode != 0
    assert "output" in (result.stderr + result.stdout).lower()


def test_template_has_exactly_one_installer_marker_in_claimci_action_reference():
    template = TEMPLATE.read_text(encoding="utf-8")
    marker = "__CLAIMCI_INSTALLER_SHA__"
    assert template.count(marker) >= 1
    marker_lines = [line for line in template.splitlines() if marker in line]
    assert all("amebaleon/ClaimCI@" in line for line in marker_lines)
    refs = re.findall(r"amebaleon/ClaimCI@([^\s\"']+)", template)
    assert refs and refs == [marker] * len(refs)
    assert yaml.safe_load(template) is not None


def test_template_is_inert_and_never_callable_from_workflows_directory():
    marker = "__CLAIMCI_INSTALLER_SHA__"
    assert TEMPLATE.parent.name == "templates"
    assert TEMPLATE.name == "claimci-external.yml.tmpl"
    workflows = ROOT / ".github" / "workflows"
    for path in workflows.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}:
            assert marker not in path.read_text(encoding="utf-8")


def test_renderer_replaces_marker_and_is_deterministic(tmp_path: Path):
    sha = "1234567890abcdef" * 2 + "12345678"
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"

    result1 = _run_renderer("--installer-sha", sha, "--output", str(first))
    result2 = _run_renderer("--installer-sha", sha, "--output", str(second))

    assert result1.returncode == 0, result1.stderr
    assert result2.returncode == 0, result2.stderr
    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    assert first_text == second_text
    assert "__CLAIMCI_INSTALLER_SHA__" not in first_text
    assert sha in first_text
    assert yaml.safe_load(first_text) is not None
