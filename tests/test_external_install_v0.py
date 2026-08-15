"""Focused contracts for the external-install v0 carrier and renderer."""

from __future__ import annotations

import subprocess
import sys
import contextlib
import io
import importlib.util
import runpy
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "action.yml"
TEMPLATE = ROOT / "templates" / "claimci-external.yml.tmpl"
ACTIVE_WORKFLOW = ROOT / ".github" / "workflows" / "claimci-external.yml"
RENDERER = ROOT / "scripts" / "render_external_workflow.py"
INSTALLER_SHA = "1b85419c4beaf7052732339fd80fb9242e75ef06"
MATERIALIZER = ROOT / "scripts" / "materialize_external_demo.py"
DEMO_SHA = "abcdef0123456789abcdef0123456789abcdef01"


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


def _run_materializer(*args: str) -> subprocess.CompletedProcess[str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(MATERIALIZER), *args]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(str(MATERIALIZER), run_name="__main__")
            except SystemExit as exc:
                code = exc.code
                returncode = code if isinstance(code, int) else 0
            else:
                returncode = 0
    finally:
        sys.argv = old_argv
    return subprocess.CompletedProcess(
        [sys.executable, str(MATERIALIZER), *args],
        returncode,
        stdout.getvalue(),
        stderr.getvalue(),
    )


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_renderer_rejects_tampered_template_refs_and_stray_marker_comments(tmp_path: Path):
    module = _load_script_module(RENDERER, "claimci_external_renderer_tampered")
    tampered = TEMPLATE.read_text(encoding="utf-8")
    tampered = tampered.replace(
        "amebaleon/ClaimCI@__CLAIMCI_INSTALLER_SHA__",
        "amebaleon/ClaimCI@main",
        1,
    )
    tampered += "\n# __CLAIMCI_INSTALLER_SHA__\n"
    module.TEMPLATE = tmp_path / "tampered.yml"
    module.TEMPLATE.write_text(tampered, encoding="utf-8")
    output = tmp_path / "rendered.yml"

    with pytest.raises(ValueError):
        module.render(INSTALLER_SHA, output)
    assert not output.exists()

    extra_path = TEMPLATE.read_text(encoding="utf-8").replace(
        "amebaleon/ClaimCI@__CLAIMCI_INSTALLER_SHA__",
        "amebaleon/ClaimCI/nested@__CLAIMCI_INSTALLER_SHA__",
        1,
    )
    module.TEMPLATE.write_text(extra_path, encoding="utf-8")
    with pytest.raises(ValueError):
        module.render(INSTALLER_SHA, output)
    assert not output.exists()


def test_materializer_uses_external_fixture_root_and_preserves_source_bytes(tmp_path: Path):
    module = _load_script_module(MATERIALIZER, "claimci_external_materializer_source")
    assert module.FIXTURE_ROOT == ROOT / "examples" / "external_install"

    output = tmp_path / "consumer"
    result = _run_materializer("--workflow-sha", DEMO_SHA, "--output", str(output))
    assert result.returncode == 0, result.stderr
    for name in module.FIXTURE_FILES:
        assert (output / name).read_bytes() == (module.FIXTURE_ROOT / name).read_bytes()


def test_materializer_rejects_malformed_or_mutable_customer_template(tmp_path: Path):
    module = _load_script_module(MATERIALIZER, "claimci_external_materializer_template")
    valid = (ROOT / "examples" / "external_install" / "customer-workflow.yml.template").read_text(
        encoding="utf-8"
    )
    rendered = module._render_customer_workflow(valid, DEMO_SHA)
    assert yaml.safe_load(rendered) is not None
    assert re.findall(r"amebaleon/ClaimCI/.github/workflows/claimci-external.yml@([^\s\"']+)", rendered) == [DEMO_SHA]

    tampered = valid.replace(
        "amebaleon/ClaimCI/.github/workflows/claimci-external.yml@__CLAIMCI_WORKFLOW_SHA__",
        "amebaleon/ClaimCI/.github/workflows/claimci-external.yml@main",
    )
    tampered += "\n# __CLAIMCI_WORKFLOW_SHA__\n"
    with pytest.raises(ValueError):
        module._render_customer_workflow(tampered, DEMO_SHA)

    wrong_kind = valid.replace(
        "amebaleon/ClaimCI/.github/workflows/claimci-external.yml@__CLAIMCI_WORKFLOW_SHA__",
        "amebaleon/ClaimCI@__CLAIMCI_WORKFLOW_SHA__",
    )
    with pytest.raises(ValueError):
        module._render_customer_workflow(wrong_kind, DEMO_SHA)

    with pytest.raises(ValueError):
        module._render_customer_workflow("not: [valid", DEMO_SHA)


def test_materializer_rejects_symlinked_allowlisted_fixture_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_script_module(MATERIALIZER, "claimci_external_materializer_symlink")
    fixture_root = tmp_path / "claimci" / "examples" / "external_install"
    fixture_root.mkdir(parents=True)
    for name in module.FIXTURE_FILES:
        (fixture_root / name).write_text(name, encoding="utf-8")
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    symlink = fixture_root / module.FIXTURE_FILES[0]
    symlink.unlink()
    try:
        symlink.symlink_to(target)
    except (OSError, NotImplementedError):
        symlink.write_text("outside", encoding="utf-8")
        monkeypatch.setattr(module.Path, "is_symlink", lambda path: path == symlink)

    module.ROOT = tmp_path / "claimci"
    module.FIXTURE_ROOT = fixture_root
    template_path = module.ROOT / "examples" / "external_install" / "customer-workflow.yml.template"
    template_path.write_text(
        "uses: amebaleon/ClaimCI/.github/workflows/claimci-external.yml@__CLAIMCI_WORKFLOW_SHA__\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module.materialize(DEMO_SHA, tmp_path / "consumer")


def test_external_install_docs_warn_about_fork_head_provider_egress():
    expected = "pull_request_target fork-authored head content may be sent to the configured provider"
    for path in (ROOT / "docs" / "external-install-v0.md", ROOT / "README.md"):
        assert expected in path.read_text(encoding="utf-8")


def _active_text() -> str:
    return ACTIVE_WORKFLOW.read_text(encoding="utf-8")


def _active_mapping() -> dict[str, object]:
    parsed = yaml.safe_load(_active_text())
    assert isinstance(parsed, dict)
    return parsed


def _active_on(workflow: dict[str, object]) -> dict[str, object]:
    value = workflow.get("on", workflow.get(True))
    assert isinstance(value, dict)
    return value


def test_active_workflow_is_rendered_at_authorized_sha_and_has_exact_call_contract():
    text = _active_text()
    workflow = _active_mapping()
    assert "__CLAIMCI_INSTALLER_SHA__" not in text
    refs = re.findall(r"amebaleon/ClaimCI@([^\s\"']+)", text)
    assert refs and refs == [INSTALLER_SHA] * len(refs)
    on = _active_on(workflow)
    call = on["workflow_call"]
    assert call["inputs"]["manifest-path"] == {
        "description": "Relative path to the research manifest in the consumer PR.",
        "required": False,
        "type": "string",
        "default": "research.yaml",
    }
    assert call["inputs"]["review-config-path"]["default"] == ".claimci/review.yaml"
    assert call["secrets"]["OPENAI_API_KEY"]["required"] is False
    assert "secrets: inherit" not in text


def test_active_workflow_separates_audit_review_and_neutral_publication():
    text = _active_text()
    jobs = _active_mapping()["jobs"]
    assert set(jobs) == {"claimci_audit", "research_review", "publish_research_review"}
    assert jobs["claimci_audit"]["permissions"] == {"contents": "read", "checks": "write"}
    assert jobs["research_review"]["permissions"] == {"contents": "read"}
    assert jobs["publish_research_review"]["permissions"] == {"checks": "write"}

    assert text.count("claimci review") == 1
    assert "--json-output claimci-review.json" in text
    assert "--markdown-output claimci-review.md" in text
    assert "--base-root consumer-base" in text
    assert "--config-root consumer-base" in text
    assert '"pull-request/$CLAIMCI_MANIFEST"' in text
    assert "--artifact-root pull-request" in text
    assert "conclusion:\"neutral\"" in text or 'conclusion: "neutral"' in text
    assert "summary_bytes" in text and "60000" in text


def test_review_handoff_is_bounded_before_and_after_base64_and_requires_valid_shape():
    text = _active_text()
    assert "payload_b64" in text
    assert "${#payload_b64}" in text
    assert "100000" in text
    assert "decode_status" in text
    assert "payload_bytes" in text
    assert "decode_status" in text and "-ne 0" in text
    assert '(keys | sort) == (["conclusion", "details_url", "head_sha", "name", "output", "status"] | sort)' in text
    assert '(.output | keys | sort) == (["summary", "title"] | sort)' in text
    assert ".output.summary | type == \"string\"" in text


def test_review_dependencies_are_probed_from_trusted_base_and_installed_advisory_only():
    text = _active_text()
    assert "load_review_config" in text
    assert "consumer-base" in text
    assert "review_config" in text
    assert "install-review-dependencies: false" in text
    assert "install-review-dependencies: true" in text
    assert "steps.review_config.outputs.enabled" in text
    assert "continue-on-error: true" in text


def test_advisory_review_setup_and_publication_cannot_fail_authoritative_audit():
    workflow = _active_mapping()
    jobs = workflow["jobs"]
    assert jobs["research_review"].get("continue-on-error") is True or "continue-on-error: true" in _active_text()
    assert jobs["publish_research_review"].get("continue-on-error") is True


def test_active_workflow_uses_explicit_base_head_checkouts_and_never_installs_consumer_code():
    text = _active_text()
    assert "github.event.pull_request.base.sha" in text
    assert text.count("github.event.pull_request.head.sha") >= 2
    assert text.count("persist-credentials: false") >= 3
    assert "path: consumer-base" in text
    assert "path: pull-request" in text
    assert "allow-unsafe-pr-checkout: true" in text

    lowered = text.casefold()
    for forbidden in (
        "pip install ./pull-request",
        "pip install -e ./pull-request",
        "python -m pip install pull-request",
        "import pull-request",
        "source pull-request/",
        "bash pull-request/",
    ):
        assert forbidden.casefold() not in lowered


def test_active_workflow_is_explicit_about_provider_secret_and_checks_credentials():
    text = _active_text()
    jobs = _active_mapping()["jobs"]
    review = jobs["research_review"]
    publisher = jobs["publish_research_review"]
    assert "OPENAI_API_KEY" in str(review)
    assert "OPENAI_API_KEY" not in str(jobs["claimci_audit"])
    assert "OPENAI_API_KEY" not in str(publisher)
    assert "gh api" in str(publisher)
    assert "GH_TOKEN" in str(publisher)
    assert "ClaimCI Audit" in text
    assert "ClaimCI Research Review" in text


def test_materializer_creates_clean_consumer_fixture_and_known_audit(tmp_path: Path):
    output = tmp_path / "consumer"
    result = _run_materializer("--workflow-sha", DEMO_SHA, "--output", str(output))
    assert result.returncode == 0, result.stderr
    expected = {
        ".github/workflows/claimci.yml",
        "research.yaml",
        "baseline-config.yaml",
        "baseline-results.json",
        "baseline-train.jsonl",
        "baseline-eval.jsonl",
        "candidate-config.yaml",
        "candidate-results.json",
        "candidate-train.jsonl",
        "candidate-eval.jsonl",
    }
    actual = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert not (output / "claimci").exists()
    assert not (output / "pyproject.toml").exists()
    assert not (output / "action.yml").exists()

    workflow_text = (output / ".github/workflows/claimci.yml").read_text(encoding="utf-8")
    assert f"amebaleon/ClaimCI/.github/workflows/claimci-external.yml@{DEMO_SHA}" in workflow_text
    assert "permissions: {}" in workflow_text
    assert "OPENAI_API_KEY:" in workflow_text
    manifest = yaml.safe_load((output / "research.yaml").read_text(encoding="utf-8"))
    for arm in ("baseline", "candidate"):
        for path_value in manifest[arm].values():
            assert isinstance(path_value, str)
            assert not Path(path_value).is_absolute()
            assert ".." not in Path(path_value).parts

    from claimci.audit import audit_research
    from claimci.models import Verdict

    audit = audit_research(output / "research.yaml", artifact_root=output)
    assert audit.verdict is Verdict.NOT_SUPPORTED
    assert {
        "CONFIG.COMPUTE_MISMATCH",
        "DATASET.EXACT_LEAKAGE",
        "RESULT.CLAIM_SUPPORTED",
        "SEED.SINGLE_RUN",
    } <= {finding.rule_id for finding in audit.findings}


def test_materializer_enable_review_writes_only_trusted_disabled_config(tmp_path: Path):
    output = tmp_path / "consumer"
    result = _run_materializer(
        "--workflow-sha", DEMO_SHA, "--output", str(output), "--enable-review"
    )
    assert result.returncode == 0, result.stderr
    config = output / ".claimci/review.yaml"
    assert config.is_file()
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert payload["enabled"] is False


@pytest.mark.parametrize("sha", ["", "0" * 40, "g" * 40, "A" * 40, "1" * 39])
def test_materializer_rejects_bad_sha_without_writing(tmp_path: Path, sha: str):
    output = tmp_path / "consumer"
    result = _run_materializer("--workflow-sha", sha, "--output", str(output))
    assert result.returncode != 0
    assert not output.exists()


def test_materializer_rejects_nonempty_or_unsafe_output(tmp_path: Path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    result = _run_materializer("--workflow-sha", DEMO_SHA, "--output", str(occupied))
    assert result.returncode != 0
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"

    nul = _run_materializer(
        "--workflow-sha", DEMO_SHA, "--output", str(tmp_path / "bad\0path")
    )
    assert nul.returncode != 0
