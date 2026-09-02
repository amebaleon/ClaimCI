"""Static contract tests for the optional Day 3 GitHub review workflow.

These checks inspect the checked-in workflow only.  They never run Actions,
invoke ``gh``, read a repository secret, or contact an external provider.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "claimci.yml"
README = Path(__file__).parents[1] / "README.md"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _documentation_text() -> str:
    """Return the workflow plus user-facing egress/fork documentation."""

    return _workflow_text() + "\n" + README.read_text(encoding="utf-8")


def _workflow_mapping() -> dict[str, Any]:
    parsed = yaml.safe_load(_workflow_text())
    assert isinstance(parsed, dict)
    return parsed


def _jobs() -> dict[str, dict[str, Any]]:
    jobs = _workflow_mapping().get("jobs")
    assert isinstance(jobs, dict)
    assert all(isinstance(name, str) and isinstance(job, dict) for name, job in jobs.items())
    return jobs


def _job_steps(job_name: str) -> list[dict[str, Any]]:
    job = _jobs().get(job_name)
    assert isinstance(job, dict)
    steps = job.get("steps")
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _steps() -> list[dict[str, Any]]:
    return [step for job in _jobs().values() for step in job.get("steps", [])]


def _step_run(step: dict[str, Any]) -> str:
    return str(step.get("run", ""))


def _review_steps() -> list[dict[str, Any]]:
    return [step for step in _steps() if "claimci review" in _step_run(step)]


def _preflight_steps() -> list[dict[str, Any]]:
    return [step for step in _review_steps() if "--preflight-only" in _step_run(step)]


def _paid_review_steps() -> list[dict[str, Any]]:
    return [step for step in _review_steps() if "--preflight-only" not in _step_run(step)]


def _audit_steps() -> list[dict[str, Any]]:
    return [step for step in _job_steps("claimci") if "claimci audit" in _step_run(step)]


def test_review_runs_one_free_preflight_then_one_paid_invocation_with_shared_views() -> None:
    """Preflight and paid review each render both views without duplicate orchestration."""

    steps = _review_steps()
    assert len(steps) == 2
    assert len(_preflight_steps()) == 1
    assert len(_paid_review_steps()) == 1
    for step in steps:
        run = _step_run(step)
        assert len(re.findall(r"(?m)^\s*claimci review(?:\s|\\|$)", run)) == 1
        lower = run.casefold()
        assert "json" in lower and "markdown" in lower
        assert re.search(r"[^\s'\"]+\.json(?:\s|\\|$)", run)
        assert re.search(r"[^\s'\"]+\.md(?:\s|\\|$)", run)
        assert lower.count("claimci review") == 1


def test_review_reads_opt_in_only_from_trusted_base_checkout() -> None:
    """PR content cannot enable review or choose the provider/data policy."""

    text = _workflow_text()
    review = _paid_review_steps()[0]
    run = _step_run(review)

    assert "review.yaml" in text
    assert "claimci-trusted" in run
    assert "pull-request/.claimci/review.yaml" not in text
    assert "pull-request\\.claimci\\review.yaml" not in text
    assert not re.search(r"pull-request[\\/][^\n]*review\.yaml", text)
    normalized_run = re.sub(r"\s+", " ", run)
    assert re.search(r"claimci-trusted.*review\.yaml", normalized_run)
    # The head checkout can be named as passive input, but must not be used as
    # the trusted package/configuration source.
    assert "--artifact-root pull-request" in text
    assert "pip install ./pull-request" not in text
    assert "pip install -e ./pull-request" not in text


def test_optional_sdk_install_uses_the_same_trusted_yaml_parser_as_runtime() -> None:
    """Valid YAML boolean spellings cannot enable review without its SDK."""

    install_steps = [step for step in _steps() if "claimci-runtime[llm]" in _step_run(step)]
    install_runs = [_step_run(step) for step in install_steps]
    assert len(install_runs) == 1
    probes = [_step_run(step) for step in _steps() if "load_review_config" in _step_run(step)]
    assert len(probes) == 1
    assert "grep -E" not in probes[0] and "grep -q" not in probes[0]
    assert "preflight" in str(install_steps[0].get("if", "")).casefold()


def test_review_runtime_install_cannot_dirty_bound_snapshot_roots() -> None:
    """Package build artifacts stay outside every repository identity root."""

    steps = _job_steps("research_review")
    runtime_checkouts = [
        step
        for step in steps
        if str(step.get("with", {}).get("path", "")) == "claimci-runtime"
    ]
    assert len(runtime_checkouts) == 1
    runtime_checkout = runtime_checkouts[0]
    assert runtime_checkout.get("with", {}).get("ref") == "${{ github.event.pull_request.base.sha }}"
    assert runtime_checkout.get("with", {}).get("persist-credentials") is False

    install_runs = "\n".join(_step_run(step) for step in steps if "pip install" in _step_run(step))
    assert "pip install ./claimci-runtime" in install_runs
    assert 'pip install "./claimci-runtime[llm]"' in install_runs
    assert "pip install ./claimci-trusted" not in install_runs
    assert "pip install ./comparison-base" not in install_runs
    assert "pip install ./pull-request" not in install_runs


def test_review_coordinates_are_exact_and_merge_base_is_unique_and_materialized() -> None:
    review_steps = _job_steps("research_review")
    checkout_steps = [step for step in review_steps if "actions/checkout@" in str(step.get("uses", ""))]
    checkout_text = "\n".join(str(step) for step in checkout_steps)
    coordinate_steps = [step for step in review_steps if step.get("id") == "coordinates"]

    assert len(coordinate_steps) == 1
    coordinate_run = _step_run(coordinate_steps[0])
    assert "github.event.pull_request.base.sha" in coordinate_run
    assert "github.event.pull_request.head.sha" in coordinate_run
    assert "merge-base --all" in coordinate_run
    assert "GIT_ALTERNATE_OBJECT_DIRECTORIES" in coordinate_run
    assert "git -C claimci-trusted merge-base --all" in coordinate_run
    assert "comparison_sha" in coordinate_run
    assert "comparison_basis" in coordinate_run
    assert "direct_base" in coordinate_run and "merge_base" in coordinate_run
    assert "comparison_root" in coordinate_run
    assert "github.sha" not in checkout_text
    assert "github.event.pull_request.base.sha" in checkout_text
    assert "github.event.pull_request.head.sha" in checkout_text
    assert checkout_text.count("fetch-depth': 0") >= 2 or checkout_text.count("fetch-depth: 0") >= 2
    comparison_checkouts = [
        step
        for step in checkout_steps
        if str(step.get("with", {}).get("path", "")) == "comparison-base"
    ]
    assert len(comparison_checkouts) == 1
    assert "merge_base" in str(comparison_checkouts[0].get("if", ""))
    assert "steps.coordinates.outputs.comparison_sha" in str(comparison_checkouts[0])

    required = (
        "--requested-base-root",
        "--comparison-base-root",
        "--requested-base-sha",
        "--comparison-base-sha",
        "--comparison-basis",
        "--head-sha",
    )
    preflight_run = _step_run(_preflight_steps()[0])
    paid_run = _step_run(_paid_review_steps()[0])
    preflight_if = str(_preflight_steps()[0].get("if", ""))
    for option in required:
        assert option in preflight_run
        assert option in paid_run
    assert "--preflight-only" in preflight_run
    assert "--preflight-only" not in paid_run
    assert "coordinates.outputs.comparison_sha != ''" not in preflight_if
    assert "0" * 40 in preflight_run
    assert str(_paid_review_steps()[0].get("if", "")).casefold().count("preflight") >= 1


def test_head_checkout_is_passive_data_and_deterministic_audit_gate_is_unchanged() -> None:
    """Review integration cannot execute PR code or replace ClaimCI Audit."""

    text = _workflow_text()
    assert "pull_request_target:" in text
    assert "github.event.pull_request.base.sha" in text
    assert "github.event.pull_request.head.sha" in text
    assert "path: claimci-trusted" in text
    assert "path: pull-request" in text
    assert "allow-unsafe-pr-checkout: true" in text
    assert "persist-credentials: false" in text
    assert "--artifact-root pull-request" in text

    audits = _audit_steps()
    assert audits, "the existing deterministic audit invocation must remain"
    audit_text = "\n".join(_step_run(step) for step in audits)
    assert audit_text.count("claimci audit") == 2
    assert "--json" in audit_text and "--markdown" in audit_text
    assert '"ClaimCI Audit"' in text
    assert "ClaimCI Research Review" in text


def test_provider_key_isolated_from_checks_write_credential() -> None:
    """Only the provider step receives OPENAI_API_KEY; only publish gets GH_TOKEN."""

    workflow = _workflow_mapping()
    assert "OPENAI_API_KEY" not in str(workflow.get("env", {}))
    jobs = workflow.get("jobs", {})
    if isinstance(jobs, dict) and isinstance(jobs.get("claimci"), dict):
        assert "OPENAI_API_KEY" not in str(jobs["claimci"].get("env", {}))

    steps = _steps()
    provider_steps = [
        step
        for step in steps
        if "OPENAI_API_KEY" in _step_run(step)
        or "OPENAI_API_KEY" in str(step.get("env", {}))
    ]
    publish_steps = [
        step
        for step in steps
        if "gh api" in _step_run(step) and "check-runs" in _step_run(step)
    ]
    assert len(provider_steps) == 1
    assert provider_steps == _paid_review_steps()
    assert "OPENAI_API_KEY" not in _step_run(_preflight_steps()[0])
    assert "OPENAI_API_KEY" not in str(_preflight_steps()[0].get("env", {}))
    assert publish_steps, "a trusted publish step must remain present"
    for provider in provider_steps:
        assert "GH_TOKEN" not in _step_run(provider)
        assert "GH_TOKEN" not in str(provider.get("env", {}))
        assert "github.token" not in _step_run(provider)
        assert "gh api" not in _step_run(provider)
    for publish in publish_steps:
        owning_job_envs = [
            str(job.get("env", {}))
            for job in _jobs().values()
            if publish in job.get("steps", [])
        ]
        assert (
            "GH_TOKEN" in _step_run(publish)
            or "GH_TOKEN" in str(publish.get("env", {}))
            or any("GH_TOKEN" in env for env in owning_job_envs)
        )
        assert "OPENAI_API_KEY" not in _step_run(publish)
        assert "OPENAI_API_KEY" not in str(publish.get("env", {}))


def test_review_publishes_a_separate_neutral_check_without_overwriting_audit() -> None:
    """The advisory Check is independently named, head-anchored, and neutral."""

    text = _workflow_text()
    steps = _steps()
    review_publish = [
        step
        for step in steps
        if "ClaimCI Research Review" in _step_run(step)
        and ("gh api" in _step_run(step) or "check-runs" in _step_run(step))
    ]
    assert review_publish, "review Check publication must be explicit"
    publish_text = "\n".join(_step_run(step) for step in review_publish)
    assert "ClaimCI Research Review" in publish_text
    assert re.search(r"conclusion\s*[:=].*neutral", publish_text, re.IGNORECASE)
    assert "github.event.pull_request.head.sha" in text
    assert "ClaimCI Audit" in text
    # A review publisher must not post under or rewrite the authoritative name.
    assert not re.search(r"name\s*[:=].*ClaimCI Audit.*Research Review", publish_text)


def test_disabled_or_missing_key_review_is_nonblocking_and_still_documented() -> None:
    """Opt-in failures remain advisory; private-content egress is disclosed."""

    text = _workflow_text()
    review = _paid_review_steps()[0]
    review_run = _step_run(review)
    # Either the step is explicitly allowed to fail or its shell command
    # converts disabled/provider/key failures to a successful advisory path.
    nonblocking = bool(review.get("continue-on-error")) or bool(
        re.search(r"\|\|\s*(true|exit\s+0)", review_run)
    )
    assert nonblocking
    assert "OPENAI_API_KEY" in review_run or "OPENAI_API_KEY" in str(review.get("env", {}))

    lower = _documentation_text().casefold()
    assert "private" in lower and "external" in lower and "provider" in lower
    assert "fork" in lower
    assert "pull_request_target" in text


def test_review_publish_runs_after_review_failures_without_affecting_audit_check() -> None:
    """Unavailable/disabled review output may publish neutral metadata, never gate PRs."""

    jobs = _jobs()
    publishers = [
        (job, step)
        for job in jobs.values()
        for step in job.get("steps", [])
        if "ClaimCI Research Review" in _step_run(step)
        and ("gh api" in _step_run(step) or "check-runs" in _step_run(step))
    ]
    assert publishers
    publish_job, publish_step = publishers[0]
    assert publish_job.get("if") in {"always()", "${{ always() }}"} or publish_step.get(
        "if"
    ) in {"always()", "${{ always() }}"}
    # No second deterministic gate may be introduced for the advisory step.
    names = [str(step.get("name", "")) for step in _steps()]
    assert "Enforce ClaimCI Research Review" not in names
