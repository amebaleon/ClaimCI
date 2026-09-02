"""Static credential-isolation contracts for the Day 3 GitHub workflow.

The provider must run in a read-only job, while Check API publication is done
by a separate trusted job.  These tests only parse the checked-in YAML; they
never execute Actions, invoke ``gh``, access a secret, or contact a provider.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "claimci.yml"

_REPOSITORY = "amebaleon/ClaimCI"
_EXTERNAL_HEAD_REPOSITORY = "amebaleon/ClaimCI-fork"
_PULL_REQUEST_EVENTS = ("opened", "synchronize", "reopened", "ready_for_review")
_RESEARCH_REVIEW_GUARD = (
    "github.event.pull_request.head.repo.full_name != github.repository"
)

# Keep this matrix deliberately bounded: each supported pull-request event is
# exercised once for an exact internal head-repository identity and once for
# an external identity.  Expected dispatch values are hand-derived from the
# security policy, not from the workflow condition under test.
_RESEARCH_REVIEW_DISPATCH_MATRIX = (
    ("opened", _REPOSITORY, False),
    ("opened", _EXTERNAL_HEAD_REPOSITORY, True),
    ("synchronize", _REPOSITORY, False),
    ("synchronize", _EXTERNAL_HEAD_REPOSITORY, True),
    ("reopened", _REPOSITORY, False),
    ("reopened", _EXTERNAL_HEAD_REPOSITORY, True),
    ("ready_for_review", _REPOSITORY, False),
    ("ready_for_review", _EXTERNAL_HEAD_REPOSITORY, True),
)


def _workflow() -> tuple[str, dict[str, Any]]:
    """Load the workflow once and reject malformed/non-mapping YAML."""

    text = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), "workflow must be a YAML mapping"
    return text, parsed


def _jobs(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_jobs = workflow.get("jobs")
    assert isinstance(raw_jobs, Mapping), "workflow must define a jobs mapping"
    jobs: dict[str, dict[str, Any]] = {}
    for job_id, raw_job in raw_jobs.items():
        assert isinstance(job_id, str), "job ids must be strings"
        assert isinstance(raw_job, Mapping), f"job {job_id!r} must be a mapping"
        jobs[job_id] = dict(raw_job)
    return jobs


def _steps(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_steps = job.get("steps", [])
    assert isinstance(raw_steps, list), "job steps must be a list"
    steps: list[dict[str, Any]] = []
    for step in raw_steps:
        assert isinstance(step, Mapping), "each workflow step must be a mapping"
        steps.append(dict(step))
    return steps


def _run_text(job: Mapping[str, Any]) -> str:
    """Return shell text while retaining step/job boundaries for inspection."""

    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _command_text(job: Mapping[str, Any]) -> str:
    """Drop shell comments before looking for executable command markers."""

    lines = []
    for line in _run_text(job).splitlines():
        if not line.lstrip().startswith("#"):
            lines.append(line)
    return "\n".join(lines)


def _review_cli_count(job: Mapping[str, Any]) -> int:
    """Count the trusted review CLI command, not prose mentioning it."""

    return len(
        re.findall(
            r"(?m)^\s*claimci\s+review(?:\s|\\|$)",
            _command_text(job),
        )
    )


def _preflight_cli_count(job: Mapping[str, Any]) -> int:
    return sum(
        _review_cli_count({"steps": [step]})
        for step in _steps(job)
        if "--preflight-only" in str(step.get("run", ""))
    )


def _paid_review_cli_count(job: Mapping[str, Any]) -> int:
    return sum(
        _review_cli_count({"steps": [step]})
        for step in _steps(job)
        if "claimci review" in str(step.get("run", ""))
        and "--preflight-only" not in str(step.get("run", ""))
    )


def _provider_jobs(jobs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        job_id: dict(job)
        for job_id, job in jobs.items()
        if _review_cli_count(job)
    }


def _review_publish_jobs(
    jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find jobs that publish the separate advisory Check through the API."""

    found: dict[str, dict[str, Any]] = {}
    for job_id, job in jobs.items():
        command = _command_text(job)
        if (
            re.search(r"\bgh\s+api\b", command)
            and "check-runs" in command
            and "--input claimci-review-check.json" in command
        ):
            found[job_id] = dict(job)
    return found


def _audit_publish_jobs(
    jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find the authoritative deterministic Check publisher(s)."""

    found: dict[str, dict[str, Any]] = {}
    for job_id, job in jobs.items():
        command = _command_text(job)
        if (
            re.search(r"\bgh\s+api\b", command)
            and "check-runs" in command
            and "--input claimci-check.json" in command
        ):
            found[job_id] = dict(job)
    return found


def _permissions(workflow: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, str]:
    """Resolve the effective job permissions for the common mapping form.

    GitHub treats an explicit job ``permissions`` map as a replacement for
    inherited permissions, so omitted scopes are not silently inherited from
    the workflow-level map.  Requiring an explicit map on credential-bearing
    jobs makes that boundary visible in the reviewed YAML.
    """

    raw = job.get("permissions")
    if raw is None:
        raw = workflow.get("permissions")
    assert isinstance(raw, Mapping), "credential-bearing jobs need a permissions map"
    return {str(name): str(value).casefold() for name, value in raw.items()}


def _env_text(workflow: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    """Include workflow-level env inherited by every job."""

    pieces: list[str] = [str(workflow.get("env", "")), str(job.get("env", ""))]
    pieces.extend(str(step.get("env", "")) for step in _steps(job))
    return "\n".join(pieces)


def _has_always_condition(value: object) -> bool:
    normalized = str(value).strip().casefold()
    normalized = re.sub(r"^\$\{\{\s*|\s*\}\}$", "", normalized).strip()
    return normalized == "always()"


def _job_or_step_always(job: Mapping[str, Any]) -> bool:
    return _has_always_condition(job.get("if")) or any(
        _has_always_condition(step.get("if")) for step in _steps(job)
    )


def _has_github_token(workflow: Mapping[str, Any], job: Mapping[str, Any]) -> bool:
    text = _env_text(workflow, job) + "\n" + _command_text(job)
    return bool(
        re.search(
            r"(?:\bGH_TOKEN\b|\bGITHUB_TOKEN\b|github\.token|secrets\.GITHUB_TOKEN)",
            text,
        )
    )


def _pull_request_target_types(workflow: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve the four supported target events from YAML 1.1/1.2 parsing."""

    raw_on = workflow.get("on")
    if raw_on is None:
        # PyYAML 6 parses the YAML 1.2 ``on`` key as ``True`` under its
        # YAML 1.1 resolver, while GitHub still consumes the literal key.
        raw_on = workflow.get(True)
    assert isinstance(raw_on, Mapping)
    pull_request_target = raw_on.get("pull_request_target")
    assert isinstance(pull_request_target, Mapping)
    raw_types = pull_request_target.get("types")
    assert isinstance(raw_types, list)
    assert all(isinstance(event, str) for event in raw_types)
    return tuple(raw_types)


def _evaluate_research_review_condition(
    condition: object,
    *,
    head_repository: str,
    repository: str,
) -> bool:
    """Evaluate only the exact repository-identity expression we require.

    This small evaluator keeps the event matrix local and deterministic while
    rejecting branch, actor, fuzzy, and secret-based substitutes for the
    repository identity guard.
    """

    assert condition == _RESEARCH_REVIEW_GUARD
    return head_repository != repository


@pytest.mark.parametrize(
    ("event", "head_repository", "expected_provider_dispatch"),
    _RESEARCH_REVIEW_DISPATCH_MATRIX,
)
def test_research_review_dispatches_only_for_external_heads_on_all_target_events(
    event: str,
    head_repository: str,
    expected_provider_dispatch: bool,
) -> None:
    """Internal PRs never reach the provider; external PRs remain enabled."""

    _workflow_text, workflow = _workflow()
    jobs = _jobs(workflow)
    assert _pull_request_target_types(workflow) == _PULL_REQUEST_EVENTS

    review = jobs.get("research_review")
    assert review is not None
    assert event in _PULL_REQUEST_EVENTS
    assert event in _pull_request_target_types(workflow)

    dispatches = _evaluate_research_review_condition(
        review.get("if"),
        head_repository=head_repository,
        repository=_REPOSITORY,
    )
    assert dispatches is expected_provider_dispatch

    # The deterministic audit remains unconditional for every event in the
    # matrix; only the optional provider job is identity-gated.
    assert jobs["claimci"].get("if") is None


def test_provider_execution_is_in_a_read_only_job_without_check_credentials() -> None:
    """The provider cannot publish or mutate Checks even if it is compromised."""

    _workflow_text, workflow = _workflow()
    jobs = _jobs(workflow)
    providers = _provider_jobs(jobs)

    assert len(providers) == 1, "exactly one job may invoke the review provider"
    provider_id, provider = next(iter(providers.items()))
    assert _review_cli_count(provider) == 2
    assert _preflight_cli_count(provider) == 1
    assert _paid_review_cli_count(provider) == 1

    review_steps = [
        step for step in _steps(provider) if "claimci review" in str(step.get("run", ""))
    ]
    preflight_steps = [
        step for step in review_steps if "--preflight-only" in str(step.get("run", ""))
    ]
    paid_steps = [
        step for step in review_steps if "--preflight-only" not in str(step.get("run", ""))
    ]
    assert len(preflight_steps) == len(paid_steps) == 1
    assert "OPENAI_API_KEY" not in str(preflight_steps[0].get("env", {}))
    assert "OPENAI_API_KEY" in str(paid_steps[0].get("env", {}))

    # A job-level map is required: a workflow-level checks:write default must
    # not accidentally become effective for the provider job.
    assert isinstance(provider.get("permissions"), Mapping), (
        f"provider job {provider_id!r} must declare least-privilege permissions"
    )
    permissions = _permissions(workflow, provider)
    assert permissions.get("contents") == "read"
    assert permissions.get("checks") != "write"
    assert "OPENAI_API_KEY" in _env_text(workflow, provider)
    key_jobs = {
        job_id
        for job_id, candidate in jobs.items()
        if "OPENAI_API_KEY" in _env_text(workflow, candidate)
    }
    assert key_jobs == {provider_id}, (
        "the provider secret must be scoped to the provider job only"
    )

    provider_text = _command_text(provider) + "\n" + _env_text(workflow, provider)
    assert not re.search(
        r"(?:\bGH_TOKEN\b|\bGITHUB_TOKEN\b|github\.token|secrets\.GITHUB_TOKEN)",
        provider_text,
    )
    assert not re.search(r"\bgh\s+api\b", _command_text(provider))

    # Keep the separation meaningful: the same job cannot also be the
    # advisory publisher, even when a future implementation renames steps.
    review_publishers = _review_publish_jobs(jobs)
    assert provider_id not in review_publishers


def test_advisory_publication_is_separate_always_running_checks_writer() -> None:
    """Only a later always-running job receives Checks write credentials."""

    _workflow_text, workflow = _workflow()
    jobs = _jobs(workflow)
    providers = _provider_jobs(jobs)
    publishers = _review_publish_jobs(jobs)

    assert len(providers) == 1
    assert len(publishers) == 1, (
        "the advisory Check must have one dedicated publication job"
    )
    provider_id = next(iter(providers))
    publisher_id, publisher = next(iter(publishers.items()))
    assert publisher_id != provider_id
    assert _has_always_condition(publisher.get("if")), (
        "the advisory publication job itself must survive provider/audit failures"
    )

    permissions = _permissions(workflow, publisher)
    assert permissions.get("checks") == "write"
    assert _has_github_token(workflow, publisher), "publisher must receive GH_TOKEN"

    publisher_text = _command_text(publisher) + "\n" + _env_text(workflow, publisher)
    assert "OPENAI_API_KEY" not in publisher_text
    assert _review_cli_count(publisher) == 0
    publish_steps = [
        step
        for step in _steps(publisher)
        if "--input claimci-review-check.json" in str(step.get("run", ""))
    ]
    assert len(publish_steps) == 1
    assert publish_steps[0].get("continue-on-error") is True, (
        "an advisory Check API outage must not make the PR workflow fail"
    )


def test_deterministic_audit_job_and_check_remain_authoritative() -> None:
    """Review isolation cannot remove or rename the blocking ClaimCI Audit."""

    text, workflow = _workflow()
    jobs = _jobs(workflow)
    audit_jobs = {
        job_id: job
        for job_id, job in jobs.items()
        if re.search(r"(?m)^\s*claimci\s+audit(?:\s|\\|$)", _command_text(job))
    }
    assert audit_jobs, "the deterministic ClaimCI Audit invocation must remain"
    assert "ClaimCI Audit" in text
    assert "--artifact-root pull-request" in text

    audit_publishers = _audit_publish_jobs(jobs)
    assert audit_publishers, "the authoritative ClaimCI Audit Check must still publish"
    for job in audit_publishers.values():
        assert _job_or_step_always(job)
        assert _permissions(workflow, job).get("checks") == "write"
        assert _has_github_token(workflow, job)

    # The provider cannot share the deterministic job, regardless of job IDs.
    provider_ids = set(_provider_jobs(jobs))
    assert provider_ids.isdisjoint(audit_jobs)


def test_review_handoff_is_bounded_and_revalidated_for_the_pr_head() -> None:
    """A compromised provider job cannot redirect or expand its neutral payload."""

    _text, workflow = _workflow()
    jobs = _jobs(workflow)
    provider = next(iter(_provider_jobs(jobs).values()))
    publisher = next(iter(_review_publish_jobs(jobs).values()))
    provider_commands = _command_text(provider)
    publisher_commands = _command_text(publisher)

    cap = re.search(r'\$\{#payload_b64\}\"?\s*-gt\s+(\d+)', provider_commands)
    # The value crosses jobs as one environment entry on Linux. Keep it below
    # MAX_ARG_STRLEN so the publisher can always start and emit its fallback.
    assert cap and int(cap.group(1)) <= 120_000
    assert '.head_sha == $head_sha' in publisher_commands
    assert '.conclusion == "neutral"' in publisher_commands
    assert "jq -j '.output.summary'" in publisher_commands
    assert "wc -c" in publisher_commands
    byte_cap = re.search(r'\$summary_bytes"?\s*-gt\s+(\d+)', publisher_commands)
    assert byte_cap and int(byte_cap.group(1)) <= 60_000
    assert ".output.summary | length" not in publisher_commands
    assert "Review unavailable; the optional advisory integration" in publisher_commands
