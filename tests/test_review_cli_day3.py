"""Wave 4 CLI contracts for opt-in advisory research review.

The review subcommand must be additive: it can render an advisory view and
write both formats in one orchestration run, while the existing ``audit``
command remains byte-for-byte and exit-code compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import claimci.cli as cli_module
import claimci.review.orchestrator as orchestrator
from claimci.cli import main
from claimci.review.models import ReviewConfig, ReviewStatus
from claimci.review.orchestrator import ResearchReview


def _write_enabled_config(root: Path, *, filename: str = ".claimci/review.yaml") -> Path:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "enabled": True,
                "policy": "advisory",
                "provider": "openai",
                "model": "fake-model",
                "limits": {
                    "max_calls": 2,
                    "max_context_chars": 60_000,
                    "max_output_chars": 12_000,
                    "max_files": 24,
                    "max_file_chars": 16_000,
                    "max_output_tokens_per_call": 2_000,
                    "timeout_seconds": 30,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _complete_review() -> ResearchReview:
    return ResearchReview(status=ReviewStatus.COMPLETE)


def _patch_run_review(monkeypatch: pytest.MonkeyPatch, result: ResearchReview, calls: list[tuple[Any, ...]]) -> None:
    """Patch either CLI import style without touching the real provider."""

    def fake_run_review(*args: Any, **kwargs: Any) -> ResearchReview:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(cli_module, "run_review", fake_run_review, raising=False)
    monkeypatch.setattr(orchestrator, "run_review", fake_run_review)


def test_review_cli_default_is_disabled_and_never_constructs_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()

    class NoProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("default-disabled review must not construct an LLM provider")

    monkeypatch.setattr(orchestrator, "OpenAIReviewerProvider", NoProvider)

    exit_code = main(["review", str(repository), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["review"] == {
        "status": "DISABLED",
        "policy": "advisory",
        "blocking": False,
    }
    assert payload["provider"]["calls"] == []
    assert "OPENAI_API_KEY" not in captured.out


def test_review_cli_reads_event_metadata_from_trusted_config_root_and_writes_both_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    trusted = tmp_path / "trusted-base"
    trusted.mkdir()
    config_path = _write_enabled_config(trusted, filename="review-config.yaml")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "Event title",
                    "body": "Event body with \"quoted\" data",
                }
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "review.json"
    markdown_output = tmp_path / "review.md"
    calls: list[tuple[Any, ...]] = []
    _patch_run_review(monkeypatch, _complete_review(), calls)

    exit_code = main(
        [
            "review",
            str(repository),
            "--base-root",
            str(base),
            "--config-root",
            str(trusted),
            "--config",
            str(config_path),
            "--event-json",
            str(event_path),
            "--pr-title",
            "Explicit title wins",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert len(calls) == 1, "one CLI invocation must orchestrate review only once"
    args, kwargs = calls[0]
    assert kwargs == {}
    inputs, config = args
    assert inputs.repository_root == repository.resolve()
    assert inputs.base_root == base.resolve()
    assert inputs.pr_title == "Explicit title wins"
    assert inputs.pr_description == 'Event body with "quoted" data'
    assert isinstance(config, ReviewConfig)
    assert config.enabled is True
    assert json.loads(json_output.read_text(encoding="utf-8"))["review"]["status"] == "COMPLETE"
    markdown = markdown_output.read_text(encoding="utf-8")
    assert markdown.startswith("## ClaimCI Research Review (Advisory)\n")


@pytest.mark.parametrize(
    "provider_error",
    [
        "OpenAI SDK is not installed; install the llm extra",
        "OPENAI_API_KEY=sk-live-secret is missing",
    ],
)
def test_review_cli_provider_or_key_failure_is_safe_advisory_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_error: str,
) -> None:
    repository = tmp_path / "pull-request"
    repository.mkdir()
    _write_enabled_config(repository)

    from claimci.review.openai_provider import OpenAIReviewerProvider

    def fail_without_network(self: OpenAIReviewerProvider) -> object:
        raise RuntimeError(provider_error)

    monkeypatch.setattr(OpenAIReviewerProvider, "_client", fail_without_network)

    exit_code = main(["review", str(repository), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["review"]["status"] == "UNAVAILABLE"
    assert payload["review"]["blocking"] is False
    assert payload["provider"]["calls"] == []
    assert "sk-live-secret" not in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_review_cli_invalid_invocation_config_and_path_are_controlled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["review"])
    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err

    missing_repository = tmp_path / "does-not-exist"
    assert main(["review", str(missing_repository)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err

    repository = tmp_path / "repo"
    repository.mkdir()
    bad_config = repository / ".claimci" / "review.yaml"
    bad_config.parent.mkdir()
    bad_config.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    assert main(["review", str(repository), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.startswith("ClaimCI error:")


def test_review_cli_rejects_conflicting_view_flags_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(SystemExit) as raised:
        main(["review", str(repository), "--json", "--markdown"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "mutually exclusive" in captured.err
    assert "Traceback" not in captured.err


def test_review_parser_does_not_change_existing_audit_bytes_or_exit(
    study_factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = study_factory(candidate_config_updates={"training_steps": 300})

    before_exit = main(["audit", str(manifest), "--json"])
    before = capsys.readouterr()

    # Exercise the newly added subcommand between two audit invocations.  A
    # parser/global-state regression must not alter deterministic report bytes.
    repository = tmp_path / "review-repository"
    repository.mkdir()
    assert main(["review", str(repository), "--json"]) == 0
    capsys.readouterr()

    after_exit = main(["audit", str(manifest), "--json"])
    after = capsys.readouterr()

    assert after_exit == before_exit == 1
    assert after.out == before.out
    assert after.err == before.err == ""
