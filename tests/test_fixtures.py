from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimci.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("fixture_name", "verdict", "exit_code"),
    [
        ("valid", "SUPPORTED", 0),
        ("compute_mismatch", "NOT_SUPPORTED", 1),
        ("seed_mismatch", "INSUFFICIENT_EVIDENCE", 2),
        ("data_leak", "NOT_SUPPORTED", 1),
        ("fake_or_unsupported_result", "NOT_SUPPORTED", 1),
        ("different_eval_dataset", "NOT_SUPPORTED", 1),
        ("missing_result", "INSUFFICIENT_EVIDENCE", 2),
        ("multiple_failures", "NOT_SUPPORTED", 1),
    ],
)
def test_named_fixture_is_runnable(
    fixture_name: str,
    verdict: str,
    exit_code: int,
    capsys,
) -> None:
    manifest = REPOSITORY_ROOT / "examples" / fixture_name / "research.yaml"

    actual_exit = main(["audit", str(manifest), "--json"])

    captured = capsys.readouterr()
    assert actual_exit == exit_code
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["verdict"] == verdict
    assert payload["manifest"] == str(manifest.resolve())
