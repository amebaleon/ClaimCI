from __future__ import annotations

import json
from pathlib import Path

from claimci.cli import main


def test_cli_human_output_and_supported_exit(study_factory, capsys) -> None:
    manifest = study_factory()

    exit_code = main(["audit", str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ClaimCI Audit" in captured.out
    assert "SUPPORTED" in captured.out
    assert captured.err == ""


def test_cli_json_output_and_not_supported_exit(study_factory, capsys) -> None:
    manifest = study_factory(candidate_config_updates={"training_steps": 300})

    exit_code = main(["audit", str(manifest), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out)["verdict"] == "NOT_SUPPORTED"
    assert captured.err == ""


def test_cli_missing_evidence_exit(study_factory, capsys) -> None:
    manifest = study_factory(write_candidate_results=False)

    exit_code = main(["audit", str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "INSUFFICIENT EVIDENCE" in captured.out
    assert "RESULT.MISSING" in captured.out


def test_cli_malformed_manifest_has_useful_error(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("not: a research manifest\n", encoding="utf-8")

    exit_code = main(["audit", str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("ClaimCI error:")
    assert "Traceback" not in captured.err


def test_cli_invalid_utf8_manifest_has_no_traceback(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_bytes(b"\xff\xfe")

    exit_code = main(["audit", str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("ClaimCI error:")
    assert "Traceback" not in captured.err
