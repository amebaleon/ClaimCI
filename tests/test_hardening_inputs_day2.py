"""Adversarial input-boundary tests for the Day 2 auditor.

These tests intentionally stay at ClaimCI's public boundaries.  A malformed
artifact must produce a concise :class:`ClaimCIError` (or an insufficient
audit finding), never a parser traceback or a silently accepted ambiguous
value.  The cases here complement the ordinary fixture tests with malformed
paths, duplicate mapping keys, booleans, and numeric extremes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimci.audit import audit_research, load_research_spec
from claimci.cli import main
from claimci.config_check import check_configs, load_config
from claimci.dataset_check import load_dataset
from claimci.models import ClaimCIError, Impact, Verdict
from claimci.result_check import check_results, load_results


METRIC = "accuracy"


def _replace_artifact_with_directory(manifest: Path, relative: str) -> Path:
    artifact = manifest.parent / relative
    artifact.unlink()
    artifact.mkdir()
    return artifact


def test_manifest_directory_path_is_a_controlled_claimci_error(tmp_path: Path) -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)manifest|read|directory"):
        load_research_spec(tmp_path)


def test_missing_manifest_path_is_a_controlled_claimci_error(tmp_path: Path) -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)manifest|not found|read"):
        load_research_spec(tmp_path / "does-not-exist.yaml")


def test_invalid_direction_is_rejected_before_artifact_validation(
    tmp_path: Path,
) -> None:
    """An invalid policy must not be hidden by a later missing-artifact error."""

    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: accuracy
  minimum_improvement: 0.05
  direction: sideways
baseline: {}
candidate: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="direction"):
        load_research_spec(manifest)


def test_manifest_nul_artifact_path_is_a_controlled_claimci_error(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_bytes(
        b"""\
claim:
  metric: accuracy
  minimum_improvement: 0.05
baseline:
  config: """ + b"\x00" + b""""
  results: base/results.json
  train_dataset: base/train.jsonl
  eval_dataset: base/eval.jsonl
candidate:
  config: candidate/config.yaml
  results: candidate/results.json
  train_dataset: candidate/train.jsonl
  eval_dataset: candidate/eval.jsonl
"""
    )

    with pytest.raises(ClaimCIError, match=r"(?i)baseline|config|path|yaml"):
        load_research_spec(manifest)


def test_manifest_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    """Last-key-wins parsing would make the declared claim ambiguous."""

    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: accuracy
  metric: loss
  minimum_improvement: 0.05
baseline:
  config: base/config.yaml
  results: base/results.json
  train_dataset: base/train.jsonl
  eval_dataset: base/eval.jsonl
candidate:
  config: candidate/config.yaml
  results: candidate/results.json
  train_dataset: candidate/train.jsonl
  eval_dataset: candidate/eval.jsonl
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match=r"(?i)duplicate|mapping|yaml"):
        load_research_spec(manifest)


def test_config_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
training_steps: 100
training_steps: 101
epochs: 4
batch_size: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match=r"(?i)duplicate|mapping|yaml|config"):
        load_config(path)


@pytest.mark.parametrize(
    ("loader", "pattern"),
    [
        (load_config, r"(?i)config|load|directory"),
        (load_dataset, r"(?i)dataset|read|directory"),
    ],
)
def test_config_and_dataset_directory_paths_are_controlled_errors(
    tmp_path: Path,
    loader,
    pattern: str,
) -> None:
    directory = tmp_path / "artifact"
    directory.mkdir()

    with pytest.raises(ClaimCIError, match=pattern):
        loader(directory)


def test_results_directory_path_is_a_controlled_error(tmp_path: Path) -> None:
    directory = tmp_path / "results.json"
    directory.mkdir()

    with pytest.raises(ClaimCIError, match=r"(?i)results|read|directory"):
        load_results(directory, METRIC)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_result_metrics_are_not_coerced_to_numeric(
    tmp_path: Path, value: bool
) -> None:
    path = tmp_path / "boolean-results.json"
    path.write_text(
        json.dumps({"runs": [{"seed": 1, METRIC: value}]}),
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match=r"(?i)metric|finite|results"):
        load_results(path, METRIC)


def test_results_reject_duplicate_json_object_keys(tmp_path: Path) -> None:
    """Duplicate metric keys must not silently change the audited value."""

    path = tmp_path / "duplicate-results.json"
    path.write_text(
        '{"runs":[{"seed":1,"accuracy":0.8,"accuracy":0.9}]}',
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match=r"(?i)duplicate|json|results"):
        load_results(path, METRIC)


def test_finite_negative_loss_supports_lower_direction_without_special_casing(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps({"runs": [{"seed": 1, "loss": -2.0}, {"seed": 2, "loss": -2.0}]}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"runs": [{"seed": 1, "loss": -3.0}, {"seed": 2, "loss": -3.0}]}),
        encoding="utf-8",
    )

    baseline = load_results(baseline_path, "loss")
    candidate = load_results(candidate_path, "loss")
    checked = check_results(baseline, candidate, "loss", 1.0, "lower")

    assert checked.absolute_improvement == pytest.approx(1.0)
    assert checked.relative_improvement == pytest.approx(0.5)
    claim = next(f for f in checked.findings if f.rule_id == "RESULT.CLAIM_SUPPORTED")
    assert claim.impact is Impact.NONE


def test_lower_direction_overflow_is_a_claimci_error(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps({"runs": [{"seed": 1, METRIC: 1.7e308}]}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"runs": [{"seed": 1, METRIC: -1.7e308}]}),
        encoding="utf-8",
    )

    baseline = load_results(baseline_path, METRIC)
    candidate = load_results(candidate_path, METRIC)

    with pytest.raises(ClaimCIError, match=r"(?i)improvement|finite"):
        check_results(baseline, candidate, METRIC, 0.0, "lower")


def test_boolean_config_numbers_become_insufficient_not_compute_advantage() -> None:
    baseline = {
        "training_steps": True,
        "epochs": 4,
        "batch_size": 8,
        "dataset": {"identifier": "train", "version": "v1"},
        "evaluation": {
            "dataset_identifier": "eval",
            "dataset_version": "v1",
            "split": "test",
        },
        "model": "base",
        "learning_rate": 0.001,
    }
    candidate = dict(baseline)

    findings = check_configs(baseline, candidate)

    missing = next(f for f in findings if f.rule_id == "CONFIG.MISSING_FIELDS")
    assert missing.impact is Impact.INSUFFICIENT
    assert "CONFIG.COMPUTE_MISMATCH" not in {f.rule_id for f in findings}


def test_missing_evaluation_field_is_insufficient_not_a_proven_mismatch() -> None:
    complete = {
        "training_steps": 100,
        "epochs": 4,
        "batch_size": 8,
        "dataset": {"identifier": "train", "version": "v1"},
        "evaluation": {
            "dataset_identifier": "eval",
            "dataset_version": "v1",
            "split": "test",
        },
        "model": "model",
        "learning_rate": 0.001,
    }
    incomplete = {
        **complete,
        "evaluation": {
            "dataset_identifier": "eval",
            "dataset_version": "v1",
        },
    }

    findings = check_configs(complete, incomplete)
    rule_ids = {finding.rule_id for finding in findings}

    assert "CONFIG.MISSING_FIELDS" in rule_ids
    assert "CONFIG.EVALUATION_MISMATCH" not in rule_ids


@pytest.mark.parametrize(
    ("relative", "rule_id"),
    [
        ("candidate/config.yaml", "CONFIG.INVALID"),
        ("candidate/results.json", "RESULT.INVALID"),
        ("candidate/train.jsonl", "DATASET.INVALID"),
        ("candidate/eval.jsonl", "DATASET.INVALID"),
    ],
)
def test_directory_artifact_is_reported_as_insufficient_finding(
    study_factory,
    relative: str,
    rule_id: str,
) -> None:
    manifest = study_factory()
    _replace_artifact_with_directory(manifest, relative)

    result = audit_research(manifest)

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    finding = next(f for f in result.findings if f.rule_id == rule_id)
    assert finding.impact is Impact.INSUFFICIENT


@pytest.mark.parametrize(
    ("relative", "rule_id"),
    [
        ("candidate/config.yaml", "CONFIG.MISSING"),
        ("candidate/results.json", "RESULT.MISSING"),
        ("candidate/train.jsonl", "DATASET.MISSING"),
        ("candidate/eval.jsonl", "DATASET.MISSING"),
    ],
)
def test_missing_artifact_is_reported_as_insufficient_finding(
    study_factory,
    relative: str,
    rule_id: str,
) -> None:
    manifest = study_factory()
    (manifest.parent / relative).unlink()

    result = audit_research(manifest)

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    finding = next(f for f in result.findings if f.rule_id == rule_id)
    assert finding.impact is Impact.INSUFFICIENT


def test_audit_relative_artifacts_are_independent_of_process_cwd(
    study_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = study_factory()
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    result = audit_research(manifest)

    assert result.verdict is Verdict.SUPPORTED


def test_cli_missing_manifest_has_no_traceback(tmp_path: Path, capsys) -> None:
    status = main(["audit", str(tmp_path / "missing-research.yaml")])

    captured = capsys.readouterr()
    assert status == 2
    assert "ClaimCI error:" in captured.err
    assert "Traceback" not in captured.err


def test_manifest_rejects_recursive_yaml_alias_graph(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: accuracy
  minimum_improvement: 0.05
extra: &loop [*loop]
baseline:
  config: base/config.yaml
  results: base/results.json
  train_dataset: base/train.jsonl
  eval_dataset: base/eval.jsonl
candidate:
  config: candidate/config.yaml
  results: candidate/results.json
  train_dataset: candidate/train.jsonl
  eval_dataset: candidate/eval.jsonl
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match=r"(?i)recursive|cycle|yaml|manifest"):
        load_research_spec(manifest)


def test_config_rejects_recursive_yaml_alias_graph(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("extra: &loop [*loop]\n", encoding="utf-8")

    with pytest.raises(ClaimCIError, match=r"(?i)recursive|cycle|yaml|config"):
        load_config(config)


def test_deeply_nested_yaml_is_a_controlled_error(tmp_path: Path) -> None:
    config = tmp_path / "deep.yaml"
    config.write_text("value: " + "[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    with pytest.raises(ClaimCIError, match=r"(?i)nested|recursion|yaml|config|load"):
        load_config(config)


def test_deeply_nested_results_json_is_a_controlled_error(tmp_path: Path) -> None:
    results = tmp_path / "deep-results.json"
    results.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    with pytest.raises(ClaimCIError, match=r"(?i)nested|recursion|json|results"):
        load_results(results, METRIC)
