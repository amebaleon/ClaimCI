"""Fairness-policy tests for explicit Day 1 configuration comparisons."""

from pathlib import Path
from typing import Any
from copy import deepcopy

import pytest

from claimci.config_check import check_configs, load_config
from claimci.models import ClaimCIError, Impact, Severity


BASE_CONFIG: dict[str, Any] = {
    "training_steps": 100,
    "epochs": 10,
    "batch_size": 10,
    "dataset": {"identifier": "train-set", "version": "v1"},
    "evaluation": {
        "dataset_identifier": "eval-set",
        "dataset_version": "v1",
        "split": "test",
    },
    "model": "baseline-model",
    "learning_rate": 0.001,
}


def _write_config(path: Path, values: dict[str, Any]) -> None:
    """Write a real YAML config without depending on dict insertion details."""

    path.write_text(
        """\
training_steps: {training_steps}
epochs: {epochs}
batch_size: {batch_size}
dataset:
  identifier: {dataset_identifier}
  version: {dataset_version}
evaluation:
  dataset_identifier: {evaluation_dataset_identifier}
  dataset_version: {evaluation_dataset_version}
  split: {evaluation_split}
model: {model}
learning_rate: {learning_rate}
""".format(
            training_steps=values["training_steps"],
            epochs=values["epochs"],
            batch_size=values["batch_size"],
            dataset_identifier=values["dataset"]["identifier"],
            dataset_version=values["dataset"]["version"],
            evaluation_dataset_identifier=values["evaluation"][
                "dataset_identifier"
            ],
            evaluation_dataset_version=values["evaluation"]["dataset_version"],
            evaluation_split=values["evaluation"]["split"],
            model=values["model"],
            learning_rate=values["learning_rate"],
        ),
        encoding="utf-8",
    )


def _configs(tmp_path: Path, candidate_updates: dict[str, Any] | None = None):
    baseline_values = {
        "training_steps": BASE_CONFIG["training_steps"],
        "epochs": BASE_CONFIG["epochs"],
        "batch_size": BASE_CONFIG["batch_size"],
        "dataset": dict(BASE_CONFIG["dataset"]),
        "evaluation": dict(BASE_CONFIG["evaluation"]),
        "model": BASE_CONFIG["model"],
        "learning_rate": BASE_CONFIG["learning_rate"],
    }
    candidate_values = {
        "training_steps": BASE_CONFIG["training_steps"],
        "epochs": BASE_CONFIG["epochs"],
        "batch_size": BASE_CONFIG["batch_size"],
        "dataset": dict(BASE_CONFIG["dataset"]),
        "evaluation": dict(BASE_CONFIG["evaluation"]),
        "model": BASE_CONFIG["model"],
        "learning_rate": BASE_CONFIG["learning_rate"],
    }
    for key, value in (candidate_updates or {}).items():
        if key in ("dataset", "evaluation"):
            candidate_values[key] = {**candidate_values[key], **value}
        else:
            candidate_values[key] = value

    baseline_path = tmp_path / "baseline.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    _write_config(baseline_path, baseline_values)
    _write_config(candidate_path, candidate_values)
    return load_config(baseline_path), load_config(candidate_path)


def _finding(findings, rule_id: str):
    matches = [finding for finding in findings if finding.rule_id == rule_id]
    assert matches, f"missing finding {rule_id!r}: {findings!r}"
    return matches[0]


def test_load_config_returns_nested_mapping_from_real_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, BASE_CONFIG)

    config = load_config(path)

    assert config["training_steps"] == 100
    assert config["dataset"] == {"identifier": "train-set", "version": "v1"}
    assert config["evaluation"]["split"] == "test"


def test_matching_configs_emit_verified_finding_without_gating_impact(
    tmp_path: Path,
) -> None:
    baseline, candidate = _configs(tmp_path)

    findings = check_configs(baseline, candidate)

    matching = _finding(findings, "CONFIG.FAIRNESS_VERIFIED")
    assert matching.severity is Severity.VERIFIED
    assert matching.impact is Impact.NONE
    assert not any(finding.impact is Impact.INVALIDATES for finding in findings)


def test_missing_supported_fields_are_insufficient_not_verified() -> None:
    findings = check_configs({}, {})

    missing = _finding(findings, "CONFIG.MISSING_FIELDS")
    assert missing.severity is Severity.WARNING
    assert missing.impact is Impact.INSUFFICIENT
    assert missing.evidence["baseline"] == [
        "training_steps",
        "epochs",
        "batch_size",
        "dataset.identifier",
        "dataset.version",
        "evaluation.dataset_identifier",
        "evaluation.dataset_version",
        "evaluation.split",
        "model",
        "learning_rate",
    ]
    assert missing.evidence["candidate"] == missing.evidence["baseline"]
    assert "CONFIG.FAIRNESS_VERIFIED" not in {
        finding.rule_id for finding in findings
    }


def test_three_x_compute_proxy_advantage_is_critical_and_invalidating(
    tmp_path: Path,
) -> None:
    baseline, candidate = _configs(tmp_path, {"training_steps": 300})

    findings = check_configs(baseline, candidate)

    compute = _finding(findings, "CONFIG.COMPUTE_MISMATCH")
    assert compute.severity is Severity.CRITICAL
    assert compute.impact is Impact.INVALIDATES
    assert compute.evidence["ratio"] == pytest.approx(3.0)


def test_compute_ratio_uses_one_common_basis_when_a_field_is_missing() -> None:
    baseline = deepcopy(BASE_CONFIG)
    candidate = deepcopy(BASE_CONFIG)
    del baseline["batch_size"]

    findings = check_configs(baseline, candidate)

    assert _finding(findings, "CONFIG.MISSING_FIELDS").impact is Impact.INSUFFICIENT
    assert "CONFIG.COMPUTE_MISMATCH" not in {
        finding.rule_id for finding in findings
    }


def test_exact_decimal_one_point_five_compute_ratio_invalidates() -> None:
    baseline = deepcopy(BASE_CONFIG)
    candidate = deepcopy(BASE_CONFIG)
    baseline.update({"training_steps": 0.1, "batch_size": 0.1})
    candidate.update({"training_steps": 0.15, "batch_size": 0.1})

    findings = check_configs(baseline, candidate)

    compute = _finding(findings, "CONFIG.COMPUTE_MISMATCH")
    assert compute.impact is Impact.INVALIDATES
    assert compute.evidence["ratio"] == pytest.approx(1.5)
    assert compute.evidence["basis"] == "training_steps*batch_size"


def test_compute_ratio_just_below_boundary_does_not_round_into_invalidation() -> None:
    baseline = deepcopy(BASE_CONFIG)
    candidate = deepcopy(BASE_CONFIG)
    baseline.update({"training_steps": 2 * 10**28, "batch_size": 1})
    candidate.update({"training_steps": 3 * 10**28 - 1, "batch_size": 1})

    findings = check_configs(baseline, candidate)

    assert "CONFIG.COMPUTE_MISMATCH" not in {
        finding.rule_id for finding in findings
    }
    difference = _finding(findings, "CONFIG.TRAINING_PARAMETER_DIFFERENCE")
    assert difference.impact is Impact.NONE


def test_huge_config_integer_is_handled_without_float_overflow() -> None:
    baseline = deepcopy(BASE_CONFIG)
    candidate = deepcopy(BASE_CONFIG)
    baseline["training_steps"] = 10**1000
    candidate["training_steps"] = 10**1000

    findings = check_configs(baseline, candidate)

    assert _finding(findings, "CONFIG.FAIRNESS_VERIFIED").impact is Impact.NONE


def test_nul_config_path_is_a_controlled_input_error() -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)config|path|load"):
        load_config(Path("\0"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", {"identifier": "other-train"}),
        ("dataset", {"version": "v2"}),
    ],
)
def test_training_dataset_difference_is_visible_but_non_gating(
    tmp_path: Path, field: str, value: dict[str, str]
) -> None:
    baseline, candidate = _configs(tmp_path, {field: value})

    findings = check_configs(baseline, candidate)

    difference = _finding(findings, "CONFIG.TRAINING_DATASET_MISMATCH")
    assert difference.severity is Severity.WARNING
    assert difference.impact is Impact.NONE
    assert not any(finding.impact is Impact.INVALIDATES for finding in findings)


def test_evaluation_mapping_difference_is_critical_and_invalidating(
    tmp_path: Path,
) -> None:
    baseline, candidate = _configs(
        tmp_path, {"evaluation": {"dataset_version": "v2"}}
    )

    findings = check_configs(baseline, candidate)

    evaluation = _finding(findings, "CONFIG.EVALUATION_MISMATCH")
    assert evaluation.severity is Severity.CRITICAL
    assert evaluation.impact is Impact.INVALIDATES


def test_model_difference_is_info_only(tmp_path: Path) -> None:
    baseline, candidate = _configs(tmp_path, {"model": "candidate-model"})

    findings = check_configs(baseline, candidate)

    model = _finding(findings, "CONFIG.MODEL_DIFFERENCE")
    assert model.severity is Severity.INFO
    assert model.impact is Impact.NONE
    assert not any(finding.impact is Impact.INVALIDATES for finding in findings)


def test_learning_rate_difference_is_warning_only(tmp_path: Path) -> None:
    baseline, candidate = _configs(tmp_path, {"learning_rate": 0.002})

    findings = check_configs(baseline, candidate)

    learning_rate = _finding(findings, "CONFIG.LEARNING_RATE_DIFFERENCE")
    assert learning_rate.severity is Severity.WARNING
    assert learning_rate.impact is Impact.NONE
    assert not any(finding.impact is Impact.INVALIDATES for finding in findings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_steps", 110),
        ("epochs", 11),
        ("batch_size", 11),
    ],
)
def test_small_training_parameter_difference_is_warning_not_invalidation(
    tmp_path: Path, field: str, value: int
) -> None:
    baseline, candidate = _configs(tmp_path, {field: value})

    findings = check_configs(baseline, candidate)

    parameter = _finding(findings, "CONFIG.TRAINING_PARAMETER_DIFFERENCE")
    assert parameter.severity is Severity.WARNING
    assert parameter.impact is Impact.NONE
    assert not any(finding.impact is Impact.INVALIDATES for finding in findings)
