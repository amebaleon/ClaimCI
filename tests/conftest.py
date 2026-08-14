from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


@pytest.fixture
def study_factory(tmp_path: Path) -> Callable[..., Path]:
    def make_study(
        *,
        baseline_config_updates: dict[str, Any] | None = None,
        candidate_config_updates: dict[str, Any] | None = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        candidate_runs: list[dict[str, Any]] | None = None,
        baseline_summary: float | None = None,
        candidate_summary: float | None = None,
        baseline_train: list[Any] | None = None,
        baseline_eval: list[Any] | None = None,
        candidate_train: list[Any] | None = None,
        candidate_eval: list[Any] | None = None,
        write_baseline_results: bool = True,
        write_candidate_results: bool = True,
        minimum_improvement: float = 0.05,
    ) -> Path:
        root = tmp_path / "study"
        base_dir = root / "base"
        candidate_dir = root / "candidate"
        base_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        base_config: dict[str, Any] = {
            "training_steps": 100,
            "epochs": 4,
            "batch_size": 8,
            "dataset": {"identifier": "toy-train", "version": "v1"},
            "evaluation": {
                "dataset_identifier": "toy-eval",
                "dataset_version": "v1",
                "split": "test",
            },
            "model": "baseline-model",
            "learning_rate": 0.001,
        }
        new_config = copy.deepcopy(base_config)
        new_config["model"] = "candidate-model"
        _deep_update(base_config, baseline_config_updates or {})
        _deep_update(new_config, candidate_config_updates or {})
        (base_dir / "config.yaml").write_text(
            yaml.safe_dump(base_config, sort_keys=True), encoding="utf-8"
        )
        (candidate_dir / "config.yaml").write_text(
            yaml.safe_dump(new_config, sort_keys=True), encoding="utf-8"
        )

        base_runs = baseline_runs or [
            {"seed": 1, "accuracy": 0.58},
            {"seed": 2, "accuracy": 0.60},
            {"seed": 3, "accuracy": 0.62},
        ]
        new_runs = candidate_runs or [
            {"seed": 1, "accuracy": 0.68},
            {"seed": 2, "accuracy": 0.70},
            {"seed": 3, "accuracy": 0.72},
        ]
        if write_baseline_results:
            payload: dict[str, Any] = {"runs": base_runs}
            if baseline_summary is not None:
                payload["summary"] = {"accuracy": baseline_summary}
            (base_dir / "results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        if write_candidate_results:
            payload = {"runs": new_runs}
            if candidate_summary is not None:
                payload["summary"] = {"accuracy": candidate_summary}
            (candidate_dir / "results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

        baseline_eval_rows = (
            baseline_eval
            if baseline_eval is not None
            else [
                {"id": "be1", "text": "base eval one"},
                {"id": "be2", "text": "base eval two"},
            ]
        )
        candidate_eval_rows = (
            candidate_eval if candidate_eval is not None else baseline_eval_rows
        )
        datasets = {
            base_dir / "train.jsonl": baseline_train
            or [{"id": "bt1", "text": "base train one"}, {"id": "bt2", "text": "base train two"}],
            base_dir / "eval.jsonl": baseline_eval_rows,
            candidate_dir / "train.jsonl": candidate_train
            or [{"id": "ct1", "text": "new train one"}, {"id": "ct2", "text": "new train two"}],
            candidate_dir / "eval.jsonl": candidate_eval_rows,
        }
        for path, rows in datasets.items():
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

        manifest = {
            "claim": {
                "metric": "accuracy",
                "minimum_improvement": minimum_improvement,
            },
            "baseline": {
                "config": "base/config.yaml",
                "results": "base/results.json",
                "train_dataset": "base/train.jsonl",
                "eval_dataset": "base/eval.jsonl",
            },
            "candidate": {
                "config": "candidate/config.yaml",
                "results": "candidate/results.json",
                "train_dataset": "candidate/train.jsonl",
                "eval_dataset": "candidate/eval.jsonl",
            },
        }
        manifest_path = root / "research.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return manifest_path

    return make_study
