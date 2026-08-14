"""Manifest schema and path-resolution contract tests."""

from pathlib import Path

import pytest

from claimci.audit import ClaimCIError, load_research_spec


def _write_artifacts(root: Path) -> None:
    for relative in (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    ):
        artifact = root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")


def _valid_manifest() -> str:
    return """\
claim:
  metric: accuracy
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
"""


def test_load_research_spec_resolves_all_relative_paths_from_manifest_directory(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    manifest = tmp_path / "nested" / "research.yaml"
    manifest.parent.mkdir()
    # Put artifacts next to the manifest so the expected base is unambiguous.
    for relative in (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    ):
        source = tmp_path / relative
        target = manifest.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.write_text(_valid_manifest(), encoding="utf-8")

    spec = load_research_spec(manifest)

    assert spec.manifest_path == manifest.resolve()
    assert spec.metric == "accuracy"
    assert spec.minimum_improvement == pytest.approx(0.05)
    assert spec.baseline.config == (manifest.parent / "base/config.yaml").resolve()
    assert spec.baseline.results == (manifest.parent / "base/results.json").resolve()
    assert spec.baseline.train_dataset == (
        manifest.parent / "base/train.jsonl"
    ).resolve()
    assert spec.baseline.eval_dataset == (
        manifest.parent / "base/eval.jsonl"
    ).resolve()
    assert spec.candidate.config == (
        manifest.parent / "candidate/config.yaml"
    ).resolve()
    assert spec.candidate.results == (
        manifest.parent / "candidate/results.json"
    ).resolve()
    assert spec.candidate.train_dataset == (
        manifest.parent / "candidate/train.jsonl"
    ).resolve()
    assert spec.candidate.eval_dataset == (
        manifest.parent / "candidate/eval.jsonl"
    ).resolve()


def test_load_research_spec_rejects_non_mapping_root(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ClaimCIError, match="mapping"):
        load_research_spec(manifest)


def test_load_research_spec_rejects_missing_required_experiment(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: accuracy
  minimum_improvement: 0.05
baseline: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="candidate"):
        load_research_spec(manifest)


def test_load_research_spec_rejects_empty_metric(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: ""
  minimum_improvement: 0.05
baseline: {}
candidate: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="metric"):
        load_research_spec(manifest)


def test_load_research_spec_rejects_negative_absolute_threshold(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        """\
claim:
  metric: accuracy
  minimum_improvement: -0.01
baseline: {}
candidate: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="minimum_improvement"):
        load_research_spec(manifest)


def test_load_research_spec_wraps_invalid_utf8(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ClaimCIError, match=r"(?i)read|utf"):
        load_research_spec(manifest)


def test_load_research_spec_rejects_unrepresentably_large_threshold(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text(
        "claim:\n"
        "  metric: accuracy\n"
        f"  minimum_improvement: {10**1000}\n"
        "baseline: {}\n"
        "candidate: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ClaimCIError, match="minimum_improvement"):
        load_research_spec(manifest)


def test_load_research_spec_wraps_nul_path_error() -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)manifest|path|read"):
        load_research_spec(Path("\0"))
