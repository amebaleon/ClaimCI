"""Integration contract for the polished Day 2 rejected-claim demo.

The demo is intentionally an apparently strong result: a single candidate
run reaches 0.90 accuracy while three baseline runs average 0.60.  This test
keeps the investor-facing story honest by requiring that ClaimCI expose the
supporting arithmetic *and* reject the claim for the independent evidence
violations (compute inflation, seed imbalance, and canonical leakage).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from claimci.audit import audit_research
from claimci.cli import main
from claimci.dataset_check import load_dataset
from claimci.models import Impact, Verdict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPOSITORY_ROOT / "examples" / "day2_demo"
DEMO_MANIFEST = DEMO_ROOT / "research.yaml"


def _finding(result, rule_id: str):
    return next(finding for finding in result.findings if finding.rule_id == rule_id)


def test_day2_demo_rejects_apparently_strong_self_contained_claim(capsys) -> None:
    """The exact demo numbers and evidence remain visible in every output."""

    assert DEMO_MANIFEST.is_file(), "the checked-in demo must be runnable as-is"

    # The demo may be copied or reviewed without depending on artifacts outside
    # its directory.  Relative paths also make its CI invocation deterministic.
    manifest_payload = yaml.safe_load(DEMO_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(manifest_payload, dict)
    for experiment in ("baseline", "candidate"):
        artifacts = manifest_payload[experiment]
        assert isinstance(artifacts, dict)
        for field in ("config", "results", "train_dataset", "eval_dataset"):
            raw_path = artifacts[field]
            assert isinstance(raw_path, str) and raw_path
            relative_path = Path(raw_path)
            assert not relative_path.is_absolute()
            resolved_path = (DEMO_ROOT / relative_path).resolve()
            assert resolved_path.is_file()
            resolved_path.relative_to(DEMO_ROOT.resolve())

    result = audit_research(DEMO_MANIFEST)

    # Both arms evaluate the exact same records. The rejection story must not
    # hide a fourth confound behind matching dataset identifiers.
    assert load_dataset(DEMO_ROOT / "baseline-eval.jsonl").hashes == load_dataset(
        DEMO_ROOT / "candidate-eval.jsonl"
    ).hashes

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert result.baseline_metrics is not None
    assert result.candidate_metrics is not None
    assert result.baseline_metrics.count == 3
    assert result.baseline_metrics.mean == pytest.approx(0.60)
    assert result.candidate_metrics.count == 1
    assert result.candidate_metrics.mean == pytest.approx(0.90)
    assert result.absolute_improvement == pytest.approx(0.30)
    assert result.relative_improvement == pytest.approx(0.50)

    claim_supported = _finding(result, "RESULT.CLAIM_SUPPORTED")
    assert claim_supported.impact is Impact.NONE
    assert claim_supported.evidence["absolute_improvement"] == pytest.approx(0.30)
    assert claim_supported.evidence["relative_improvement"] == pytest.approx(0.50)

    compute = _finding(result, "CONFIG.COMPUTE_MISMATCH")
    assert compute.impact is Impact.INVALIDATES
    assert compute.evidence["ratio"] == pytest.approx(3.0)

    single_run = _finding(result, "SEED.SINGLE_RUN")
    assert single_run.evidence["experiment"] == "candidate"
    imbalance = _finding(result, "SEED.IMBALANCE")
    assert imbalance.evidence["baseline_count"] == 3
    assert imbalance.evidence["candidate_count"] == 1
    assert imbalance.evidence["ratio"] == pytest.approx(3.0)

    leakage = _finding(result, "DATASET.EXACT_LEAKAGE")
    assert leakage.evidence["experiment"] == "candidate"
    assert leakage.evidence["overlap_count"] == 1
    assert leakage.evidence["eval_count"] == 2
    assert leakage.evidence["overlap_rate"] == pytest.approx(0.5)

    candidate_overlap = next(
        overlap for overlap in result.overlaps if overlap.experiment == "candidate"
    )
    assert candidate_overlap.overlap_count == 1
    assert candidate_overlap.eval_count == 2
    assert candidate_overlap.overlap_rate == pytest.approx(0.5)

    # JSON and Markdown are two views of the same AuditResult.  Keep this
    # assertion at the CLI boundary so the demo also exercises exit semantics.
    json_exit = main(["audit", str(DEMO_MANIFEST), "--json"])
    json_capture = capsys.readouterr()
    assert json_exit == 1
    assert json_capture.err == ""
    payload = json.loads(json_capture.out)
    assert payload["verdict"] == "NOT_SUPPORTED"
    assert payload["metrics"]["baseline"]["count"] == 3
    assert payload["metrics"]["baseline"]["mean"] == pytest.approx(0.60)
    assert payload["metrics"]["candidate"]["count"] == 1
    assert payload["metrics"]["candidate"]["mean"] == pytest.approx(0.90)
    assert payload["metrics"]["absolute_improvement"] == pytest.approx(0.30)
    assert payload["metrics"]["relative_improvement"] == pytest.approx(0.50)

    markdown_exit = main(["audit", str(DEMO_MANIFEST), "--markdown"])
    markdown_capture = capsys.readouterr()
    assert markdown_exit == 1
    assert markdown_capture.err == ""
    markdown = markdown_capture.out
    normalized_markdown = markdown.upper().replace("_", " ")
    for rule_id in (
        "RESULT.CLAIM_SUPPORTED",
        "CONFIG.COMPUTE_MISMATCH",
        "SEED.SINGLE_RUN",
        "SEED.IMBALANCE",
        "DATASET.EXACT_LEAKAGE",
    ):
        assert rule_id in markdown
    assert "NOT SUPPORTED" in normalized_markdown
    assert "0.600000" in markdown
    assert "0.900000" in markdown
    assert "0.300000" in markdown
    assert "50.00%" in markdown
