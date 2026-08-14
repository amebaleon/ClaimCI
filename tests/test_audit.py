from __future__ import annotations

from pathlib import Path

import pytest

from claimci.audit import audit_research, load_research_spec
from claimci.models import ClaimCIError, Verdict


def test_valid_audit_recomputes_metrics_and_supports_claim(study_factory) -> None:
    manifest = study_factory(
        baseline_summary=0.60,
        candidate_summary=0.70,
    )

    result = audit_research(manifest)

    assert result.verdict is Verdict.SUPPORTED
    assert result.baseline_metrics is not None
    assert result.candidate_metrics is not None
    assert result.baseline_metrics.count == 3
    assert result.baseline_metrics.mean == pytest.approx(0.60)
    assert result.baseline_metrics.standard_deviation == pytest.approx(0.02)
    assert result.candidate_metrics.mean == pytest.approx(0.70)
    assert result.candidate_metrics.standard_deviation == pytest.approx(0.02)
    assert result.absolute_improvement == pytest.approx(0.10)
    assert result.relative_improvement == pytest.approx(1 / 6)
    assert "RESULT.CLAIM_SUPPORTED" in {finding.rule_id for finding in result.findings}
    assert len([f for f in result.findings if f.rule_id == "DATASET.NO_LEAKAGE"]) == 2
    assert len(
        [f for f in result.findings if f.rule_id == "DATASET.EVALUATION_ALIGNED"]
    ) == 1


def test_audit_invalidates_claim_when_evaluation_record_multisets_differ(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_eval=[{"id": "shared"}, {"id": "duplicate"}, {"id": "duplicate"}],
        candidate_eval=[{"id": "shared"}, {"id": "candidate-only"}],
    )

    result = audit_research(manifest)

    assert result.verdict is Verdict.NOT_SUPPORTED
    mismatch = next(
        finding
        for finding in result.findings
        if finding.rule_id == "DATASET.EVALUATION_MISMATCH"
    )
    assert mismatch.evidence["baseline_eval_count"] == 3
    assert mismatch.evidence["candidate_eval_count"] == 2
    assert mismatch.evidence["matching_count"] == 1


def test_audit_does_not_claim_evaluation_mismatch_when_one_eval_is_unavailable(
    study_factory,
) -> None:
    manifest = study_factory()
    (manifest.parent / "candidate" / "eval.jsonl").unlink()

    result = audit_research(manifest)

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    rule_ids = {finding.rule_id for finding in result.findings}
    assert "DATASET.MISSING" in rule_ids
    assert "DATASET.EVALUATION_ALIGNED" not in rule_ids
    assert "DATASET.EVALUATION_MISMATCH" not in rule_ids


def test_audit_aggregates_failures_and_invalidators_win(study_factory) -> None:
    leaked = {"id": "leak", "text": "same canonical row"}
    manifest = study_factory(
        candidate_config_updates={"training_steps": 300},
        write_candidate_results=False,
        candidate_train=[leaked, {"id": "train-only"}],
        candidate_eval=[{"text": "same canonical row", "id": "leak"}],
    )

    result = audit_research(manifest)

    assert result.verdict is Verdict.NOT_SUPPORTED
    rule_ids = {finding.rule_id for finding in result.findings}
    assert "CONFIG.COMPUTE_MISMATCH" in rule_ids
    assert "RESULT.MISSING" in rule_ids
    assert "DATASET.EXACT_LEAKAGE" in rule_ids


def test_invalid_manifest_is_a_concise_input_error(tmp_path: Path) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text("claim: []\n", encoding="utf-8")

    with pytest.raises(ClaimCIError, match="claim"):
        load_research_spec(manifest)


def test_finding_order_and_rule_ids_are_deterministic(study_factory) -> None:
    manifest = study_factory(candidate_config_updates={"learning_rate": 0.002})

    first = audit_research(manifest)
    second = audit_research(manifest)

    assert [finding.rule_id for finding in first.findings] == [
        finding.rule_id for finding in second.findings
    ]
    assert [finding.evidence for finding in first.findings] == [
        finding.evidence for finding in second.findings
    ]


def test_absolute_threshold_is_not_replaced_by_relative_improvement(study_factory) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, "accuracy": 0.49},
            {"seed": 2, "accuracy": 0.50},
            {"seed": 3, "accuracy": 0.51},
        ],
        candidate_runs=[
            {"seed": 1, "accuracy": 0.515},
            {"seed": 2, "accuracy": 0.525},
            {"seed": 3, "accuracy": 0.535},
        ],
        minimum_improvement=0.05,
    )

    result = audit_research(manifest)

    assert result.absolute_improvement == pytest.approx(0.025)
    assert result.relative_improvement == pytest.approx(0.05)
    assert result.verdict is Verdict.NOT_SUPPORTED


def test_overflowing_raw_statistics_become_insufficient_evidence(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, "accuracy": 1.7e308},
            {"seed": 2, "accuracy": -1.7e308},
        ],
        candidate_runs=[
            {"seed": 1, "accuracy": 0.7},
            {"seed": 2, "accuracy": 0.8},
        ],
    )

    result = audit_research(manifest)

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    invalid = next(
        finding for finding in result.findings if finding.rule_id == "RESULT.INVALID"
    )
    assert invalid.evidence["experiment"] == "baseline"


def test_nonfinite_derived_improvement_becomes_insufficient_evidence(
    study_factory,
) -> None:
    manifest = study_factory(
        baseline_runs=[
            {"seed": 1, "accuracy": -1.0e308},
        ],
        candidate_runs=[
            {"seed": 1, "accuracy": 1.0e308},
        ],
    )

    result = audit_research(manifest)

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    invalid = next(
        finding for finding in result.findings if finding.rule_id == "RESULT.INVALID"
    )
    assert invalid.evidence["experiment"] == "comparison"
