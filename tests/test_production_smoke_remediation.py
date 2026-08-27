"""Controlled production-smoke fixture and deterministic planning regressions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from claimci.analysis import (
    ArtifactSnapshotRole,
    EvidenceObligationReason,
    EvidenceObligationState,
    GitCommitSha,
    PlanningState,
    RepositoryIdentity,
    passive_artifact_from_snapshot,
    plan_ephemeral_audit,
    planning_request_from_discovery,
)
from claimci.analysis.adapters import extract_registered_artifact
from claimci.analysis.discovery import discover_repository
from claimci.review.models import ClaimDirection


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "production_smoke"
REPOSITORY = RepositoryIdentity("amebaleon", "claimci-production-smoke-fixture")
HEAD_SHA = GitCommitSha("d" * 40)
EXACT_CLAIM = (
    "The candidate improves accuracy from 0.60 to 0.70 under the same "
    "configuration and evaluation dataset."
)
EXPECTED_HEAD_SHA256 = {
    "README.md": "245906ae491e68f661e4b4571122bb3a9372211708af56bb07f28db87981e1bb",
    "research.yaml": "8c7121eeb7603b816de26699640cd28be1ca2192ecfc057a1a8a0daf4cdd3d16",
    "baseline-config.yaml": "e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d",
    "candidate-config.yaml": "e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d",
    "baseline-eval.jsonl": "62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600",
    "candidate-eval.jsonl": "62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600",
    "baseline-results.json": "8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493",
    "candidate-results.json": "7ea7c722aaae5e2739ae13800daa5f5ea12bd1675fda6e04994c95e22ce54dc3",
    "baseline-train.jsonl": "7ace1dd7aa40731aaee3c6e40d1d5d67f053e654488361e849b3d2b065626417",
    "candidate-train.jsonl": "1381bb732892ff2d772a5d06a0b7da2f3f43336195102073c7029799f390dc44",
}
BASE_CANDIDATE_RESULTS_SHA256 = (
    "8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493"
)


def test_controlled_fixture_bytes_match_approved_sha256_map() -> None:
    for relative, expected in EXPECTED_HEAD_SHA256.items():
        observed = hashlib.sha256((FIXTURE_ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, relative

    base_candidate = FIXTURE_ROOT / "candidate-results-base.json"
    assert hashlib.sha256(base_candidate.read_bytes()).hexdigest() == (
        BASE_CANDIDATE_RESULTS_SHA256
    )
    assert base_candidate.read_bytes() == (FIXTURE_ROOT / "baseline-results.json").read_bytes()


def _reconstruct_tree(root: Path, *, base: bool) -> None:
    for relative in EXPECTED_HEAD_SHA256:
        source = FIXTURE_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if base and relative == "candidate-results.json":
            destination.write_bytes((FIXTURE_ROOT / "candidate-results-base.json").read_bytes())
        else:
            destination.write_bytes(source.read_bytes())


def test_controlled_fixture_reaches_honest_threshold_free_planning_boundary(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "base"
    head_root = tmp_path / "head"
    base_root.mkdir()
    head_root.mkdir()
    _reconstruct_tree(base_root, base=True)
    _reconstruct_tree(head_root, base=False)

    discovery = discover_repository(
        head_root,
        base_root=base_root,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=1,
        pr_title="Production smoke: verify supported result",
        pr_description=EXACT_CLAIM,
    )

    assert len(discovery.claims) == 1
    assert discovery.claims[0].reference.text == EXACT_CLAIM
    assert discovery.claims[0].metric == "accuracy"
    assert discovery.claims[0].direction is ClaimDirection.HIGHER
    assert discovery.claims[0].baseline_value is not None
    assert discovery.claims[0].baseline_value.value == 0.60
    assert discovery.claims[0].candidate_value is not None
    assert discovery.claims[0].candidate_value.value == 0.70
    assert discovery.claims[0].minimum_improvement is None

    evidence = []
    for candidate in discovery.artifacts:
        content = (head_root / str(candidate.path)).read_bytes()
        passive = passive_artifact_from_snapshot(
            REPOSITORY,
            ArtifactSnapshotRole.HEAD,
            HEAD_SHA,
            candidate,
            content,
        )
        extracted = extract_registered_artifact(passive)
        assert extracted is not None, candidate.path
        evidence.append(extracted)

    assert {item.artifact.path for item in evidence} == {
        item.path for item in discovery.artifacts
    }
    assert evidence
    assert all(
        re.fullmatch(r"evidence-v2-[0-9a-f]{64}", item.evidence_id)
        for item in evidence
    )
    config_ids = {
        str(item.artifact.path): item.evidence_id
        for item in evidence
        if str(item.artifact.path)
        in {"baseline-config.yaml", "candidate-config.yaml"}
    }
    assert config_ids["baseline-config.yaml"] != config_ids["candidate-config.yaml"]

    request = planning_request_from_discovery(
        discovery,
        claim_id=discovery.claims[0].reference.claim_id,
        normalized_evidence=tuple(evidence),
    )
    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == "required_threshold_not_recovered"
    assert outcome.plan is None
    assert outcome.mapping_question is None
    assert outcome.evidence_obligations is not None
    assert outcome.evidence_obligations.blocking_obligation_ids == (
        "claim.threshold",
    )
    threshold = next(
        item
        for item in outcome.evidence_obligations.obligations
        if item.obligation_id == "claim.threshold"
    )
    assert threshold.state is EvidenceObligationState.MISSING
    assert threshold.reason is EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED
