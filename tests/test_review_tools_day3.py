"""Wave 2 contract tests for the one-way deterministic-tool bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from claimci.audit import audit_research
from claimci.models import AuditResult, Finding, Impact, Severity, Verdict
from claimci.review.tools import (
    ManifestAuditPlan,
    discover_manifests,
    plan_manifest_audits,
    run_manifest_audits,
    snapshot_audit_result,
)
from claimci.review.models import ReviewLimits

def test_manifest_discovery_is_bounded_sorted_and_repository_confined(
    tmp_path: Path, study_factory
) -> None:
    """Only indexed manifest candidates beneath the checkout are audited."""

    manifest = study_factory()
    root = manifest.parent
    _manifest = root / "nested" / "research.yaml"
    _manifest.parent.mkdir(parents=True)
    _manifest.write_text("not a complete manifest\n", encoding="utf-8")
    outside = root.parent / "outside-research.yaml"
    outside.write_text("claim: outside\n", encoding="utf-8")

    paths = ["nested/research.yaml", "research.yaml", "../outside-research.yaml", "README.md"]
    discovered = discover_manifests(root, paths, max_manifests=1)
    assert discovered == ("research.yaml",)
    assert all(not Path(path).is_absolute() for path in discovered)
    assert all(".." not in Path(path).parts for path in discovered)

    # Shuffling the path index must not change selection or order.
    assert discover_manifests(root, list(reversed(paths)), max_manifests=4) == (
        "research.yaml",
        "nested/research.yaml",
    )


def test_manifest_audits_use_artifact_root_bridge_and_actual_audit_snapshots(
    study_factory,
) -> None:
    """Discovered manifests flow through audit_research(..., artifact_root=root)."""

    manifest = study_factory()
    root = manifest.parent
    snapshots = run_manifest_audits(root, ("research.yaml",))
    expected = audit_research(manifest, artifact_root=root)

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.verdict == expected.verdict.value
    assert snapshot.metric == expected.metric
    assert snapshot.minimum_improvement == expected.minimum_improvement
    assert snapshot.manifest_path == "research.yaml"
    assert tuple(item.rule_id for item in snapshot.findings) == tuple(
        item.rule_id for item in expected.findings
    )


def test_malformed_discovered_manifest_is_omitted_without_aborting_review_tools(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "research.yaml"
    manifest.write_text("claim: [unterminated", encoding="utf-8")

    assert run_manifest_audits(tmp_path, ("research.yaml",)) == ()


def test_manifest_audit_respects_global_file_and_per_file_review_budgets(
    study_factory,
) -> None:
    """The advisory bridge never lets one manifest bypass review inspection caps."""

    manifest = study_factory()
    root = manifest.parent

    snapshots = run_manifest_audits(
        root,
        ("research.yaml",),
        limits=ReviewLimits(max_files=1, max_file_chars=1),
    )

    assert snapshots == ()


def test_manifest_planning_records_unissued_dependencies_without_opening_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An issued manifest may name a path that remains outside the read boundary."""

    manifest = tmp_path / "research.yaml"
    artifact_paths = (
        "private/baseline-config.yaml",
        "private/baseline-results.json",
        "private/baseline-train.jsonl",
        "private/baseline-eval.jsonl",
        "private/candidate-config.yaml",
        "private/candidate-results.json",
        "private/candidate-train.jsonl",
        "private/candidate-eval.jsonl",
    )
    manifest.write_text(
        """claim:
  metric: accuracy
  minimum_improvement: 0.05
baseline:
  config: private/baseline-config.yaml
  results: private/baseline-results.json
  train_dataset: private/baseline-train.jsonl
  eval_dataset: private/baseline-eval.jsonl
candidate:
  config: private/candidate-config.yaml
  results: private/candidate-results.json
  train_dataset: private/candidate-train.jsonl
  eval_dataset: private/candidate-eval.jsonl
""",
        encoding="utf-8",
    )
    for relative in artifact_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("UNISSUED_DEPENDENCY_MUST_NOT_BE_READ\n", encoding="utf-8")

    opened: list[str] = []
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        relative = path.resolve().relative_to(tmp_path.resolve()).as_posix()
        opened.append(relative)
        if relative != "research.yaml":
            raise AssertionError(f"unissued dependency opened: {relative}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    plans = plan_manifest_audits(
        tmp_path,
        ("research.yaml",),
        issued_paths=("research.yaml",),
    )

    assert plans == (
        ManifestAuditPlan(
            manifest_path="research.yaml",
            paths=("research.yaml", *artifact_paths),
        ),
    )
    assert opened and set(opened) == {"research.yaml"}


def test_manifest_audit_rejects_internal_symlink_artifacts(study_factory) -> None:
    """Deterministic tools cannot route excluded in-repository files via links."""

    manifest = study_factory()
    root = manifest.parent
    link = root / "candidate" / "config.yaml"
    private = root / "private" / "real-config.yaml"
    private.parent.mkdir()
    private.write_bytes(link.read_bytes())
    link.unlink()
    try:
        link.symlink_to(private)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")

    snapshots = run_manifest_audits(root, ("research.yaml",))

    assert snapshots == ()


def test_snapshot_factory_accepts_only_actual_audit_result_and_is_read_only(
    study_factory,
) -> None:
    """Provider-shaped data cannot manufacture or mutate deterministic authority."""

    manifest = study_factory()
    root = manifest.parent
    result = audit_research(manifest, artifact_root=root)
    snapshot = snapshot_audit_result(result, repository_root=root)

    assert snapshot.verdict == result.verdict.value
    assert snapshot.minimum_improvement == result.minimum_improvement
    assert isinstance(snapshot.findings, tuple)
    original_evidence = dict(result.findings[0].evidence) if result.findings else {}
    if result.findings:
        # Finding is frozen, but its legacy evidence mapping is mutable; the
        # review snapshot must not retain that live authority-owned mapping.
        result.findings[0].evidence["__provider_mutation__"] = "ignored"
        assert dict(snapshot.findings[0].evidence) == original_evidence
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        snapshot.verdict = "NOT_SUPPORTED"
    if snapshot.findings:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            snapshot.findings[0].rule_id = "invented.rule"
        with pytest.raises((TypeError, AttributeError)):
            snapshot.findings[0].evidence["invented"] = "provider text"

    with pytest.raises((TypeError, ValueError)):
        snapshot_audit_result(
            SimpleNamespace(
                verdict="SUPPORTED",
                findings=(),
                metric="accuracy",
                minimum_improvement=0.0,
            ),
            repository_root=root,
        )
    with pytest.raises((TypeError, ValueError)):
        snapshot_audit_result(
            {"verdict": "NOT_SUPPORTED", "findings": [], "minimum_improvement": 999},
            repository_root=root,
        )


def test_snapshot_copies_authority_values_without_provider_override(study_factory) -> None:
    """Verdict/finding severity-impact/threshold/evidence come only from AuditResult."""

    manifest = study_factory(minimum_improvement=0.125)
    root = manifest.parent
    result = audit_research(manifest, artifact_root=root)
    snapshot = snapshot_audit_result(result, repository_root=root)

    assert snapshot.verdict == result.verdict.value
    assert snapshot.minimum_improvement == 0.125
    assert [finding.severity for finding in snapshot.findings] == [
        finding.severity.value for finding in result.findings
    ]
    assert [finding.impact for finding in snapshot.findings] == [
        finding.impact.value for finding in result.findings
    ]
    assert [finding.evidence for finding in snapshot.findings] == [
        dict(finding.evidence) for finding in result.findings
    ]
    # The review bridge has no setter/override channel for authority fields.
    assert not hasattr(snapshot, "verdict_override")
    assert not hasattr(snapshot, "threshold_override")


def test_snapshot_fields_are_plain_review_data_not_deterministic_model_objects(
    study_factory,
) -> None:
    """The model layer sees copied scalars, never live Verdict/Finding objects."""

    manifest = study_factory()
    root = manifest.parent
    result = audit_research(manifest, artifact_root=root)
    snapshot = snapshot_audit_result(result, repository_root=root)

    assert not isinstance(snapshot.verdict, Verdict)
    assert all(not isinstance(item.severity, Severity) for item in snapshot.findings)
    assert all(not isinstance(item.impact, Impact) for item in snapshot.findings)
    assert all(not isinstance(item, Finding) for item in snapshot.findings)
    assert isinstance(result, AuditResult)
