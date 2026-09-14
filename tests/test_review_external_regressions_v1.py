"""Frozen external-scope regressions and provider-free replay safety tests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claimci.review.evidence import (
    EvidenceKind,
    EvidenceProvenance,
    discover_evidence,
)
from claimci.review.inventory import (
    build_git_change_inventory,
    exact_git_material_omission_matches,
)
from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ClaimType,
    ComparisonBasis,
    ExactMaterialOmission,
    GateDisposition,
    ProviderUsage,
    ReviewConfig,
    ReviewLimits,
    ReviewMaterialKind,
    ReviewStatus,
    ScientificClaim,
    SnapshotIdentity,
    SnapshotRole,
    SourceKind,
    SourceLocation,
)
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.path_policy import classify_review_material, is_source_file
from claimci.review.preflight import build_review_scope, preflight_review
from claimci.review.provider import ProviderResponse, StructuredRequest
from scripts.replay_review_preflight import (
    CandidateInputLayout,
    FrozenCandidate,
    _digest_metadata_tree,
    _reject_unsafe_source_config,
    _review_config,
    _source_identity,
    load_frozen_manifests,
    replay_candidate,
    replay_study,
    resolve_replay_paths,
    serialize_preflight,
)


FIXTURES = Path(__file__).parent / "fixtures" / "review-preflight-v1"
EXPECTED_COORDINATES = {
    "A-InferenceX-2630": (
        "bdca939fe5ca04cf23bb06cbc83cbdfe95030203",
        "bdca939fe5ca04cf23bb06cbc83cbdfe95030203",
        "c38fb02378565de853fed5860c7459473f655a83",
        ComparisonBasis.DIRECT_BASE,
        58,
        Counter(added=10, modified=16, deleted=32),
    ),
    "B-Transformers-48455": (
        "8e35c981dbdb2caf75b341023e243d834f829b2b",
        "8e35c981dbdb2caf75b341023e243d834f829b2b",
        "7d6d2ca28ac3be64204d49b89cf0651bd8ed923f",
        ComparisonBasis.DIRECT_BASE,
        2,
        Counter(modified=2),
    ),
    "C-Weave-7801": (
        "625a8241cd46898f86b8290a937f3f1d3ef71729",
        "c31ce92835cd2415bbb3c7361e756b12ce2133f9",
        "a1c2bc8e82f96980651de1c6eb8b331cb804a144",
        ComparisonBasis.MERGE_BASE,
        12,
        Counter(added=2, modified=10),
    ),
    "D-OpenASR-205": (
        "d1e99b25524814332d6868a5645e568670834cfb",
        "d1e99b25524814332d6868a5645e568670834cfb",
        "9e7a2dfeda0eb3284a87a0105e8dedd45f7967d6",
        ComparisonBasis.DIRECT_BASE,
        1,
        Counter(modified=1),
    ),
    "E-NVCF-1425": (
        "72485b3e08a08461474fef5d0977456a6544e73a",
        "812a162ad964c077357a86a69f8f08ba11f87d9b",
        "c2a1dd54782e17a2eb443b428cd01eb9964abdd1",
        ComparisonBasis.MERGE_BASE,
        10,
        Counter(added=9, modified=1),
    ),
}


def _credential_free_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ):
        env.pop(name, None)
    return env


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
            "-c",
            "protocol.file.allow=always",
            *args,
        ],
        cwd=root,
        env=_credential_free_env(),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _write(root: Path, relative: str, text: str) -> None:
    target = root / Path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    sha = _git(root, "rev-parse", "HEAD")
    _stabilize_index_metadata(root)
    return sha


def _stabilize_index_metadata(root: Path) -> None:
    index = Path(
        _git(root, "rev-parse", "--path-format=absolute", "--git-path", "index")
    )
    metadata = index.stat()
    os.utime(
        index,
        ns=(
            metadata.st_atime_ns,
            max(metadata.st_mtime_ns, time.time_ns()) + 1_000_000,
        ),
    )


def _new_repo(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")


def _identity(role: SnapshotRole, root: Path, sha: str) -> SnapshotIdentity:
    if (root / ".git").exists():
        _stabilize_index_metadata(root)
    return SnapshotIdentity(role, root.resolve(), sha)


def _scope_inputs(root: Path, *, title: str, description: str = "") -> Any:
    return SimpleNamespace(
        repository_root=root.resolve(),
        base_root=None,
        pr_title=title,
        pr_description=description,
        requested_base=None,
        comparison_base=None,
        head=None,
        inventory=None,
        coordinates=None,
        inventory_failure=None,
    )


def test_frozen_manifests_have_exact_coordinates_counts_statuses_and_sorted_paths() -> None:
    manifests = load_frozen_manifests(FIXTURES)

    assert tuple(item.candidate for item in manifests) == tuple(EXPECTED_COORDINATES)
    for candidate in manifests:
        requested, comparison, head, basis, count, statuses = EXPECTED_COORDINATES[
            candidate.candidate
        ]
        assert (
            candidate.requested_base_sha,
            candidate.comparison_base_sha,
            candidate.head_sha,
            candidate.comparison_basis,
            candidate.declared_entry_count,
        ) == (requested, comparison, head, basis, count)
        assert Counter(entry.status.value for entry in candidate.entries) == statuses
        assert tuple(entry.path for entry in candidate.entries) == tuple(
            sorted(entry.path for entry in candidate.entries)
        )
        assert candidate.inventory.entries == candidate.entries


def test_frozen_manifests_materialize_only_declared_non_deleted_paths_without_disappearance(
    tmp_path: Path,
) -> None:
    scopes = {}
    for candidate in load_frozen_manifests(FIXTURES):
        root = tmp_path / candidate.candidate
        root.mkdir()
        for entry in candidate.entries:
            if entry.status is not ChangeStatus.DELETED:
                _write(root, entry.path, "benchmark accuracy improves by 5%\n")
        inputs = _scope_inputs(
            root,
            title="Benchmark accuracy improves by 5% for the model",
            description="The changed evaluation configuration is reproducible.",
        )
        first = build_review_scope(inputs, ReviewConfig(enabled=True), candidate.inventory)
        second = build_review_scope(inputs, ReviewConfig(enabled=True), candidate.inventory)
        assert first == second
        assert first.inventory.entries == candidate.entries
        declared_non_deleted = {
            entry.path
            for entry in candidate.entries
            if entry.status is not ChangeStatus.DELETED
        }
        assert set(first.issued_paths).issubset(declared_non_deleted)
        scopes[candidate.candidate] = first

    transformer = scopes["B-Transformers-48455"]
    assert transformer.issued_paths == (
        "src/transformers/models/glm5_next/modeling_glm5_next.py",
        "src/transformers/models/glm5_next/modular_glm5_next.py",
    )


def test_declared_inventory_never_reads_unrelated_comparison_or_huge_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    for index in range(513):
        _write(root, f"unrelated/{index:04d}.py", "same\n")
    huge = root / "tests/integrations/cassette.yaml"
    huge.parent.mkdir(parents=True)
    huge.write_bytes(b"x" * 1_048_577)
    _write(root, "zz_target/benchmark.py", "throughput improves by 10%\n")
    inventory = ChangeInventory(
        schema_version=1,
        requested_base_sha="1" * 40,
        comparison_base_sha="1" * 40,
        head_sha="2" * 40,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=1,
        complete=True,
        entries=(ChangeEntry("zz_target/benchmark.py", ChangeStatus.MODIFIED),),
    )
    import claimci.review.preflight as preflight_module

    opened: list[str] = []
    real_capture = preflight_module.capture_confined_regular_file

    def traced_capture(base: Path, relative: str, **kwargs: object):
        if relative.startswith("unrelated/") or relative.endswith("cassette.yaml"):
            raise AssertionError("unrelated checkout content was opened")
        opened.append(relative)
        return real_capture(base, relative, **kwargs)

    monkeypatch.setattr(preflight_module, "capture_confined_regular_file", traced_capture)
    scope = build_review_scope(
        _scope_inputs(root, title="Benchmark throughput improves by 10%"),
        ReviewConfig(enabled=True),
        inventory,
    )

    assert opened == ["zz_target/benchmark.py"]
    assert scope.selected_paths == ("zz_target/benchmark.py",)


def test_required_oversize_changed_material_is_honestly_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    required = root / "results/required.json"
    required.parent.mkdir(parents=True)
    with required.open("wb") as stream:
        stream.seek(16 * 1024 * 1024)
        stream.write(b"x")
    inventory = ChangeInventory(
        schema_version=1,
        requested_base_sha="1" * 40,
        comparison_base_sha="1" * 40,
        head_sha="2" * 40,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=1,
        complete=True,
        entries=(ChangeEntry("results/required.json", ChangeStatus.ADDED),),
    )

    scope = build_review_scope(
        _scope_inputs(
            root,
            title="Benchmark accuracy improves by 5%",
            description="See results/required.json for the result.",
        ),
        ReviewConfig(enabled=True),
        inventory,
    )

    assert scope.complete is False
    assert scope.selected_paths == ()
    assert any(
        issue.code == "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"
        and issue.path == "results/required.json"
        for issue in scope.issues
    )


def test_openasr_shell_is_passive_supporting_config_never_source_or_result(
    tmp_path: Path,
) -> None:
    manifest = next(
        item
        for item in load_frozen_manifests(FIXTURES)
        if item.candidate == "D-OpenASR-205"
    )
    path = manifest.entries[0].path
    _write(tmp_path, path, "MODEL_IDs+=(bosonai/Qwen3-ASR-1.7B-hf-orze)\n")
    claim = ScientificClaim(
        claim_id="openasr-model",
        source_text="The model reports mean WER 4.33.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="Qwen3 ASR WER",
        source=SourceLocation(
            source_id="pr-description",
            kind=SourceKind.PULL_REQUEST_DESCRIPTION,
            path=None,
            start_line=1,
            end_line=1,
        ),
        evidence_hints=(path,),
    )

    bundle = discover_evidence(
        tmp_path,
        (claim,),
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
    )

    reference = bundle.references[0]
    assert classify_review_material(path) is ReviewMaterialKind.SUBMISSION_CONFIG
    assert is_source_file(path) is False
    assert reference.kind is EvidenceKind.CONFIG
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT


@pytest.mark.parametrize("candidate_name", ["C-Weave-7801", "E-NVCF-1425"])
def test_merge_base_candidates_fail_when_verified_as_direct_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
) -> None:
    candidate = next(
        item for item in load_frozen_manifests(FIXTURES) if item.candidate == candidate_name
    )
    root = tmp_path.resolve()
    inputs = ReviewInputs(
        repository_root=root,
        pr_title="Benchmark throughput improves by 10%",
        requested_base=_identity(
            SnapshotRole.REQUESTED_BASE, root, candidate.requested_base_sha
        ),
        comparison_base=_identity(
            SnapshotRole.COMPARISON_BASE, root, candidate.comparison_base_sha
        ),
        head=_identity(SnapshotRole.HEAD, root, candidate.head_sha),
        inventory=candidate.inventory,
    )
    wrong = ChangeInventory(
        schema_version=1,
        requested_base_sha=candidate.requested_base_sha,
        comparison_base_sha=candidate.requested_base_sha,
        head_sha=candidate.head_sha,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=len(candidate.entries),
        complete=True,
        entries=candidate.entries,
    )
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory", lambda *_args: wrong
    )

    result = preflight_review(inputs, ReviewConfig(enabled=True))

    assert result.ready_for_provider is False
    assert result.gates[0].disposition is GateDisposition.FAIL
    assert [reason.code for reason in result.gates[0].reasons] == [
        "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH"
    ]


class _TwoCallProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[StructuredRequest] = []

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        assert self.events == ["preflight_pass", "preflight_pass"]
        self.events.append("extract")
        self.calls.append(request)
        source = request.payload["sources"][0]
        return ProviderResponse(
            output_text=json.dumps(
                {
                    "claims": [
                        {
                            "source_text": source["text"],
                            "claim_type": "metric_improvement",
                            "subject": "candidate accuracy",
                            "metric": "accuracy",
                            "direction": "higher",
                            "claimed_magnitude": None,
                            "qualifiers": [],
                            "source": {
                                "source_id": source["source_id"],
                                "start_line": 1,
                                "end_line": 1,
                            },
                            "confidence": 0.9,
                            "evidence_hints": ["results.json"],
                        }
                    ]
                }
            ),
            provider="fake",
            model="fake",
            request_id="fake-1",
            usage=ProviderUsage(),
        )

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        assert self.events == ["preflight_pass", "preflight_pass", "extract"]
        self.events.append("synthesize")
        self.calls.append(request)
        claim_id = request.payload["claims"][0]["claim_id"]
        citations = [item["evidence_id"] for item in request.payload["evidence"]]
        return ProviderResponse(
            output_text=json.dumps(
                {
                    "interpretations": [
                        {
                            "claim_id": claim_id,
                            "interpretation": "Advisory only.",
                            "citations": citations[:1],
                            "missing_evidence": [],
                            "unsupported_inferences": [],
                            "confidence": 0.6,
                        }
                    ]
                }
            ),
            provider="fake",
            model="fake",
            request_id="fake-2",
            usage=ProviderUsage(),
        )


def test_provider_is_injected_only_after_pass_and_never_exceeds_two_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "head"
    root.mkdir()
    base_root = tmp_path / "base"
    base_root.mkdir()
    _write(root, "results.json", '{"accuracy": 0.95}\n')
    sha = "1" * 40
    inventory = ChangeInventory(
        schema_version=1,
        requested_base_sha=sha,
        comparison_base_sha=sha,
        head_sha="2" * 40,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
        declared_entry_count=1,
        complete=True,
        entries=(ChangeEntry("results.json", ChangeStatus.ADDED),),
    )
    inputs = ReviewInputs(
        repository_root=root.resolve(),
        pr_title="Benchmark accuracy improves by 5% in results.json",
        requested_base=_identity(SnapshotRole.REQUESTED_BASE, base_root, sha),
        comparison_base=_identity(SnapshotRole.COMPARISON_BASE, base_root, sha),
        head=_identity(SnapshotRole.HEAD, root, "2" * 40),
        inventory=inventory,
    )
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory", lambda *_args: inventory
    )
    monkeypatch.setattr(
        "claimci.review.orchestrator.verify_git_material_identities",
        lambda *_args, **_kwargs: None,
    )
    import claimci.review.preflight as preflight_module

    real_preflight = preflight_module._plan_preflight_review
    events: list[str] = []

    def observed_preflight(*args: object, **kwargs: object):
        result = real_preflight(*args, **kwargs)
        assert result.ready_for_provider
        events.append("preflight_pass")
        return result

    monkeypatch.setattr(
        preflight_module, "_plan_preflight_review", observed_preflight
    )
    provider = _TwoCallProvider(events)

    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert events == [
        "preflight_pass",
        "preflight_pass",
        "extract",
        "synthesize",
    ]
    assert len(provider.calls) == 2
    assert len(result.provider_calls) == 2


def test_selected_material_is_bound_to_exact_head_blob_under_repeated_read_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-consistent worktree reads cannot substitute bytes outside declared HEAD."""

    import claimci.review.orchestrator as orchestrator_module
    import claimci.review.preflight as preflight_module

    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    _write(repository, "README.md", "base\n")
    base_sha = _commit(repository, "base")
    committed = b"Benchmark accuracy improves by 5 percent.\n"
    transient = b"TRANSIENT non-HEAD accuracy improves by 50 percent.\n"
    study = repository / "study.md"
    study.write_bytes(committed)
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested,
        comparison,
        head,
        ComparisonBasis.DIRECT_BASE,
    )
    inputs = ReviewInputs(
        repository_root=repository,
        pr_title="Benchmark accuracy improves by 5%; see study.md",
        requested_base=requested,
        comparison_base=comparison,
        head=head,
        inventory=inventory,
    )

    def wrap_reader(reader):
        def transient_read(root: Path, path: str, *, max_bytes: int):
            if Path(root).resolve() != repository or path != "study.md":
                return reader(root, path, max_bytes=max_bytes)
            study.write_bytes(transient)
            try:
                return reader(root, path, max_bytes=max_bytes)
            finally:
                study.write_bytes(committed)

        return transient_read

    monkeypatch.setattr(
        preflight_module,
        "capture_confined_regular_file",
        wrap_reader(preflight_module.capture_confined_regular_file),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "inspect_confined_regular_file",
        wrap_reader(orchestrator_module.inspect_confined_regular_file),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "capture_confined_regular_file",
        wrap_reader(orchestrator_module.capture_confined_regular_file),
    )
    free_preflight = preflight_review(inputs, ReviewConfig(enabled=True))
    assert free_preflight.ready_for_provider is False
    assert free_preflight.scope is None
    assert free_preflight.gates[0].disposition is GateDisposition.FAIL
    provider = _TwoCallProvider([])

    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert result.status is ReviewStatus.UNAVAILABLE
    assert result.error_code == "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"
    assert result.preflight is not None
    assert result.preflight.scope is None
    assert result.provider_calls == ()
    assert provider.calls == []
    assert study.read_bytes() == committed
    assert not _git(repository, "status", "--porcelain")


@pytest.mark.parametrize("source_path", ["src/Stable.java", "Stable.java"])
def test_exact_unchanged_source_literal_is_a_bounded_deterministic_supplement(
    tmp_path: Path,
    source_path: str,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    _write(repository, source_path, "final class Stable { int score = 95; }\n")
    base_sha = _commit(repository, "base")
    _write(repository, "results/metrics.json", '{"accuracy": 0.95}\n')
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5%",
            pr_description=(
                f"The exact implementation is {source_path} and measurements "
                "are in results/metrics.json."
            ),
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
    )

    assert result.scope is not None
    assert result.scope.selected_paths == (
        "results/metrics.json",
        source_path,
    )
    assert result.scope.issued_changed_paths == ("results/metrics.json",)
    assert not any(
        issue.code == "PREFLIGHT_G2_OUT_OF_SCOPE_PATH"
        and issue.path == source_path
        for issue in result.scope.issues
    )


def test_changed_manifest_issues_unchanged_declared_dependencies_as_supplements(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    dependencies = (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    )
    for relative in dependencies:
        _write(repository, relative, "{}\n")
    manifest = """claim:
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
    _write(repository, "research.yaml", manifest.replace("0.05", "0.04"))
    base_sha = _commit(repository, "base")
    _write(repository, "research.yaml", manifest)
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5% with reproducible configuration",
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
    )

    assert result.scope is not None
    assert result.scope.issued_changed_paths == ("research.yaml",)
    assert result.scope.selected_paths == ("research.yaml", *sorted(dependencies))


def test_unchanged_root_manifest_is_supplemented_when_its_dependency_changes(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    dependencies = (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    )
    for relative in dependencies:
        _write(repository, relative, "{}\n")
    _write(
        repository,
        "research.yaml",
        """claim:
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
""",
    )
    base_sha = _commit(repository, "base")
    _write(repository, "candidate/results.json", '{"accuracy": 0.95}\n')
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5% with reproducible configuration",
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
    )

    assert result.scope is not None
    assert result.scope.issued_changed_paths == ("candidate/results.json",)
    assert result.scope.selected_paths == (
        "candidate/results.json",
        "research.yaml",
        *sorted(set(dependencies) - {"candidate/results.json"}),
    )
    assert result.scope.atomic_path_groups == (
        ("research.yaml", *sorted(dependencies)),
    )


def test_manifest_bundle_is_rejected_atomically_when_dependencies_exceed_file_cap(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    dependencies = (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    )
    for relative in dependencies:
        _write(repository, relative, "{}\n")
    manifest = """claim:
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
    _write(repository, "research.yaml", manifest.replace("0.05", "0.04"))
    base_sha = _commit(repository, "base")
    _write(repository, "research.yaml", manifest)
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5% in research.yaml",
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True, limits=ReviewLimits(max_files=3)),
    )

    assert result.ready_for_provider is False
    assert result.gates[1].disposition is GateDisposition.FAIL
    assert result.gates[2].disposition is GateDisposition.NOT_EVALUATED
    assert result.scope is not None
    assert result.scope.selected_paths == ()
    assert result.scope.atomic_path_groups == ()
    assert any(
        issue.code == "PREFLIGHT_G2_OUT_OF_SCOPE_PATH"
        and issue.path == "research.yaml"
        for issue in result.scope.issues
    )


def test_nested_exact_manifest_literal_brings_its_complete_atomic_dependency_scope(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    manifest_path = "studies/demo/research.yaml"
    dependency_names = (
        "base-config.yaml",
        "base-results.json",
        "base-train.jsonl",
        "base-eval.jsonl",
        "candidate-config.yaml",
        "candidate-results.json",
        "candidate-train.jsonl",
        "candidate-eval.jsonl",
    )
    for name in dependency_names:
        _write(repository, f"studies/demo/{name}", "{}\n")
    _write(
        repository,
        manifest_path,
        """claim:
  metric: accuracy
  minimum_improvement: 0.05
baseline:
  config: base-config.yaml
  results: base-results.json
  train_dataset: base-train.jsonl
  eval_dataset: base-eval.jsonl
candidate:
  config: candidate-config.yaml
  results: candidate-results.json
  train_dataset: candidate-train.jsonl
  eval_dataset: candidate-eval.jsonl
""",
    )
    base_sha = _commit(repository, "base")
    _write(repository, "results/metrics.json", '{"accuracy": 0.95}\n')
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5%",
            pr_description=f"Audit inputs are declared by {manifest_path}.",
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
    )

    expected_group = (
        manifest_path,
        *sorted(f"studies/demo/{name}" for name in dependency_names),
    )
    assert result.scope is not None
    assert result.scope.atomic_path_groups == (expected_group,)
    assert set(expected_group).issubset(result.scope.selected_paths)


def test_deleted_root_manifest_dependency_is_bound_as_atomic_audit_absence(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    dependencies = (
        "base/config.yaml",
        "base/results.json",
        "base/train.jsonl",
        "base/eval.jsonl",
        "candidate/config.yaml",
        "candidate/results.json",
        "candidate/train.jsonl",
        "candidate/eval.jsonl",
    )
    for relative in dependencies:
        _write(repository, relative, "{}\n")
    _write(
        repository,
        "research.yaml",
        """claim:
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
""",
    )
    base_sha = _commit(repository, "base")
    (repository / "candidate" / "eval.jsonl").unlink()
    _write(repository, "results/metrics.json", '{"accuracy": 0.95}\n')
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )

    result = preflight_review(
        ReviewInputs(
            repository_root=repository,
            pr_title="Accuracy improves by 5%",
            requested_base=requested,
            comparison_base=comparison,
            head=head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
    )

    assert result.scope is not None
    assert "research.yaml" in result.scope.selected_paths
    assert "candidate/eval.jsonl" not in result.scope.selected_paths
    assert result.scope.atomic_path_groups == (
        ("research.yaml", *sorted(dependencies)),
    )
    assert not any(
        issue.code == "PREFLIGHT_G2_OUT_OF_SCOPE_PATH"
        and issue.path == "candidate/eval.jsonl"
        for issue in result.scope.issues
    )


def test_deleted_manifest_dependency_remains_relevant_to_runtime_audit(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    fixture = Path(__file__).parents[1] / "examples" / "day2_demo"
    audit_paths = (
        "research.yaml",
        "baseline-config.yaml",
        "baseline-results.json",
        "baseline-train.jsonl",
        "baseline-eval.jsonl",
        "candidate-config.yaml",
        "candidate-results.json",
        "candidate-train.jsonl",
        "candidate-eval.jsonl",
    )
    for relative in audit_paths:
        _write(
            repository,
            relative,
            (fixture / relative).read_text(encoding="utf-8"),
        )
    base_sha = _commit(repository, "base")
    (repository / "candidate-eval.jsonl").unlink()
    _write(repository, "claim-results.json", '{"accuracy": 0.95}\n')
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )
    inputs = ReviewInputs(
        repository_root=repository,
        pr_title="Accuracy improves by 5%",
        requested_base=requested,
        comparison_base=comparison,
        head=head,
        inventory=inventory,
    )
    provider = _TwoCallProvider(["preflight_pass", "preflight_pass"])

    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert ChangeEntry("candidate-eval.jsonl", ChangeStatus.DELETED) in inventory.entries
    assert result.preflight is not None and result.preflight.scope is not None
    assert result.preflight.scope.atomic_path_groups == (
        ("research.yaml", *sorted(audit_paths[1:])),
    )
    assert "candidate-eval.jsonl" not in result.preflight.scope.selected_paths
    extraction_paths = {
        source["path"]
        for source in provider.calls[0].payload["sources"]
        if source["path"] is not None
    }
    assert "candidate-eval.jsonl" not in extraction_paths
    assert len(result.deterministic_audits) == 1
    audit = result.deterministic_audits[0]
    assert audit.verdict == "NOT_SUPPORTED"
    assert "DATASET.MISSING" in {finding.rule_id for finding in audit.findings}
    assert [call.task for call in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def test_manifest_discovery_cap_is_explicit_without_changing_audit_selection(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    _write(repository, "README.md", "base\n")
    base_sha = _commit(repository, "base")
    for relative in (
        "a/research.yaml",
        "a/research.yml",
        "b/research.yaml",
        "b/research.yml",
    ):
        _write(repository, relative, "not: [valid\n")
    fixture = Path(__file__).parents[1] / "examples" / "day2_demo"
    valid_paths = (
        "research.yaml",
        "baseline-config.yaml",
        "baseline-results.json",
        "baseline-train.jsonl",
        "baseline-eval.jsonl",
        "candidate-config.yaml",
        "candidate-results.json",
        "candidate-train.jsonl",
        "candidate-eval.jsonl",
    )
    for relative in valid_paths:
        _write(
            repository,
            f"c/{relative}",
            (fixture / relative).read_text(encoding="utf-8"),
        )
    head_sha = _commit(repository, "head")
    base_root = (tmp_path / "base").resolve()
    _git(repository, "worktree", "add", "-q", "--detach", str(base_root), base_sha)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base_sha)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base_sha)
    head = _identity(SnapshotRole.HEAD, repository, head_sha)
    inventory = build_git_change_inventory(
        requested, comparison, head, ComparisonBasis.DIRECT_BASE
    )
    inputs = ReviewInputs(
        repository_root=repository,
        pr_title="Candidate improves accuracy from 0.60 to 0.90",
        pr_description="The bounded deterministic inputs are in c/research.yaml.",
        requested_base=requested,
        comparison_base=comparison,
        head=head,
        inventory=inventory,
    )
    provider = _TwoCallProvider(["preflight_pass", "preflight_pass"])

    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert result.preflight is not None and result.preflight.scope is not None
    assert result.status is ReviewStatus.PARTIAL
    assert result.preflight.scope.complete is False
    assert "c/research.yaml" in result.preflight.scope.selected_paths
    assert result.preflight.scope.atomic_path_groups == ()
    assert result.deterministic_audits == ()
    assert result.preflight.gates[2].disposition is GateDisposition.PASS_PARTIAL
    assert "PREFLIGHT_G3_AUDIT_PLAN_OMITTED" in {
        issue.code for issue in result.preflight.gates[2].reasons
    }
    assert [call.task for call in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]


def _same_head_graph(tmp_path: Path) -> dict[str, Any]:
    repository = (tmp_path / "repository").resolve()
    _new_repo(repository)
    _write(repository, "README.md", "baseline\n")
    common = _commit(repository, "common")
    _git(repository, "branch", "requested", common)
    _write(repository, "src/earlier.py", "earlier = 1\n")
    direct = _commit(repository, "direct")
    _write(repository, "src/later.py", "later = 2\n")
    head = _commit(repository, "head")
    _git(repository, "checkout", "-q", "requested")
    _write(repository, "docs/requested.md", "requested branch\n")
    requested = _commit(repository, "requested")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    roots = {}
    for name, sha in (
        ("common", common),
        ("direct", direct),
        ("head", head),
        ("requested", requested),
    ):
        root = (snapshots / name).resolve()
        _git(repository, "worktree", "add", "-q", "--detach", str(root), sha)
        roots[name] = root
    return {
        "roots": roots,
        "common": common,
        "direct": direct,
        "head": head,
        "requested": requested,
    }


def test_same_head_different_base_coordinates_do_not_share_scope_or_cache(
    tmp_path: Path,
) -> None:
    graph = _same_head_graph(tmp_path)
    roots = graph["roots"]
    direct_inventory = build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, roots["direct"], graph["direct"]),
        _identity(SnapshotRole.COMPARISON_BASE, roots["direct"], graph["direct"]),
        _identity(SnapshotRole.HEAD, roots["head"], graph["head"]),
        ComparisonBasis.DIRECT_BASE,
    )
    merge_inventory = build_git_change_inventory(
        _identity(
            SnapshotRole.REQUESTED_BASE, roots["requested"], graph["requested"]
        ),
        _identity(SnapshotRole.COMPARISON_BASE, roots["common"], graph["common"]),
        _identity(SnapshotRole.HEAD, roots["head"], graph["head"]),
        ComparisonBasis.MERGE_BASE,
    )

    def run(inventory: ChangeInventory, requested_root: Path, comparison_root: Path):
        inputs = ReviewInputs(
            repository_root=roots["head"],
            pr_title="The model implementation improves by 5%",
            pr_description="Review src/earlier.py and src/later.py.",
            requested_base=_identity(
                SnapshotRole.REQUESTED_BASE,
                requested_root,
                inventory.requested_base_sha,
            ),
            comparison_base=_identity(
                SnapshotRole.COMPARISON_BASE,
                comparison_root,
                inventory.comparison_base_sha,
            ),
            head=_identity(SnapshotRole.HEAD, roots["head"], graph["head"]),
            inventory=inventory,
        )
        return preflight_review(inputs, ReviewConfig(enabled=True))

    first_direct = run(direct_inventory, roots["direct"], roots["direct"])
    merge = run(merge_inventory, roots["requested"], roots["common"])
    second_direct = run(direct_inventory, roots["direct"], roots["direct"])

    assert direct_inventory.head_sha == merge_inventory.head_sha
    assert direct_inventory.entries != merge_inventory.entries
    assert first_direct.scope is not None and merge.scope is not None
    assert first_direct.scope.selected_paths != merge.scope.selected_paths
    assert serialize_preflight(first_direct) != serialize_preflight(merge)
    assert serialize_preflight(first_direct) == serialize_preflight(second_direct)


def test_replay_paths_reject_urls_and_outputs_inside_study(tmp_path: Path) -> None:
    study = tmp_path / "ClaimCI-External-Study-2026-09-02"
    study.mkdir()
    outside = tmp_path / "replay.json"

    with pytest.raises(ValueError, match="local"):
        resolve_replay_paths("https://example.invalid/study", outside)
    with pytest.raises(ValueError, match="outside"):
        resolve_replay_paths(study, study / "report" / "replay.json")
    assert resolve_replay_paths(study, outside) == (
        study.resolve(),
        outside.resolve(),
    )


def test_source_identity_detects_unreferenced_object_store_addition(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source").resolve()
    _new_repo(source)
    _write(source, "tracked.txt", "tracked\n")
    _commit(source, "baseline")
    empty_hooks = (tmp_path / "empty-hooks").resolve()
    empty_hooks.mkdir()
    before = _source_identity(source, empty_hooks)

    completed = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=source,
        env=_credential_free_env(),
        input=b"unreferenced frozen-source mutation\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")

    after = _source_identity(source, empty_hooks)

    assert before != after
    assert before["head_sha"] == after["head_sha"]
    assert before["tracked_file_identity_sha256"] == after[
        "tracked_file_identity_sha256"
    ]
    assert after["git_object_store_entry_count"] > before[
        "git_object_store_entry_count"
    ]
    assert after["git_object_store_sha256"] != before["git_object_store_sha256"]


def test_replay_rejects_executable_alternate_refs_config_before_clone(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source").resolve()
    _new_repo(source)
    _git(
        source,
        "config",
        "core.alternateRefsCommand",
        "claimci-test-sentinel-command",
    )
    empty_hooks = (tmp_path / "empty-hooks").resolve()
    empty_hooks.mkdir()

    with pytest.raises(ValueError, match="not passive"):
        _reject_unsafe_source_config(source, empty_hooks)


def test_metadata_digest_prunes_top_level_objects_before_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = (tmp_path / "git-metadata").resolve()
    objects = metadata / "objects"
    objects.mkdir(parents=True)
    (objects / "must-not-be-visited").mkdir()
    (objects / "must-not-be-visited" / "payload").write_bytes(b"object payload")
    config = metadata / "config"
    config.write_bytes(b"[core]\n\tbare = false\n")
    real_scandir = os.scandir

    def reject_object_descent(path):
        scanned = Path(path).resolve(strict=True)
        if scanned == objects:
            raise AssertionError("metadata traversal entered the object store")
        return real_scandir(path)

    monkeypatch.setattr(
        "scripts.replay_review_preflight.os.scandir", reject_object_descent
    )
    digest = hashlib.sha256()

    count, size = _digest_metadata_tree(metadata, namespace="common", sink=digest)

    assert count == 1
    assert size == len(b"[core]\n\tbare = false\n")


def test_review_config_provenance_parses_the_same_snapshot_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (Path(__file__).parents[1] / ".claimci" / "review.yaml").read_bytes()
    transient = original.replace(b"model: gpt-5.6-terra", b"model: transient-model")
    assert transient != original
    config_path = tmp_path / ".claimci" / "review.yaml"
    config_path.parent.mkdir()
    config_path.write_bytes(transient)

    monkeypatch.setattr("scripts.replay_review_preflight._WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.replay_review_preflight._read_bounded",
        lambda path: original if Path(path) == config_path else Path(path).read_bytes(),
    )

    config, provenance = _review_config()

    assert config.model == "gpt-5.6-terra"
    assert provenance["effective"]["model"] == "gpt-5.6-terra"
    assert provenance["source_sha256"] == hashlib.sha256(original).hexdigest()


def test_replay_publication_never_truncates_an_outside_hardlink_to_study_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    study.mkdir()
    frozen = study / "frozen-source.txt"
    frozen_bytes = b"frozen study content must remain byte-identical\n"
    frozen.write_bytes(frozen_bytes)
    destination = (tmp_path / "replay.json").resolve()
    os.link(frozen, destination)
    aliased_inode = frozen.stat().st_ino

    def provider_free_result(_study, candidate, _layout, _config, *, temp_parent=None):
        assert temp_parent is None
        return {
            "candidate": candidate.candidate,
            "ready": True,
            "provider_calls": 0,
        }

    monkeypatch.setattr(
        "scripts.replay_review_preflight.replay_candidate", provider_free_result
    )
    runtime_code = {
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
        "worktree_clean": True,
    }
    monkeypatch.setattr(
        "scripts.replay_review_preflight._runtime_code_identity",
        lambda: runtime_code,
    )

    payload = replay_study(study, destination, fixture_directory=FIXTURES)

    assert payload["ready_count"] == 5
    assert payload["provider_calls"] == 0
    assert payload["runtime_code"] == runtime_code
    config_bytes = (
        Path(__file__).parents[1] / ".claimci" / "review.yaml"
    ).read_bytes()
    assert payload["review_config"] == {
        "source_path": ".claimci/review.yaml",
        "source_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "effective": {
            "schema_version": 1,
            "enabled": True,
            "policy": "advisory",
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "limits": {
                "max_calls": 2,
                "max_context_chars": 60_000,
                "max_output_chars": 24_000,
                "max_files": 24,
                "max_file_chars": 16_000,
                "max_claims": 16,
                "extraction_max_output_tokens": 5_000,
                "synthesis_max_output_tokens": 4_000,
                "timeout_seconds": 90.0,
                "retries": 0,
            },
        },
    }
    assert frozen.read_bytes() == frozen_bytes
    assert frozen.stat().st_ino == aliased_inode
    assert destination.stat().st_ino != aliased_inode
    assert json.loads(destination.read_text(encoding="utf-8"))[
        "mode"
    ] == "provider_free_frozen_local_replay"
    assert not tuple(destination.parent.glob(".claimci-replay-*.tmp"))


def _merge_replay_source(study: Path) -> tuple[FrozenCandidate, CandidateInputLayout]:
    source = study / "paid" / "X-Synthetic-1" / "head"
    _new_repo(source)
    _write(source, "README.md", "baseline\n")
    comparison = _commit(source, "comparison")
    _git(source, "branch", "requested", comparison)
    _write(source, "results/metric.json", '{"throughput": 10}\n')
    head = _commit(source, "head")
    _git(source, "checkout", "-q", "requested")
    _write(source, "requested-only.txt", "request\n")
    requested = _commit(source, "requested")
    _git(source, "checkout", "-q", "--detach", head)
    metadata = study / "paid" / "X-Synthetic-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark throughput improves by 10%`\n", encoding="utf-8"
    )
    (metadata / "pr-description.md").write_text(
        "The result is in results/metric.json and is reproducible.\n",
        encoding="utf-8",
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Synthetic-1",
        repository="example/synthetic",
        pull_request=1,
        requested_base_sha=requested,
        comparison_base_sha=comparison,
        head_sha=head,
        comparison_basis=ComparisonBasis.MERGE_BASE,
        declared_entry_count=1,
        entries=(ChangeEntry("results/metric.json", ChangeStatus.ADDED),),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Synthetic-1/head",
        metadata_relative="paid/X-Synthetic-1/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Synthetic-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    return candidate, layout


def test_replay_uses_disposable_local_clone_materializes_exact_merge_base_and_preserves_source(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    study.mkdir()
    candidate, layout = _merge_replay_source(study)
    temp_parent = (tmp_path / "replay-temporary").resolve()
    temp_parent.mkdir()

    result = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert result["provider_calls"] == 0
    assert result["source_unchanged"] is True
    assert result["source_before"] == result["source_after"]
    assert result["materialized_coordinates"] == {
        "requested_base_sha": candidate.requested_base_sha,
        "comparison_base_sha": candidate.comparison_base_sha,
        "head_sha": candidate.head_sha,
    }
    assert result["preflight"]["coordinates"]["comparison_basis"] == "merge_base"
    assert not any(temp_parent.iterdir())


def test_replay_hydrates_exact_unchanged_literal_before_bound_preflight(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Supplement-1" / "head"
    _new_repo(source)
    _write(source, "src/accuracy.py", "EXPECTED_ACCURACY = 0.90\n")
    base = _commit(source, "base")
    _write(source, "results/metric.json", '{"accuracy": 0.90}\n')
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Supplement-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    (metadata / "pr-description.md").write_text(
        "The implementation is in src/accuracy.py and results/metric.json.\n",
        encoding="utf-8",
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Supplement-1",
        repository="example/supplement",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=1,
        entries=(ChangeEntry("results/metric.json", ChangeStatus.ADDED),),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Supplement-1/head",
        metadata_relative="paid/X-Supplement-1/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Supplement-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    temp_parent = (tmp_path / "supplement-replay-temporary").resolve()
    temp_parent.mkdir()

    result = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert result["provider_calls"] == 0
    assert result["ready"] is True
    assert result["preflight"]["scope"] is not None
    assert set(result["preflight"]["scope"]["selected_paths"]) == {
        "results/metric.json",
        "src/accuracy.py",
    }
    assert result["source_before"] == result["source_after"]
    assert not any(temp_parent.iterdir())


def test_replay_hydrates_unchanged_manifest_and_its_dependency_closure(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Supplement-2" / "head"
    _new_repo(source)
    fixture = Path(__file__).parents[1] / "examples" / "day2_demo"
    audit_paths = (
        "research.yaml",
        "baseline-config.yaml",
        "baseline-results.json",
        "baseline-train.jsonl",
        "baseline-eval.jsonl",
        "candidate-config.yaml",
        "candidate-results.json",
        "candidate-train.jsonl",
        "candidate-eval.jsonl",
    )
    for relative in audit_paths:
        _write(source, relative, (fixture / relative).read_text(encoding="utf-8"))
    base = _commit(source, "base")
    _write(source, "candidate-results.json", '{"accuracy": 0.95}\n')
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Supplement-2" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Candidate accuracy improves from 0.60 to 0.95`\n",
        encoding="utf-8",
    )
    (metadata / "pr-description.md").write_text(
        "The bounded deterministic inputs are declared in research.yaml.\n",
        encoding="utf-8",
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Supplement-2",
        repository="example/supplement",
        pull_request=2,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=1,
        entries=(ChangeEntry("candidate-results.json", ChangeStatus.MODIFIED),),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Supplement-2/head",
        metadata_relative="paid/X-Supplement-2/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Supplement-2/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    temp_parent = (tmp_path / "manifest-replay-temporary").resolve()
    temp_parent.mkdir()

    result = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert result["provider_calls"] == 0
    assert result["ready"] is True
    assert result["preflight"]["scope"] is not None
    assert set(audit_paths).issubset(
        result["preflight"]["scope"]["selected_paths"]
    )
    assert result["source_before"] == result["source_after"]
    assert not any(temp_parent.iterdir())


def test_gate3_trim_drops_unbound_path_scoped_locality_facts(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "trim-head").resolve()
    _new_repo(source)
    _write(source, "README.md", "base\n")
    base = _commit(source, "base")
    _write(source, "results/a.json", '{"accuracy":0.95}\n')
    for index in range(4):
        _write(
            source,
            f"docs/d{index}.md",
            "".join(
                f"background doc {index} line {line}\n"
                for line in range(2_000)
            ),
        )
    head = _commit(source, "head")
    base_root = (tmp_path / "trim-base").resolve()
    _git(source, "worktree", "add", "-q", "--detach", str(base_root), base)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base)
    exact_head = _identity(SnapshotRole.HEAD, source, head)
    inventory = build_git_change_inventory(
        requested,
        comparison,
        exact_head,
        ComparisonBasis.DIRECT_BASE,
    )

    review = run_review(
        ReviewInputs(
            repository_root=source,
            base_root=base_root,
            pr_title="Benchmark accuracy improves by 5%",
            pr_description="Benchmark evidence: results/a.json",
            requested_base=requested,
            comparison_base=comparison,
            head=exact_head,
            inventory=inventory,
        ),
        ReviewConfig(
            enabled=True,
            limits=ReviewLimits(
                max_context_chars=38_000,
                max_file_chars=16_000,
            ),
        ),
        preflight_only=True,
    )

    assert review.preflight is not None and review.preflight.ready_for_provider
    assert review.preflight.scope is not None
    scope = review.preflight.scope
    document_paths = {f"docs/d{index}.md" for index in range(4)}
    trimmed = document_paths - set(scope.selected_paths)
    assert trimmed
    assert trimmed.isdisjoint(scope.selected_paths)
    assert {
        issue.code for issue in scope.issues if issue.path in trimmed
    } == {"PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT"}
    assert not [
        issue
        for gate in review.preflight.gates
        for issue in gate.reasons
        if issue.path in trimmed
        and issue.code != "PREFLIGHT_G3_ROUTABLE_PATH_INDEX_LIMIT"
    ]


def test_replay_does_not_hydrate_large_unselected_changed_other_file(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Sparse-1" / "head"
    _new_repo(source)
    _write(source, "README.md", "baseline\n")
    base = _commit(source, "base")
    unrelated = source / "assets" / "unrelated.bin"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"x" * 16_777_217)
    _write(source, "results/metric.json", '{"accuracy": 0.90}\n')
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Sparse-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    (metadata / "pr-description.md").write_text(
        "The result is in results/metric.json.\n",
        encoding="utf-8",
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Sparse-1",
        repository="example/sparse",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=2,
        entries=(
            ChangeEntry("assets/unrelated.bin", ChangeStatus.ADDED),
            ChangeEntry("results/metric.json", ChangeStatus.ADDED),
        ),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Sparse-1/head",
        metadata_relative="paid/X-Sparse-1/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Sparse-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    temp_parent = (tmp_path / "sparse-replay-temporary").resolve()
    temp_parent.mkdir()

    result = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert result["provider_calls"] == 0
    assert result["ready"] is True
    assert result["preflight"]["scope"] is not None
    assert result["preflight"]["scope"]["selected_paths"] == [
        "results/metric.json"
    ]
    assert result["source_before"] == result["source_after"]
    assert not any(temp_parent.iterdir())


@pytest.mark.parametrize(
    ("omission", "safe_alternative", "expected_code"),
    (
        ("non_utf8", True, "PREFLIGHT_G2_CANDIDATE_UNREADABLE"),
        ("too_large", True, "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"),
        ("non_utf8", False, "PREFLIGHT_G2_CANDIDATE_UNREADABLE"),
        ("too_large", False, "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"),
    ),
)
def test_sparse_replay_preserves_exact_changed_candidate_omissions(
    tmp_path: Path,
    omission: str,
    safe_alternative: bool,
    expected_code: str,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Omission-1" / "head"
    _new_repo(source)
    _write(source, "README.md", "baseline\n")
    base = _commit(source, "base")
    omitted_path = source / "results" / "a-omitted.json"
    omitted_path.parent.mkdir()
    if omission == "non_utf8":
        omitted_path.write_bytes(b'{"accuracy": "\xff"}\n')
    else:
        omitted_path.write_bytes(b"x" * 16_777_217)
    entries = [ChangeEntry("results/a-omitted.json", ChangeStatus.ADDED)]
    if safe_alternative:
        _write(source, "results/z-safe.json", '{"accuracy": 0.90}\n')
        entries.append(ChangeEntry("results/z-safe.json", ChangeStatus.ADDED))
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Omission-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    description = (
        "Inspect the changed results for the reported accuracy improvement.\n"
    )
    (metadata / "pr-description.md").write_text(description, encoding="utf-8")
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Omission-1",
        repository="example/omission",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=len(entries),
        entries=tuple(entries),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Omission-1/head",
        metadata_relative="paid/X-Omission-1/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Omission-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    base_root = (tmp_path / f"{omission}-{safe_alternative}-base").resolve()
    _git(source, "worktree", "add", "-q", "--detach", str(base_root), base)
    inventory = build_git_change_inventory(
        _identity(SnapshotRole.REQUESTED_BASE, base_root, base),
        _identity(SnapshotRole.COMPARISON_BASE, base_root, base),
        _identity(SnapshotRole.HEAD, source, head),
        ComparisonBasis.DIRECT_BASE,
    )
    direct = run_review(
        ReviewInputs(
            repository_root=source,
            base_root=base_root,
            pr_title="Benchmark accuracy improves by 5%",
            pr_description=description,
            requested_base=_identity(SnapshotRole.REQUESTED_BASE, base_root, base),
            comparison_base=_identity(
                SnapshotRole.COMPARISON_BASE, base_root, base
            ),
            head=_identity(SnapshotRole.HEAD, source, head),
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
        preflight_only=True,
    )
    assert direct.preflight is not None
    temp_parent = (tmp_path / f"{omission}-{safe_alternative}-temporary").resolve()
    temp_parent.mkdir()

    replay = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert replay["provider_calls"] == 0
    assert replay["ready"] is safe_alternative
    assert replay["preflight"] == serialize_preflight(direct.preflight)
    assert replay["preflight"]["gates"][1]["disposition"] == (
        "pass_partial" if safe_alternative else "fail"
    )
    assert expected_code in {
        reason["code"]
        for reason in replay["preflight"]["gates"][1]["issues"]
    }
    assert replay["source_before"] == replay["source_after"]
    assert not any(temp_parent.iterdir())


def test_sparse_replay_preserves_oversized_comparison_candidate_omission(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Comparison-Omission-1" / "head"
    _new_repo(source)
    oversized = source / "src" / "metric.py"
    oversized.parent.mkdir()
    oversized.write_bytes(b"x" * 16_777_217)
    base = _commit(source, "base")
    _write(source, "src/metric.py", "ACCURACY = 0.90\n")
    _write(source, "results/safe.json", '{"accuracy": 0.90}\n')
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Comparison-Omission-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    description = "Inspect src/metric.py and results/safe.json for the result.\n"
    (metadata / "pr-description.md").write_text(
        description, encoding="utf-8"
    )
    entries = (
        ChangeEntry("results/safe.json", ChangeStatus.ADDED),
        ChangeEntry("src/metric.py", ChangeStatus.MODIFIED),
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Comparison-Omission-1",
        repository="example/comparison-omission",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=len(entries),
        entries=entries,
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Comparison-Omission-1/head",
        metadata_relative=(
            "paid/X-Comparison-Omission-1/inputs/frozen-coordinates.md"
        ),
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Comparison-Omission-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    base_root = (tmp_path / "comparison-omission-base").resolve()
    _git(source, "worktree", "add", "-q", "--detach", str(base_root), base)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base)
    exact_head = _identity(SnapshotRole.HEAD, source, head)
    inventory = build_git_change_inventory(
        requested,
        comparison,
        exact_head,
        ComparisonBasis.DIRECT_BASE,
    )
    direct = run_review(
        ReviewInputs(
            repository_root=source,
            base_root=base_root,
            pr_title="Benchmark accuracy improves by 5%",
            pr_description=description,
            requested_base=requested,
            comparison_base=comparison,
            head=exact_head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
        preflight_only=True,
    )
    assert direct.preflight is not None
    temp_parent = (tmp_path / "comparison-omission-temporary").resolve()
    temp_parent.mkdir()

    replay = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert replay["provider_calls"] == 0
    assert replay["ready"] is True
    assert replay["preflight"] == serialize_preflight(direct.preflight)
    assert replay["preflight"]["gates"][1]["disposition"] == "pass_partial"
    assert "PREFLIGHT_G2_COMPARISON_CANDIDATE_TOO_LARGE" in {
        reason["code"]
        for reason in replay["preflight"]["gates"][1]["issues"]
    }
    assert replay["source_before"] == replay["source_after"]
    assert not any(temp_parent.iterdir())


@pytest.mark.parametrize("route", ("exact_literal", "manifest_dependency"))
def test_sparse_replay_preserves_oversized_unchanged_supplement_omission(
    tmp_path: Path,
    route: str,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Unchanged-Omission-1" / "head"
    _new_repo(source)
    if route == "exact_literal":
        omitted_relative = "src/reference.py"
        description = (
            "Inspect src/reference.py and results/safe.json for the result.\n"
        )
    else:
        fixture = Path(__file__).parents[1] / "examples" / "day2_demo"
        for relative in (
            "research.yaml",
            "baseline-config.yaml",
            "baseline-results.json",
            "baseline-train.jsonl",
            "baseline-eval.jsonl",
            "candidate-config.yaml",
            "candidate-results.json",
            "candidate-train.jsonl",
            "candidate-eval.jsonl",
        ):
            _write(
                source,
                relative,
                (fixture / relative).read_text(encoding="utf-8"),
            )
        omitted_relative = "candidate-eval.jsonl"
        description = (
            "Audit inputs are declared in research.yaml; inspect "
            "results/safe.json for the result.\n"
        )
    omitted = source / omitted_relative
    omitted.parent.mkdir(parents=True, exist_ok=True)
    omitted.write_bytes(b"x" * 16_777_217)
    base = _commit(source, "base")
    _write(source, "results/safe.json", '{"accuracy": 0.90}\n')
    head = _commit(source, "head")
    metadata = study / "paid" / "X-Unchanged-Omission-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    (metadata / "pr-description.md").write_text(description, encoding="utf-8")
    entry = ChangeEntry("results/safe.json", ChangeStatus.ADDED)
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Unchanged-Omission-1",
        repository="example/unchanged-omission",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=1,
        entries=(entry,),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Unchanged-Omission-1/head",
        metadata_relative=(
            "paid/X-Unchanged-Omission-1/inputs/frozen-coordinates.md"
        ),
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Unchanged-Omission-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    base_root = (tmp_path / f"{route}-unchanged-base").resolve()
    _git(source, "worktree", "add", "-q", "--detach", str(base_root), base)
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base)
    exact_head = _identity(SnapshotRole.HEAD, source, head)
    inventory = build_git_change_inventory(
        requested,
        comparison,
        exact_head,
        ComparisonBasis.DIRECT_BASE,
    )
    direct = run_review(
        ReviewInputs(
            repository_root=source,
            base_root=base_root,
            pr_title="Benchmark accuracy improves by 5%",
            pr_description=description,
            requested_base=requested,
            comparison_base=comparison,
            head=exact_head,
            inventory=inventory,
        ),
        ReviewConfig(enabled=True),
        preflight_only=True,
    )
    assert direct.preflight is not None
    temp_parent = (tmp_path / f"{route}-unchanged-temporary").resolve()
    temp_parent.mkdir()

    replay = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert replay["provider_calls"] == 0
    assert replay["ready"] is True
    assert replay["preflight"] == serialize_preflight(direct.preflight)
    assert replay["preflight"]["gates"][1]["disposition"] == "pass_partial"
    assert {
        "code": "PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
        "path": omitted_relative,
        "observed": 16_777_217,
        "limit": 16_777_216,
    } in replay["preflight"]["gates"][1]["issues"]
    assert replay["source_before"] == replay["source_after"]
    assert not any(temp_parent.iterdir())


def test_forged_sparse_oversize_fact_is_rejected_by_exact_source_size(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "forged-omission-source").resolve()
    _new_repo(source)
    _write(source, "README.md", "baseline\n")
    base = _commit(source, "base")
    _write(source, "results/metric.json", '{"accuracy": 0.90}\n')
    head = _commit(source, "head")
    base_root = (tmp_path / "forged-omission-base").resolve()
    _git(source, "worktree", "add", "-q", "--detach", str(base_root), base)
    sparse_root = (tmp_path / "forged-omission-sparse-head").resolve()
    sparse_root.mkdir()
    requested = _identity(SnapshotRole.REQUESTED_BASE, base_root, base)
    comparison = _identity(SnapshotRole.COMPARISON_BASE, base_root, base)
    exact_head = _identity(SnapshotRole.HEAD, source, head)
    inventory = build_git_change_inventory(
        requested,
        comparison,
        exact_head,
        ComparisonBasis.DIRECT_BASE,
    )
    object_id = _git(source, "rev-parse", f"{head}:results/metric.json")
    forged = ExactMaterialOmission(
        role=SnapshotRole.HEAD,
        source=exact_head,
        path="results/metric.json",
        object_id=object_id,
        observed=16_777_217,
        limit=16_777_216,
        code="PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
    )
    assert not exact_git_material_omission_matches(forged, exact_head)

    scope = build_review_scope(
        ReviewInputs(
            repository_root=sparse_root,
            base_root=base_root,
            pr_title="Benchmark accuracy improves by 5%",
            requested_base=requested,
            comparison_base=comparison,
            head=exact_head,
            inventory=inventory,
            exact_material_omissions=(forged,),
        ),
        ReviewConfig(enabled=True),
        inventory,
    )

    assert "PREFLIGHT_G1_MATERIAL_PATH_NOT_REGULAR" in {
        issue.code for issue in scope.issues
    }
    assert "PREFLIGHT_G2_CANDIDATE_TOO_LARGE" not in {
        issue.code for issue in scope.issues
    }


def test_replay_partial_source_never_fetches_missing_unrelated_blob(
    tmp_path: Path,
) -> None:
    study = (tmp_path / "ClaimCI-External-Study-2026-09-02").resolve()
    source = study / "paid" / "X-Partial-1" / "head"
    _new_repo(source)
    _write(source, "unrelated/large.bin", "UNRELATED_NEVER_FETCH\n")
    base = _commit(source, "base")
    _write(source, "results/metric.json", '{"accuracy": 0.9}\n')
    head = _commit(source, "head")
    missing_blob = _git(source, "rev-parse", f"{head}:unrelated/large.bin")
    missing_object = source / ".git" / "objects" / missing_blob[:2] / missing_blob[2:]
    assert missing_object.is_file()
    missing_object.chmod(stat.S_IREAD | stat.S_IWRITE)
    missing_object.unlink()
    _git(source, "config", "remote.origin.url", "https://example.invalid/never-fetch")
    _git(source, "config", "remote.origin.promisor", "true")
    _git(source, "config", "remote.origin.partialclonefilter", "blob:none")
    metadata = study / "paid" / "X-Partial-1" / "inputs"
    metadata.mkdir()
    (metadata / "frozen-coordinates.md").write_text(
        "- Title: `Benchmark accuracy improves by 5%`\n", encoding="utf-8"
    )
    (metadata / "pr-description.md").write_text(
        "See results/metric.json for the reproducible result.\n", encoding="utf-8"
    )
    candidate = FrozenCandidate(
        schema_version=1,
        candidate="X-Partial-1",
        repository="example/partial",
        pull_request=1,
        requested_base_sha=base,
        comparison_base_sha=base,
        head_sha=head,
        comparison_basis=ComparisonBasis.DIRECT_BASE,
        declared_entry_count=1,
        entries=(ChangeEntry("results/metric.json", ChangeStatus.ADDED),),
    )
    layout = CandidateInputLayout(
        source_relative="paid/X-Partial-1/head",
        metadata_relative="paid/X-Partial-1/inputs/frozen-coordinates.md",
        metadata_kind="frozen_coordinates",
        body_relative="paid/X-Partial-1/inputs/pr-description.md",
        body_capture="raw_pr_body",
    )
    temp_parent = (tmp_path / "partial-replay-temporary").resolve()
    temp_parent.mkdir()

    result = replay_candidate(
        study,
        candidate,
        layout,
        ReviewConfig(enabled=True),
        temp_parent=temp_parent,
    )

    assert result["provider_calls"] == 0
    assert result["source_unchanged"] is True
    assert not any(temp_parent.iterdir())


def test_cold_provider_free_preflight_imports_no_openai_adapter_or_sdk(tmp_path: Path) -> None:
    code = r'''
import pathlib
import sys
import tempfile
sys.path.insert(0, str(pathlib.Path.cwd()))
import scripts.replay_review_preflight as replay
with tempfile.TemporaryDirectory() as raw:
    root = pathlib.Path(raw).resolve()
    inputs = replay.ReviewInputs(
        repository_root=root, base_root=None, pr_title="Documentation cleanup",
        pr_description="", requested_base=None, comparison_base=None, head=None,
        inventory=None, coordinates=None, inventory_failure=None,
    )
    review = replay.run_review(
        inputs, replay.ReviewConfig(enabled=True), preflight_only=True
    )
    assert review.preflight is not None
    assert not review.preflight.ready_for_provider
    assert review.provider_lifecycle.value == "not_attempted"
    assert review.provider_attempt_count == 0
    assert review.provider_calls == ()
assert "claimci.review.openai_provider" not in sys.modules
assert "openai" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=Path(__file__).parents[1],
        env=_credential_free_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_direct_replay_script_resolves_the_current_workspace_package() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/replay_review_preflight.py", "--help"],
        cwd=Path(__file__).parents[1],
        env=_credential_free_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Replay all five frozen local ClaimCI preflights" in completed.stdout
