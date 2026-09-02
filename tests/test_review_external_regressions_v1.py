"""Frozen external-scope regressions and provider-free replay safety tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
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
from claimci.review.inventory import build_git_change_inventory
from claimci.review.models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ClaimType,
    ComparisonBasis,
    GateDisposition,
    ProviderUsage,
    ReviewConfig,
    ReviewMaterialKind,
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
    return _git(root, "rev-parse", "HEAD")


def _new_repo(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@claimci.invalid")
    _git(root, "config", "user.name", "ClaimCI Tests")


def _identity(role: SnapshotRole, root: Path, sha: str) -> SnapshotIdentity:
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
        assert self.events == ["preflight_pass"]
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
        assert self.events == ["preflight_pass", "extract"]
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
        requested_base=_identity(SnapshotRole.REQUESTED_BASE, root, sha),
        comparison_base=_identity(SnapshotRole.COMPARISON_BASE, root, sha),
        head=_identity(SnapshotRole.HEAD, root, "2" * 40),
        inventory=inventory,
    )
    monkeypatch.setattr(
        "claimci.review.preflight.build_git_change_inventory", lambda *_args: inventory
    )
    import claimci.review.preflight as preflight_module

    real_preflight = preflight_module.preflight_review
    events: list[str] = []

    def observed_preflight(*args: object, **kwargs: object):
        result = real_preflight(*args, **kwargs)
        assert result.ready_for_provider
        events.append("preflight_pass")
        return result

    monkeypatch.setattr(preflight_module, "preflight_review", observed_preflight)
    provider = _TwoCallProvider(events)

    result = run_review(inputs, ReviewConfig(enabled=True), provider=provider)

    assert events == ["preflight_pass", "extract", "synthesize"]
    assert len(provider.calls) == 2
    assert len(result.provider_calls) == 2


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
    assert first_direct.scope.materialized_path_chars != merge.scope.materialized_path_chars
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

    payload = replay_study(study, destination, fixture_directory=FIXTURES)

    assert payload["ready_count"] == 5
    assert payload["provider_calls"] == 0
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
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path.cwd()))
import scripts.replay_review_preflight as replay
with tempfile.TemporaryDirectory() as raw:
    root = pathlib.Path(raw).resolve()
    inputs = SimpleNamespace(
        repository_root=root, base_root=None, pr_title="Documentation cleanup",
        pr_description="", requested_base=None, comparison_base=None, head=None,
        inventory=None, coordinates=None, inventory_failure=None,
    )
    result = replay.preflight_review(inputs, replay.ReviewConfig(enabled=True))
    assert not result.ready_for_provider
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
