"""Wave 2 contract tests for deterministic, bounded review evidence.

These tests intentionally exercise the trusted evidence boundary rather than a
provider implementation.  A provider may suggest paths, but only the review
package may resolve them, read regular files, and issue content-addressed
references.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from claimci.review.evidence import (
    EvidenceKind,
    EvidenceLocality,
    EvidenceProvenance,
    discover_evidence,
)
from claimci.review.models import (
    ClaimType,
    ReviewError,
    ReviewLimits,
    ScientificClaim,
    SourceKind,
    SourceLocation,
)


def _claim(claim_type: str, claim_id: str = "claim-1") -> ScientificClaim:
    """Build a fully trusted claim without making evidence tests parse claims."""

    return ScientificClaim(
        claim_id=claim_id,
        source_text=f"The experiment makes a {claim_type} claim.",
        claim_type=ClaimType(claim_type),
        subject="candidate model",
        source=SourceLocation(
            source_id="README.md",
            kind=SourceKind.REPOSITORY_FILE,
            path="README.md",
            start_line=1,
            end_line=1,
        ),
    )


def _limits(**updates: int | float) -> ReviewLimits:
    return replace(ReviewLimits(), **updates)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _path_set(bundle, claim_id: str) -> set[str]:
    return {
        str(reference.path).replace("\\", "/")
        for reference in bundle.references
        if claim_id in reference.claim_ids
    }


def test_claim_type_routing_is_selective_for_all_supported_types(tmp_path: Path) -> None:
    """Each normalized claim type selects its evidence family, not the tree."""

    files = {
        "manifests/research.yaml": "claim:\n  metric: accuracy\n",
        "results/metrics.json": '{"accuracy": 0.82}\n',
        "configs/train.yaml": "training_steps: 100\nbatch_size: 8\n",
        "runs/metadata.json": '{"seed": 3, "steps": 100}\n',
        "data/eval.jsonl": '{"id": "held-out-1"}\n',
        "benchmarks/latency.json": '{"latency_ms": 4.2}\n',
        "ablations/results.json": '{"without_component": 0.70}\n',
        "configs/reward.yaml": "external_reward: false\n",
        "src/model.py": "class Candidate: pass\n",
        "tests/test_model.py": "def test_candidate(): pass\n",
        # This file deliberately resembles useful evidence but is not indexed.
        "private/secret_results.json": "do-not-send\n",
    }
    for path, text in files.items():
        _write(tmp_path, path, text)

    repository_paths = sorted(path for path in files if not path.startswith("private/"))
    claims = [
        _claim("metric_improvement", "metric"),
        _claim("compute_equivalence", "compute"),
        _claim("held_out_evaluation", "held-out"),
        _claim("resource_reduction", "resource"),
        _claim("component_causality", "causal"),
        _claim("no_external_reward", "reward"),
        _claim("implementation_claim", "implementation"),
        _claim("other_scientific", "other"),
    ]

    bundle = discover_evidence(tmp_path, claims, repository_paths)

    # Routing-specific positives.
    assert "manifests/research.yaml" in _path_set(bundle, "metric")
    assert "results/metrics.json" in _path_set(bundle, "metric")
    assert "runs/metadata.json" in _path_set(bundle, "compute")
    assert "configs/train.yaml" in _path_set(bundle, "compute")
    assert "data/eval.jsonl" in _path_set(bundle, "held-out")
    assert "benchmarks/latency.json" in _path_set(bundle, "resource")
    assert "ablations/results.json" in _path_set(bundle, "causal")
    assert "configs/reward.yaml" in _path_set(bundle, "reward")
    assert "src/model.py" in _path_set(bundle, "implementation")
    assert "tests/test_model.py" in _path_set(bundle, "implementation")

    # A path omitted from the trusted index must never be smuggled into output.
    assert all("private/secret_results.json" not in str(ref.path) for ref in bundle.references)
    assert all("do-not-send" not in ref.excerpt for ref in bundle.references)


def test_route_terms_match_path_tokens_instead_of_unrelated_substrings(
    tmp_path: Path,
) -> None:
    """Short route words must not select private files through collisions."""

    files = {
        "private/latest_credentials.txt": "LATEST_SECRET_NEVER_EGRESS\n",
        "docs/glossary_private.md": "GLOSSARY_SECRET_NEVER_EGRESS\n",
    }
    for path, text in files.items():
        _write(tmp_path, path, text)

    bundle = discover_evidence(
        tmp_path,
        [
            _claim("held_out_evaluation", "held-out"),
            _claim("no_external_reward", "reward"),
        ],
        tuple(files),
    )

    assert bundle.references == ()
    assert all(
        "SECRET_NEVER_EGRESS" not in reference.excerpt
        for reference in bundle.references
    )


def test_provider_hint_must_still_match_the_trusted_claim_route(
    tmp_path: Path,
) -> None:
    """An extraction hint cannot turn the model into an arbitrary file reader."""

    _write(tmp_path, "results/metrics.json", '{"accuracy": 0.9}\n')
    _write(tmp_path, "private/credentials.txt", "PRIVATE_SECRET_NEVER_EGRESS\n")
    claim = _claim("metric_improvement")

    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["results/metrics.json", "private/credentials.txt"],
        suggested_paths={claim.claim_id: ["private/credentials.txt"]},
    )

    assert "results/metrics.json" in _path_set(bundle, claim.claim_id)
    assert "private/credentials.txt" not in _path_set(bundle, claim.claim_id)
    assert any(
        item.requested_path == "private/credentials.txt"
        and item.reason == "route_mismatch"
        for item in bundle.missing
    )
    assert all(
        "PRIVATE_SECRET_NEVER_EGRESS" not in reference.excerpt
        for reference in bundle.references
    )


@pytest.mark.parametrize(
    "path",
    (
        "score.pem",
        "benchmark/private.pem",
        "config/credentials.txt",
        "artifacts/manifest.pem",
    ),
)
def test_changed_hint_cannot_promote_an_arbitrary_suffix_to_evidence(
    tmp_path: Path,
    path: str,
) -> None:
    """Path/category tokens must not turn an arbitrary file type into evidence."""

    _write(tmp_path, path, "PRIVATE_KEY_MATERIAL_NEVER_EGRESS\n")
    claim = ScientificClaim(
        claim_id="claim-arbitrary-suffix",
        source_text="The benchmark accuracy score improves by five percent.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="benchmark accuracy",
        metric="score",
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

    assert bundle.references == ()
    assert any(
        item.claim_id == claim.claim_id
        and item.requested_path == path
        and item.reason == "route_mismatch"
        for item in bundle.missing
    )


@pytest.mark.parametrize(
    ("path", "expected_kind"),
    [
        ("configs/opaque.ini", EvidenceKind.CONFIG),
        ("outputs/opaque.json", EvidenceKind.RESULTS),
        ("manifests/opaque.lock", EvidenceKind.MANIFEST),
        ("benchmarks/opaque.txt", EvidenceKind.BENCHMARK),
        ("transformers/submit_jobs_qwen3asr.sh", EvidenceKind.CONFIG),
    ],
)
def test_exact_changed_issued_supporting_artifact_bypasses_only_semantic_route(
    tmp_path: Path,
    path: str,
    expected_kind: EvidenceKind,
) -> None:
    """Removing the constrained bypass recreates real exact-hint false gaps."""

    _write(tmp_path, path, "OPAQUE_LOCAL_SUPPORTING_CONFIGURATION\n")
    claim = ScientificClaim(
        claim_id="claim-exact-supporting-artifact",
        source_text="The candidate improves accuracy by five percent.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="candidate accuracy",
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
        [claim],
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
    )

    reference = next(item for item in bundle.references if item.path == path)
    assert reference.claim_ids == (claim.claim_id,)
    assert reference.kind is expected_kind
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert reference.excerpt_locality is EvidenceLocality.COMPLETE_FILE
    assert not any(item.reason == "route_mismatch" for item in bundle.missing)
    assert bundle.routing_incomplete is False


def test_unissued_exact_hint_is_not_opened_and_cannot_expand_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact provider path outside the issued shortlist remains metadata only."""

    issued = "notes/issued.txt"
    outside = "private/outside-results.json"
    _write(tmp_path, issued, "UNRELATED_ISSUED_NOTE\n")
    _write(tmp_path, outside, "SECRET_OUTSIDE_SCOPE_NEVER_READ\n")
    claim = _claim("metric_improvement")
    claim = replace(claim, evidence_hints=(outside,))

    import claimci.review.evidence as evidence_module

    opened: list[str] = []
    original_capture = evidence_module.capture_confined_regular_file

    def traced_capture(root: Path, relative: str, **kwargs: object):
        opened.append(relative)
        return original_capture(root, relative, **kwargs)

    monkeypatch.setattr(
        evidence_module, "capture_confined_regular_file", traced_capture
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (issued,),
        selected_paths=(issued,),
        changed_paths=(issued,),
    )

    assert outside not in opened
    assert all(
        "SECRET_OUTSIDE_SCOPE_NEVER_READ" not in ref.excerpt
        for ref in bundle.references
    )
    assert any(
        item.requested_path == outside
        and item.reason == "unresolved_provider_hint"
        for item in bundle.missing
    )
    assert bundle.routing_incomplete is True


def test_arbitrary_shell_hint_is_not_promoted_to_review_evidence(tmp_path: Path) -> None:
    """Shell support is a fixed submission-runner exception, not a suffix allowlist."""

    path = "scripts/metrics.sh"
    _write(tmp_path, path, "echo ARBITRARY_SHELL_MUST_NOT_ROUTE\n")
    claim = _claim("metric_improvement")
    claim = replace(claim, evidence_hints=(path,))

    bundle = discover_evidence(
        tmp_path,
        [claim],
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
    )

    assert bundle.references == ()
    assert any(
        item.requested_path == path and item.reason == "route_mismatch"
        for item in bundle.missing
    )
    assert bundle.routing_incomplete is True


def test_test_path_submission_runner_is_config_and_requires_changed_identity(
    tmp_path: Path,
) -> None:
    """A test-like directory must not grant shell runners SOURCE/TEST bypass."""

    path = "tests/eval.sh"
    _write(tmp_path, path, "python run_eval.py --model candidate\n")
    claim = _claim("held_out_evaluation")
    claim = replace(claim, evidence_hints=(path,))

    unchanged = discover_evidence(
        tmp_path,
        [claim],
        (path,),
        selected_paths=(path,),
        changed_paths=(),
    )

    assert unchanged.references == ()
    assert any(
        item.requested_path == path and item.reason == "route_mismatch"
        for item in unchanged.missing
    )
    assert unchanged.routing_incomplete is True

    changed = discover_evidence(
        tmp_path,
        [claim],
        (path,),
        selected_paths=(path,),
        changed_paths=(path,),
    )

    reference = next(item for item in changed.references if item.path == path)
    assert reference.kind is EvidenceKind.CONFIG
    assert reference.provenance is EvidenceProvenance.SUPPORTING_ARTIFACT
    assert reference.claim_ids == (claim.claim_id,)
    assert changed.routing_incomplete is False


def test_provider_normalization_terms_cannot_expand_beyond_the_exact_quote(
    tmp_path: Path,
) -> None:
    """Hallucinated subject/qualifier tokens are not trusted routing inputs."""

    _write(tmp_path, "private/credentials.txt", "CREDENTIAL_SECRET_NEVER_EGRESS\n")
    claim = ScientificClaim(
        claim_id="claim-hallucinated-routing",
        source_text="Accuracy improves on the evaluation set.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="credentials",
        metric="accuracy",
        qualifiers=("private",),
        source=SourceLocation(
            source_id="source-title",
            kind=SourceKind.PULL_REQUEST_TITLE,
            path=None,
            start_line=1,
            end_line=1,
        ),
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["private/credentials.txt"],
    )

    assert bundle.references == ()


def test_literal_routing_ignores_stopword_only_overlap(tmp_path: Path) -> None:
    """A trusted quote does not make generic words a private-file selector."""

    _write(tmp_path, "private/the/credentials.txt", "STOPWORD_SECRET_NEVER_EGRESS\n")
    claim = ScientificClaim(
        claim_id="claim-stopword-routing",
        source_text="The candidate improved accuracy.",
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="the candidate",
        metric="accuracy",
        source=SourceLocation(
            source_id="source-title",
            kind=SourceKind.PULL_REQUEST_TITLE,
            path=None,
            start_line=1,
            end_line=1,
        ),
    )

    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["private/the/credentials.txt"],
    )

    assert bundle.references == ()


def test_evidence_kind_and_routing_use_tokens_not_embedded_substrings(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "runs/metadata.json", '{"steps": 100}\n')
    _write(tmp_path, "docs/costume.md", "COSTUME_SECRET_NEVER_EGRESS\n")

    bundle = discover_evidence(
        tmp_path,
        [
            _claim("compute_equivalence", "compute"),
            _claim("resource_reduction", "resource"),
        ],
        ["runs/metadata.json", "docs/costume.md"],
    )

    metadata = next(
        reference
        for reference in bundle.references
        if reference.path == "runs/metadata.json"
    )
    assert metadata.kind.value != "dataset"
    assert all(reference.path != "docs/costume.md" for reference in bundle.references)


def test_literal_routing_preserves_short_digit_metrics_and_unicode_terms(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "artifacts/f1.json", '{"f1": 0.9}\n')
    _write(tmp_path, "artifacts/정확도.json", '{"value": 0.9}\n')
    claims = [
        ScientificClaim(
            claim_id="claim-f1",
            source_text="F1 improves.",
            claim_type=ClaimType.METRIC_IMPROVEMENT,
            subject="classifier",
            metric="F1",
            source=SourceLocation(
                source_id="source-f1",
                kind=SourceKind.PULL_REQUEST_TITLE,
                path=None,
                start_line=1,
                end_line=1,
            ),
        ),
        ScientificClaim(
            claim_id="claim-unicode",
            source_text="정확도 결과를 보고합니다.",
            claim_type=ClaimType.OTHER_SCIENTIFIC,
            subject="정확도",
            source=SourceLocation(
                source_id="source-unicode",
                kind=SourceKind.PULL_REQUEST_DESCRIPTION,
                path=None,
                start_line=1,
                end_line=1,
            ),
        ),
    ]

    bundle = discover_evidence(
        tmp_path,
        claims,
        ["artifacts/f1.json", "artifacts/정확도.json"],
    )

    assert "artifacts/f1.json" in _path_set(bundle, "claim-f1")
    assert "artifacts/정확도.json" in _path_set(bundle, "claim-unicode")


def test_unchanged_source_and_test_paths_are_not_sent_for_implementation_claims(
    tmp_path: Path,
) -> None:
    """Source-code routes use the trusted base/head change set."""

    base = tmp_path / "base"
    head = tmp_path / "head"
    _write(base, "src/model.py", "MODEL = 'unchanged'\n")
    _write(head, "src/model.py", "MODEL = 'unchanged'\n")
    from claimci.review.sources import collect_review_sources

    sources = collect_review_sources(head, base_root=base)
    claim = _claim("implementation_claim")
    bundle = discover_evidence(
        head,
        [claim],
        sources.repository_paths,
        changed_paths=sources.changed_paths,
    )

    assert "src/model.py" not in _path_set(bundle, claim.claim_id)


def test_changed_only_policy_wins_over_source_filename_keywords(tmp_path: Path) -> None:
    """Names such as config.py stay source code, not unchanged config evidence."""

    paths = [
        "src/config.py",
        "src/metadata.py",
        "src/results.py",
        "src/cost.py",
        "tests/test_config.py",
    ]
    for path in paths:
        _write(tmp_path, path, "UNCHANGED_SOURCE_SECRET = True\n")

    claim = _claim("implementation_claim")
    bundle = discover_evidence(
        tmp_path,
        [claim],
        paths,
        changed_paths=(),
    )

    assert bundle.references == ()


def test_file_limit_missing_evidence_is_aggregated_per_claim(tmp_path: Path) -> None:
    """A large routed index cannot balloon advisory output with one row per path."""

    paths = []
    for index in range(40):
        relative = f"src/component-{index:02d}.py"
        _write(tmp_path, relative, "COMPONENT = True\n")
        paths.append(relative)
    claim = _claim("implementation_claim")

    bundle = discover_evidence(
        tmp_path,
        [claim],
        paths,
        limits=_limits(max_files=1),
    )

    limited = [item for item in bundle.missing if item.reason == "file_limit"]
    assert len(limited) == 1
    assert limited[0].claim_id == claim.claim_id


def test_evidence_is_sorted_bounded_and_content_addressed(tmp_path: Path) -> None:
    """File count, per-file chars, total chars, lines, and hashes are hard bounds."""

    _write(tmp_path, "results/z.json", "z-value\n" + "z" * 80 + "\n")
    _write(tmp_path, "results/a.json", "a-value\n" + "a" * 80 + "\n")
    _write(tmp_path, "results/m.json", "m-value\n" + "m" * 80 + "\n")
    claim = _claim("metric_improvement")
    limits = _limits(max_files=2, max_file_chars=12, max_context_chars=20)
    paths = ["results/z.json", "results/a.json", "results/m.json"]

    first = discover_evidence(tmp_path, [claim], paths, limits=limits)
    second = discover_evidence(tmp_path, [claim], list(reversed(paths)), limits=limits)

    assert first.references == second.references
    assert len(first.references) <= 2
    assert first.total_chars <= limits.max_context_chars
    assert all(len(reference.excerpt) <= limits.max_file_chars for reference in first.references)
    assert [str(reference.path) for reference in first.references] == sorted(
        str(reference.path) for reference in first.references
    )
    for reference in first.references:
        raw = (tmp_path / Path(str(reference.path))).read_bytes()
        assert reference.sha256 == hashlib.sha256(raw).hexdigest()
        assert reference.size == len(raw)
        assert reference.start_line >= 1
        assert reference.end_line >= reference.start_line


def test_trusted_priority_preempts_heuristics_without_weakening_confinement(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a-decoy-results.json", '{"unrelated": true}\n')
    _write(tmp_path, "z-declared-results.json", '{"accuracy": 0.9}\n')
    claim = _claim("metric_improvement")
    paths = ["a-decoy-results.json", "z-declared-results.json"]

    bundle = discover_evidence(
        tmp_path,
        [claim],
        paths,
        limits=_limits(max_files=1),
        priority_paths={claim.claim_id: ["z-declared-results.json"]},
    )

    assert [reference.path for reference in bundle.references] == [
        "z-declared-results.json"
    ]
    with pytest.raises(ReviewError, match="unsafe or not indexed"):
        discover_evidence(
            tmp_path,
            [claim],
            paths,
            priority_paths={claim.claim_id: ["../outside.json"]},
        )
    with pytest.raises(ReviewError, match="unknown claim"):
        discover_evidence(
            tmp_path,
            [claim],
            paths,
            priority_paths={"claim-invented": ["z-declared-results.json"]},
        )


def test_evidence_only_reads_indexed_regular_files_and_preserves_relative_paths(
    tmp_path: Path,
) -> None:
    """Repository indexing is an information boundary, not a hint to walk everything."""

    _write(tmp_path, "results/metrics.json", "accuracy: 0.9\n")
    _write(tmp_path, "unindexed/secret.txt", "PRIVATE_TOKEN=should-not-appear\n")
    claim = _claim("metric_improvement")
    bundle = discover_evidence(tmp_path, [claim], ["results/metrics.json"])

    assert [str(reference.path).replace("\\", "/") for reference in bundle.references] == [
        "results/metrics.json"
    ]
    assert all(not Path(str(reference.path)).is_absolute() for reference in bundle.references)
    assert all("PRIVATE_TOKEN" not in reference.excerpt for reference in bundle.references)


@pytest.mark.parametrize(
    "hint",
    [
        "C:\\outside\\results.json",
        "/outside/results.json",
        "../outside/results.json",
        "results\\..\\outside.json",
        "results/metrics.json\x00escape",
        "missing/results.json",
    ],
)
def test_unsafe_or_missing_provider_hints_become_missing_evidence(
    tmp_path: Path, hint: str
) -> None:
    """Untrusted paths are recorded as missing and can never cause an arbitrary read."""

    _write(tmp_path, "results/metrics.json", "safe: true\n")
    claim = _claim("metric_improvement")
    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["results/metrics.json"],
        suggested_paths={claim.claim_id: [hint]},
    )

    assert any(item.claim_id == claim.claim_id for item in bundle.missing)
    assert any(item.requested_path == hint for item in bundle.missing)
    assert all(hint not in str(reference.path) for reference in bundle.references)


def test_directory_and_special_file_hints_are_rejected(tmp_path: Path) -> None:
    """Only regular files are eligible evidence, including when a provider names one."""

    _write(tmp_path, "results/metrics.json", "safe: true\n")
    (tmp_path / "results/directory").mkdir(parents=True)
    claim = _claim("metric_improvement")
    hints = ["results/directory"]

    special = tmp_path / "results" / "fifo"
    try:
        os.mkfifo(special)
    except (AttributeError, NotImplementedError, OSError):
        # Windows CI may not expose named pipes as filesystem entries.  The
        # directory assertion still protects the regular-file boundary there.
        special = None
    if special is not None:
        hints.append("results/fifo")

    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["results/metrics.json"],
        suggested_paths={claim.claim_id: hints},
    )
    requested = {item.requested_path for item in bundle.missing}
    assert "results/directory" in requested
    if special is not None:
        assert "results/fifo" in requested


def test_symlink_escape_is_rejected_without_reading_target(tmp_path: Path) -> None:
    """A link beneath the checkout cannot exfiltrate a file outside its root."""

    _write(tmp_path, "results/metrics.json", "safe: true\n")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-secret.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    link = tmp_path / "results" / "linked.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this test environment")

    claim = _claim("metric_improvement")
    bundle = discover_evidence(
        tmp_path,
        [claim],
        ["results/metrics.json"],
        suggested_paths={claim.claim_id: ["results/linked.json"]},
    )

    assert any(item.requested_path == "results/linked.json" for item in bundle.missing)
    assert all("outside-secret" not in ref.excerpt for ref in bundle.references)
