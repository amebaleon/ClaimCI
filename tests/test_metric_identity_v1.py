"""Bounded, exact-pair metric identity for passive multi-metric evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldProvenance,
    FieldMapping,
    GitCommitSha,
    MetricBinding,
    MetricBindingAuthority,
    MetricBindingState,
    MetricCandidate,
    MetricCandidateLimitError,
    MetricCandidateLimits,
    MetricIdentity,
    MetricIdentityAuditContext,
    MetricIdentityKind,
    MappingCandidate,
    MappingTrust,
    MaterializationUnavailable,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    RepositoryIdentity,
    RepoMapping,
    Sha256Digest,
    SelectorKind,
    TablePredicate,
    TableScalarType,
    TableSelector,
    extract_metric_candidates,
    approve_metric_binding,
    derive_ephemeral_plan_id,
    execute_ephemeral_audit,
    extract_bound_metric_evidence,
    identify_metric_candidate,
    integrate_metric_binding,
    plan_ephemeral_audit,
    planning_request_from_discovery,
    PlanningState,
    resolve_metric_binding,
)


def _passive(
    content: bytes,
    *,
    path: str,
    kind: ArtifactKind = ArtifactKind.RESULTS,
    relevant_claim_ids: tuple[str, ...] = ("claim-accuracy",),
) -> PassiveArtifact:
    candidate = ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.95),
        discovery_reason="metric identity fixture",
        relevant_claim_ids=relevant_claim_ids,
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "metric identity fixture bytes",
            RepositoryPath(path),
            "fixture:metric-identity",
        ),
    )
    return PassiveArtifact(candidate, content)


def test_json_multi_metric_candidate_extraction_is_bounded_and_exact() -> None:
    passive = _passive(
        b'{"seed":7,"acc":0.91,"loss":0.2}',
        path="results/baseline.json",
    )

    first = extract_metric_candidates(
        passive,
        role=ExperimentRole.BASELINE,
    )
    second = extract_metric_candidates(
        passive,
        role=ExperimentRole.BASELINE,
    )

    assert first == second
    assert tuple(item.raw_metric_name for item in first) == ("acc", "loss")
    assert tuple(item.selector.expression for item in first) == ("/acc", "/loss")
    assert all(item.experiment_role is ExperimentRole.BASELINE for item in first)
    assert all(item.artifact_path == passive.candidate.path for item in first)
    assert all(item.artifact_sha256 == passive.candidate.sha256 for item in first)
    assert all(item.adapter_id == "claimci-json-v1" for item in first)
    assert len({item.candidate_id for item in first}) == 2


def test_candidate_contract_is_factory_only_and_fails_closed_at_its_bound() -> None:
    content = json.dumps(
        {f"metric_{index}": index / 10 for index in range(5)},
        separators=(",", ":"),
    ).encode("utf-8")
    passive = _passive(content, path="results/too-many.json")

    with pytest.raises(MetricCandidateLimitError, match="candidate bound"):
        extract_metric_candidates(
            passive,
            role=ExperimentRole.BASELINE,
            limits=MetricCandidateLimits(max_candidates_per_artifact=4),
        )
    with pytest.raises(TypeError, match="fixed adapter|extraction"):
        MetricCandidate()
    assert extract_metric_candidates(
        _passive(
            b'{"accuracy":0.9}',
            path="notes/metrics.json",
            kind=ArtifactKind.DOCUMENT,
        ),
        role=ExperimentRole.BASELINE,
    ) == ()


def test_provider_proposal_cannot_construct_a_metric_candidate_or_audit_context() -> None:
    passive = _passive(b'{"acc":0.9}', path="results/provider.json")
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "provider proposed a selector",
        passive.candidate.path,
        "provider:metric-selector",
    )
    match = AdapterMatch(
        "claimci-json-v1",
        passive.candidate.path,
        Confidence(1),
        (
            FieldMapping(
                "metric_value",
                EvidenceSelector(SelectorKind.JSON_POINTER, "/acc", provenance),
                provenance,
            ),
        ),
        (provenance,),
    )

    with pytest.raises(AnalysisContractError, match="fixed adapter extraction"):
        extract_metric_candidates(
            passive,
            role=ExperimentRole.BASELINE,
            adapter_match=match,
        )
    with pytest.raises(TypeError, match="materialization"):
        MetricIdentityAuditContext()


def test_tsv_multi_metric_candidates_use_the_fixed_tsv_adapter() -> None:
    candidates = extract_metric_candidates(
        _passive(
            b"seed\tacc\tloss\n1\t0.81\t0.4\n2\t0.83\t0.3\n",
            path="results/candidate.tsv",
        ),
        role=ExperimentRole.CANDIDATE,
    )

    assert tuple(item.raw_metric_name for item in candidates) == ("acc", "loss")
    assert all(item.adapter_id == "claimci-tsv-v1" for item in candidates)


def test_csv_multi_metric_candidates_preserve_exact_columns() -> None:
    passive = _passive(
        b"seed,acc,loss\n1,0.81,0.4\n2,0.83,0.3\n",
        path="results/candidate.csv",
    )

    candidates = extract_metric_candidates(
        passive,
        role=ExperimentRole.CANDIDATE,
    )

    assert tuple(item.raw_metric_name for item in candidates) == ("acc", "loss")
    assert tuple(item.selector.expression for item in candidates) == ("acc", "loss")
    assert all(item.adapter_id == "claimci-csv-v1" for item in candidates)


def test_selected_table_candidate_commits_the_complete_role_specific_selector() -> None:
    passive = _passive(
        b"commit,acc,loss\nbaseline,0.71,0.4\ncandidate,0.82,0.3\n",
        path="benchmarks/shared.csv",
    )

    def selected(role: ExperimentRole, key: str):
        provenance = FieldProvenance(
            ProvenanceKind.ADAPTER_EXTRACTION,
            f"exact table selector; sha256={passive.candidate.sha256}",
            passive.candidate.path,
            f"metric-table:{key}",
        )
        selector = TableSelector(
            "acc",
            (TablePredicate("commit", TableScalarType.STRING, key),),
            1,
            provenance,
        )
        match = AdapterMatch(
            "claimci-csv-v1",
            passive.candidate.path,
            Confidence(0.95),
            (FieldMapping("metric_value", selector, provenance),),
            (provenance,),
        )
        values = extract_metric_candidates(
            passive,
            role=role,
            adapter_match=match,
        )
        assert len(values) == 1
        return values[0]

    baseline = selected(ExperimentRole.BASELINE, "baseline")
    candidate = selected(ExperimentRole.CANDIDATE, "candidate")

    assert baseline.raw_metric_name == candidate.raw_metric_name == "acc"
    assert isinstance(baseline.selector, TableSelector)
    assert baseline.selector.predicates[0].value == "baseline"
    assert candidate.selector.predicates[0].value == "candidate"
    assert baseline.candidate_id != candidate.candidate_id


@pytest.mark.parametrize(
    ("content", "path", "adapter_id"),
    [
        (
            b'{"seed":1,"acc":0.81,"loss":0.4}\n'
            b'{"seed":2,"acc":0.83,"loss":0.3}\n',
            "results/runs.jsonl",
            "claimci-jsonl-v1",
        ),
        (
            b'{"runs":[{"seed":1,"acc":0.81,"loss":0.4},'
            b'{"seed":2,"acc":0.83,"loss":0.3}]}',
            "results/native.json",
            "claimci-native-results-v1",
        ),
    ],
)
def test_record_result_adapters_extract_only_common_metric_candidates(
    content: bytes,
    path: str,
    adapter_id: str,
) -> None:
    candidates = extract_metric_candidates(
        _passive(content, path=path),
        role=ExperimentRole.BASELINE,
    )

    assert tuple(item.raw_metric_name for item in candidates) == ("acc", "loss")
    assert tuple(item.selector.expression for item in candidates) == ("/acc", "/loss")
    assert all(item.adapter_id == adapter_id for item in candidates)


def test_metric_identity_is_conservative_and_candidate_scoped() -> None:
    candidates = extract_metric_candidates(
        _passive(
            b'{"acc":0.91,"accuracy":0.90,"acct":0.89,"loss":0.2}',
            path="results/baseline.json",
        ),
        role=ExperimentRole.BASELINE,
    )
    by_name = {item.raw_metric_name: item for item in candidates}

    alias = identify_metric_candidate(by_name["acc"], canonical_metric="accuracy")
    exact = identify_metric_candidate(
        by_name["accuracy"],
        canonical_metric="accuracy",
    )

    assert alias is not None
    assert alias.kind is MetricIdentityKind.LEXICAL_ALIAS
    assert alias.candidate == by_name["acc"]
    assert alias.canonical_metric == "accuracy"
    assert exact is not None
    assert exact.kind is MetricIdentityKind.EXACT
    assert identify_metric_candidate(
        by_name["acct"],
        canonical_metric="accuracy",
    ) is None
    assert identify_metric_candidate(
        by_name["loss"],
        canonical_metric="accuracy",
    ) is None
    with pytest.raises(TypeError, match="factory|identify"):
        MetricIdentity()


def test_explicit_raw_metric_name_binds_the_numeric_selector_without_guessing() -> None:
    candidates = extract_metric_candidates(
        _passive(
            b'{"metric_name":"acc","value":0.91,"seed":7}',
            path="results/named.json",
        ),
        role=ExperimentRole.BASELINE,
    )

    assert len(candidates) == 1
    assert candidates[0].raw_metric_name == "acc"
    assert candidates[0].selector.expression == "/value"
    identity = identify_metric_candidate(
        candidates[0],
        canonical_metric="accuracy",
    )
    assert identity is not None
    assert identity.kind is MetricIdentityKind.LEXICAL_ALIAS


def test_acc_pair_requires_one_atomic_approval_over_two_exact_candidates() -> None:
    repository = RepositoryIdentity("amebaleon", "ClaimCI")
    head_sha = GitCommitSha("a" * 40)
    baseline = extract_metric_candidates(
        _passive(
            b'{"acc":0.71,"loss":0.4}',
            path="results/baseline.json",
        ),
        role=ExperimentRole.BASELINE,
    )
    candidate = extract_metric_candidates(
        _passive(
            b'{"acc":0.82,"loss":0.3}',
            path="results/candidate.json",
        ),
        role=ExperimentRole.CANDIDATE,
    )

    resolution = resolve_metric_binding(
        (*baseline, *candidate),
        repository=repository,
        head_sha=head_sha,
        canonical_metric="accuracy",
        relevant_claim_id="claim-accuracy",
    )

    assert resolution.state is MetricBindingState.APPROVAL_REQUIRED
    assert resolution.binding is None
    assert resolution.question is not None
    assert len(resolution.question.proposals) == 1
    proposal = resolution.question.proposals[0]
    assert proposal.label == "[acc]"
    assert proposal.baseline_identity.candidate.raw_metric_name == "acc"
    assert proposal.candidate_identity.candidate.raw_metric_name == "acc"

    binding = approve_metric_binding(
        resolution.question,
        proposal_id=proposal.proposal_id,
        approved_by="owner:amebaleon",
    )

    assert binding.authority is MetricBindingAuthority.USER_APPROVED
    assert binding.approved_by == "owner:amebaleon"
    assert binding.baseline_candidate == proposal.baseline_identity.candidate
    assert binding.candidate_candidate == proposal.candidate_identity.candidate
    assert binding.baseline_candidate.candidate_id != binding.candidate_candidate.candidate_id
    assert resolve_metric_binding(
        (*baseline, *candidate),
        repository=repository,
        head_sha=head_sha,
        canonical_metric="accuracy",
        relevant_claim_id="claim-accuracy",
        approved_binding=binding,
    ).binding is binding
    with pytest.raises(TypeError, match="factory|approval"):
        MetricBinding()


def test_unique_exact_pair_is_automatic_but_approved_alias_never_inherits() -> None:
    repository = RepositoryIdentity("amebaleon", "ClaimCI")
    original_head = GitCommitSha("b" * 40)
    baseline = extract_metric_candidates(
        _passive(b'{"accuracy":0.71}', path="results/baseline.json"),
        role=ExperimentRole.BASELINE,
    )
    candidate = extract_metric_candidates(
        _passive(b'{"accuracy":0.82}', path="results/candidate.json"),
        role=ExperimentRole.CANDIDATE,
    )

    exact = resolve_metric_binding(
        (*baseline, *candidate),
        repository=repository,
        head_sha=original_head,
        canonical_metric="accuracy",
    )

    assert exact.state is MetricBindingState.BOUND
    assert exact.binding is not None
    assert exact.binding.authority is MetricBindingAuthority.DETERMINISTIC_EXACT
    assert exact.binding.approved_by is None

    alias_baseline = extract_metric_candidates(
        _passive(b'{"acc":0.71}', path="results/baseline.json"),
        role=ExperimentRole.BASELINE,
    )
    alias_candidate = extract_metric_candidates(
        _passive(b'{"acc":0.82}', path="results/candidate.json"),
        role=ExperimentRole.CANDIDATE,
    )
    pending = resolve_metric_binding(
        (*alias_baseline, *alias_candidate),
        repository=repository,
        head_sha=original_head,
        canonical_metric="accuracy",
    )
    assert pending.question is not None
    approved = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )

    later_head = resolve_metric_binding(
        (*alias_baseline, *alias_candidate),
        repository=repository,
        head_sha=GitCommitSha("c" * 40),
        canonical_metric="accuracy",
        approved_binding=approved,
    )
    changed_baseline = extract_metric_candidates(
        _passive(b'{"acc":0.72}', path="results/baseline.json"),
        role=ExperimentRole.BASELINE,
    )
    changed_pair = resolve_metric_binding(
        (*changed_baseline, *alias_candidate),
        repository=repository,
        head_sha=original_head,
        canonical_metric="accuracy",
        approved_binding=approved,
    )

    for stale in (later_head, changed_pair):
        assert stale.state is MetricBindingState.APPROVAL_REQUIRED
        assert stale.binding is None
        assert stale.question is not None
        assert stale.question.prompt.startswith("Re-approve")
        assert len(stale.question.proposals) == 1


def test_atomic_binding_integrates_exact_selectors_and_canonicalizes_evidence() -> None:
    repository = RepositoryIdentity("amebaleon", "ClaimCI")
    head_sha = GitCommitSha("d" * 40)
    baseline_passive = _passive(
        b'{"seed":1,"acc":0.71,"loss":0.4}',
        path="results/baseline.json",
    )
    candidate_passive = _passive(
        b'{"seed":2,"acc":0.82,"loss":0.3}',
        path="results/candidate.json",
    )
    metric_candidates = (
        *extract_metric_candidates(
            baseline_passive,
            role=ExperimentRole.BASELINE,
        ),
        *extract_metric_candidates(
            candidate_passive,
            role=ExperimentRole.CANDIDATE,
        ),
    )
    pending = resolve_metric_binding(
        metric_candidates,
        repository=repository,
        head_sha=head_sha,
        canonical_metric="accuracy",
    )
    assert pending.question is not None
    binding = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )
    mapping_provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "exact baseline/candidate result paths",
    )
    mapping = MappingCandidate(
        mapping_id="mapping-paths",
        bindings=(
            ArtifactBinding(
                baseline_passive.candidate.path,
                ArtifactKind.RESULTS,
                ExperimentRole.BASELINE,
                None,
                (),
                mapping_provenance,
            ),
            ArtifactBinding(
                candidate_passive.candidate.path,
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
                None,
                (),
                mapping_provenance,
            ),
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=mapping_provenance,
    )

    integrated = integrate_metric_binding(mapping, binding)
    integrated_approved = integrate_metric_binding(
        RepoMapping.approve(
            repository,
            mapping,
            approved_by="owner:repository-mapping",
        ),
        binding,
    )
    baseline_evidence = extract_bound_metric_evidence(
        baseline_passive,
        binding,
        role=ExperimentRole.BASELINE,
    )
    candidate_evidence = extract_bound_metric_evidence(
        candidate_passive,
        binding,
        role=ExperimentRole.CANDIDATE,
    )

    assert {
        item.role: (
            item.adapter_id,
            tuple(
                (field.target_field, field.selector.expression)
                for field in item.mappings
            ),
        )
        for item in integrated.bindings
    } == {
        ExperimentRole.BASELINE: ("claimci-json-v1", (("metric_value", "/acc"),)),
        ExperimentRole.CANDIDATE: ("claimci-json-v1", (("metric_value", "/acc"),)),
    }
    assert type(integrated_approved) is RepoMapping
    assert integrated_approved.approved_by == "owner:repository-mapping"
    assert all(
        any(item.target_field == "metric_value" for item in artifact.mappings)
        for artifact in integrated_approved.bindings
    )
    for evidence, expected_value, expected_seed in (
        (baseline_evidence, 0.71, 1),
        (candidate_evidence, 0.82, 2),
    ):
        metrics = tuple(
            item for item in evidence.observations if item.metric_name is not None
        )
        assert len(metrics) == 1
        assert metrics[0].metric_name == "accuracy"
        assert metrics[0].metric_value == pytest.approx(expected_value)
        assert metrics[0].seed == expected_seed
        assert next(
            item
            for item in evidence.adapter_match.mappings
            if item.target_field == "metric_value"
        ).selector.expression == "/acc"
    assert binding.baseline_candidate.raw_metric_name == "acc"
    assert binding.candidate_candidate.raw_metric_name == "acc"


def test_planning_carries_atomic_binding_and_commits_it_to_plan_identity() -> None:
    from test_ephemeral_planner_v03 import (
        CLAIM_ID,
        HEAD_SHA,
        REPOSITORY,
        _binding,
        _claim,
        _complete_evidence,
        _discovery,
    )

    baseline_passive = _passive(
        b'{"seed":1,"acc":0.71,"loss":0.4}',
        path="results/baseline.json",
        relevant_claim_ids=(CLAIM_ID,),
    )
    candidate_passive = _passive(
        b'{"seed":2,"acc":0.82,"loss":0.3}',
        path="results/candidate.json",
        relevant_claim_ids=(CLAIM_ID,),
    )
    candidates = (
        *extract_metric_candidates(
            baseline_passive,
            role=ExperimentRole.BASELINE,
        ),
        *extract_metric_candidates(
            candidate_passive,
            role=ExperimentRole.CANDIDATE,
        ),
    )
    pending = resolve_metric_binding(
        candidates,
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        canonical_metric="accuracy",
        relevant_claim_id=CLAIM_ID,
    )
    assert pending.question is not None
    metric_binding = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )
    baseline_evidence = extract_bound_metric_evidence(
        baseline_passive,
        metric_binding,
        role=ExperimentRole.BASELINE,
    )
    candidate_evidence = extract_bound_metric_evidence(
        candidate_passive,
        metric_binding,
        role=ExperimentRole.CANDIDATE,
    )
    ordinary = _complete_evidence()[2:]
    evidence = (baseline_evidence, candidate_evidence, *ordinary)
    path_provenance = FieldProvenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        "complete exact-head artifact roles",
    )
    source_mapping = MappingCandidate(
        mapping_id="mapping-metric-planning-source",
        bindings=(
            ArtifactBinding(
                baseline_passive.candidate.path,
                ArtifactKind.RESULTS,
                ExperimentRole.BASELINE,
                None,
                (),
                path_provenance,
            ),
            ArtifactBinding(
                candidate_passive.candidate.path,
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
                None,
                (),
                path_provenance,
            ),
            *(
                _binding(item, item.observations[0].experiment_role)
                for item in ordinary
            ),
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=path_provenance,
    )
    discovery = _discovery(_claim(), evidence, (source_mapping,))

    request = planning_request_from_discovery(
        discovery,
        claim_id=CLAIM_ID,
        normalized_evidence=evidence,
        metric_binding=metric_binding,
    )
    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.metric_binding is metric_binding
    assert derive_ephemeral_plan_id(outcome.plan) == outcome.plan.plan_id
    without_binding = dataclasses.replace(outcome.plan, metric_binding=None)
    assert derive_ephemeral_plan_id(without_binding) != outcome.plan.plan_id


def _metric_materialization_fixture(tmp_path):
    from test_ephemeral_materialize_v03 import (
        _binding,
        _plan_fixture,
        _write_artifact,
    )

    plan, runtime, checkout, scratch = _plan_fixture(tmp_path)
    baseline_content = b'{"seed":1,"acc":0.51,"loss":0.4}'
    candidate_content = b'{"seed":11,"acc":0.91,"loss":0.3}'
    baseline_artifact = _write_artifact(
        checkout,
        "results/metric-baseline.json",
        ArtifactKind.RESULTS,
        baseline_content,
    )
    candidate_artifact = _write_artifact(
        checkout,
        "results/metric-candidate.json",
        ArtifactKind.RESULTS,
        candidate_content,
    )
    baseline_passive = PassiveArtifact(baseline_artifact, baseline_content)
    candidate_passive = PassiveArtifact(candidate_artifact, candidate_content)
    candidates = (
        *extract_metric_candidates(
            baseline_passive,
            role=ExperimentRole.BASELINE,
        ),
        *extract_metric_candidates(
            candidate_passive,
            role=ExperimentRole.CANDIDATE,
        ),
    )
    pending = resolve_metric_binding(
        candidates,
        repository=plan.repository,
        head_sha=plan.head_sha,
        canonical_metric="accuracy",
        relevant_claim_id=plan.claim.claim_id,
    )
    assert pending.question is not None
    metric_binding = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )
    baseline_result = extract_bound_metric_evidence(
        baseline_passive,
        metric_binding,
        role=ExperimentRole.BASELINE,
    )
    candidate_result = extract_bound_metric_evidence(
        candidate_passive,
        metric_binding,
        role=ExperimentRole.CANDIDATE,
    )
    assert plan.selected_mapping is not None
    non_result_bindings = tuple(
        item
        for item in plan.selected_mapping.bindings
        if item.kind is not ArtifactKind.RESULTS
    )
    mapping = MappingCandidate(
        mapping_id="mapping-metric-materialization",
        bindings=(
            _binding(baseline_result, ExperimentRole.BASELINE),
            _binding(candidate_result, ExperimentRole.CANDIDATE),
            *non_result_bindings,
        ),
        confidence=plan.selected_mapping.confidence,
        trust=plan.selected_mapping.trust,
        provenance=plan.selected_mapping.provenance,
    )
    changed = dataclasses.replace(
        plan,
        baseline_evidence=(baseline_result, *plan.baseline_evidence[1:]),
        candidate_evidence=(candidate_result, *plan.candidate_evidence[1:]),
        selected_mapping=mapping,
        mapping_provenance=(mapping.provenance,),
        metric_binding=metric_binding,
    )
    changed = dataclasses.replace(
        changed,
        plan_id=derive_ephemeral_plan_id(changed),
    )
    return changed, runtime, scratch


@pytest.mark.parametrize(
    "stale_role",
    [ExperimentRole.BASELINE, ExperimentRole.CANDIDATE],
)
def test_materialization_invalidates_the_whole_pair_when_either_identity_is_stale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stale_role: ExperimentRole,
) -> None:
    import claimci.analysis.metric_identity as metric_identity

    plan, runtime, scratch = _metric_materialization_fixture(tmp_path)
    original = metric_identity.extract_metric_candidates
    observed_roles: list[ExperimentRole] = []

    def one_stale(
        passive,
        *,
        role,
        limits=metric_identity.MetricCandidateLimits(),
        adapter_match=None,
    ):
        observed_roles.append(role)
        if role is stale_role:
            return ()
        return original(
            passive,
            role=role,
            limits=limits,
            adapter_match=adapter_match,
        )

    monkeypatch.setattr(metric_identity, "extract_metric_candidates", one_stale)

    with pytest.raises(
        MaterializationUnavailable,
        match="entire metric pair|re-approval",
    ):
        execute_ephemeral_audit(plan, runtime)

    assert set(observed_roles) == {
        ExperimentRole.BASELINE,
        ExperimentRole.CANDIDATE,
    }
    assert not tuple(scratch.iterdir())


def test_exact_head_pair_revalidation_is_visible_in_native_audit(tmp_path) -> None:
    plan, runtime, scratch = _metric_materialization_fixture(tmp_path)

    result = execute_ephemeral_audit(plan, runtime)

    finding = next(
        item
        for item in result.findings
        if item.rule_id == "RESULT.METRIC_IDENTITY_VERIFIED"
    )
    assert finding.impact.value == "NONE"
    assert finding.evidence["binding_id"] == plan.metric_binding.binding_id
    assert finding.evidence["canonical_metric"] == "accuracy"
    assert finding.evidence["authority"] == "user_approved"
    assert finding.evidence["baseline"] == {
        "candidate_id": plan.metric_binding.baseline_candidate.candidate_id,
        "experiment_role": "baseline",
        "artifact_path": "results/metric-baseline.json",
        "artifact_sha256": str(plan.metric_binding.baseline_candidate.artifact_sha256),
        "adapter_id": "claimci-json-v1",
        "selector": ("field", "json_pointer", "/acc"),
        "raw_metric_name": "acc",
    }
    assert finding.evidence["candidate"]["candidate_id"] == (
        plan.metric_binding.candidate_candidate.candidate_id
    )
    assert finding.evidence["candidate"]["raw_metric_name"] == "acc"
    assert not tuple(scratch.iterdir())
