"""Deterministic Benchmark Evidence Profile v0 contracts and lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AdapterMatch,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    ArtifactEvidenceSlot,
    BenchmarkResultForm,
    BenchmarkVariant,
    CanonicalScientificClaim,
    Confidence,
    ConfigValue,
    DatasetReference,
    DatasetSplit,
    EvidenceProfileDefinition,
    EvidenceProfileId,
    EvidenceProfileSelection,
    EvidenceObligationDecision,
    ProfileEvidenceSupport,
    ProfileEvidenceTarget,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    MappingCandidate,
    MappingTrust,
    NormalizedEvidence,
    NormalizedObservation,
    ProfileSelectionFact,
    ProfileSelectionState,
    ProfileSupportKind,
    ProfiledEvidencePolicy,
    PlanningRequest,
    PlanningState,
    ProvenanceKind,
    RepositoryPath,
    RepositoryIdentity,
    RepoMapping,
    SelectorKind,
    Sha256Digest,
    GitCommitSha,
    MaterializationUnavailable,
    PassiveArtifact,
    RuntimeExecutionContext,
    TablePredicate,
    TableScalarType,
    TableSelector,
    TraceCompleteness,
    TraceRecordKind,
    derive_ephemeral_plan_id,
    evidence_profile_registry,
    assess_evidence_obligations,
    assess_profiled_evidence_obligations,
    compile_audit_claim,
    claim_evidence_policy,
    evidence_obligations_json_bytes,
    profile_evidence_target,
    profiled_evidence_policy,
    recover_scientific_claim,
    select_evidence_profile,
    plan_ephemeral_audit,
    execute_ephemeral_audit_with_trace,
    trace_json_bytes,
    to_jsonable,
    validated_profile_evidence_support,
)
from claimci.analysis.adapters import (
    CsvAdapter,
    PassiveJsonLinesDatasetAdapter,
    YamlConfigAdapter,
)
from claimci.benchmark_audit import audit_benchmark
from claimci.models import ClaimCIError, Impact, Severity, Verdict
from claimci.analysis.contracts import ClaimReference, EvidenceSelector
from claimci.measurement import MeasurementDriftState


CLAIM_ID = "claim-profile-1"
REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI-Benchmark-Fixture")


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
    *,
    path: str = "CLAIM.md",
    source_id: str = "profile-fixture",
) -> FieldProvenance:
    return FieldProvenance(
        kind,
        "fixed benchmark profile fixture",
        RepositoryPath(path),
        source_id,
    )


def _claim(text: str) -> CanonicalScientificClaim:
    reference = ClaimReference(
        claim_id=CLAIM_ID,
        text=text,
        source_path=RepositoryPath("CLAIM.md"),
        confidence=Confidence(0.95),
        provenance=_provenance(),
    )
    result = recover_scientific_claim(reference)
    assert type(result) is CanonicalScientificClaim
    return result


def _evidence(
    path: str,
    kind: ArtifactKind,
    role: ExperimentRole,
    *,
    metric: str | None = None,
    value: float | None = None,
    config: tuple[tuple[str, object], ...] = (),
    split: DatasetSplit | None = None,
) -> tuple[NormalizedEvidence, ArtifactBinding]:
    content = (path + repr((metric, value, config, split))).encode()
    provenance = _provenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        path=path,
        source_id=f"adapter:{path}:{role.value}:{split}",
    )
    mappings: list[FieldMapping] = []
    if metric is not None:
        mappings.append(
            FieldMapping(
                "metric_value",
                EvidenceSelector(
                    SelectorKind.DOTTED_PATH,
                    "fixture.value",
                    provenance,
                ),
                provenance,
            )
        )
    mappings.extend(
        FieldMapping(
            key,
            EvidenceSelector(SelectorKind.DOTTED_PATH, key, provenance),
            provenance,
        )
        for key, _value in config
    )
    artifact = ArtifactCandidate(
        RepositoryPath(path),
        kind,
        Sha256Digest(hashlib.sha256(content).hexdigest()),
        len(content),
        Confidence(0.95),
        "profile fixture",
        (CLAIM_ID,),
        provenance,
    )
    match = AdapterMatch(
        "fixture-profile-v1",
        artifact.path,
        Confidence(0.95),
        tuple(mappings),
        (provenance,),
    )
    if kind is ArtifactKind.DATASET:
        observation = NormalizedObservation(
            provenance,
            experiment_role=role,
            dataset_references=(
                DatasetReference(artifact.path, split.value if split else None, provenance),
            ),
        )
    elif metric is not None:
        observation = NormalizedObservation(
            provenance,
            metric_name=metric,
            metric_value=value,
            run_id=f"{role.value}-run",
            experiment_role=role,
            config_values=tuple(ConfigValue(key, item, provenance) for key, item in config),
        )
    else:
        observation = NormalizedObservation(
            provenance,
            experiment_role=role,
            config_values=tuple(ConfigValue(key, item, provenance) for key, item in config),
        )
    normalized = NormalizedEvidence(
        f"evidence-{hashlib.sha256(content).hexdigest()[:16]}",
        artifact,
        match,
        (observation,),
    )
    binding = ArtifactBinding(
        artifact.path,
        kind,
        role,
        match.adapter_id,
        match.mappings,
        provenance,
        split,
    )
    return normalized, binding


def _mapping(
    bindings: tuple[ArtifactBinding, ...],
    *,
    provenance: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> MappingCandidate:
    return MappingCandidate(
        "mapping-" + hashlib.sha256(repr(bindings).encode()).hexdigest()[:16],
        bindings,
        Confidence(0.95),
        MappingTrust.INFERRED,
        _provenance(provenance, source_id="mapping-profile-fixture"),
    )


def _benchmark_mapping(
    metric: str = "latency",
    *,
    path_prefix: str = "bench",
) -> tuple[tuple[NormalizedEvidence, ...], MappingCandidate]:
    evidence: list[NormalizedEvidence] = []
    bindings: list[ArtifactBinding] = []
    for role, value, subject in (
        (ExperimentRole.BASELINE, 100.0, "sequential"),
        (ExperimentRole.CANDIDATE, 80.0, "parallel"),
    ):
        result, result_binding = _evidence(
            f"{path_prefix}/{role.value}.csv",
            ArtifactKind.BENCHMARK,
            role,
            metric=metric,
            value=value,
            config=(
                ("benchmark.subject.id", subject),
                ("benchmark.workload.id", "shared-workload-v1"),
                ("benchmark.workload.model", "model-v1"),
                ("benchmark.workload.input_shape", "1x1024"),
                ("benchmark.workload.corpus", "eval-corpus-v1"),
                ("benchmark.workload.sequence_length", 1024),
                ("benchmark.workload.batch_size", 1),
                ("benchmark.workload.concurrency", 1),
                ("benchmark.workload.tensor_shape", "1024x1024"),
                ("benchmark.workload.dtype", "float16"),
                ("benchmark.workload.layout", "contiguous"),
                ("benchmark.workload.tensor_generation", "seeded_uniform_v1"),
                ("benchmark.workload.workload_volume", 1000),
                ("benchmark.workload.token_volume", 100000),
                ("benchmark.measurement_config.timing_boundary", "wall_clock"),
                ("benchmark.measurement_config.measurement_window", "2026-08"),
                ("benchmark.measurement_config.memory_collection", "peak_allocated"),
                ("benchmark.measurement_config.network_boundary", "excluded"),
                ("benchmark.run_protocol.result_form", "RAW_RUN_SERIES"),
                ("benchmark.run_protocol.sample_count", 2),
                ("benchmark.run_protocol.step_count", 100),
                ("benchmark.run_protocol.aggregation", "arithmetic_mean_v1"),
                ("benchmark.run_protocol.statistic", "mean"),
                ("benchmark.run_protocol.warmup", 1),
                ("benchmark.run_protocol.retry_policy", "none"),
                ("benchmark.run_protocol.timing_method", "synchronized_wall_clock"),
                ("benchmark.run_protocol.generation_seed", 42),
                ("benchmark.environment.hardware", "A100"),
                ("benchmark.environment.software", "benchmark-suite-v1"),
                ("benchmark.environment.runtime", "cuda-12.4"),
                ("benchmark.environment.compiler", "nvcc-12.4"),
                ("benchmark.environment.region", "us-east"),
                ("benchmark.environment.provider", "provider-v1"),
                ("benchmark.environment.sku", "gpu-a100"),
                ("benchmark.environment.pricing_snapshot", "2026-08-01"),
                ("benchmark.environment.pricing_version", "v1"),
                ("benchmark.evaluator.id", "claimci-benchmark-v0"),
                ("benchmark.evaluator.version", "v1"),
                ("benchmark.evaluator.scoring", "exact_match_v1"),
                ("benchmark.evaluator.metric_definition", metric),
                ("benchmark.evaluator.metric_config", "default-v1"),
                ("benchmark.evaluator.normalization", "none"),
                ("benchmark.evaluator.postprocessing", "none"),
                ("benchmark.evaluator.formula_family", "multiply_v1"),
                ("benchmark.evaluator.unit_value", 1),
                ("benchmark.evaluator.quantity", 1),
                ("benchmark.evaluator.included_costs", "compute"),
                ("benchmark.evaluator.excluded_costs", "network"),
                ("measurement.metric_identity.definition", metric),
                ("measurement.metric_identity.configuration", "default-v1"),
                ("measurement.metric_identity.version", "v1"),
            ),
        )
        first = result.observations[0]
        result = replace(
            result,
            observations=(
                replace(
                    first,
                    metric_value=value,
                    run_id=f"{role.value}-run-1",
                    seed=1,
                ),
                replace(
                    first,
                    metric_value=value + (2 if role is ExperimentRole.BASELINE else 1),
                    run_id=f"{role.value}-run-2",
                    seed=2,
                ),
            ),
        )
        evidence.append(result)
        bindings.append(result_binding)
    return tuple(evidence), _mapping(tuple(bindings))


def _training_mapping() -> tuple[tuple[NormalizedEvidence, ...], MappingCandidate]:
    evidence: list[NormalizedEvidence] = []
    bindings: list[ArtifactBinding] = []
    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        result, result_binding = _evidence(
            f"results/{role.value}.json",
            ArtifactKind.RESULTS,
            role,
            metric="accuracy",
            value=0.7 if role is ExperimentRole.BASELINE else 0.8,
        )
        config, config_binding = _evidence(
            f"config/{role.value}.yaml",
            ArtifactKind.CONFIG,
            role,
            config=(("training.steps", 10),),
        )
        evidence.extend((result, config))
        bindings.extend((result_binding, config_binding))
        for split in (DatasetSplit.TRAIN, DatasetSplit.EVAL):
            dataset, dataset_binding = _evidence(
                f"data/{role.value}-{split.value}.jsonl",
                ArtifactKind.DATASET,
                role,
                split=split,
            )
            evidence.append(dataset)
            bindings.append(dataset_binding)
    return tuple(evidence), _mapping(tuple(bindings))


def test_profile_registry_is_immutable_and_authority_contracts_are_factory_only() -> None:
    registry = evidence_profile_registry()

    assert tuple(registry) == (
        EvidenceProfileId.TRAINING_EXPERIMENT_V0,
        EvidenceProfileId.BENCHMARK_MEASUREMENT_V0,
    )
    assert registry[EvidenceProfileId.TRAINING_EXPERIMENT_V0].version == 0
    with pytest.raises(TypeError):
        registry[EvidenceProfileId.TRAINING_EXPERIMENT_V0] = object()  # type: ignore[index]
    for contract in (
        EvidenceProfileDefinition,
        EvidenceProfileSelection,
        ProfileSelectionFact,
        ProfiledEvidencePolicy,
    ):
        with pytest.raises(TypeError):
            contract()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            type("Forged", (contract,), {})


@pytest.mark.parametrize(
    ("claim_text", "variant"),
    (
        ("Latency improved from 100 to 80 by at least 10.", BenchmarkVariant.LATENCY),
        ("Throughput improved from 100 to 150 by at least 10.", BenchmarkVariant.THROUGHPUT_RESOURCE),
        ("Peak VRAM improved from 80 to 60 by at least 10.", BenchmarkVariant.THROUGHPUT_RESOURCE),
        ("Kernel time improved from 8 to 6 by at least 1.", BenchmarkVariant.KERNEL),
        ("Cost per request improved from 2 to 1 by at least 0.5.", BenchmarkVariant.COST),
    ),
)
def test_performance_metric_routes_to_benchmark_without_negative_dataset_inference(
    claim_text: str,
    variant: BenchmarkVariant,
) -> None:
    outcome = select_evidence_profile(
        _claim(claim_text),
        normalized_evidence=(),
        mapping_candidates=(),
    )

    assert outcome.state is ProfileSelectionState.SELECTED
    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.selection.benchmark_variant is variant
    assert outcome.selection.complete is False


def test_positive_benchmark_facts_select_raw_latency_profile() -> None:
    evidence, mapping = _benchmark_mapping()

    outcome = select_evidence_profile(
        _claim("Latency improved from 100 to 80 by at least 10."),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )

    assert outcome.state is ProfileSelectionState.SELECTED
    selection = outcome.selection
    assert selection is not None
    assert selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert selection.benchmark_variant is BenchmarkVariant.LATENCY
    assert selection.result_form is BenchmarkResultForm.RAW_RUN_SERIES
    assert selection.complete is True
    assert selection.facts


def test_repeated_raw_runs_do_not_amplify_profile_fact_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.profiles as profiles

    evidence, mapping = _benchmark_mapping()
    repeated = tuple(
        replace(item, observations=item.observations * 128)
        for item in evidence
    )
    original = profiles._selection_fact
    calls = 0

    def counting_fact(*args: object, **kwargs: object) -> ProfileSelectionFact:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(profiles, "_selection_fact", counting_fact)

    select_evidence_profile(
        _claim("Latency improved from 100 to 80 by at least 10."),
        normalized_evidence=repeated,
        mapping_candidates=(mapping,),
    )

    assert calls <= 192


@pytest.mark.parametrize(
    ("claim_text", "required_key", "variant"),
    (
        (
            "Latency improved from 100 to 80 by at least 10.",
            "benchmark.measurement_config.timing_boundary",
            BenchmarkVariant.LATENCY,
        ),
        (
            "Throughput improved from 100 to 150 by at least 10.",
            "benchmark.environment.runtime",
            BenchmarkVariant.THROUGHPUT_RESOURCE,
        ),
        (
            "Kernel time improved from 8 to 6 by at least 1.",
            "benchmark.workload.tensor_generation",
            BenchmarkVariant.KERNEL,
        ),
        (
            "Accuracy improved from 0.70 to 0.80 by at least 0.05.",
            "benchmark.evaluator.scoring",
            BenchmarkVariant.MODEL_QUALITY,
        ),
        (
            "Cost per request improved from 2 to 1 by at least 0.5.",
            "benchmark.evaluator.included_costs",
            BenchmarkVariant.COST,
        ),
    ),
)
def test_variant_required_component_is_not_satisfied_by_generic_benchmark_shape(
    claim_text: str,
    required_key: str,
    variant: BenchmarkVariant,
) -> None:
    claim = _claim(claim_text)
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    changed_evidence: list[NormalizedEvidence] = []
    changed_bindings: list[ArtifactBinding] = []
    for item, binding in zip(evidence, mapping.bindings, strict=True):
        mappings = tuple(
            field
            for field in item.adapter_match.mappings
            if field.target_field != required_key
        )
        observations = tuple(
            replace(
                observation,
                config_values=tuple(
                    config
                    for config in observation.config_values
                    if config.key != required_key
                ),
            )
            for observation in item.observations
        )
        changed_evidence.append(
            replace(
                item,
                adapter_match=replace(item.adapter_match, mappings=mappings),
                observations=observations,
            )
        )
        changed_bindings.append(replace(binding, mappings=mappings))
    changed_mapping = _mapping(tuple(changed_bindings))

    outcome = select_evidence_profile(
        claim,
        normalized_evidence=tuple(changed_evidence),
        mapping_candidates=(changed_mapping,),
    )

    assert outcome.selection is not None
    assert outcome.selection.benchmark_variant is variant
    assert outcome.selection.complete is False


def test_required_profile_integer_must_be_independently_canonical() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    changed = tuple(
        replace(
            item,
            observations=tuple(
                replace(
                    observation,
                    config_values=tuple(
                        replace(value, value="many")
                        if value.key == "benchmark.run_protocol.warmup"
                        else value
                        for value in observation.config_values
                    ),
                )
                for observation in item.observations
            ),
        )
        for item in evidence
    )

    outcome = plan_ephemeral_audit(
        _planning_request(claim, changed, (mapping,))
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == "clarification_not_bounded"
    assert outcome.mapping_question is None


def test_conflicting_profile_values_within_one_role_fail_before_audit() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    baseline = evidence[0]
    second = replace(
        baseline.observations[1],
        config_values=tuple(
            replace(value, value=2)
            if value.key == "benchmark.run_protocol.warmup"
            else value
            for value in baseline.observations[1].config_values
        ),
    )
    changed = (
        replace(baseline, observations=(baseline.observations[0], second)),
        *evidence[1:],
    )

    outcome = plan_ephemeral_audit(
        _planning_request(claim, changed, (mapping,))
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None


def test_exact_legacy_training_shape_selects_training_profile() -> None:
    evidence, mapping = _training_mapping()

    outcome = select_evidence_profile(
        _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05."),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )

    assert outcome.state is ProfileSelectionState.SELECTED
    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0
    assert outcome.selection.complete is True


def test_incomplete_benchmark_shape_cannot_displace_complete_legacy_training() -> None:
    claim = _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05.")
    training_evidence, training_mapping = _training_mapping()
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")
    incomplete_benchmark = tuple(
        replace(
            item,
            observations=tuple(
                replace(
                    observation,
                    config_values=tuple(
                        value
                        for value in observation.config_values
                        if value.key != "benchmark.evaluator.scoring"
                    ),
                )
                for observation in item.observations
            ),
        )
        for item in benchmark_evidence
    )

    outcome = select_evidence_profile(
        claim,
        normalized_evidence=(*training_evidence, *incomplete_benchmark),
        mapping_candidates=(training_mapping, benchmark_mapping),
    )

    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0
    assert outcome.selection.complete is True
    policy = profiled_evidence_policy(_claim("Accuracy improved from 0.70 to 0.80 by at least 0.05."), outcome.selection)
    assert policy.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0
    assert "artifact.baseline.dataset.train" in policy.profile_template_ids


def test_provider_provenance_cannot_become_profile_authority() -> None:
    evidence, mapping = _benchmark_mapping()
    provider_mapping = _mapping(
        mapping.bindings,
        provenance=ProvenanceKind.PROVIDER_PROPOSAL,
    )

    outcome = select_evidence_profile(
        _claim("Latency improved from 100 to 80 by at least 10."),
        normalized_evidence=evidence,
        mapping_candidates=(provider_mapping,),
    )

    assert outcome.selection is not None
    assert all(
        fact.provenance_kind is not ProvenanceKind.PROVIDER_PROPOSAL
        for fact in outcome.selection.facts
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        outcome.selection.profile_id = EvidenceProfileId.TRAINING_EXPERIMENT_V0  # type: ignore[misc]


def test_provider_injected_normalized_values_cannot_satisfy_profile_after_approval() -> None:
    evidence, native = _benchmark_mapping()
    provider = _provenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        source_id="provider:forged-profile-values",
    )
    forged = tuple(
        replace(
            item,
            observations=tuple(
                replace(
                    observation,
                    provenance=provider,
                    config_values=tuple(
                        replace(config, provenance=provider)
                        for config in observation.config_values
                    ),
                )
                for observation in item.observations
            ),
        )
        for item in evidence
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        _mapping(
            native.bindings,
            provenance=ProvenanceKind.PROVIDER_PROPOSAL,
        ),
        approved_by="pilot-user",
    )

    outcome = select_evidence_profile(
        _claim("Latency improved from 100 to 80 by at least 10."),
        normalized_evidence=forged,
        mapping_candidates=(),
        approved_mapping=approved,
    )

    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.selection.complete is False


def test_complete_training_and_benchmark_shapes_are_semantically_ambiguous() -> None:
    training_evidence, training_mapping = _training_mapping()
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")

    outcome = select_evidence_profile(
        _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05."),
        normalized_evidence=(*training_evidence, *benchmark_evidence),
        mapping_candidates=(training_mapping, benchmark_mapping),
    )

    assert outcome.state is ProfileSelectionState.AMBIGUOUS
    assert outcome.selection is None
    assert outcome.reason == "evidence_profile_ambiguous"


def test_training_mapping_shell_without_exact_evidence_cannot_force_ambiguity() -> None:
    _training_evidence, training_mapping = _training_mapping()
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")

    outcome = select_evidence_profile(
        _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05."),
        normalized_evidence=benchmark_evidence,
        mapping_candidates=(training_mapping, benchmark_mapping),
    )

    assert outcome.state is ProfileSelectionState.SELECTED
    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.selection.complete is True


def test_provider_mapping_cannot_ambiguate_an_independently_complete_profile() -> None:
    training_evidence, training_mapping = _training_mapping()
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")
    provider_training = _mapping(
        training_mapping.bindings,
        provenance=ProvenanceKind.PROVIDER_PROPOSAL,
    )

    outcome = select_evidence_profile(
        _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05."),
        normalized_evidence=(*training_evidence, *benchmark_evidence),
        mapping_candidates=(provider_training, benchmark_mapping),
    )

    assert outcome.state is ProfileSelectionState.SELECTED
    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.selection.complete is True


def test_profile_target_is_additive_and_legacy_artifact_slot_is_unchanged() -> None:
    evidence, mapping = _benchmark_mapping()
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None

    target = profile_evidence_target(outcome.selection, "benchmark.workload")

    assert type(target) is ProfileEvidenceTarget
    assert target.slot_id == "benchmark.workload"
    with pytest.raises(TypeError):
        ProfileEvidenceTarget()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ProfileEvidenceSupport()  # type: ignore[call-arg]
    with pytest.raises(Exception):
        ArtifactEvidenceSlot(ArtifactKind.BENCHMARK, ExperimentRole.BASELINE)
    with pytest.raises(Exception):
        ArtifactEvidenceSlot(ArtifactKind.RESULTS, ExperimentRole.REFERENCE)


def test_profile_support_requires_trusted_mapping_and_exact_recovered_component() -> None:
    evidence, mapping = _benchmark_mapping()
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None
    target = profile_evidence_target(outcome.selection, "benchmark.workload")

    support = validated_profile_evidence_support(
        target=target,
        selection=outcome.selection,
        normalized_evidence=evidence,
        selected_mapping=mapping,
        claim_metric="latency",
    )

    assert support.slot_id == "benchmark.workload"
    assert support.source_binding_ids
    assert support.semantic_projection
    serialized = to_jsonable(support)
    assert "semantic_projection" not in serialized
    assert set(serialized) == {
        "support_id",
        "profile_id",
        "slot_id",
        "role_mode",
        "support_kind",
        "source_binding_ids",
    }

    provider_mapping = _mapping(
        mapping.bindings,
        provenance=ProvenanceKind.PROVIDER_PROPOSAL,
    )
    with pytest.raises(Exception):
        validated_profile_evidence_support(
            target=target,
            selection=outcome.selection,
            normalized_evidence=evidence,
            selected_mapping=provider_mapping,
            claim_metric="latency",
        )


def test_unselected_normalized_scalar_cannot_satisfy_profile_component() -> None:
    evidence, mapping = _benchmark_mapping()
    stripped_evidence: list[NormalizedEvidence] = []
    stripped_bindings: list[ArtifactBinding] = []
    for item, binding in zip(evidence, mapping.bindings, strict=True):
        retained = tuple(
            field
            for field in item.adapter_match.mappings
            if field.target_field != "benchmark.workload.id"
        )
        stripped_evidence.append(
            replace(
                item,
                adapter_match=replace(item.adapter_match, mappings=retained),
            )
        )
        stripped_bindings.append(replace(binding, mappings=retained))
    stripped_mapping = _mapping(tuple(stripped_bindings))

    outcome = select_evidence_profile(
        _claim("Latency improved from 100 to 80 by at least 10."),
        normalized_evidence=tuple(stripped_evidence),
        mapping_candidates=(stripped_mapping,),
    )

    assert outcome.selection is not None
    assert outcome.selection.complete is False


def test_training_profile_assessment_is_exact_legacy_bundle_compatibility() -> None:
    evidence, mapping = _training_mapping()
    claim = _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05.")
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None
    policy = profiled_evidence_policy(claim, outcome.selection)
    kwargs = dict(
        claim=claim,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
        selected_mapping=mapping,
        mapping_question=None,
        ambiguity_reason=None,
    )

    legacy = assess_evidence_obligations(policy=policy.claim_policy, **kwargs)
    profiled = assess_profiled_evidence_obligations(policy=policy, **kwargs)

    assert evidence_obligations_json_bytes(profiled) == evidence_obligations_json_bytes(legacy)


@pytest.mark.parametrize(
    ("claim_text", "expected_count"),
    (
        ("Latency improved from 100 to 80 by at least 10.", 11),
        ("Throughput improved from 100 to 150 by at least 10.", 11),
        ("Kernel time improved from 8 to 6 by at least 1.", 11),
        ("Accuracy improved from 0.70 to 0.80 by at least 0.05.", 11),
        ("Cost per request improved from 2 to 1 by at least 0.5.", 12),
    ),
    ids=("vessl-latency", "unsloth-throughput", "kernel", "model-quality", "cerebrium-cost"),
)
def test_every_benchmark_variant_stays_within_thirteen_obligations(
    claim_text: str,
    expected_count: int,
) -> None:
    claim = _claim(claim_text)
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None
    assert outcome.selection.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.selection.complete
    policy = profiled_evidence_policy(claim, outcome.selection)

    bundle = assess_profiled_evidence_obligations(
        claim=claim,
        policy=policy,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
        selected_mapping=mapping,
        mapping_question=None,
        ambiguity_reason=None,
    )

    assert len(bundle.obligations) == expected_count
    assert len(bundle.obligations) <= 13
    assert sum(item.target is not None for item in bundle.obligations) <= 12
    assert bundle.decision is EvidenceObligationDecision.READY

    planned = plan_ephemeral_audit(_planning_request(claim, evidence, (mapping,)))
    assert planned.state is PlanningState.READY
    assert planned.plan is not None
    result = audit_benchmark(
        audit_claim=planned.plan.audit_claim,
        selection=planned.plan.profiled_policy.profile_selection,
        baseline_evidence=planned.plan.baseline_evidence,
        candidate_evidence=planned.plan.candidate_evidence,
        reference_evidence=planned.plan.reference_evidence,
    )
    assert result.verdict in {
        Verdict.SUPPORTED,
        Verdict.NOT_SUPPORTED,
        Verdict.INSUFFICIENT_EVIDENCE,
    }


def test_native_benchmark_audit_rejects_an_incomplete_profile_selection() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=(),
        mapping_candidates=(),
    )
    assert outcome.selection is not None
    assert outcome.selection.complete is False

    with pytest.raises(ClaimCIError, match="complete evidence profile"):
        audit_benchmark(
            audit_claim=compile_audit_claim(claim),
            selection=outcome.selection,
            baseline_evidence=(),
            candidate_evidence=(),
        )


def test_cost_variant_worst_case_four_claim_fields_plus_eight_profile_groups_fits_cap() -> None:
    claim = _claim(
        "On the held-out workload, cost per request improved from 2 to 1 by at least 0.5."
    )
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete
    policy = profiled_evidence_policy(claim, outcome.selection)

    bundle = assess_profiled_evidence_obligations(
        claim=claim,
        policy=policy,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
        selected_mapping=mapping,
        mapping_question=None,
        ambiguity_reason=None,
    )

    assert len(policy.claim_template_ids) == 4
    assert len(policy.profile_template_ids) == 8
    assert len(bundle.obligations) == 13


def test_cost_with_explicit_reduction_still_fits_grouped_thirteen_obligations() -> None:
    claim = _claim(
        "On the held-out workload, cost per request improved from 2 to 1 by at "
        "least 0.5 using the arithmetic mean across supplied runs."
    )
    evidence, mapping = _benchmark_mapping(claim.primary.metric)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete
    policy = profiled_evidence_policy(claim, outcome.selection)

    bundle = assess_profiled_evidence_obligations(
        claim=claim,
        policy=policy,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
        selected_mapping=mapping,
        mapping_question=None,
        ambiguity_reason=None,
    )

    assert len(policy.claim_template_ids) == 4
    assert len(policy.profile_template_ids) == 8
    assert len(bundle.obligations) == 13
    assert bundle.decision is EvidenceObligationDecision.READY


def _planning_request(
    claim: CanonicalScientificClaim,
    evidence: tuple[NormalizedEvidence, ...],
    mappings: tuple[MappingCandidate, ...],
    *,
    approved: RepoMapping | None = None,
) -> PlanningRequest:
    return PlanningRequest(
        repository=REPOSITORY,
        pr_number=20,
        head_sha=GitCommitSha("a" * 40),
        claim=claim.reference,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=mappings,
        approved_mapping=approved,
        scientific_claim=claim,
        claim_policy=claim_evidence_policy(claim),
    )


def test_benchmark_planner_reaches_ready_without_train_eval_datasets() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()

    outcome = plan_ephemeral_audit(_planning_request(claim, evidence, (mapping,)))

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.profiled_policy is not None
    assert outcome.plan.profiled_policy.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    assert outcome.plan.reference_evidence == ()
    assert all(item.artifact.kind is not ArtifactKind.DATASET for item in (*outcome.plan.baseline_evidence, *outcome.plan.candidate_evidence))


def test_shared_reference_workload_satisfies_profile_without_becoming_subject() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    paired_evidence: list[NormalizedEvidence] = []
    paired_bindings: list[ArtifactBinding] = []
    for item, binding in zip(evidence, mapping.bindings, strict=True):
        mappings = tuple(
            field
            for field in item.adapter_match.mappings
            if field.target_field != "benchmark.workload.id"
        )
        observations = tuple(
            replace(
                observation,
                config_values=tuple(
                    config
                    for config in observation.config_values
                    if config.key != "benchmark.workload.id"
                ),
            )
            for observation in item.observations
        )
        paired_evidence.append(
            replace(
                item,
                adapter_match=replace(item.adapter_match, mappings=mappings),
                observations=observations,
            )
        )
        paired_bindings.append(replace(binding, mappings=mappings))
    reference, reference_binding = _evidence(
        "bench/shared-workload.yaml",
        ArtifactKind.CONFIG,
        ExperimentRole.REFERENCE,
        config=(("benchmark.workload.id", "shared-workload-v1"),),
    )
    all_evidence = (*paired_evidence, reference)
    selected = _mapping((*paired_bindings, reference_binding))

    planned = plan_ephemeral_audit(
        _planning_request(claim, all_evidence, (selected,))
    )

    assert planned.state is PlanningState.READY
    assert planned.plan is not None
    assert planned.plan.reference_evidence == (reference,)
    assert all(
        item.observations[0].config_values[0].key == "benchmark.workload.id"
        for item in planned.plan.reference_evidence
    )


def test_model_quality_reference_jsonl_is_profile_workload_not_train_eval() -> None:
    claim = _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05.")
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")
    paired_without_corpus = tuple(
        replace(
            item,
            observations=tuple(
                replace(
                    observation,
                    config_values=tuple(
                        value
                        for value in observation.config_values
                        if value.key
                        not in {"benchmark.workload.id", "benchmark.workload.corpus"}
                    ),
                )
                for observation in item.observations
            ),
        )
        for item in benchmark_evidence
    )
    workload, workload_binding = _evidence(
        "benchmarks/evaluation-corpus.jsonl",
        ArtifactKind.DATASET,
        ExperimentRole.REFERENCE,
    )
    workload_identity, workload_identity_binding = _evidence(
        "benchmarks/evaluation-corpus.yaml",
        ArtifactKind.CONFIG,
        ExperimentRole.REFERENCE,
        config=(("benchmark.workload.id", "quality-corpus-v1"),),
    )
    mapping = _mapping(
        (
            *benchmark_mapping.bindings,
            workload_binding,
            workload_identity_binding,
        )
    )
    evidence = (*paired_without_corpus, workload, workload_identity)

    outcome = plan_ephemeral_audit(_planning_request(claim, evidence, (mapping,)))

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert set(outcome.plan.reference_evidence) == {workload, workload_identity}
    assert workload_binding.dataset_split is None
    assert not any(
        binding.kind is ArtifactKind.DATASET
        and binding.dataset_split in {DatasetSplit.TRAIN, DatasetSplit.EVAL}
        for binding in outcome.plan.selected_mapping.bindings
    )
    support = validated_profile_evidence_support(
        target=profile_evidence_target(
            outcome.plan.profiled_policy.profile_selection,
            "benchmark.workload",
        ),
        selection=outcome.plan.profiled_policy.profile_selection,
        normalized_evidence=evidence,
        selected_mapping=mapping,
        claim_metric=claim.primary.metric,
    )
    assert support.support_kind is ProfileSupportKind.ARTIFACT_BINDING


def test_explicit_arithmetic_mean_claim_is_supported_by_benchmark_run_protocol() -> None:
    claim = _claim(
        "Latency improved from 100 to 80 by at least 10 using the arithmetic mean across supplied runs."
    )
    evidence, mapping = _benchmark_mapping(claim.primary.metric)

    planned = plan_ephemeral_audit(_planning_request(claim, evidence, (mapping,)))

    assert planned.state is PlanningState.READY
    assert planned.plan is not None
    assert planned.evidence_obligations is not None
    procedure = next(
        item
        for item in planned.evidence_obligations.obligations
        if item.obligation_id == "benchmark.run_protocol"
    )
    assert procedure.state.value == "satisfied"
    assert len(procedure.support_references) == 1
    support = procedure.support_references[0]
    assert type(support) is ProfileEvidenceSupport
    assert (
        "required_upstream_aggregation",
        "arithmetic_mean_v1",
    ) in support.semantic_projection


def test_performance_metric_without_positive_benchmark_facts_is_profile_partial() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")

    outcome = plan_ephemeral_audit(_planning_request(claim, (), ()))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == "required_profile_evidence_not_recovered"
    assert not outcome.missing_evidence


def test_quality_metric_without_either_positive_shape_is_profile_partial() -> None:
    claim = _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05.")

    selection = select_evidence_profile(
        claim,
        normalized_evidence=(),
        mapping_candidates=(),
    )
    planned = plan_ephemeral_audit(_planning_request(claim, (), ()))

    assert selection.state is ProfileSelectionState.UNRESOLVED
    assert selection.selection is None
    assert selection.reason == "required_profile_evidence_not_recovered"
    assert planned.state is PlanningState.PARTIAL
    assert planned.reason == "required_profile_evidence_not_recovered"
    assert not planned.missing_evidence


def test_profile_semantic_ambiguity_is_partial_not_a_profile_question() -> None:
    claim = _claim("Accuracy improved from 0.70 to 0.80 by at least 0.05.")
    training_evidence, training_mapping = _training_mapping()
    benchmark_evidence, benchmark_mapping = _benchmark_mapping("accuracy")

    outcome = plan_ephemeral_audit(
        _planning_request(
            claim,
            (*training_evidence, *benchmark_evidence),
            (training_mapping, benchmark_mapping),
        )
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == "evidence_profile_ambiguous"
    assert outcome.mapping_question is None


def test_provider_benchmark_mapping_requires_explicit_repo_mapping_approval() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, native = _benchmark_mapping()
    provider = _mapping(native.bindings, provenance=ProvenanceKind.PROVIDER_PROPOSAL)

    proposed = plan_ephemeral_audit(_planning_request(claim, evidence, (provider,)))
    assert proposed.state is PlanningState.MAPPING_NEEDED
    assert proposed.mapping_question is not None
    assert proposed.plan is None

    approved = RepoMapping.approve(REPOSITORY, provider, approved_by="pilot-user")
    executable = plan_ephemeral_audit(
        _planning_request(claim, evidence, (provider,), approved=approved)
    )
    assert executable.state is PlanningState.READY
    assert executable.plan is not None
    assert type(executable.plan.selected_mapping) is RepoMapping


def test_multi_result_provider_mapping_keeps_one_bounded_approval_question() -> None:
    claim = _claim("Latency improved from 100 to 80 by at least 10.")
    evidence, native = _benchmark_mapping()
    adjusted = tuple(
        replace(
            item,
            observations=tuple(
                replace(
                    observation,
                    config_values=tuple(
                        replace(value, value=3)
                        if value.key == "benchmark.run_protocol.sample_count"
                        else value
                        for value in observation.config_values
                    ),
                )
                for observation in item.observations
            ),
        )
        for item in evidence
    )
    extra_evidence: list[NormalizedEvidence] = []
    extra_bindings: list[ArtifactBinding] = []
    for role, value in (
        (ExperimentRole.BASELINE, 99.0),
        (ExperimentRole.CANDIDATE, 79.0),
    ):
        extra, binding = _evidence(
            f"bench/extra-{role.value}.csv",
            ArtifactKind.BENCHMARK,
            role,
            metric="latency",
            value=value,
        )
        extra_evidence.append(extra)
        extra_bindings.append(binding)
    provider = _mapping(
        (*native.bindings, *extra_bindings),
        provenance=ProvenanceKind.PROVIDER_PROPOSAL,
    )
    all_evidence = (*adjusted, *extra_evidence)

    proposed = plan_ephemeral_audit(
        _planning_request(claim, all_evidence, (provider,))
    )

    assert proposed.state is PlanningState.MAPPING_NEEDED
    assert proposed.mapping_question is not None
    assert len(proposed.mapping_question.choices) == 2
    approved = RepoMapping.approve(REPOSITORY, provider, approved_by="pilot-user")
    executable = plan_ephemeral_audit(
        _planning_request(
            claim,
            all_evidence,
            (provider,),
            approved=approved,
        )
    )
    assert executable.state is PlanningState.READY
    assert executable.plan is not None
    assert len(executable.plan.baseline_evidence) == 2
    assert len(executable.plan.candidate_evidence) == 2


def test_raw_benchmark_uses_existing_arithmetic_reduction_and_audit_authority() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=tuple(
            item for item in evidence if item.observations[0].experiment_role is ExperimentRole.BASELINE
        ),
        candidate_evidence=tuple(
            item for item in evidence if item.observations[0].experiment_role is ExperimentRole.CANDIDATE
        ),
    )

    assert result.verdict is Verdict.SUPPORTED
    assert result.baseline_metrics is not None
    assert result.baseline_metrics.mean == 101.0
    assert result.candidate_metrics is not None
    assert result.candidate_metrics.mean == 80.5
    assert any(item.rule_id == "RESULT.RECOMPUTED" for item in result.findings)


def test_missing_benchmark_procedure_is_native_insufficient_rule_fact() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(),
        candidate_evidence=(evidence[1],),
    )

    finding = next(
        item
        for item in result.findings
        if item.rule_id == "BENCHMARK.PROCEDURE_INSUFFICIENT"
    )
    assert finding.severity is Severity.WARNING
    assert finding.impact is Impact.INSUFFICIENT
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_known_workload_mismatch_is_native_invalidating_finding() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    candidate = evidence[1]
    candidate_observations = tuple(
        replace(
            observation,
            config_values=tuple(
                replace(item, value="different-workload")
                if item.key == "benchmark.workload.id"
                else item
                for item in observation.config_values
            ),
        )
        for observation in candidate.observations
    )
    changed = replace(candidate, observations=candidate_observations)
    changed_evidence = (evidence[0], changed)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=changed_evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(changed_evidence[0],),
        candidate_evidence=(changed_evidence[1],),
    )

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.WORKLOAD_MISMATCH"
        and item.impact.value == "INVALIDATES"
        for item in result.findings
    )


@pytest.mark.parametrize(
    "key",
    (
        "benchmark.measurement_config.timing_boundary",
        "benchmark.run_protocol.retry_policy",
    ),
)
def test_known_measurement_config_or_procedure_mismatch_reaches_native_audit(
    key: str,
) -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    candidate = evidence[1]
    changed = replace(
        candidate,
        observations=tuple(
            replace(
                observation,
                config_values=tuple(
                    replace(item, value="changed-v2") if item.key == key else item
                    for item in observation.config_values
                ),
            )
            for observation in candidate.observations
        ),
    )
    changed_evidence = (evidence[0], changed)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=changed_evidence,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(changed_evidence[0],),
        candidate_evidence=(changed_evidence[1],),
    )

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.CONFIG_MISMATCH"
        and item.impact.value == "INVALIDATES"
        for item in result.findings
    )


def test_claim_metric_and_recovered_measurement_metric_remain_separate() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping("runtime")

    planned = plan_ephemeral_audit(_planning_request(claim, evidence, (mapping,)))

    assert planned.state is PlanningState.READY
    assert planned.plan is not None
    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=planned.plan.profiled_policy.profile_selection,  # type: ignore[union-attr]
        baseline_evidence=planned.plan.baseline_evidence,
        candidate_evidence=planned.plan.candidate_evidence,
    )
    assert result.verdict is Verdict.NOT_SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.METRIC_MISMATCH"
        and item.impact.value == "INVALIDATES"
        for item in result.findings
    )


def test_different_recovered_role_metrics_never_emit_protocol_verified() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    candidate = replace(
        evidence[1],
        observations=tuple(
            replace(observation, metric_name="runtime")
            for observation in evidence[1].observations
        ),
    )
    changed = (evidence[0], candidate)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=changed,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(changed[0],),
        candidate_evidence=(changed[1],),
    )

    assert result.verdict is Verdict.NOT_SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.METRIC_MISMATCH"
        for item in result.findings
    )
    assert not any(
        item.rule_id == "BENCHMARK.PROTOCOL_VERIFIED"
        for item in result.findings
    )


def _with_benchmark_form(
    evidence: tuple[NormalizedEvidence, ...],
    form: BenchmarkResultForm,
    *,
    remove: frozenset[str] = frozenset(),
    extras: tuple[tuple[str, object], ...] = (),
    one_metric: bool = False,
) -> tuple[NormalizedEvidence, ...]:
    changed: list[NormalizedEvidence] = []
    for item in evidence:
        observations = item.observations[:1] if one_metric else item.observations
        rewritten = []
        for observation in observations:
            values = {
                config.key: config
                for config in observation.config_values
                if config.key not in remove
                and config.key != "benchmark.run_protocol.result_form"
            }
            values["benchmark.run_protocol.result_form"] = ConfigValue(
                "benchmark.run_protocol.result_form",
                form.value,
                observation.provenance,
            )
            for key, value in extras:
                values[key] = ConfigValue(key, value, observation.provenance)
            rewritten.append(
                replace(
                    observation,
                    config_values=tuple(values[key] for key in sorted(values)),
                )
            )
        changed.append(replace(item, observations=tuple(rewritten)))
    return tuple(changed)


def test_reported_aggregate_is_compared_without_raw_run_reinterpretation() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    aggregate = _with_benchmark_form(
        evidence,
        BenchmarkResultForm.REPORTED_AGGREGATE,
        extras=(
            ("benchmark.run_protocol.sample_count", 50),
            ("benchmark.run_protocol.aggregation", "median_v1"),
            ("benchmark.run_protocol.statistic", "p50"),
        ),
        one_metric=True,
    )
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=aggregate,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None
    assert outcome.selection.complete is True

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(aggregate[0],),
        candidate_evidence=(aggregate[1],),
    )

    assert result.verdict is Verdict.SUPPORTED
    assert any(item.rule_id == "RESULT.AGGREGATE_VERIFIED" for item in result.findings)
    assert not any(item.rule_id == "RESULT.RECOMPUTED" for item in result.findings)
    assert not any(item.rule_id.startswith("SEED.") for item in result.findings)


def test_different_complete_result_form_bindings_require_one_mapping_choice() -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    raw_evidence, raw_mapping = _benchmark_mapping(path_prefix="bench/raw")
    aggregate_evidence, aggregate_mapping = _benchmark_mapping(
        path_prefix="bench/aggregate"
    )
    aggregate_evidence = _with_benchmark_form(
        aggregate_evidence,
        BenchmarkResultForm.REPORTED_AGGREGATE,
        extras=(
            ("benchmark.run_protocol.sample_count", 50),
            ("benchmark.run_protocol.aggregation", "median_v1"),
            ("benchmark.run_protocol.statistic", "p50"),
        ),
        one_metric=True,
    )

    outcome = plan_ephemeral_audit(
        _planning_request(
            claim,
            (*raw_evidence, *aggregate_evidence),
            (raw_mapping, aggregate_mapping),
        )
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None
    assert len(outcome.mapping_question.choices) == 2

    approved = RepoMapping.approve(
        REPOSITORY,
        aggregate_mapping,
        approved_by="pilot-user",
    )
    resolved = plan_ephemeral_audit(
        _planning_request(
            claim,
            (*raw_evidence, *aggregate_evidence),
            (raw_mapping, aggregate_mapping),
            approved=approved,
        )
    )

    assert resolved.state is PlanningState.READY
    assert resolved.plan is not None
    assert (
        resolved.plan.profiled_policy.profile_selection.result_form
        is BenchmarkResultForm.REPORTED_AGGREGATE
    )


@pytest.mark.parametrize(
    "missing_key",
    (
        "benchmark.run_protocol.sample_count",
        "benchmark.run_protocol.aggregation",
    ),
)
def test_aggregate_missing_required_metadata_is_pre_audit_partial(
    missing_key: str,
) -> None:
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    evidence, mapping = _benchmark_mapping()
    aggregate = _with_benchmark_form(
        evidence,
        BenchmarkResultForm.REPORTED_AGGREGATE,
        remove=frozenset({missing_key}),
        extras=tuple(
            item
            for item in (
                ("benchmark.run_protocol.aggregation", "median_v1"),
                ("benchmark.run_protocol.statistic", "p50"),
            )
            if item[0] != missing_key
        ),
        one_metric=True,
    )

    outcome = plan_ephemeral_audit(_planning_request(claim, aggregate, (mapping,)))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.plan is None


def test_fixed_cost_derivation_recomputes_inputs_and_rejects_arbitrary_formula() -> None:
    claim = _claim("Cost decreased from 2 to 1 by at least 0.5.")
    evidence, mapping = _benchmark_mapping("cost")
    derived = _with_benchmark_form(
        evidence,
        BenchmarkResultForm.DETERMINISTIC_DERIVATION,
        extras=(
            ("benchmark.evaluator.formula_family", "multiply_v1"),
            ("benchmark.evaluator.unit_value", 1),
            ("benchmark.evaluator.quantity", 1),
        ),
        one_metric=True,
    )
    baseline = replace(
        derived[0],
        observations=(replace(derived[0].observations[0], metric_value=2.0),),
    )
    candidate = replace(
        derived[1],
        observations=(replace(derived[1].observations[0], metric_value=1.0),),
    )
    role_inputs = []
    for item, unit in ((baseline, 2), (candidate, 1)):
        observation = item.observations[0]
        configs = tuple(
            replace(config, value=unit)
            if config.key == "benchmark.evaluator.unit_value"
            else config
            for config in observation.config_values
        )
        role_inputs.append(
            replace(item, observations=(replace(observation, config_values=configs),))
        )
    exact = tuple(role_inputs)
    outcome = select_evidence_profile(
        claim,
        normalized_evidence=exact,
        mapping_candidates=(mapping,),
    )
    assert outcome.selection is not None and outcome.selection.complete

    derivation_support = validated_profile_evidence_support(
        target=profile_evidence_target(
            outcome.selection,
            "benchmark.evaluator",
        ),
        selection=outcome.selection,
        normalized_evidence=exact,
        selected_mapping=mapping,
        claim_metric=claim.primary.metric,
    )
    assert (
        derivation_support.support_kind
        is ProfileSupportKind.DETERMINISTIC_DERIVATION
    )

    result = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(exact[0],),
        candidate_evidence=(exact[1],),
    )
    assert result.verdict is Verdict.SUPPORTED
    assert not any(
        item.rule_id == "BENCHMARK.DERIVATION_MISMATCH" for item in result.findings
    )

    forged_value = replace(
        exact[0],
        observations=(
            replace(exact[0].observations[0], metric_value=999.0),
        ),
    )
    mismatch = audit_benchmark(
        audit_claim=compile_audit_claim(claim),
        selection=outcome.selection,
        baseline_evidence=(forged_value,),
        candidate_evidence=(exact[1],),
    )
    assert mismatch.verdict is Verdict.NOT_SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.DERIVATION_MISMATCH"
        and item.impact.value == "INVALIDATES"
        for item in mismatch.findings
    )

    arbitrary = _with_benchmark_form(
        evidence,
        BenchmarkResultForm.DETERMINISTIC_DERIVATION,
        extras=(
            ("benchmark.evaluator.formula_family", "python:price * tokens"),
            ("benchmark.evaluator.unit_value", 1),
            ("benchmark.evaluator.quantity", 1),
        ),
        one_metric=True,
    )
    blocked = plan_ephemeral_audit(_planning_request(claim, arbitrary, (mapping,)))
    assert blocked.state is PlanningState.PARTIAL
    assert blocked.plan is None


def _shared_latency_table_plan(
    tmp_path: Path,
    *,
    candidate_workload: str = "requests-v1",
    measured_metric: str = "latency",
    artifact_kind: ArtifactKind = ArtifactKind.BENCHMARK,
    result_form: BenchmarkResultForm = BenchmarkResultForm.RAW_RUN_SERIES,
    reference_workload: bool = False,
    reference_dataset: bool = False,
) -> tuple[object, RuntimeExecutionContext, Path, ArtifactCandidate]:
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    checkout.mkdir()
    scratch.mkdir()
    marker = checkout / "customer-code-ran"
    (checkout / "customer_benchmark.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    path = RepositoryPath("benchmarks/latency.csv")
    if result_form is BenchmarkResultForm.RAW_RUN_SERIES:
        rows = (
            "baseline,b-1,1,100,sequential,requests-v1,wall_clock,RAW_RUN_SERIES,2,arithmetic_mean_v1,mean,1,A100,bench-v1,provider-prose-must-not-persist\n"
            "baseline,b-2,2,102,sequential,requests-v1,wall_clock,RAW_RUN_SERIES,2,arithmetic_mean_v1,mean,1,A100,bench-v1,provider-prose-must-not-persist\n"
            f"candidate,c-1,1,80,parallel,{candidate_workload},wall_clock,RAW_RUN_SERIES,2,arithmetic_mean_v1,mean,1,A100,bench-v1,provider-prose-must-not-persist\n"
            f"candidate,c-2,2,81,parallel,{candidate_workload},wall_clock,RAW_RUN_SERIES,2,arithmetic_mean_v1,mean,1,A100,bench-v1,provider-prose-must-not-persist\n"
        )
        expected_rows = 2
    else:
        rows = (
            "baseline,b-aggregate,1,100,sequential,requests-v1,wall_clock,REPORTED_AGGREGATE,50,median_v1,p50,1,A100,bench-v1,provider-prose-must-not-persist\n"
            f"candidate,c-aggregate,1,80,parallel,{candidate_workload},wall_clock,REPORTED_AGGREGATE,50,median_v1,p50,1,A100,bench-v1,provider-prose-must-not-persist\n"
        )
        expected_rows = 1
    content = (
        f"role,run,seed,{measured_metric},subject,workload,timing,form,samples,aggregation,statistic,warmup,hardware,evaluator,ignored\n"
        + rows
    ).encode()
    destination = checkout / str(path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)
    artifact = ArtifactCandidate(
        path,
        artifact_kind,
        Sha256Digest(hashlib.sha256(content).hexdigest()),
        len(content),
        Confidence(0.99),
        "exact benchmark table fixture",
        (CLAIM_ID,),
        _provenance(path=str(path), source_id="artifact:latency-table"),
    )
    passive = passive_artifact(artifact, content)
    adapter = CsvAdapter()
    fields = (
        ("metric_value", measured_metric),
        ("benchmark.subject.id", "subject"),
        *((() if reference_workload else (("benchmark.workload.id", "workload"),))),
        ("benchmark.measurement_config.timing_boundary", "timing"),
        ("benchmark.run_protocol.result_form", "form"),
        ("benchmark.run_protocol.sample_count", "samples"),
        ("benchmark.run_protocol.aggregation", "aggregation"),
        ("benchmark.run_protocol.statistic", "statistic"),
        ("benchmark.run_protocol.warmup", "warmup"),
        ("benchmark.environment.hardware", "hardware"),
        ("benchmark.evaluator.id", "evaluator"),
    )
    evidence: list[NormalizedEvidence] = []
    bindings: list[ArtifactBinding] = []
    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        provenance = FieldProvenance(
            ProvenanceKind.ADAPTER_EXTRACTION,
            f"exact table selector; sha256={artifact.sha256}",
            path,
            f"selector:{role.value}",
        )
        predicates = (
            TablePredicate("role", TableScalarType.STRING, role.value),
        )
        mappings = tuple(
            FieldMapping(
                target,
                TableSelector(column, predicates, expected_rows, provenance),
                provenance,
            )
            for target, column in fields
        ) + (
            FieldMapping(
                "run_id",
                EvidenceSelector(SelectorKind.COLUMN, "run", provenance),
                provenance,
            ),
            FieldMapping(
                "seed",
                EvidenceSelector(SelectorKind.COLUMN, "seed", provenance),
                provenance,
            ),
        )
        normalized = adapter.extract(
            passive,
            AdapterMatch(
                adapter.adapter_id,
                path,
                Confidence(0.99),
                mappings,
                (provenance,),
            ),
        )
        evidence.append(normalized)
        bindings.append(
            ArtifactBinding(
                path,
                artifact_kind,
                role,
                normalized.adapter_match.adapter_id,
                normalized.adapter_match.mappings,
                provenance,
            )
        )
    artifacts = [artifact]
    if reference_workload:
        reference_content = (
            b"benchmark:\n  workload:\n    id: requests-v1\n"
            b"provider_prose: provider-yaml-prose-must-not-persist\n"
        )
        reference_path = RepositoryPath("benchmarks/workload.yaml")
        reference_destination = checkout / str(reference_path)
        reference_destination.write_bytes(reference_content)
        reference_artifact = ArtifactCandidate(
            reference_path,
            ArtifactKind.CONFIG,
            Sha256Digest(hashlib.sha256(reference_content).hexdigest()),
            len(reference_content),
            Confidence(0.99),
            "exact shared workload fixture",
            (CLAIM_ID,),
            _provenance(
                path=str(reference_path),
                source_id="artifact:shared-workload",
            ),
        )
        reference_passive = passive_artifact(reference_artifact, reference_content)
        config_adapter = YamlConfigAdapter()
        reference_match = config_adapter.probe(reference_passive)
        assert reference_match is not None
        reference_evidence = config_adapter.extract(
            reference_passive,
            reference_match,
        )
        evidence.append(reference_evidence)
        bindings.append(
            ArtifactBinding(
                reference_path,
                ArtifactKind.CONFIG,
                ExperimentRole.REFERENCE,
                reference_evidence.adapter_match.adapter_id,
                reference_evidence.adapter_match.mappings,
                _provenance(
                    path=str(reference_path),
                    source_id="binding:shared-workload",
                ),
            )
        )
        artifacts.append(reference_artifact)
    if reference_dataset:
        dataset_content = b'{"request":"fixed-pilot-input"}\n'
        dataset_path = RepositoryPath("benchmarks/reference-workload.jsonl")
        dataset_destination = checkout / str(dataset_path)
        dataset_destination.write_bytes(dataset_content)
        dataset_artifact = ArtifactCandidate(
            dataset_path,
            ArtifactKind.DATASET,
            Sha256Digest(hashlib.sha256(dataset_content).hexdigest()),
            len(dataset_content),
            Confidence(0.99),
            "exact shared benchmark input corpus",
            (CLAIM_ID,),
            _provenance(
                path=str(dataset_path),
                source_id="artifact:reference-workload",
            ),
        )
        dataset_passive = passive_artifact(dataset_artifact, dataset_content)
        dataset_adapter = PassiveJsonLinesDatasetAdapter()
        dataset_match = dataset_adapter.probe(dataset_passive)
        assert dataset_match is not None
        dataset_evidence = dataset_adapter.extract(dataset_passive, dataset_match)
        evidence.append(dataset_evidence)
        bindings.append(
            ArtifactBinding(
                dataset_path,
                ArtifactKind.DATASET,
                ExperimentRole.REFERENCE,
                dataset_evidence.adapter_match.adapter_id,
                (),
                _provenance(
                    path=str(dataset_path),
                    source_id="binding:reference-workload",
                ),
            )
        )
        artifacts.append(dataset_artifact)
    mapping = _mapping(tuple(bindings))
    claim = _claim("Latency decreased from 100 to 80 by at least 10.")
    request = PlanningRequest(
        repository=REPOSITORY,
        pr_number=20,
        head_sha=GitCommitSha("a" * 40),
        claim=claim.reference,
        audit_claim=compile_audit_claim(claim),
        artifacts=tuple(artifacts),
        normalized_evidence=tuple(evidence),
        mapping_candidates=(mapping,),
        scientific_claim=claim,
        claim_policy=claim_evidence_policy(claim),
    )
    outcome = plan_ephemeral_audit(request)
    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    runtime = RuntimeExecutionContext(
        repository=REPOSITORY,
        checkout_root=checkout,
        head_sha=GitCommitSha("a" * 40),
        scratch_root=scratch,
    )
    return outcome.plan, runtime, scratch, artifact


def test_benchmark_materialization_recaptures_one_table_and_emits_complete_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, scratch, artifact = _shared_latency_table_plan(tmp_path)
    original = materialize._capture_artifact
    captures: list[RepositoryPath] = []

    def counting_capture(*args: object, **kwargs: object) -> PassiveArtifact:
        candidate = args[0]
        assert isinstance(candidate, ArtifactCandidate)
        captures.append(candidate.path)
        return original(*args, **kwargs)

    monkeypatch.setattr(materialize, "_capture_artifact", counting_capture)

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.SUPPORTED
    assert execution.audit_result.measurement_drift is not None
    assert (
        execution.audit_result.measurement_drift.state
        is MeasurementDriftState.VERIFIED
    )
    assert (
        execution.audit_result.measurement_drift.claimci_verification_reduction.value
        == "arithmetic_mean_v1"
    )
    assert (
        execution.audit_result.measurement_drift.baseline_semantic_protocol_id
        == execution.audit_result.measurement_drift.candidate_semantic_protocol_id
    )
    assert (
        execution.audit_result.measurement_drift.baseline_source_snapshot_id
        != execution.audit_result.measurement_drift.candidate_source_snapshot_id
    )
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    assert captures == [artifact.path]
    assert any(
        entry.detail_code == "benchmark.selected_measurement_values"
        for entry in execution.trace.entries
    )
    assert b"provider-prose-must-not-persist" not in trace_json_bytes(execution.trace)
    assert not (runtime.checkout_root / "customer-code-ran").exists()
    assert not tuple(scratch.iterdir())


def test_measurement_source_snapshot_commits_exact_table_row_selector(
    tmp_path: Path,
) -> None:
    from claimci.analysis.measurement import (
        measurement_audit_context_from_materialization,
    )

    plan, runtime, _scratch, artifact = _shared_latency_table_plan(tmp_path)
    assert plan.selected_mapping is not None
    baseline_binding = next(
        item
        for item in plan.selected_mapping.bindings
        if item.role is ExperimentRole.BASELINE
    )
    candidate_binding = next(
        item
        for item in plan.selected_mapping.bindings
        if item.role is ExperimentRole.CANDIDATE
    )
    captured = {
        str(artifact.path): passive_artifact(
            artifact,
            (runtime.checkout_root / str(artifact.path)).read_bytes(),
        )
    }
    original = measurement_audit_context_from_materialization(
        plan,
        (
            (plan.baseline_evidence[0], baseline_binding),
            (plan.candidate_evidence[0], candidate_binding),
        ),
        captured,
    )
    first = baseline_binding.mappings[0]
    assert type(first.selector) is TableSelector
    changed_predicate = replace(
        first.selector.predicates[0],
        value="different-issued-row",
    )
    changed_selector = replace(
        first.selector,
        predicates=(changed_predicate,),
    )
    changed_binding = replace(
        baseline_binding,
        mappings=(replace(first, selector=changed_selector), *baseline_binding.mappings[1:]),
    )

    changed = measurement_audit_context_from_materialization(
        plan,
        (
            (plan.baseline_evidence[0], changed_binding),
            (plan.candidate_evidence[0], candidate_binding),
        ),
        captured,
    )

    assert (
        original.baseline_source.source_snapshot_id
        != changed.baseline_source.source_snapshot_id
    )


def test_benchmark_materialization_rejects_forged_normalized_value(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(tmp_path)
    baseline = plan.baseline_evidence[0]
    forged = replace(
        baseline,
        observations=(
            replace(baseline.observations[0], metric_value=1.0),
        ),
    )
    changed = replace(
        plan,
        baseline_evidence=(forged,),
    )

    with pytest.raises(MaterializationUnavailable, match="table evidence"):
        execute_ephemeral_audit_with_trace(changed, runtime)


def test_benchmark_trace_classifies_native_comparability_finding(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        candidate_workload="requests-v2",
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.NOT_SUPPORTED
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    derived = tuple(
        item
        for item in execution.trace.entries
        if item.detail_code == "derived.benchmark"
    )
    assert len(derived) == 1
    assert "BENCHMARK.WORKLOAD_MISMATCH" in derived[0].consumer_rule_ids


@pytest.mark.parametrize(
    "rule_id",
    (
        "BENCHMARK.METRIC_MISMATCH",
        "BENCHMARK.WORKLOAD_MISMATCH",
        "BENCHMARK.CONFIG_MISMATCH",
        "BENCHMARK.PROCEDURE_INSUFFICIENT",
        "BENCHMARK.DERIVATION_MISMATCH",
        "BENCHMARK.PROTOCOL_VERIFIED",
        "RESULT.AGGREGATE_VERIFIED",
    ),
)
def test_every_new_native_rule_is_classified_by_evidence_trace(
    tmp_path: Path,
    rule_id: str,
) -> None:
    import claimci.analysis.materialize as materialize

    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(tmp_path)
    execution = execute_ephemeral_audit_with_trace(plan, runtime)
    passive = tuple(
        item
        for item in execution.trace.entries
        if item.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE
    )
    template = execution.audit_result.findings[0]
    projected = replace(
        execution.audit_result,
        findings=(replace(template, rule_id=rule_id),),
    )

    derived = materialize._derived_trace_entries(plan, projected, passive)

    assert rule_id in {
        consumer
        for entry in derived
        for consumer in entry.consumer_rule_ids
    }


def test_runtime_rejects_profile_selection_from_different_bound_evidence(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(first_root)
    other, _other_runtime, _other_scratch, _other_artifact = (
        _shared_latency_table_plan(
            second_root,
            candidate_workload="requests-v2",
        )
    )
    changed = replace(
        plan,
        profiled_policy=other.profiled_policy,
        evidence_obligations=other.evidence_obligations,
    )
    changed = replace(changed, plan_id=derive_ephemeral_plan_id(changed))

    with pytest.raises(MaterializationUnavailable, match="profile selection"):
        execute_ephemeral_audit_with_trace(changed, runtime)


def test_measured_metric_mismatch_is_audited_and_traced_from_exact_table(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        measured_metric="runtime",
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.NOT_SUPPORTED
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    assert execution.audit_result.measurement_drift is not None
    metric_drift = next(
        item
        for item in execution.audit_result.measurement_drift.findings
        if item.component_kind.value == "metric"
    )
    assert metric_drift.state is MeasurementDriftState.VERIFIED
    assert metric_drift.native_rule_ids == ()
    assert any(
        item.rule_id == "BENCHMARK.METRIC_MISMATCH"
        for item in execution.audit_result.findings
    )


def test_benchmark_profile_results_kind_retains_mismatch_trace_provenance(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        measured_metric="runtime",
        artifact_kind=ArtifactKind.RESULTS,
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.NOT_SUPPORTED
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    assert any(
        item.detail_code == "benchmark.selected_measurement_values"
        and "BENCHMARK.METRIC_MISMATCH" in item.consumer_rule_ids
        for item in execution.trace.entries
    )


def test_reported_aggregate_does_not_claim_raw_run_verification_reduction(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        result_form=BenchmarkResultForm.REPORTED_AGGREGATE,
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.SUPPORTED
    assert execution.audit_result.measurement_drift is not None
    assert (
        execution.audit_result.measurement_drift.claimci_verification_reduction.value
        == "not_applicable"
    )
    assert any(
        item.rule_id == "RESULT.AGGREGATE_VERIFIED"
        for item in execution.audit_result.findings
    )


def test_reference_workload_is_freshly_reextracted_by_fixed_config_adapter(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        reference_workload=True,
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.SUPPORTED
    assert any(
        item.rule_id == "BENCHMARK.PROTOCOL_VERIFIED"
        for item in execution.audit_result.findings
    )
    assert len(plan.reference_evidence) == 1
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
    assert any(
        item.detail_code == "config.selected_scalars"
        and item.role is ExperimentRole.REFERENCE
        for item in execution.trace.entries
    )
    assert any(
        "BENCHMARK.PROTOCOL_VERIFIED" in item.consumer_rule_ids
        for item in execution.trace.entries
    )
    assert b"provider-yaml-prose-must-not-persist" not in trace_json_bytes(
        execution.trace
    )
    assert b"provider_prose" not in trace_json_bytes(execution.trace)


def test_reference_jsonl_is_passive_benchmark_input_without_train_eval_split(
    tmp_path: Path,
) -> None:
    plan, runtime, _scratch, _artifact = _shared_latency_table_plan(
        tmp_path,
        reference_workload=True,
        reference_dataset=True,
    )

    execution = execute_ephemeral_audit_with_trace(plan, runtime)

    assert execution.audit_result.verdict is Verdict.SUPPORTED
    dataset_bindings = tuple(
        binding
        for binding in plan.selected_mapping.bindings
        if binding.kind is ArtifactKind.DATASET
    )
    assert len(dataset_bindings) == 1
    assert dataset_bindings[0].role is ExperimentRole.REFERENCE
    assert dataset_bindings[0].dataset_split is None
    assert execution.trace.completeness is TraceCompleteness.COMPLETE
