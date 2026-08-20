# Evaluator and Measurement Drift v1 Design

## Goal and authority

Measurement Drift v1 answers one bounded question: did the baseline and candidate use the same measurement procedure that ClaimCI can independently recover from passive evidence, or did the represented ruler change?

The authority sequence is fixed:

```text
Evidence Obligations / planning
  -> required procedure semantics present and executable?
  -> exact-head passive materialization
  -> independently recover semantic measurement fields
  -> native Audit comparison
  -> AuditResult findings and verdict
```

Provider output may propose a claim or mapping, and `RepoMapping.approve()` may authorize a binding selection. Neither can construct a measurement identity, assign component scope, assert drift, emit a finding, or affect a verdict. Customer repository data remains passive and is never executed. `AuditResult` remains the sole deterministic verdict authority.

## Identity split

`MeasurementProtocolIdentity` carries two independent commitments.

`semantic_protocol_id` is the SHA-256 commitment to:

- protocol domain and schema version;
- fixed ClaimCI measurement policy ID and version;
- trusted component schema IDs and versions;
- component kind and trusted scope;
- component coverage;
- independently recovered bounded semantic fields;
- metric identity/definition/configuration;
- evaluation dataset declared identity/version/split and canonical record-multiset commitment;
- represented evaluation/evaluator/procedure semantics.

It explicitly excludes repository/PR identity, head SHA, paths, raw hashes and sizes, selectors, adapters, provider confidence or prose, approval actor, claim direction, claim improvement threshold, findings, and verdict.

`source_snapshot_id` is the SHA-256 commitment to:

- repository identity and exact head SHA;
- every exact passive measurement binding: path, artifact kind, SHA-256, size, adapter ID/version, selectors, experiment role, and dataset split.

Source bindings and components are canonically ordered. A source-only change can therefore alter `source_snapshot_id` without altering `semantic_protocol_id`. It is provenance drift only and never automatic invalidation. V1 does not attempt arbitrary source-code semantic equivalence.

## Contracts and bounds

`claimci.measurement` owns frozen, validated, final value objects:

- `MeasurementComponentKind`: the ten approved component kinds.
- `MeasurementComponentScope`: `MEASUREMENT_PROCEDURE`, `SYSTEM_UNDER_TEST`, `UNKNOWN`.
- `MeasurementCoverage`: `RECOVERED`, `BYTE_ONLY`, `MISSING`, `NOT_APPLICABLE`.
- `MeasurementDriftState`: `VERIFIED`, `WARNING`, `INSUFFICIENT`, `INVALIDATES`, `NOT_ASSESSED`.
- `MeasurementSemanticField`: one canonical bounded scalar field.
- `MeasurementSourceSelector` and `MeasurementSourceBinding`: exact passive source identity.
- factory-created `MeasurementSourceSnapshot` and `MeasurementProtocolIdentity`.
- `MeasurementProtocolPair`, `MeasurementDriftFinding`, and `MeasurementDriftReport`.
- `ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1`, represented separately from upstream procedure identity.

The production v1 recovery path emits exactly one component per known kind; the reusable identity contract permits a bounded subset but never duplicate kinds. Each component has at most 32 scalar fields, each source snapshot has at most 16 bindings, and all identifiers and strings are bounded. Canonical JSON uses finite JSON scalars and deterministic ordering. No generic provider deserializer exists. Component scope and schema are fixed by ClaimCI's v1 policy, and drift findings/reports are comparator-factory-only. Machine output exposes bounded IDs, coverage, states, and referenced native rule IDs; it does not persist evaluator source, prompt text, dataset rows, or duplicate detailed Trace evidence.

## Pilot semantic recovery

The exact-head analysis builder records source snapshots only after plan, mapping, artifact hash, adapter, selector, role, and split revalidation. Native Audit then derives semantic components from the confined native inputs that it already safely parses:

- `METRIC`: exact audited metric name plus fixed supported `evaluation.metric*` scalar configuration when represented.
- `EVALUATION_DATASET`: declared dataset identifier/version/split plus an order-independent canonical evaluation-record multiset commitment and record count.
- `EVALUATION_CONFIG`: fixed bounded scalar projection of the represented evaluation mapping.
- `RETRY_AGGREGATION`: independently recovered fixed aggregation/retry fields when represented.
- other component kinds only when fixed ClaimCI key policy classifies represented `evaluation.*` fields into that component; otherwise `NOT_APPLICABLE`.

Provider values are never inputs. Direction and minimum improvement remain only in `AuditClaimSpec`. Changing either therefore cannot change a semantic protocol ID.

The default Pilot policy does not require evaluator source, prompt, decoding, filtering, normalization, or benchmark metadata. Their absence is `NOT_APPLICABLE` / `NOT_ASSESSED` and has no lifecycle or verdict effect. A byte-only evaluator component can report `WARNING` when its source commitment differs, but never invalidates on hash inequality alone.

## Native comparison and non-duplication

V1 adds no `MEASUREMENT.*` Audit rule. It reuses current native authority:

- changed canonical evaluation population is represented by `DATASET.EVALUATION_MISMATCH`;
- changed represented evaluation/metric/procedure config is represented by `CONFIG.EVALUATION_MISMATCH`;
- missing/invalid native config or dataset inputs remain current insufficient-evidence findings.

`MeasurementDriftReport` references those rule IDs and deterministically reduces component states, but cannot add or override a verdict. Known semantic mismatch is `INVALIDATES` only when the corresponding native invalidating finding exists. Missing native semantics are `INSUFFICIENT` only when native Audit already reports insufficient evidence. Equal recovered semantics are `VERIFIED`; optional absent dimensions are `NOT_ASSESSED`; byte-only source change is `WARNING` with no impact.

Because no new native rule prefix ships, Evidence Trace's classifier remains unchanged. The stable Audit projection commits the bounded measurement report, while existing passive Trace entries retain exact path/hash/selector provenance. Trace construction failure remains non-authoritative and cannot alter the result or verdict. Any later `MEASUREMENT.*` rule must add its Trace classification in the same change.

## Upstream procedure obligations

ClaimCI's existing reduction over captured raw runs is explicitly named:

```text
claimci_verification_reduction = arithmetic_mean_v1
```

This is not evidence of the upstream evaluator's aggregation. Legacy `MetricImprovementClaim` retains its current arithmetic-mean-over-supplied-runs semantics when no explicit upstream procedure is stated.

A fixed source-bound parser recognizes only bounded, explicit aggregation/procedure phrases. Unknown explicit reduction phrases are classified as unsupported rather than silently receiving legacy arithmetic-mean semantics. It attaches a `measurement.retry_aggregation` policy template without changing the primary claim or `AuditClaimSpec`. Evidence Obligations adds:

- `MeasurementComponentTarget` for the exact required procedure;
- factory-only `MeasurementProcedureSupport`, derived only from selected, trusted config evidence and exact bindings;
- fixed missing, ambiguous, and unsupported reason codes.

V1 natively supports only explicitly represented `arithmetic_mean_v1`. Both baseline and candidate must independently recover the supported procedure from fixed config keys. A missing requirement yields pre-Audit `PARTIAL`; a represented unsupported median, weighted mean, best-of-N, retry filtering, or adjudication procedure yields pre-Audit `PARTIAL / UNSUPPORTED`; a mapping ambiguity yields `MAPPING_NEEDED` only when one existing bounded question has a choice that can produce complete native inputs and exact procedure support. Missing or unsupported still dominates ambiguity. No branch silently substitutes ClaimCI's arithmetic mean.

## Compatibility

Direct `audit_research`, CLI, manifest behavior, existing plan identity, `AuditClaimSpec`, and legacy `AuditResult` JSON remain byte-compatible when no measurement context is supplied. The `AuditResult` measurement companion is additive and omitted from reports when absent. Hosted no-manifest planning supplies the context from its existing exact-head passive path; no `research.yaml` prerequisite or fallback is introduced. No new LLM call, adapter, evaluator execution, Replay, Staleness, or Regression behavior is added.

## Acceptance

Tests cover frozen/bounded contracts, canonical ordering, independent semantic/source ID stability, direction/threshold exclusion, same-protocol verification, current config/dataset invalidation without duplicate findings, source-only warning, default optional dimensions, explicit procedure lifecycle reductions, provider authority attacks, exact-head runtime drift, trace completeness and bounds, legacy byte compatibility, CLI compatibility, and the unchanged passive execution boundary.
