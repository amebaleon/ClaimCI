# Benchmark Evidence Profile v0 Design

## Goal and authority

Benchmark Evidence Profile v0 lets the existing `MetricImprovementClaim` use one of two deterministic evidence-obligation families:

- `TRAINING_EXPERIMENT_V0`
- `BENCHMARK_MEASUREMENT_V0`

It does not create a new claim taxonomy or verdict system. The authority chain remains:

```text
CanonicalScientificClaim
  -> ClaimEvidencePolicy
  -> deterministic profile selection
  -> ProfiledEvidencePolicy
  -> Evidence Obligations
  -> exact-head passive materialization
  -> Measurement Drift
  -> native Audit checks
  -> determine_verdict()
  -> AuditResult
```

Provider output may propose claims, selectors, or mappings. It cannot construct a profile definition, selection fact, selected profile, profile support, scope, requiredness, finding, or verdict. `RepoMapping.approve()` authorizes only exact bindings. Customer content remains passive data and is never executed.

## Contracts

`claimci.analysis.profiles` owns frozen, final contracts and a fixed registry:

- `EvidenceProfileId`: `TRAINING_EXPERIMENT_V0`, `BENCHMARK_MEASUREMENT_V0`.
- `BenchmarkVariant`: `LATENCY`, `THROUGHPUT_RESOURCE`, `KERNEL`, `MODEL_QUALITY`, `COST`.
- `BenchmarkResultForm`: `RAW_RUN_SERIES`, `REPORTED_AGGREGATE`, `DETERMINISTIC_DERIVATION`.
- `ProfileRoleMode`: `PAIRED_BASELINE_CANDIDATE`, `SHARED_REFERENCE`, `SHARED_REFERENCE_OR_PAIRED`.
- `ProfileSupportKind`: `ARTIFACT_BINDING`, `NORMALIZED_METRIC`, `STRUCTURED_COMPONENT`, `DETERMINISTIC_DERIVATION`.
- factory-only `EvidenceProfileDefinition`, `EvidenceProfileSelection`, `ProfileSelectionFact`, `ProfileSelectionOutcome`, and `ProfiledEvidencePolicy`.

The registry is an immutable mapping. Public constructors for authority-bearing records raise. Deterministic factories validate exact canonical claims, issued evidence, exact adapter selectors, and validated or approved bindings. There is no generic provider deserializer.

`ClaimEvidencePolicy` remains unchanged for compatibility. `ProfiledEvidencePolicy` wraps its exact claim policy and separates claim-owned templates from the selected profile's templates. Thus the legacy object can retain its historical template projection while new execution gives profile ownership to artifact and measurement obligations.

## Additive obligation target

The existing `ArtifactEvidenceSlot` remains byte- and behavior-compatible: only `RESULTS`, `CONFIG`, or `DATASET`; only baseline or candidate; and train/eval required only for datasets.

`ObligationTarget` gains `ProfileEvidenceTarget` without removing the existing measurement-procedure target. A profile target contains:

- fixed `profile_id`;
- fixed grouped `slot_id`;
- fixed `role_mode`;
- immutable allowed artifact kinds;
- immutable allowed support kinds;
- fixed activation-policy ID.

The eight Benchmark slots are exactly:

1. `benchmark.subject`
2. `benchmark.results`
3. `benchmark.workload`
4. `benchmark.measurement_config`
5. `measurement.metric_identity`
6. `benchmark.run_protocol`
7. `benchmark.environment`
8. `benchmark.evaluator`

Factory-only `ProfileEvidenceSupport` is created from one selected mapping and freshly normalized issued evidence. It commits support kind, slot, role, source binding identities, and a bounded canonical structured projection. Every projected scalar must be named by an exact selected field mapping; merely placing a value in a normalized object is not support. Provider provenance cannot create support. Detailed values remain in Evidence Trace rather than obligation prose.

## Metric separation

`claim.metric` is the canonical semantic assertion owned by Claim Type. `measurement.metric_identity` is the represented metric definition/configuration recovered from result, evaluator, and config evidence. They are separate obligations. A satisfied claim field never satisfies measurement identity. Before Audit, each role must expose one independently recovered measured-metric identity. A known disagreement with the claim or between roles reaches the native `BENCHMARK.METRIC_MISMATCH` check and invalidates there; it is not misreported as missing evidence. Claim direction and improvement threshold are excluded from measurement-protocol identity.

## Selection policy

Selection uses only canonical claim semantics, fixed policy, independently recovered facts, and exact validated/approved mappings. Provider profile names and prose are ignored.

Fixed metric routing is:

- latency, runtime, and cold-start -> Benchmark `LATENCY`;
- throughput, tokens/s, samples/s, peak memory, VRAM, and resource -> Benchmark `THROUGHPUT_RESOURCE`;
- kernel time and bandwidth -> Benchmark `KERNEL`;
- cost, cost/request, and cost/token -> Benchmark `COST`;
- accuracy, F1, BLEU, perplexity, and loss -> evidence-shape-dependent `MODEL_QUALITY` Benchmark or legacy Training.

The exact historical results/config/train/eval mapping is positive Training evidence. Benchmark selection requires positive, independently recovered benchmark component facts; a `BENCHMARK` artifact kind alone and absence of train/eval data are never positive facts. Performance metrics route to Benchmark even when facts are incomplete, yielding Benchmark-specific `PARTIAL` rather than a Training JSONL complaint.

For quality metrics, one complete shape selects its profile. A legacy baseline/candidate results-plus-config footprint remains an incomplete Training selection so its historical dataset obligations and mapping lifecycle stay compatible. If neither that positive Training footprint nor positive Benchmark facts exist, selection remains `UNRESOLVED` and planning returns `PARTIAL` with `required_profile_evidence_not_recovered`; it does not default to Training or invent train/eval blockers. If both Training and Benchmark shapes are independently complete and canonical semantics/fixed facts do not resolve them, planning returns `PARTIAL` with exact reason `evidence_profile_ambiguous`. It never asks an authoritative Training-or-Benchmark question. `MAPPING_NEEDED` is permitted only when a bounded issued binding/selector choice indirectly yields one executable profile; after approval selection reruns from the fresh mapping.

## Benchmark facts and Table Adapter

Benchmark Table Adapter v0 remains CSV/TSV only. JSON and JSONL continue through existing registered adapters. Exact row-key `TableSelector` predicates identify selected cells; row positions and first matches are forbidden. Fixed profile-field targets may project only selected scalar cells into bounded `ConfigValue` components. Types are not guessed: selected table scalars remain exact text until a fixed profile parser validates a required canonical integer/decimal/enum.

One physical table, whether discovered as `BENCHMARK` or an existing `RESULTS` artifact, can supply baseline and candidate result rows plus workload/config cells through distinct selector-scoped evidence identities. Distinct syntax alone is insufficient: cross-role selectors must address different target columns or prove row disjointness through mutually exclusive typed equality predicates on a shared predicate column. Subset/superset predicates and typed aliases that can match the same target cell are conflicting role bindings and fail closed before approval or materialization. Materialization captures the physical bytes once, reruns the fixed adapter for every selected variant, and never persists complete rows. Measurement source snapshots commit a SHA-256 projection of the complete canonical table selector, including predicates and expected cardinality; a column name alone is not selector identity.

Structured components use fixed bounded scalar keys under the eight slot namespaces. Each Benchmark variant has an activation policy defining required keys and role mode. Subject fields are `SYSTEM_UNDER_TEST` and excluded from protocol equality. Procedure fields are `MEASUREMENT_PROCEDURE`; unresolved scope is `UNKNOWN` and blocks where required. Provider output cannot assign scope.

## Result forms

`RAW_RUN_SERIES` requires exact passive scalar observations for both roles. ClaimCI applies `claimci_verification_reduction=arithmetic_mean_v1`; this does not assert the upstream evaluator used that aggregation. Historical Training claims explicitly defined as the mean of supplied runs remain unchanged.

`REPORTED_AGGREGATE` requires independently recovered statistic identity, aggregate value, sample count, and aggregation identity for both roles. Audit compares the represented aggregate directly and records that it was aggregate evidence. It never reinterprets it as raw runs. Missing sample count or aggregation blocks planning.

For reported aggregates and deterministic derivations, the public Measurement Drift companion records `claimci_verification_reduction=not_applicable`; only raw-run verification reports `arithmetic_mean_v1`.

`DETERMINISTIC_DERIVATION` permits only a fixed ClaimCI formula family. V0 supports bounded multiplication (`unit_value * quantity`) for cost/resource evidence. Exact passive inputs are recomputed; arbitrary expressions, scripts, provider formulas, and customer execution are forbidden.

## Variant activation matrix

All variants require subject, results, workload, measurement configuration, metric identity, and run protocol.

- `LATENCY`: timing boundary, warmup, sample count, aggregation, workload/request shape, hardware; represented region/network boundary is procedure evidence.
- `THROUGHPUT_RESOURCE`: model/workload, batch or concurrency/sequence shape, sample/step count, hardware/runtime, timing or memory collection.
- `KERNEL`: tensor shape, dtype/layout, generation seed/procedure, compiler/runtime, hardware, warmup, timing method.
- `MODEL_QUALITY`: evaluation corpus/workload, evaluator/scorer identity, metric configuration; evaluator-scoped prompt/template/decoding only when represented. It does not require a training dataset.
- `COST`: workload volume, provider/SKU/resource configuration, included cost components, pricing snapshot/version, measurement window, and reported aggregation or the fixed deterministic formula.

Environment is required for latency, throughput/resource, kernel, and cost. Evaluator is required for model quality and every cost form because included/excluded cost components are measurement-critical. Optional represented components are still traced and compared.

## Obligation bound

The bundle cap remains 16. The maximum executable activation is four claim fields, eight grouped profile slots, and one comparison composite: 13. No scalar field becomes a separate obligation. The ordinary v0 activations are 11 for latency, throughput/resource, kernel, and model quality; 12 for cost; and exactly 13 for a cost claim with the fourth supported claim-field constraint. Parameterized tests prove all of these counts. Legacy Training calls continue through the existing obligation function and preserve exact IDs, states, serialization, and plan identity.

## Planning, materialization, and Audit

Planning first recovers a profile outcome, then evaluates only profile-valid mappings. Missing or unsupported profile facts become profile-owned blockers. Ambiguous bindings can produce the existing bounded mapping question. Semantic profile ambiguity is a reason-only `PARTIAL`.

`EphemeralAuditPlan` carries an additive factory-created profiled policy plus optional reference evidence. Training plans use the historical evidence projection in plan-ID derivation, preserving exact IDs. Benchmark plan IDs additionally commit profile ID/version, variant, result form, activation policy, and reference bindings.

Materialization revalidates repository/head, path, kind, SHA-256, adapter, selector, role, mapping, and profile support against fresh passive bytes. Reference evidence is folded into both measurement source snapshots only where fixed policy marks it shared procedure evidence. Any drift is `UNAVAILABLE`.

One profile-aware dispatcher calls the unchanged Training Audit for Training. Benchmark Audit consumes only freshly materialized normalized evidence, applies the selected result-form comparison, invokes Measurement Drift with a profile/version/variant/activation policy ID, emits ordinary `Finding` values, calls the existing `determine_verdict()`, and returns ordinary `AuditResult`. There is no Benchmark verdict type or second verdict function.

V0 adds only the smallest truthful native rule surface:

- `BENCHMARK.METRIC_MISMATCH` (`INVALIDATES`)
- `BENCHMARK.WORKLOAD_MISMATCH` (`INVALIDATES`)
- `BENCHMARK.CONFIG_MISMATCH` (`INVALIDATES`)
- `BENCHMARK.PROCEDURE_INSUFFICIENT` (`INSUFFICIENT`)
- `BENCHMARK.DERIVATION_MISMATCH` (`INVALIDATES`)
- `BENCHMARK.PROTOCOL_VERIFIED` (`NONE`, informational)
- `RESULT.AGGREGATE_VERIFIED` (`NONE`, informational; exact reported aggregate source)

Existing numeric `RESULT.*` claim rules remain the numeric claim authority where truthful. Every new `BENCHMARK.*` and `RESULT.AGGREGATE_VERIFIED` finding is classified by Evidence Trace in this same change; no unclassified native rule may ship.

## Compatibility and exclusions

Direct `audit_research`, `research.yaml`, CLI output, legacy Training findings/verdicts, `AuditClaimSpec`, historical plan IDs, Hosted provider-call count, and customer-code isolation remain unchanged. No new Hosted prerequisite is introduced.

V0 does not execute evaluator code; infer semantic equivalence from arbitrary source diffs; download datasets or benchmark suites; infer missing hardware, pricing, workload, aggregation, sample counts, or metric definitions; support arbitrary formulas; validate vendor invoices; establish external benchmark authenticity; or implement Replay, Staleness, or Verdict Regression. Missing passive representation fails closed.

## Acceptance

Tests cover final/factory-only contracts, immutable registry, provider authority attacks, profile routing, ambiguity lifecycle, exact legacy compatibility, all eight grouped slots and obligation counts, claim/measured-metric separation, CSV/TSV selector-scoped component projection, shared physical capture, all result forms, fixed derivation, variant fixtures modeled after VESSL/Unsloth/Cerebrium/kernel/model-quality cases, Measurement Drift scopes, new rule classification, no raw-row/provider-prose persistence, runtime drift, and no customer execution.
