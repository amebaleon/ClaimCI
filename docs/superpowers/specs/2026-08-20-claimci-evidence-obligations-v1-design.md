# Evidence Obligations and Specific Missing Evidence v1 Design

## Goal

Add one deterministic pre-Audit obligations engine that answers which source-bound claim fields and exact passive evidence slots are required, whether each is supported strongly enough, and whether planning may proceed. Obligation satisfaction means only that a required input is bound; it never means that the scientific claim is supported.

The lifecycle reduction is fixed:

```text
all required obligations satisfied                  -> READY
only resolvable ambiguity + one useful bounded ask  -> MAPPING_NEEDED
any missing or unsupported obligation               -> PARTIAL
missing/unsupported plus ambiguity                   -> PARTIAL, no question
exact-head/runtime/hash integrity failure            -> UNAVAILABLE
Audit has started                                    -> native Audit authority
```

`AuditResult` remains the sole verdict authority. Repository and provider content remain passive data.

## Contracts and bounds

Create `claimci.analysis.obligations` with frozen, validated contracts:

- `EvidenceObligationState`: `SATISFIED`, `MISSING`, `AMBIGUOUS`, `UNSUPPORTED`.
- `EvidenceObligationReason`: the fixed reason codes in the approved brief.
- `EvidenceObligationEffect`: `NONE`, `BLOCKS_PARTIAL`, `REQUIRES_MAPPING`.
- `EvidenceObligationDecision`: `READY`, `PARTIAL`, `MAPPING_NEEDED`.
- `ClaimFieldKind`: `METRIC`, `DIRECTION`, `THRESHOLD`, `QUANTITATIVE_BOUND`, `EVALUATION_CONSTRAINT`.
- `ClaimFieldTarget` and `ArtifactEvidenceSlot` form `ObligationTarget`.
- factory-only `ClaimFieldSupport` and `ArtifactEvidenceSupport` form `ObligationSupportReference`.
- `EvidenceObligation` represents either one target leaf or one dependency-derived composite.
- factory-only `EvidenceObligationBundle` carries the selected claim/policy, compiler assessment, stable obligations, blocker IDs, lifecycle decision, and optional linked mapping-question ID.

The graph permits at most 16 obligations, 16 dependencies per composite, and 32 support references in total. IDs are canonical, unique, and deterministically ordered. Every dependency must resolve and the graph must be acyclic. Composite state, reason, and effect are recomputed from children; callers cannot assert them independently.

Serialization is a deterministic minimal JSON projection of IDs, states, reasons, effects, targets, support references, dependencies, blockers, and linkage. It contains no duplicate artifact path, hash, selector, adapter, value, or provenance detail already owned by Evidence Trace. Core imposes structural bounds but no independent Hosted byte allocation.

## Trust model

`claim_field_support()` is the sole supported `ClaimFieldSupport` factory. It accepts only an exact `CanonicalScientificClaim`, re-runs source recovery, and creates support only for a field actually present in the recovered semantics. Thresholds are claim-field support, never fake `MANIFEST` evidence. A `HELD_OUT` constraint is retained as `EVALUATION_CONSTRAINT` support but gains no Audit or verdict authority.

`validated_artifact_support()` is the sole supported `ArtifactEvidenceSupport` factory. It accepts only exact issued `NormalizedEvidence`, an exact binding in the selected mapping, and a currently revalidated deterministic/manifest mapping or an explicit `RepoMapping`. It commits to the artifact hash, role/split, adapter, selectors, evidence ID, and mapping identity in a bounded digest, while serializing only support/evidence/binding IDs. Unapproved provider provenance cannot create support. `RepoMapping.approve()` remains the only elevation of provider-proposed bindings and remains selection authority, not verdict authority.

Provider payloads are never deserialized into obligation contracts. Obligation templates, requiredness, dependencies, reasons, effects, supports, and decisions are generated only from the canonical policy and validated Core inputs.

## Policy instantiation

`ClaimEvidencePolicy` gains stable obligation template IDs owned by the Claim Type System. Metric improvement instantiates:

- claim metric;
- claim direction;
- absolute improvement threshold;
- optional recovered evaluation constraint;
- baseline and candidate result metric slots;
- baseline and candidate config slots;
- baseline/candidate train/eval dataset slots;
- one comparison-readiness composite depending on every required executable leaf.

The composite indicates only that the native comparison inputs are ready. Equality, overlap, and scientific comparability remain native Audit decisions.

Recognized unsupported `AbsoluteMetricClaim`, `GeneralizationClaim`, and `GenericQuantitativeClaim` get a structured policy-level `UNSUPPORTED / claim_type_not_executable / BLOCKS_PARTIAL` assessment and only their real source-bound field supports. They do not instantiate executable artifact slots and retain the exact public partial reason `unsupported_deterministic_claim_compiler`.

## Planning integration

Planner continues to own mapping trust selection and exact runtime-validity checks. It supplies the obligations engine with the selected mapping, viable candidates, and at most one useful bounded question. The engine owns the final READY/PARTIAL/MAPPING_NEEDED reduction.

A mapping question is useful only when at least one issued choice can complete the native input set after exact evidence/adapter/selector/role/split validation. Returned questions gain `blocking_obligation_ids`; questions are suppressed whenever any required obligation is missing or unsupported. Invalid/unissued selectors cannot be made usable merely by approval.

`PlanningOutcome`, `EphemeralAuditPlan`, and `UnifiedAnalysisResult` carry the canonical obligation bundle additively. Ready plans require a `READY` bundle. The bundle is excluded from plan identity so existing metric-improvement plan IDs and Audit results remain unchanged.

Legacy `MissingEvidence` is derived only from blocking artifact leaf obligations where a truthful artifact projection exists. Claim-field absence is represented canonically in the obligation bundle and reason; no prose parsing and no fake artifact are permitted.

Executable plans can only be constructed with a `READY` bundle. Materialization then performs all existing exact-head/path/hash/adapter/selector/mapping checks. A typed, pre-Audit native-representation limitation derives an exact `UNSUPPORTED` artifact obligation and returns `PARTIAL`; runtime identity/integrity failures stay `UNAVAILABLE`. Native Audit findings and `INSUFFICIENT_EVIDENCE` remain unchanged.

## Evidence Trace and compatibility

Evidence Obligations reference claim IDs, evidence IDs, and binding support IDs. Evidence Trace continues to own detailed path/hash/selector/adapter/provenance/value records. No new Audit rules or trace classifier categories are introduced.

Existing `AuditResult`, `AuditClaimSpec`, CLI, manifest behavior, Hosted no-manifest path, Review authority separation, and passive execution boundary remain unchanged. If a future Hosted projection cannot fit, it may bound or omit the obligations companion but must preserve an already-created deterministic outcome.

## Acceptance

Tests cover claim-field factories, exact artifact slots, HELD_OUT retention, graph validation, deterministic serialization, every lifecycle reduction, unsupported claim types, provider authority attacks, explicit approval, stale mapping, legacy projection, byte-identical executable plans/results, and Audit non-reachability before READY.
