# ClaimCI Evidence Trace and Provenance/Authority v1 Design

## Goal and boundary

ClaimCI must make every value consumed by the deterministic Audit traceable to
the exact repository snapshot and validated passive evidence that supplied it.
The trace is a companion to the existing deterministic result. It does not
change `AuditResult`, `Verdict`, `render_json()`, CLI exit behavior, or any
deterministic rule.

The first slice is backend-only:

```text
Core exact-head materialization
  -> immutable EvidenceTraceBundle
  -> Hosted runner result
  -> Worker validation
  -> existing deterministic_json persistence
```

GitHub Check and Dashboard projection remain unchanged. Their future rendering
must consume a deliberately redacted public projection rather than the internal
trace object.

## Canonical ownership

The canonical trace belongs in `claimci.analysis`, beside the planner and
materializer. This is the only layer that simultaneously has the exact head
SHA, repository-relative artifact identities, selected mapping, fresh passive
captures, adapter selectors, normalized evidence, and the actual
`AuditResult`.

`AuditResult` must not gain trace fields. The native Audit operates on a
generated ephemeral tree and intentionally knows nothing about customer
repository identity. Reconstructing trace in Hosted would duplicate Core trust
logic and could launder provider data after source identity has been flattened.

## Authority model

Trace record kind and authority class are separate typed concepts.

Record kinds distinguish:

- natural-language claim input;
- LLM semantic proposal;
- passive source evidence;
- deterministic derivation;
- mapping approval;
- deterministic Audit result;
- advisory Research Review.

Authority classes distinguish:

- `NON_AUTHORITATIVE_INPUT` for claim text and provider proposals;
- `PASSIVE_SOURCE` for freshly revalidated repository evidence;
- `DETERMINISTIC_DERIVATION` for computations derived by the native Audit;
- `EXPLICIT_MAPPING_APPROVAL` for `RepoMapping.approve()` only;
- `DETERMINISTIC_VERDICT` for an actual `AuditResult` only;
- `ADVISORY_INTERPRETATION` for Research Review only.

Mapping approval is permission to use one exact binding. It is never
scientific evidence or verdict authority. Provider provenance is never valid
for `PASSIVE_SOURCE`, `DETERMINISTIC_DERIVATION`, or `DETERMINISTIC_VERDICT`.

## Core contracts

Add immutable, deeply validated, JSON-safe contracts under
`claimci.analysis.trace`:

```python
TraceRecordKind
TraceAuthorityClass
TraceCompleteness
BoundedValueRepresentation
EvidenceTraceEntry
EvidenceTraceBundle
EphemeralAuditExecution
```

`EphemeralAuditExecution` contains the unchanged `AuditResult` plus its trace.
It is created only by the trusted materialization path. The deterministic
authority record is created from the exact `AuditResult`, following the same
factory-only principle as `DeterministicAuditOutcome`.

`PlanningRequest` and `EphemeralAuditPlan` carry one optional
`semantic_proposal_provenance` field. It accepts only `PROVIDER_PROPOSAL`
provenance, changes no selection or plan identity, and exists solely so an
independently confirmed Hosted semantic preflight is not erased when Core
deduplicates an equivalent deterministic claim or mapping.

The bundle records:

- repository commit SHA;
- claim ID, claim-text digest, source path/source ID when available, and claim
  provenance without treating the claim as evidence;
- selected mapping identity, mapping provenance, and whether the selected
  mapping is an explicit `RepoMapping`;
- artifact repository path, SHA-256, byte size, kind, role, and dataset split;
- fixed adapter identity and separately parsed version when the registered ID
  carries one;
- validated target fields and selector kind/expression;
- bounded source and normalized-value representations;
- actual `Finding.rule_id` values that consumed each evidence category;
- deterministic derivations and their source trace IDs;
- the exact deterministic verdict authority record;
- an advisory record only when Research Review exists.

Current Audit rule IDs are classified explicitly by the Core trace builder.
An emitted rule ID that has no declared consumption classification makes the
trace incomplete; it does not alter or reinterpret the Audit verdict.

## Exact source and value representation

Trace construction begins only after `_evidence_by_binding()` and
`_capture_plan_artifacts()` have revalidated mapping trust, repository-relative
paths, sizes, SHA-256 digests, adapter identities, selectors, roles, and exact
captured bytes. Dataset identity continues to use the fixed passive adapter.
No repository code is executed.

The trace does not create a general customer-source persistence surface.

- Finite numeric values, booleans, and null may be represented directly.
- Metric names, config keys, target fields, and validated selectors are bounded
  identity metadata.
- Repository strings, run IDs, dataset rows, collections, and natural-language
  prose are not persisted as arbitrary raw content. They use type, count or
  length, canonical SHA-256 commitment, and only a narrowly justified preview.
- V1 uses no prose preview and no dataset-row preview. String scalar values are
  committed by digest rather than copied.
- Large result series and collections are represented by count, canonical
  sequence digest, and a small numeric preview. Artifact path/hash plus selector
  remains the exact retrieval identity.

All canonical digests use duplicate-free, finite, sorted-key JSON with UTF-8
encoding and explicit type tags so distinct source types cannot collide.

## Completeness and failure behavior

`TraceCompleteness` is `complete`, `bounded`, or `unavailable`.

- `complete` means every value consumed by current Audit rules is represented
  directly or by a canonical commitment and every emitted rule ID is classified.
- `bounded` preserves bundle identity, counts, commitments, and omissions when
  detailed groups exceed a transport limit.
- `unavailable` carries a fixed reason code and no provider-controlled message.

Trace construction and projection never create or replace a verdict. The
existing Audit runs exactly once and its `AuditResult` remains authoritative.
A trace problem produces an explicitly incomplete trace, or a Hosted `partial`
result that still carries the same deterministic outcome if even the minimal
trace representation cannot fit. Pilot fixtures must produce `complete`.

## Bounds and transport

The existing Hosted result codec is limited to 65,536 UTF-8 bytes, and D1
limits `deterministic_json` to 65,536 JSON characters. V1 reserves at most
16,384 canonical UTF-8 bytes for the trace: one quarter of the existing result
envelope, leaving 49,152 bytes for the pre-existing result identity,
deterministic findings, advisory lane, missing evidence, and usage.

Additional structural bounds are:

- 32 trace entries;
- 32 selector/value groups per artifact entry;
- 32 consumer rule IDs per entry;
- 8 direct numeric preview values per group;
- 512 Unicode code points per bounded identity/representation field.

Core independently enforces the 16,384-byte trace ceiling. The Python Hosted
codec independently revalidates the same structure and ceiling. The Worker
codec repeats both checks before persistence. The runner also checks the full
65,536-byte result envelope. If the complete Core trace does not fit the
remaining result budget, it emits a bounded projection with a root commitment;
if the minimal projection cannot fit, the deterministic lane is preserved in a
typed partial result and no alternative verdict is manufactured.

The trace is an optional additive field in the existing internal v1
deterministic object for rolling compatibility. Deployment ordering is Worker
acceptance first, runner emission second. This branch changes both together but
tests old trace-absent persisted results.

## Compatibility seams

Add:

```python
execute_ephemeral_audit_with_trace(plan, runtime) -> EphemeralAuditExecution
```

Keep:

```python
execute_ephemeral_audit(plan, runtime) -> AuditResult
```

The compatibility function delegates to the traced execution and returns the
same `AuditResult`. Existing callers, CLI behavior, `render_json()`, and
deterministic snapshots remain unchanged. `UnifiedAnalysisResult` gains an
optional trace field at the end of its constructor; ordinary legacy
construction remains valid, while the hosted `run_unified_analysis()` path
always attempts trace creation for a deterministic result.

Hosted stores the validated trace inside existing `deterministic_json`; no D1
migration is required. Public `Analysis`, GitHub Check rendering, and Dashboard
types omit it in this slice.

## Acceptance criteria

- Equivalent fixtures produce byte-identical `render_json(AuditResult)` output
  and identical verdicts with traced and compatibility execution.
- Complete Pilot fixtures trace claim policy, result metrics/seeds, config
  scalars, dataset identities, derived metric/overlap values, and every emitted
  rule ID.
- Path, hash, adapter, selector, role, split, or head drift fails before passive
  evidence can enter the trace.
- Provider-proposed values and selectors cannot acquire passive or
  deterministic authority merely by being returned by a provider.
- `RepoMapping.approve()` is represented only as an explicit binding trust
  transition.
- Hostile over-bound, malformed, duplicate-key, non-finite, corrupt, and
  authority-confused traces fail closed in both Python and TypeScript codecs.
- Old Hosted rows without trace still load and project exactly as before.
- No Check or Dashboard output changes in this slice.
