# Verification Input Snapshot and Replay Recipe v1 Design

## Goal and authority

Verification Input Snapshot and Replay Recipe v1 add two immutable companions
around the existing deterministic Audit:

```text
exact-head materialization
  -> VerificationInputSnapshot
  -> native deterministic Audit
  -> ReplayAuditCommitment.from_audit_result()
  -> Evidence Trace reference
  -> ReplayRecipe
```

The snapshot commits only trusted, freshly captured deterministic verification
inputs. The recipe adds exact engine provenance and the actual Audit commitment.
Neither object creates a verdict, executes repository code, downloads evidence,
or authorizes experiment replay. `RepoMapping.approve()` remains binding
selection authority only. `AuditResult` remains the sole verdict authority.

Provider or LLM payloads have no deserializer into snapshot, compatibility,
commitment, trace-reference, or recipe contracts. A provider proposal may reach
the existing planner and mapping lifecycle, but only independently revalidated
exact-head materialization can construct a snapshot.

## Canonical contracts

`claimci.analysis.verification` owns pre-Audit contracts:

- `VerificationSnapshotCapability`: `COMPLETE`, `IDENTITY_ONLY`, `UNAVAILABLE`.
- `VerificationSelectorIdentity`: the complete validated selector identity,
  including exact table predicates and cardinality where applicable.
- factory-only `VerificationArtifactIdentity`.
- factory-only `VerificationClaimIdentity`.
- factory-only `VerificationMappingIdentity`.
- factory-only `EvidenceProfileIdentity`.
- factory-only `EvidenceObligationIdentity`.
- factory-only `MeasurementProtocolReference`.
- factory-only `AuditSemanticsCompatibility`.
- factory-only `VerificationInputSnapshot`.

`claimci.analysis.replay` owns post-Audit contracts:

- factory-created `ReplayEngineProvenance`.
- factory-only `ReplayAuditCommitment` from an exact `AuditResult`.
- factory-only `ReplayTraceReference` from an exact `EvidenceTraceBundle` and
  matching Audit commitment.
- factory-only `ReplayRecipe`.

Authority-bearing classes are frozen, final, and constructor-sealed. Factories
require exact base types rather than subclasses. Canonical JSON is finite,
sorted-key, duplicate-free UTF-8 with fixed domain/version tags.

## Verification artifact identity

Each selected binding becomes one `VerificationArtifactIdentity` with:

- repository-relative path;
- artifact kind;
- baseline, candidate, or reference role;
- train/eval split where applicable;
- exact source SHA-256;
- deterministic extraction SHA-256;
- Audit-semantic SHA-256;
- fixed adapter ID;
- separately recovered adapter semantic version;
- fixed evidence-projector version;
- complete canonical selectors.

The three commitments are distinct:

1. `source_sha256` commits exact captured bytes.
2. `extraction_sha256` commits the selected adapter projection and normalized
   evidence after fresh fixed-adapter validation. It excludes formatting-only
   source bytes.
3. `audit_semantics_sha256` commits only the values and metadata consumed by
   the current deterministic Audit for that exact binding.

The full verification-evidence identity commits all fields above, including
role, split, and selector. One physical Benchmark table selected through
different baseline/candidate predicates therefore produces distinct evidence
identities even though `path` and `source_sha256` are equal.

Training result semantics contain only the audited metric observations and
integer seeds consumed by the native result check. Run IDs remain extraction
and full provenance only because the current deterministic Audit does not
consume them.
Training config semantics contain the exact generated scalar mapping consumed
by config checks. Training dataset semantics contain canonical JSONL record
hashes with counts, so formatting-only JSON changes do not alter Audit
semantics. Benchmark result semantics contain claim-metric values, integer
seeds only for raw-run-series policy, represented non-claim metric names
without their ignored values, and canonical deduplicated profile components.
They therefore commit the exact selector-scoped fields consumed by the
selected `audit_benchmark()` result form without treating ignored observation
metadata as Audit semantics.

## Claim, mapping, profile, and obligation identities

`VerificationClaimIdentity` contains the selected claim ID, a canonical
scientific-claim commitment, and the exact `AuditClaimSpec` commitment. Claim
direction and threshold belong here because the native Audit consumes them;
they remain excluded from Measurement Protocol identity.

`VerificationMappingIdentity` commits the selected mapping trust class, source
mapping ID, exact canonical bindings, and approval-provenance commitment when
the selection is a `RepoMapping`. Approval grants only binding authority. It
never contributes a verdict or scientific result.

`EvidenceProfileIdentity` commits profile ID/version, Benchmark variant, result
form, activation policy, selection ID, and a separate profile-semantics digest.
The semantics digest excludes source-specific selection facts.

`EvidenceObligationIdentity` commits the exact bounded obligation bundle and a
separate semantic projection of policy, targets, states, effects, and
dependencies. Source support IDs remain in the full snapshot but do not affect
Audit-semantic compatibility.

Each role's `MeasurementProtocolReference` contains only the already factory-
created `semantic_protocol_id` and `source_snapshot_id`. The semantic ID feeds
Audit comparison commitments; source-only measurement provenance feeds only
the complete snapshot.

## Audit semantics compatibility

`AuditSemanticsCompatibility` is separate from exact Core build provenance. It
commits versioned behavior that can alter deterministic interpretation:

- Audit policy semantics;
- deterministic rule-set semantics;
- claim compiler semantics;
- evidence-projector versions;
- adapter semantic versions;
- selected evidence-profile semantics;
- Benchmark compiler/policy semantics when applicable.

Its `audit_semantics_sha256` excludes package version, Git revision,
distribution digest, repository/head identity, and source-only artifact data.
An exact Core build change therefore does not automatically make outcomes
incomparable when these semantics remain the same.

## Snapshot identity and capabilities

`VerificationInputSnapshot` contains:

- repository, PR, exact head, and optional trusted base SHA;
- claim and `AuditClaimSpec` identity;
- selected mapping identity and exact bindings;
- verification artifact identities;
- profile identity;
- Evidence Obligation identity;
- baseline and candidate Measurement Protocol references;
- Audit Semantics Compatibility;
- `captured_audit_input_sha256`;
- `comparison_frame_sha256`;
- `input_snapshot_sha256`;
- capability and bounded omission commitments.

`COMPLETE` retains all bounded component records. If the canonical complete
projection exceeds the Core safety ceiling, the factory computes the same full
`input_snapshot_sha256` and returns `IDENTITY_ONLY`, retaining counts and root
commitments while omitting verbose component records. This is a Core resource
bound, not a reserved Hosted transport allocation. Hosted must choose its final
projection from one global result-envelope budget. `UNAVAILABLE` carries only
fixed reason identity and cannot proceed to Audit.

The snapshot is created after all runtime path/hash/adapter/selector/role/split
and profile checks, after ephemeral native inputs are prepared, and before the
Audit call. The materializer executes Audit only with a `COMPLETE` or
`IDENTITY_ONLY` snapshot.

## Digest invariants

All digests use a fixed domain tag, explicit version, canonical sorted-key JSON,
UTF-8, `ensure_ascii=True`, compact separators, and non-finite rejection.

### `captured_audit_input_sha256`

Includes:

- canonical `AuditClaimSpec`;
- profile semantics;
- obligation semantics;
- mapping role/split/selector semantics;
- role-indexed artifact `audit_semantics_sha256` values;
- baseline/candidate measurement `semantic_protocol_id` values;
- `AuditSemanticsCompatibility.audit_semantics_sha256`.

Excludes:

- repository/head/base identity;
- file paths and formatting-only source bytes;
- exact Core revision/distribution;
- measurement `source_snapshot_id`;
- provider prose, confidence, and approval actor.

### `comparison_frame_sha256`

Includes:

- canonical `AuditClaimSpec`;
- baseline and shared-reference Audit-semantic evidence;
- profile semantics;
- baseline/candidate measurement semantic-protocol IDs;
- Audit Semantics Compatibility.

Excludes:

- candidate result values;
- exact Core build provenance;
- source-only measurement provenance;
- repository/head/path identity.

Candidate protocol/config semantics remain included because they define the
ruler, while candidate measured result values are excluded.

### `input_snapshot_sha256`

Commits the complete exact pre-Audit snapshot projection: repository/PR/head/
base, claim, mapping trust and bindings, all verification artifacts, profile,
obligations, measurement references, compatibility, and both narrower
commitments. It is calculated before complete details are reduced to an
identity-only representation.

## Replay companions

`ReplayEngineProvenance` records:

- ClaimCI package version;
- exact source revision when supplied by the trusted host;
- distribution SHA-256 when supplied by the trusted host.

These values describe the build and do not feed Audit-semantic compatibility.

`ReplayAuditCommitment.from_audit_result()` accepts only an exact
`AuditResult`. It records the real verdict, the stable Audit projection digest,
and sorted applied rule IDs. It reuses the established stable Audit projection
that removes invocation-private scratch paths.

`ReplayTraceReference` commits the exact deterministic Evidence Trace canonical
bytes produced by materialization before any later advisory Review attachment,
plus version, completeness, reason/omission identity, head, and deterministic
Audit digest. Its factory rejects a trace whose deterministic authority does
not match the Replay Audit commitment. `EphemeralAuditExecution` also rejects
a recipe that references a different same-head trace. A later unified result
may expose the trace augmented with advisory metadata or a trace-only fallback;
that does not rewrite the already committed deterministic Replay reference.

`ReplayRecipe` contains the input snapshot, engine provenance, actual Audit
commitment, and Trace reference. `recipe_sha256` commits those identities and
the fixed recipe version. `experiment_replay_supported` is permanently false.
There is no API that materializes or executes a repository from a recipe.

## Integration and compatibility

`execute_ephemeral_audit_with_trace()` becomes the canonical constructor of:

```text
VerificationInputSnapshot
AuditResult
EvidenceTraceBundle
ReplayRecipe
```

`EphemeralAuditExecution` gains optional snapshot/recipe companions at the end
of its positional shape. Legacy direct construction with only Audit and Trace
remains valid. Canonical materialization always returns both companions.
`execute_ephemeral_audit()` still returns the unchanged `AuditResult`.

`UnifiedAnalysisResult` carries the companions additively only when an actual
deterministic result exists. They cannot appear in mapping-needed, partial, or
unavailable states and cannot alter `authoritative_verdict`.

Existing `AuditResult`, `render_json()`, CLI text/JSON/Markdown, research.yaml,
legacy Training behavior, Benchmark Audit semantics, Evidence Trace semantics,
and provider call limits remain unchanged when replay companions are absent.
No Hosted transport allocation, database schema, deployment, Stale Evidence,
Verdict Regression, or experiment replay is implemented.

Both `TRAINING_EXPERIMENT_V0` and `BENCHMARK_MEASUREMENT_V0` are supported.
Benchmark selector identities remain exact and selector-scoped. No customer
repository code is executed.

## Evidence Trace dependency

Replay references the existing stabilized Evidence Trace and its real Audit
commitment. This branch introduces no new `MEASUREMENT.*` Audit finding.
Future `MEASUREMENT.*` findings remain blocked until the trace classifier
explicitly classifies them; unclassified native rule IDs must continue to make
Trace incomplete rather than silently disappear.

## Acceptance

Tests must cover constructor sealing and deep immutability; exact digest golden
vectors; formatting-only source changes; selector-scoped Benchmark identities;
profile and obligation inclusion; semantic/source measurement separation;
Core build versus semantic compatibility separation; provider injection;
mapping approval remaining non-verdict authority; actual-Audit-only replay
commitment; over-bound identity-only reduction; trace/commitment matching;
Training and Benchmark integration; legacy Audit/CLI/report compatibility; and
the passive no-execution boundary.
