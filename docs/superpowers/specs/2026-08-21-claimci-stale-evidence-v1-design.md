# Stale Evidence Detection v1 Design

## Goal and authority

Stale Evidence Detection v1 compares one previous `ReplayRecipe.input_snapshot`
with one current `VerificationInputSnapshot` and reports whether ClaimCI's
represented, versioned deterministic verification inputs changed materially.
It is deterministic metadata only. It never creates a `Finding`, `Impact`,
`Verdict`, `AuditResult`, mapping approval, or replay authority.

`NO_REVERIFY_TRIGGER_DETECTED` means only that no material change requiring
reverification was detected inside the represented captured scope. It does not
prove global experiment stability or reproducibility and never authorizes an
Audit skip. A previously scheduled Audit always continues.

## Public contracts

`claimci.analysis.staleness` owns:

- `StalenessState`: `FRESH`, `BYTE_ONLY_CHANGE`, `IDENTITY_RELOCATION`,
  `MATERIAL_CHANGE`, `INDETERMINATE`, `LEGACY_UNAVAILABLE`.
- `ReverificationDecision`: `NO_REVERIFY_TRIGGER_DETECTED`,
  `REVERIFY_REQUIRED`, `UNDETERMINED`.
- `RequiredAction`: `NONE`, `REVERIFY`, `REMAP`, `SUPPLY_PROVENANCE`.
- fixed `StalenessReason` codes.
- immutable `ArtifactRelocation` metadata.
- immutable, constructor-sealed `StaleEvidenceAssessment`.
- `assess_stale_evidence(previous_recipe, current_snapshot, *,
  issued_candidates=())`.

The assessment records previous/current snapshot commitments, fixed reason
codes, bounded relocation metadata, a deterministic comparison digest, and the
fixed version. It exposes `scheduled_audit_continues=True` and
`authorizes_audit_skip=False` as class-level policy, not caller-controlled
fields.

Only an exact `ReplayRecipe`, exact current `VerificationInputSnapshot`, and
exact factory-created `VerificationArtifactIdentity` relocation candidates are
accepted. Provider dictionaries, subclasses, paths, values, selectors, or
claimed equivalence have no construction path into the comparison.

## Comparison order

The comparison is conservative and evaluates independent identity layers:

1. snapshot capability and required semantic coverage;
2. repository identity;
3. canonical claim statement and `AuditClaimSpec` commitments;
4. Evidence Profile semantics;
5. Evidence Obligation semantics;
6. baseline/candidate Measurement Protocol semantic and source IDs;
7. `AuditSemanticsCompatibility`;
8. selected binding, selector, adapter, projector, extraction, Audit-semantic,
   and exact-byte identities;
9. relocation candidates for a previous path that disappeared.

Head and base SHA changes alone are ignored when all represented input
identities, including measurement source snapshot references, remain equal.
An opaque measurement source snapshot change is retained as provenance/source
change even when the head also changed; v1 does not guess that the head was its
only cause. Replay engine package version, source revision, and distribution
digest are intentionally excluded; fixed `AuditSemanticsCompatibility` is the
comparison boundary for Core behavior.

## Artifact comparison

Artifacts are first paired by exact `(path, kind, role, split)`. Their fields
are compared independently:

- different `audit_semantics_sha256` is material;
- different selectors, role, split, or selected kind is material;
- different adapter/projector semantics is material when both versions are
  complete, otherwise indeterminate;
- different extraction identity with equal Audit semantics is indeterminate
  unless a fixed compatibility contract explicitly covers it;
- different source bytes with equal extraction and Audit semantics is
  byte-only change;
- exact source/extraction/Audit semantics with only head/base change is fresh.

Same-table Benchmark evidence remains selector-scoped. A changed table
predicate, cardinality, target field, or selected metric column is material
even when the physical CSV/TSV bytes are unchanged.

Formatting or row-order changes that preserve the fixed order-insensitive
projector's extraction and Audit-semantic identities are byte-only changes.

## Relocation

When a previous path is absent, the comparator searches only the current exact
snapshot identities plus supplied exact issued candidates. A relocation match
must preserve required kind, role, split, and independently recovered
`audit_semantics_sha256`; selector and adapter/projector compatibility are then
checked independently.

- exactly one compatible match: `IDENTITY_RELOCATION`,
  `NO_REVERIFY_TRIGGER_DETECTED`, `REMAP`;
- multiple matches: `INDETERMINATE`, `UNDETERMINED`, `REMAP`;
- no match while a complete current slot has different Audit semantics:
  `MATERIAL_CHANGE`, `REVERIFY_REQUIRED`, `REVERIFY`;
- no match without complete comparable evidence: `INDETERMINATE`,
  `UNDETERMINED`, `SUPPLY_PROVENANCE`.

Provider suggestions cannot enter the exact candidate tuple and cannot
establish equivalence.

## Component rules

- claim statement, claim ID, or Audit claim-spec change is material;
- Evidence Profile semantic change is material in v1; no compatibility
  exception registry is introduced;
- Evidence Obligation semantic change is material; source-support-only bundle
  changes are non-authoritative provenance metadata;
- measurement `semantic_protocol_id` change is material;
- measurement `source_snapshot_id` change with equal semantic protocol is
  source/provenance-only and never semantic measurement drift;
- changed `AuditSemanticsCompatibility.audit_semantics_sha256` is material;
- exact Core build provenance changes with unchanged compatibility do not
  trigger reverification;
- an unavailable or unequal identity-only snapshot, incomplete profile,
  missing required identity root, unversioned required adapter, or otherwise
  incomplete projector coverage is indeterminate;
- no prior Replay recipe is `LEGACY_UNAVAILABLE` with
  `SUPPLY_PROVENANCE`.

Legacy absence and incomplete required semantic coverage are gates. Once both
snapshots are complete, state precedence is material change, indeterminate,
relocation, byte-only change, then fresh.

## Integration boundary

V1 is an additive pure Core API. It is not invoked automatically by the
materializer and does not change `execute_ephemeral_audit*`, `audit_research`,
CLI behavior, Trace, Replay, deterministic findings, or unified result state.
Web/D1 storage and historical lookup are out of scope. A later host may call
the comparator before or alongside an already scheduled Audit, but its result
cannot suppress that Audit.

## Acceptance

Tests cover unrelated head changes, byte-only changes, order-insensitive row
reordering, table-selector changes, unique and ambiguous relocation, claim and
Audit-spec changes, profile and obligation changes, measurement semantic and
source-only changes, Core build changes with equal compatibility, legacy
recipes, provider-shaped attacks, unavailable/identity-only snapshots, and
incomplete adapter/projector coverage. Full Core verification, compileall, and
diff checks must pass before the stacked Draft PR is opened.
