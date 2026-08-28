# Claim Recovery and Evidence Occurrence Identity V2 Design

**Status:** Approved for implementation on 2026-08-27

**Scope:** ClaimCI Core only. The Hosted integration, Core pin, provider-routing regressions, and Pilot-credit correction live in the companion ClaimCI-Web design.

## Goal

Allow the unchanged controlled production-smoke fixture to recover its natural subject-first metric-improvement claim and reach deterministic planning, while replacing collision-prone evidence identifiers with deterministic IDs bound to trusted physical artifact occurrences and canonical extraction selectors.

## Frozen constraints

- `recover_scientific_claim()` remains the sole natural-language grammar authority.
- Deterministic discovery projects canonical recovery; it does not maintain a second metric-improvement grammar.
- The extension is bounded to explicit subject-first metric-improvement forms.
- No metric, baseline, candidate, direction, or threshold may be inferred from arithmetic or unrelated nearby numbers.
- Physical artifact occurrence identity remains separate from scientific baseline/candidate/reference role.
- `ArtifactCandidate` is unchanged.
- Repository, snapshot role, and exact commit scope are attached at a trusted artifact wrapper/factory boundary.
- Provider and mapping payload schemas gain no occurrence fields and cannot manufacture occurrence scope.
- Evidence ID v2 excludes provider output, claim prose, approval actor, PR number, and scientific role labels.
- Existing completed result JSON remains readable; `evidence_id` remains a bounded string.
- No D1 migration is needed or permitted by this design.
- If implementation proves a schema migration is necessary, stop with `EVIDENCE_ID_SCHEMA_GATE`.
- Existing pending mapping approvals continue to authorize only their exact repository/head/path/kind/SHA/size/adapter/selector material. The authoritative cross-run verifier is Hosted's public mapping question/choice identity, which the companion PR versions and hardens; Core does not introduce a second approval authority. Any regenerated question/choice mismatch fails closed through the existing path.
- AuditResult, verdict, Review, mapping trust, and scientific authority contracts are unchanged.

## Canonical claim grammar

### Accepted forms

The existing metric-first form remains accepted:

```text
accuracy improved from 0.60 to 0.70
```

The canonical parser additionally accepts only these bounded subject-first shapes, case-insensitively, with the existing numeric and unit grammar:

```text
candidate improves accuracy from 0.60 to 0.70
the candidate improves accuracy from 0.60 to 0.70
model improves accuracy from 0.60 to 0.70
candidate increases accuracy from 0.60 to 0.70
candidate reduces loss from 0.40 to 0.30
```

The subject token is limited to `candidate` or `model`, with optional leading `the`. The positive verb set is `improves`, `increases`, `raises`, and `grows`; the negative verb set is `reduces`, `decreases`, `lowers`, and `drops`. This is an explicit grammar, not a general NLP parser.

### Locality and ambiguity

The metric token must immediately follow the subject-first verb. The existing value pair must be locally attached to that metric clause. Existing semicolon threshold continuation remains valid:

```text
candidate improves accuracy from 0.60 to 0.70; improved by at least 0.05
```

An explicit threshold is recovered only from the existing threshold grammar and locality checks. `0.70 - 0.60`, a second numeric clause, or any unrelated nearby number never becomes a threshold. Multiple candidate metric clauses, multiple locally plausible value pairs, or multiple thresholds remain ambiguous and recover as `None`.

### Discovery projection

`claimci.analysis.discovery.claims` removes its metric-improvement regexes and numeric reparsing. For every source line it uses a bounded provisional reference to identify the canonical claim kind, derives the final existing claim ID, and lets the final exact `ClaimReference` recover through `recover_scientific_claim()`. It projects the returned `MetricImprovementClaim` into `DiscoveredClaim`:

- metric from `primary.metric`;
- direction from `primary.direction`;
- baseline/candidate/minimum values from the corresponding canonical `ClaimQuantity` values;
- source line, exact source text, confidence, and provenance from the issued source record/reference;
- subject remains the scientific role label `candidate`, not a parser-derived physical identity.

Legacy deterministic categories that are outside the canonical scientific compiler may keep their existing non-metric classifiers. They may not recognize or reparse metric-improvement prose.

The Core provider-discovery projection also consumes the canonical claim stored on its exact `ClaimReference` for baseline, candidate, and minimum values. It does not retain `_claimed_values` or another numeric natural-language parser. This does not give provider data authority: the source-bound canonical recovery remains the only value source.

## Trusted artifact occurrence

### Contract

Add an immutable `ArtifactOccurrence` companion and `ArtifactSnapshotRole` enum to the Core analysis contracts.

```python
class ArtifactSnapshotRole(str, Enum):
    HEAD = "head"
    BASE = "base"

@dataclass(frozen=True, slots=True, init=False)
class ArtifactOccurrence:
    repository: RepositoryIdentity
    snapshot_role: ArtifactSnapshotRole
    commit: GitCommitSha
    candidate: ArtifactCandidate
```

`ArtifactOccurrence` has no public value constructor. It is issued through `artifact_occurrence_from_snapshot(...)`, which validates concrete ClaimCI contract types. `PassiveArtifact` contains an issued occurrence plus immutable bytes and exposes `candidate` as a read-only compatibility property. `ArtifactSource` contains the same occurrence and retains read-only `repository`, `head_sha`, and `candidate` compatibility properties.

Factories are the only production boundary that attaches occurrence scope:

```python
artifact_occurrence_from_snapshot(repository, snapshot_role, commit, candidate)
passive_artifact_from_snapshot(repository, snapshot_role, commit, candidate, content)
artifact_source_from_snapshot(repository, commit, root, candidate, *, snapshot_role=ArtifactSnapshotRole.HEAD)
```

The default `HEAD` on the existing `ArtifactSource` factory preserves its historical exact-head meaning. Callers that open a base tree must pass `BASE` explicitly. Provider/mapping payload decoding never calls these factories and receives no matching schema fields.

### Why `ArtifactCandidate` stays unchanged

An artifact candidate describes bytes discovered at a repository path. It is used by discovery, provider mapping proposals, and historical result JSON. Repository/snapshot/commit scope is execution context, not candidate data. Keeping scope on factory-issued wrappers avoids reinterpreting candidate equality, widening provider contracts, or changing historical decoders.

## Evidence ID v2

### Canonical material

Every built-in adapter has a fixed trusted `semantic_version = "1"`. Extraction generates a single versioned ID from this canonical JSON material:

```json
{
  "schema_version": 2,
  "repository": {"owner": "amebaleon", "name": "repository"},
  "snapshot": {"role": "head", "commit": "<40 lowercase hex>"},
  "artifact": {
    "path": "candidate-config.yaml",
    "kind": "config",
    "sha256": "<full 64 lowercase hex>",
    "size": 271
  },
  "adapter": {"id": "claimci-yaml-config-v1", "version": "1"},
  "selector": ["<canonical provenance-free mapping material>"]
}
```

Serialization uses UTF-8, `ensure_ascii=True`, `sort_keys=True`, separators `(",", ":")`, and `allow_nan=False`. Mappings are sorted by `field_mapping_identity`; each mapping uses the existing provenance-free `field_mapping_material`. An extraction with no selectors uses an empty list.

The output is:

```text
evidence-v2-<full lowercase SHA-256 of canonical JSON>
```

The ID is 76 characters and remains inside the existing 128-character bound.

### Included and excluded authority

Included:

- schema version;
- repository owner/name;
- physical snapshot role;
- exact commit;
- artifact path/kind/full SHA/size;
- trusted adapter ID and semantic version;
- canonical selector identity.

Excluded:

- provider response or provider prose;
- claim text;
- mapping/approval actor and provenance detail;
- pull-request number;
- scientific baseline/candidate/reference role;
- random values and timestamps.

Provider-proposed mappings may select only selectors that pass existing adapter validation. Their raw payload, provenance text, and any alleged occurrence metadata are not hashed. Repository/snapshot/commit scope comes exclusively from the issued artifact wrapper.

### Identity invariants

- Re-reading the same exact occurrence with the same adapter and selector produces the same ID.
- Different repository, physical snapshot role, exact commit, path, kind, full SHA, size, adapter version, or canonical selector produces a different ID.
- Byte-identical baseline/candidate configuration files at different paths produce different IDs.
- The same path/bytes in base and head snapshots produces different IDs even if the commit strings were accidentally equal, because snapshot role is explicit.
- Assigning one occurrence a different scientific role does not change its ID.
- A true duplicate extraction of the same occurrence/selector keeps one identity. Existing planner uniqueness checks continue to fail closed if callers try to multiply that identity; conflicting outputs are never silently deduplicated.

## Planner and materialization behavior

Adapters generate v2 IDs before `PlanningRequest`. Therefore distinct byte-identical paths no longer collide at `PlanningRequest.__post_init__` or `EphemeralAuditPlan.__post_init__`. The planner keeps its duplicate-ID integrity checks: they prevent a repeated representation from multiplying scientific authority.

Materialization recaptures the same repository/head occurrence through the trusted factory. Fresh adapter extraction must reproduce the exact v2 ID and normalized evidence. Any changed path, bytes, commit, selector, or adapter version fails the existing exact-head revalidation path.

Scientific role remains attached through `ArtifactBinding` and normalized observations. It is never mixed into physical occurrence identity.

## Compatibility and migration analysis

- `ArtifactCandidate`, `AdapterMatch`, `ArtifactBinding`, `NormalizedEvidence`, provider claim/mapping JSON, and public result JSON shapes are unchanged.
- Historical `evidence-*` identifiers remain valid bounded strings when completed result JSON is decoded.
- Core has no completed-result JSON decoder; that persistence boundary is Hosted/Cloud-owned. Core proves bounded legacy `NormalizedEvidence` construction/serialization remains valid, while the companion PR must decode one immutable completed legacy result through both the Hosted Python contract decoder and Cloud TypeScript repository/parser path.
- New extraction emits only `evidence-v2-*`; no legacy value is rewritten in storage.
- Core's mapping values remain in-memory planning data and are not treated as cross-run authorization. Hosted's versioned public mapping question/choice identity binds exact repository/head and artifact/adapter/selector material, contains no evidence-ID alias, and rejects every regenerated mismatch. This companion-owned verifier prevents an already stored approval from silently retargeting v2 evidence.
- There is no D1 column keyed by evidence ID and no migration is required.
- The direct Python construction pattern for `PassiveArtifact(candidate, content)` is intentionally replaced by the trusted snapshot factory. This is internal execution-contract hardening, not a provider/public schema expansion.

## Exact controlled fixture gate

Commit a byte-for-byte synthetic copy of controlled tree `90266b8ad6b63cda4f766e31039c75e0967f9b1c` under Core test fixtures, plus the exact base candidate result. The integration test reconstructs base/head trees, uses PR title `Production smoke: verify supported result`, and uses the exact claim text:

```text
The candidate improves accuracy from 0.60 to 0.70 under the same configuration and evaluation dataset.
```

It must prove:

- deterministic scientific claim count is exactly 1;
- the selected canonical metric is `accuracy`, direction is `higher`, baseline is `0.60`, candidate is `0.70`, and no threshold is inferred;
- all normalized evidence IDs use v2;
- byte-identical baseline/candidate configs have different IDs;
- `planning_request_from_discovery()` succeeds;
- `plan_ephemeral_audit()` returns the legitimate bounded `PARTIAL` reason `required_threshold_not_recovered`: the exact approved prose has no explicit threshold, the manifest mapping does not become claim-prose authority, and no threshold is synthesized. If implementation instead exposes an independently real `READY` or `MAPPING_NEEDED` state, the report must prove the exact existing authority path; the result must never be `scientific_routing_uncertain`.

The approved manifest supplies the eight artifact bindings but does not supply canonical claim-field support. The fixture gate therefore freezes claim recovery and the current honest pre-Audit boundary without broadening manifest or mapping authority.

## Verification

Focused stages:

1. canonical claim grammar and deterministic discovery projection;
2. occurrence factory, v2 identity matrix, adapters, planner collision, and materialization revalidation;
3. exact controlled fixture integration.

Final Core verification runs once after the branch is complete:

- full Core pytest suite;
- `python -m compileall claimci tests`;
- repository-supported lint/type checks if configured;
- `git diff --check`;
- scoped security review of provider authority, path/snapshot trust, selector canonicalization, collision/fail-closed behavior, and historical decoding.

## Out of scope

- Hosted/Cloud changes;
- provider retry or routing behavior;
- Audit/verdict/Review changes;
- D1 migrations;
- production deployment or production runs;
- `/app` gate changes.

## Addendum — detached minimum-improvement source binding (2026-08-28)

`recover_scientific_claim()` remains the sole grammar authority. Deterministic
discovery may supply it an invocation-private certificate for the exact
Review-issued source document and the exact primary line span. The parser alone
recognizes a detached declaration only when its entire source line has one of
these ASCII-case-insensitive labels, with horizontal whitespace allowed only
around the declaration:

```text
Declared minimum improvement:
Minimum improvement:
Required minimum improvement:
```

The value/unit grammar is the existing Core numeric grammar. A declaration
must be finite and nonnegative, has no trailing prose, and is associated only
when its document has exactly one recoverable `MetricImprovement` and exactly
one valid declaration. Multiple claims or declarations leave the normal claim
unattached. An inline and detached threshold must have the same value and
unit; disagreement makes canonical recovery fail closed.

The parser retains an immutable private binding containing the issued document
identity/hash/text, primary span, and declaration span. Its `ClaimQuantity`
keeps the deterministic source provenance. Re-validation by the compiler and
claim-field-obligation path reuses that private certificate. The binding is
not added to provider schemas and is explicitly excluded from `to_jsonable`,
so public JSON retains its historical shape.

The controlled fixture keeps the original scientific claim line byte-for-byte
unchanged and adds `Declared minimum improvement: 0.05` as the next line of
the same trusted PR-description document. It must now reach `READY` and the
existing native Audit must return `SUPPORTED`; no provider, mapping, or
manifest field becomes claim-threshold authority.
