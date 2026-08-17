# ClaimCI v0.3 Ephemeral Planner and Unified Analysis Design

Date: 2026-08-18

## Objective

Implement the deterministic planning and integration portion of ClaimCI v0.3's
zero-configuration primary path:

> validated discovered claim + validated mappings + normalized evidence ->
> ephemeral audit plan -> existing deterministic ClaimCI Audit -> advisory
> Research Review -> one unified analysis result.

Ordinary users do not need to author `research.yaml` or convert existing result
files into ClaimCI-native JSON. A user-authored manifest remains an optional,
high-confidence hint and advanced override. The legacy
`claimci audit research.yaml` path retains its existing behavior.

## Non-goals

This branch does not implement Claim Discovery, Evidence Discovery, concrete
production adapters, a GitHub App, hosted SaaS, workflow changes, or persistence
of inferred mappings. It does not execute pull-request code, download external
datasets, change deterministic audit rules, change existing CLI behavior, or
modify the existing two-call `run_review()` state machine.

The compatibility materializer is not a second scientific implementation. It
only translates sufficiently complete validated evidence into the existing
native file shapes so the mature Audit remains the sole deterministic authority.

## Package and public boundaries

The implementation extends the shared analysis package with focused modules:

```text
claimci/analysis/
    claims.py
    planner.py
    materialize.py
    integration.py
claimci/review/
    analysis_bridge.py
```

`claims.py` owns deterministic claim-to-test contracts. `planner.py` selects a
trusted mapping and emits one planning state. `materialize.py` owns the
confined, ephemeral compatibility tree and invokes the existing Audit.
`integration.py` is the hosted-runner entry point. `analysis_bridge.py` is the
only Review-to-analysis conversion and one-call synthesis boundary.

The shared `claimci.analysis` export surface exposes the new contracts and
entry points. The deterministic planner API does not import or accept
`claimci.review.models.ScientificClaim`.

## Audit claim contract

`AuditClaimSpec` is an immutable, validated description of the claim to test:

```python
@dataclass(frozen=True, slots=True)
class AuditClaimSpec:
    claim_id: str
    metric: str
    direction: Direction
    minimum_absolute_improvement: float | None
    metric_provenance: FieldProvenance
    direction_provenance: FieldProvenance
    threshold_provenance: FieldProvenance | None
    claimed_values: tuple[ClaimedMetricValue, ...] = ()
```

It contains no verdict, severity, impact, blocking conclusion, or advisory
interpretation. Construction means only that a claim has been validated as a
candidate for deterministic testing. It does not imply that the claim is
supported.

The metric is bounded, non-empty, and free of control characters. Direction is
the existing deterministic `Direction.HIGHER` or `Direction.LOWER`. A threshold
is finite and non-negative, and appears if and only if threshold provenance is
present.

`minimum_absolute_improvement` may be populated only when the validated source
text explicitly states bounded threshold semantics such as "at least",
"minimum", `>=`, or a narrowly allowlisted equivalent. An observed change such
as "accuracy improved from 71% to 79%" is not a minimum threshold. A vague
claim such as "accuracy improved" also has no threshold. The current native
Audit requires a numeric threshold and does not implement strict
directional-only semantics; therefore either case remains `PARTIAL` until the
deterministic engine gains that capability. The planner never inserts zero,
`0.05`, an epsilon, or a difference calculated from descriptive values.

`ClaimedMetricValue` records a separately validated baseline or candidate value,
its optional exact raw token, optional unit, and field provenance. The raw token
is nullable because Auto Discovery's current `ClaimedValue` contract preserves
the numeric value, unit, and provenance while the full exact quotation remains
in `ClaimReference.text`. It is descriptive
claim evidence only. It never feeds `minimum_improvement`, native result
observations, plan identity, or verdict logic merely because it numerically
matches discovered artifacts. Unstructured values remain preserved in the
separate `ClaimReference.text` rather than being guessed into typed roles.

## Explicit claim conversion boundaries

Provider-originated `ScientificClaim` data cannot enter the planner directly.
The Review-side conversion boundary is:

```python
validate_scientific_claim_for_audit(
    claim: ScientificClaim,
    sources: SourceBundle,
) -> AuditClaimSpec
```

It requires an exact validated `ScientificClaim` and its issued source bundle,
revalidates the source identifier and span, requires source text to match the
issued text, anchors metric and direction in that span, and applies the strict
threshold rules above. Provider provenance remains visible as provider
provenance; validation does not silently upgrade its origin or grant authority.
Unsupported claim types, ambiguous units, unanchored fields, and inconsistent
magnitude values produce a controlled validation limitation rather than a
quantitative guess.

The native-manifest boundary is independent:

```python
audit_claim_spec_from_research_spec(spec: ResearchSpec) -> AuditClaimSpec
```

It copies the already parsed native metric, direction, and explicit threshold
with `MANIFEST_HINT` field provenance. This representation supports common
analysis consumers, but the legacy audit continues to execute its user-authored
manifest through the existing native path.

### Auto Discovery conversion

The implemented Auto Discovery contracts enter planning through one explicit
boundary:

```python
planning_request_from_discovery(
    discovery: DiscoveryResult,
    *,
    claim_id: str,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> PlanningRequest
```

The converter requires exact `DiscoveryResult` and `DiscoveredClaim` values. It
selects exactly one issued claim ID, reuses its validated `ClaimReference`, and
constructs `AuditClaimSpec` from the claim's metric, direction, claimed values,
and field provenance. It accepts only `ClaimType.METRIC_IMPROVEMENT` with a
present metric and `HIGHER` or `LOWER` direction. Other discovered claim types
remain `PARTIAL` or unavailable to the current deterministic engine rather than
being coerced into metric-improvement semantics.

`DiscoveredClaim.baseline_value` and `candidate_value` become descriptive
`ClaimedMetricValue` records with explicit roles. They do not become result
observations or thresholds. `DiscoveredClaim.minimum_improvement` becomes a
threshold only after the bridge independently confirms bounded requirement
language in `ClaimReference.text` and verifies numeric/unit consistency. A
from-to value pair alone never supplies that threshold.

Repository identity, head SHA, PR number, artifacts, mapping candidates,
approved mapping, and any upstream bounded mapping question are carried forward
without trust elevation. The repository-wide `DiscoveryResult` is narrowed to
the selected claim ID: only artifacts whose `relevant_claim_ids` contain that
claim, normalized evidence issued for those artifacts, mapping bindings over
those artifacts, and a question relevant to that claim may cross the boundary.
An unrelated artifact or question for another discovered claim cannot reject or
block the selected claim. Normalized evidence must still refer to an artifact
issued by the exact concrete `DiscoveryResult`; structurally similar duck-typed
objects are rejected. Artifact issues may inform controlled missing-evidence
reporting but are never treated as observations. The converter does not use
`DiscoveryResult.preferred_mapping` as an authorization shortcut; the planner
applies the full mapping policy itself.

The conversion preserves provider provenance. A provider-discovered claim still
requires this validation boundary, and a provider-originated mapping still
requires `RepoMapping.approve()` before deterministic use.

## Planning input and outcome

`PlanningRequest` contains repository identity, optional PR number, head SHA,
`ClaimReference`, `AuditClaimSpec`, normalized evidence, inferred or manifest
mapping candidates, and applicable approved repository mappings. Claim IDs must
match. The request contains metadata and immutable evidence contracts, not a
repository root or executable callback.

`plan_ephemeral_audit(request)` returns a frozen `PlanningOutcome` with exactly
one state:

- `READY`: one fully validated `EphemeralAuditPlan`;
- `MAPPING_NEEDED`: one bounded `MappingQuestion` that can materially unblock
  an otherwise viable plan;
- `PARTIAL`: typed `MissingEvidence` or an unsupported scientific/evidence
  representation that mapping clarification cannot repair;
- `UNAVAILABLE`: analysis cannot meaningfully start.

State precedence is explicit:

1. reject structurally invalid public inputs;
2. identify critical evidence or representation limits no mapping answer can
   solve and return `PARTIAL`;
3. resolve approved mappings, manifest hints, and inferred mapping trust;
4. return `MAPPING_NEEDED` only when one bounded clarification can produce a
   viable plan;
5. return `READY` only with a fully validated plan.

The planner does not ask a mapping question when the answer cannot overcome an
already-known missing threshold, missing artifact category, or unsupported
native representation.

## Mapping trust and automatic selection

An applicable, runtime-valid `RepoMapping` is a separate, explicit
user-approval transition and has precedence over every manifest, inferred, or
provider proposal. Unapproved alternatives neither override nor block that
mapping. A stale or runtime-invalid approved binding fails closed and prevents
fallback to lower-trust candidates. A current-head manifest hint is
an optional high-confidence advanced input. In the hosted path,
`MappingTrust.MANIFEST_HINT` and confidence such as `0.99` do not elevate the
candidate to `USER_APPROVED`, create a reusable repository mapping, or bypass
runtime validation. A sole, structurally valid, current-head manifest hint may
guide the ephemeral plan, but conflicts with approved mappings or other strong
candidates are never silently resolved. The legacy native manifest path remains
independent and continues to honor the user-authored manifest directly.

An inferred mapping is automatically usable only when all seven conditions hold:

1. it is the only viable candidate;
2. confidence is at least `0.90`;
3. provenance is not provider-originated;
4. path, SHA-256, selector, and artifact kind can be revalidated at runtime;
5. it does not conflict with an applicable approved `RepoMapping`;
6. there is no contradictory manifest hint or other strong candidate;
7. the resulting ephemeral selection is bound to the PR's current head SHA.

If any condition fails, the planner returns a useful bounded
`MappingQuestion`, or `PARTIAL` when clarification cannot solve the known
limitation. A provider proposal is never auto-used, even at confidence `1.0`.
It can reach deterministic planning only after explicit user approval through
`RepoMapping.approve()`.

## Ephemeral audit plan and stable identity

`EphemeralAuditPlan` is extended to carry its `AuditClaimSpec`, the selected
immutable mapping bindings/trust/provenance, selected artifact digests, and the
existing repository, PR, head SHA, claim, evidence, missing-evidence, and
confidence fields. The claim-reference ID and audit-claim ID must match.

The deterministic `plan_id` is derived from canonical JSON containing only the
audit-relevant projection:

- repository owner/name;
- current head SHA and PR number when present;
- claim ID, validated metric, direction, and explicit threshold;
- selected bindings, roles, artifact kinds, adapter IDs, and selectors;
- selected artifact paths and SHA-256 digests;
- mapping trust and stable provenance identifiers needed to audit selection.

Canonical key ordering and compact JSON encoding are fixed before SHA-256
derivation. `ClaimedMetricValue`, claim prose, provider explanations, discovery
reasons, Review output, Python object representations, timestamps, and other
descriptive or unstable fields are excluded. Identical validated audit inputs
therefore produce the same plan identity without allowing descriptive values
to influence deterministic execution.

The plan is always ephemeral. It exposes no repository write or commit method,
does not persist approval, and never implies that its generated manifest was
authored or approved by the customer.

## Runtime snapshot and final revalidation

Immediately before materialization, the executor consumes a trusted
`RuntimeExecutionContext` containing trusted repository identity, checkout
root, current head SHA, and invocation-private scratch root. The repository
identity and head SHA must exactly match the plan. The executor then revalidates
mapping trust, bindings, path, artifact kind, adapter ID, and selectors. Final
validation produces an internal immutable captured snapshot; callers cannot
supply a preconstructed captured snapshot that skips these checks.

Every selected customer artifact is captured exactly once into immutable
`PassiveArtifact.content` during this final validation:

1. resolve the canonical repository-relative path under the trusted checkout;
2. verify the destination remains confined and every lexical path component is
   non-symlink;
3. open with no-follow semantics where the platform provides them;
4. compare file identity and regular-file status around the open operation;
5. read at most the configured per-file and aggregate bounds;
6. hash the bytes that were actually read and compare them with the selected
   `ArtifactCandidate.sha256`;
7. recheck available file identity/path metadata before accepting the snapshot.

Substitution, stale head, digest change, symlink substitution, confinement
failure, artifact-kind mismatch, or selector/adapter mismatch invalidates the
trusted execution snapshot and returns `UNAVAILABLE`. Once capture succeeds,
the materializer consumes only those immutable bytes and normalized evidence.
It never reopens a customer path.

## Confined compatibility materialization

The caller supplies an invocation-private scratch root outside the customer
repository. The materializer creates exactly one exclusive child named by the
plan ID:

```text
<private-scratch>/<plan-id>/
    research.yaml
    baseline/
        config.yaml
        results.json
        train.jsonl
        eval.jsonl
    candidate/
        config.yaml
        results.json
        train.jsonl
        eval.jsonl
```

The relative filenames and structure are deterministic. The plan tree is
created exclusively and is never reused. A collision is `UNAVAILABLE`.
Individual files use exclusive creation. The resolved scratch root and plan
tree must not overlap the customer repository or any selected source artifact.
No symlink is accepted or created.

One outer cleanup boundary removes only the exclusively created
`<plan-id>` tree on success and every controlled or exceptional failure. It
never deletes or recursively modifies the caller-owned scratch root.

The materializer imports no customer code and runs no customer command, hook,
build, script, or executable. It performs no network request or external
dataset download. It writes only the fixed YAML, JSON, and JSON Lines files.

### Results conversion

Only validated normalized observations whose role and metric match the plan are
emitted. Values remain finite numbers. Integer seeds are preserved. A genuinely
absent seed remains absent. A present seed with a representation the native
loader cannot preserve is a representation failure and returns `PARTIAL`; it is
not silently converted to missing. No run, seed, metric, claimed summary, or
aggregate summary is invented.

### Config conversion

Only validated `ConfigValue` fields are emitted. Dotted keys become nested
mappings using a deterministic ordering. Duplicate values, scalar/mapping path
collisions, or unsupported values return `PARTIAL`. `ComputeEvidence` is not
silently mapped into config fields.

A present config artifact with only partial supported fields may be
materialized faithfully. The existing Audit alone decides whether those fields
produce `CONFIG.MISSING_FIELDS` and `INSUFFICIENT_EVIDENCE`; the compatibility
layer does not duplicate that scientific policy. An entirely missing config
artifact does not become an empty placeholder.

### Dataset conversion

Exactly one unambiguous train and evaluation artifact per role is required for
the current native Audit. Their verified passive bytes are copied unchanged.
Because the shared normalized contract does not contain normalized dataset
rows, a non-JSONL dataset cannot be converted losslessly in this branch and
returns `PARTIAL`. No external data is fetched and no dataset identity is
invented.

### Generated manifest

The generated `research.yaml` contains only the validated metric, direction,
explicit threshold, and fixed internal relative artifact paths. A claim without
a legitimate explicit threshold is `PARTIAL`; the materializer does not insert
zero or another value to satisfy the legacy schema.

The generated manifest is an implementation detail. User-facing results must
not describe it as customer-authored, committed, persisted, or approved.

## Deterministic execution and legacy compatibility

After successful materialization, the executor invokes the existing
`audit_research()` with the plan tree as its confined artifact root. It does not
reimplement `check_configs`, `check_results`, leakage checks, alignment checks,
or verdict policy.

Only the exact `AuditResult` returned by that invocation can cross
`DeterministicAuditOutcome.from_audit_result()`. No advisory object, normalized
observation, claimed value, generated YAML payload, or provider output can
construct or replace deterministic authority.

The user-authored manifest path remains separate and unchanged. Existing
`claimci audit research.yaml` and Review manifest auditing continue to call the
native loader and Audit directly, including the current engine's ability to
emit findings for declared-but-missing artifacts. Zero-configuration preflight
rules are not imposed on legacy manifests. Where applicable, the legacy path's
verdict and rendered bytes remain identical.

## Single-synthesis Research Review boundary

The immutable `AnalysisReviewContext` contains:

- `ClaimReference` and `AuditClaimSpec`;
- issued evidence provenance;
- the actual `DeterministicAuditOutcome`, when Audit completed;
- typed missing evidence.

The Review-side entry point is:

```python
run_analysis_review(
    context: AnalysisReviewContext,
    config: ReviewConfig,
    *,
    provider: ReviewerProvider | None = None,
) -> AdvisoryResearchInterpretation
```

This is a separate one-call synthesis path. It does not call or fall back to
`run_review()`, extract claims, alter mappings, materialize files, or invoke
another Audit. It makes at most one provider call and performs zero retries.
Its request and response schemas contain no verdict, severity, impact, blocking,
plan mutation, or artifact-generation fields.

The existing two-call `run_review()` path remains byte- and behavior-compatible.
The new context is frozen; Research Review may explain the deterministic result
and missing evidence but cannot regenerate evidence, modify the plan, or
override the verdict.

## Unified hosted-runner entry point

The hosted runner calls:

```python
run_unified_analysis(
    request: PlanningRequest,
    runtime: RuntimeExecutionContext,
    review_config: ReviewConfig,
    *,
    provider: ReviewerProvider | None = None,
) -> UnifiedAnalysisResult
```

Every invocation terminates in one typed result for expected operational
conditions. Planning states are translated without manufacturing authority.
The unified result gains typed `missing_evidence` while retaining separate
deterministic and advisory lanes.

`AnalysisState` describes pipeline completeness, not authority:

- `MAPPING_NEEDED` has one useful bounded question and no guessed verdict;
- `PARTIAL` may have no deterministic outcome when pre-Audit evidence is
  missing or unsupported;
- `PARTIAL` may retain a valid deterministic outcome when Audit completed but
  Research Review was disabled, failed, or was unavailable;
- `COMPLETE` means planning, real Audit, and advisory Review completed, even if
  the Audit verdict is `INSUFFICIENT_EVIDENCE`;
- `UNAVAILABLE` means analysis could not meaningfully start or the trusted
  runtime snapshot became invalid.

`authoritative_verdict` continues to derive solely from the presence of the
real deterministic outcome, independent of pipeline state or advisory text.
There is no successful Audit-only product mode: an unavailable Review produces
`PARTIAL`, not `COMPLETE`, while preserving any valid Audit authority.

## Error classification

Known scientific, evidence, or native-representation limitations are
`PARTIAL`. Runtime integrity failures—including stale head SHA, changed digest,
symlink substitution, confinement failure, selector/adapter mismatch, and
scratch collision—are `UNAVAILABLE`. Unexpected materializer or Audit
exceptions fail closed as `UNAVAILABLE`; they are not mislabeled as missing
scientific evidence. Provider failure after Audit is `PARTIAL` with the
deterministic outcome retained.

Expected states are returned as typed values. Incorrect public argument types
and impossible contract construction remain explicit `TypeError` or
`AnalysisContractError` rather than being hidden as operational states. Raw
provider or artifact content is not copied into controlled public error text.

## Test strategy

### Acceptance scenarios

The branch must pass these end-to-end acceptance scenarios through public
interfaces:

- **A. Zero-manifest full audit:** registered CSV/YAML adapters provide valid
  normalized results and config evidence plus bounded selected passive JSONL
  dataset identity; the hosted entry point creates an ephemeral plan, runs the
  existing Audit, runs one advisory synthesis, and returns one unified result.
- **B. Native-manifest compatibility:** an existing user-authored
  `research.yaml` follows the unchanged native path and produces the same
  deterministic verdict and rendered bytes as before where applicable.
- **C. Ambiguous candidate mapping:** two viable candidate-result mappings
  produce one bounded `MappingQuestion` and no guessed deterministic verdict.
- **D. Missing dataset:** an entirely absent required dataset produces typed
  missing evidence and pre-Audit `PARTIAL`; no placeholder dataset, manifest,
  or verdict is manufactured.
- **E. Advisory disagreement:** an LLM interpretation that disagrees with the
  Audit cannot change `authoritative_verdict`.
- **F. Explicit threshold:** a source claim with validated bounded threshold
  semantics produces the corresponding explicit absolute threshold in
  `AuditClaimSpec` and the generated native manifest.
- **G. Vague improvement:** a claim such as "accuracy improved" receives no
  `0.05`, zero, epsilon, or derived threshold and remains `PARTIAL` for the
  current engine.
- **H. Stable retries:** identical canonical inputs produce the same plan ID,
  plan serialization, state, deterministic verdict/findings, and advisory test
  result, excluding only explicitly documented host-path diagnostics.
- **I. Hostile provider mapping:** unsafe paths/selectors, authority fields,
  trust escalation, or provider-originated mappings—including confidence
  `1.0`—are rejected from deterministic planning unless the exact validated
  candidate crosses `RepoMapping.approve()`.

Focused tests cover:

1. `AuditClaimSpec`, claimed-value isolation, immutability, JSON serialization,
   explicit threshold language, and vague/from-to claims with no threshold;
2. exact Review-source conversion, source anchoring, provider bypass rejection,
   and manifest conversion;
3. planner state precedence and every automatic-mapping condition;
4. provider proposal rejection at confidence `1.0` and explicit approval;
5. stable plan identity from the audit-relevant projection only;
6. JSON/CSV-origin normalized observations reaching the existing Audit through
   trivial test adapters/fakes;
7. integer, absent, and unsupported seed behavior;
8. partial config behavior delegated to the existing Audit;
9. no empty placeholders for missing artifacts;
10. exact passive dataset-byte copying and unsupported representation handling;
11. stale head, path, symlink, digest, kind, selector, adapter, size, overlap,
    and scratch-collision failures;
12. single-capture behavior with no customer-path reopen;
13. deterministic tree layout and cleanup after success and injected failures;
14. legacy manifest verdict and rendered-byte compatibility;
15. one synthesis call, zero retries, no extraction/fallback/second Audit;
16. advisory disagreement leaving deterministic authority unchanged;
17. completed `INSUFFICIENT_EVIDENCE` Audit plus Review yielding `COMPLETE`;
18. all requested scenarios A through I through the hosted entry point.

Remaining dataset/provider fakes implement only bounded shared interfaces. They
do not become production discovery algorithms or concrete adapters. Tests
perform no provider, network, workflow, or customer-code execution.

The pre-merge integration gate additionally uses the concrete Auto Discovery
and Adapter implementations. It exercises `discover_repository()`, registered
CSV/YAML adapter `probe()` and `extract()`, generic `UNSPECIFIED` normalized
observations plus explicit mapping roles, planner conversion, confined
materialization, the real `audit_research()`, and `run_unified_analysis()`.
Because the current registered adapters intentionally do not parse dataset
formats, those integration tests use only a bounded passive dataset-identity
stub; production dataset bytes remain selected and copied without execution.
The concrete regression matrix includes a repository with multiple claims,
approved-versus-inferred precedence, the manifest automatic-use gate, explicit
provider approval, and hosted scenarios A through I.

The final branch gate is:

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

An independent final review must compare the implementation, tests, public
exports, legacy behavior, authority boundary, materialization lifecycle, and
all A-I scenarios against this design before publication.

## Compatibility and consumer guidance

- Existing CLI and user-authored manifest behavior remain unchanged.
- Auto Discovery supplies concrete `DiscoveryResult`, `DiscoveredClaim`,
  `ClaimReference`, optional `ClaimedMetricValue`, scoped artifacts, mapping
  proposals, and provenance; it does not grant mapping or verdict authority.
- Registered adapters supply typed selectors and normalized evidence from
  passive bytes with `UNSPECIFIED` experiment roles where the artifact itself
  does not encode a role. The selected mapping assigns baseline/candidate roles;
  adapters must not execute repository content.
- The hosted runner supplies a trusted current-head `RuntimeExecutionContext`
  with repository identity, checkout root, and invocation-private scratch root,
  then calls `run_unified_analysis()`.
- A stored mapping becomes trusted only through `RepoMapping.approve()`.
- Consumers must account for the added audit-claim and selected-mapping fields
  on `EphemeralAuditPlan` and typed missing evidence on
  `UnifiedAnalysisResult`.
