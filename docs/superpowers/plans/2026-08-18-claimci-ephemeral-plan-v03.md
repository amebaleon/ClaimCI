# ClaimCI v0.3 Ephemeral Planner and Unified Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert validated zero-configuration claims, mappings, and normalized evidence into a confined ephemeral native Audit, one advisory Research Review synthesis, and one authority-safe unified result while preserving the legacy manifest path.

**Architecture:** Add immutable claim/planning contracts, a pure planner, a single-capture confined compatibility materializer that delegates to the existing `audit_research()`, and a separate one-call Review synthesis bridge. The hosted entry point composes those units and maps expected failures into typed pipeline states without manufacturing a verdict.

**Tech Stack:** Python 3.11+, frozen/slotted dataclasses, standard library path/hash/JSON/file APIs, existing PyYAML, existing ClaimCI Audit and Review provider protocols, pytest.

## Global Constraints

- Work only on `feat/ephemeral-plan-v03` in `C:\ClaimCI-worktrees\ephemeral-plan-v03`.
- Preserve `claimci audit research.yaml`, existing deterministic rules, existing `run_review()`, provider max-two-call/zero-retry limits, workflows, and CLI behavior.
- Integrate the concrete Auto Discovery and Adapter implementations without
  duplicating their algorithms or adding new production adapters in this branch.
- Pull-request content remains passive bytes. Never import or execute customer code, invoke customer hooks/builds/scripts, or download external data.
- Provider claims/mappings never gain deterministic authority from confidence. `RepoMapping.approve()` is the only provider-mapping approval transition.
- Only a real `AuditResult` returned by the existing Audit may create `DeterministicAuditOutcome`.
- A missing legitimate explicit threshold is `PARTIAL`; never manufacture zero, `0.05`, epsilon, or a difference from claimed values.
- Runtime integrity failures are `UNAVAILABLE`; known scientific/evidence/representation limitations are `PARTIAL`.
- Materialize only into an invocation-private scratch root outside the customer repository, exclusively create the plan tree/files, capture each customer artifact once, and remove only the created plan tree on every exit.
- Use test-driven development: add one observable failing test, verify RED for the intended missing behavior, implement minimally, and verify GREEN before the next behavior.
- Integrate Auto Discovery commit `6d4c1240c0c09668c04b66035a5ce2d8598ab813`
  and Adapter commit `f928986` directly for the pre-merge gate. The conversion
  boundary must require their concrete immutable contracts and scope a
  repository-wide result to one selected claim.

---

## File structure

- Create `claimci/analysis/claims.py`: native-manifest claim conversion and audit-relevant claim projection.
- Create `claimci/analysis/planner.py`: planning request/outcome contracts, Auto Discovery conversion boundary, mapping selection, evidence completeness, and stable plan identity.
- Create `claimci/analysis/materialize.py`: runtime context, single-capture integrity validation, deterministic tree generation, native Audit invocation, and cleanup.
- Create `claimci/analysis/integration.py`: hosted-runner orchestration and typed state resolution.
- Create `claimci/review/analysis_bridge.py`: validated `ScientificClaim` conversion and one-call advisory synthesis.
- Modify `claimci/analysis/contracts.py`: `ClaimedMetricValue`, `AuditClaimSpec`, executable plan fields, and typed unified missing evidence.
- Modify `claimci/analysis/__init__.py`: documented public exports.
- Modify `claimci/review/__init__.py` only if the package already re-exports public Review entry points; otherwise import the bridge directly by module.
- Create `tests/test_analysis_claims_v03.py`.
- Create `tests/test_ephemeral_planner_v03.py`.
- Create `tests/test_ephemeral_materialize_v03.py`.
- Create `tests/test_analysis_review_bridge_v03.py`.
- Create `tests/test_unified_analysis_v03.py`.
- Modify `tests/test_analysis_contracts_v03.py` only for additive shared-contract compatibility assertions.

---

### Task 1: Audit claim and unified-state contracts

**Files:**
- Create: `claimci/analysis/claims.py`
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/__init__.py`
- Test: `tests/test_analysis_claims_v03.py`
- Test: `tests/test_analysis_contracts_v03.py`

**Interfaces:**
- Produces: `ClaimedMetricValue`, `AuditClaimSpec`, `audit_claim_spec_from_research_spec(spec)`, `audit_relevant_claim_projection(spec)`, additive `EphemeralAuditPlan.audit_claim`, additive `EphemeralAuditPlan.selected_mapping`, and `UnifiedAnalysisResult.missing_evidence`.
- Consumes: existing `Direction`, `ResearchSpec`, `ExperimentRole`, `FieldProvenance`, `MappingCandidate`, `RepoMapping`, and `MissingEvidence`.

- [ ] **Step 1: Add failing contract tests for descriptive claimed values and strict threshold invariants**

  Add literal tests that construct baseline/candidate descriptive values, assert frozen behavior and JSON-safe serialization, reject non-finite values and `UNSPECIFIED` roles, require threshold/provenance co-presence, and prove changing `claimed_values` does not change `audit_relevant_claim_projection()`.

  ```python
  first = AuditClaimSpec(
      claim_id="claim-1",
      metric="accuracy",
      direction=Direction.HIGHER,
      minimum_absolute_improvement=0.05,
      metric_provenance=source,
      direction_provenance=source,
      threshold_provenance=source,
      claimed_values=(
          ClaimedMetricValue(ExperimentRole.BASELINE, 0.71, "%", source),
          ClaimedMetricValue(ExperimentRole.CANDIDATE, 0.79, "%", source),
      ),
  )
  second = dataclasses.replace(first, claimed_values=())
  assert audit_relevant_claim_projection(first) == audit_relevant_claim_projection(second)
  ```

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_analysis_claims_v03.py`

  Expected: collection/import failure because the new contracts and helpers do not exist.

- [ ] **Step 3: Implement minimal claim contracts and projection**

  Add the frozen records to `contracts.py`. The projection returns a literal dict containing only `claim_id`, `metric`, `direction.value`, and `minimum_absolute_improvement`. Add `claims.py` manifest conversion using `MANIFEST_HINT` provenance and no descriptive values.

  ```python
  def audit_relevant_claim_projection(spec: AuditClaimSpec) -> dict[str, object]:
      return {
          "claim_id": spec.claim_id,
          "metric": spec.metric,
          "direction": spec.direction.value,
          "minimum_absolute_improvement": spec.minimum_absolute_improvement,
      }
  ```

- [ ] **Step 4: Add failing tests for additive plan and unified partial semantics**

  Assert a plan with `audit_claim` requires matching claim IDs and exact mapping type. Assert `PARTIAL` can carry a deterministic outcome plus typed missing evidence, while `authoritative_verdict` remains the real Audit verdict. Preserve construction of legacy plan/result fixtures through defaults where existing tests rely on the previous positional shape.

- [ ] **Step 5: Run the new contract tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_analysis_contracts_v03.py tests/test_analysis_claims_v03.py`

  Expected: failures name missing plan fields/unified typed missing-evidence behavior.

- [ ] **Step 6: Implement additive plan/result fields and invariants**

  Append defaulted fields to preserve the merged v0.3 constructor shape:

  ```python
  audit_claim: AuditClaimSpec | None = None
  selected_mapping: MappingCandidate | RepoMapping | None = None
  missing_evidence: tuple[MissingEvidence, ...] = ()  # UnifiedAnalysisResult
  ```

  Planner-produced plans will always set both optional plan fields; the executor rejects representational legacy plans lacking them. Update `to_jsonable` through the existing dataclass path and public exports.

- [ ] **Step 7: Run focused tests and verify GREEN**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_analysis_contracts_v03.py tests/test_analysis_claims_v03.py`

  Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 1**

  ```powershell
  git add claimci/analysis/contracts.py claimci/analysis/claims.py claimci/analysis/__init__.py tests/test_analysis_contracts_v03.py tests/test_analysis_claims_v03.py
  git commit -m "feat: add deterministic audit claim contracts"
  ```

---

### Task 2: Discovery conversion and pure planner

**Files:**
- Create: `claimci/analysis/planner.py`
- Modify: `claimci/analysis/__init__.py`
- Test: `tests/test_ephemeral_planner_v03.py`

**Interfaces:**
- Consumes: Task 1 contracts plus existing discovery-shaped fields, `NormalizedEvidence`, mapping contracts, repository/head identity, and `Confidence`.
- Produces: `PlanningState`, `PlanningRequest`, `PlanningOutcome`, `planning_request_from_discovery(...)`, and `plan_ephemeral_audit(request)`.

- [ ] **Step 1: Add failing tests for the Auto Discovery boundary**

  Use the concrete immutable `DiscoveryResult`, `DiscoveredClaim`, and
  `ClaimedValue` contracts. Assert the converter reuses `ClaimReference`, maps
  `ClaimDirection` to deterministic `Direction`, preserves baseline/candidate
  values descriptively, revalidates an explicit `at least` threshold, rejects a
  from-to pair as threshold, scopes artifacts/evidence/mappings/questions to the
  selected claim, and rejects evidence not issued by discovery.

  ```python
  request = planning_request_from_discovery(
      discovery,
      claim_id="claim-1",
      normalized_evidence=(baseline_results, candidate_results),
  )
  assert request.claim is discovered.reference
  assert request.audit_claim.minimum_absolute_improvement == 0.05
  assert request.mapping_candidates == discovery.mapping_candidates
  ```

- [ ] **Step 2: Run boundary tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_planner_v03.py -k discovery`

  Expected: import failure for the missing planner boundary.

- [ ] **Step 3: Implement the isolated discovery-shaped conversion**

  Use `TYPE_CHECKING` imports for concrete discovery names. Runtime validation must inspect every expected field and require shared nested types, unique claim ID, matching repository/head/PR metadata, issued artifact equality, and claim relevance. Do not import or copy production discovery code.

- [ ] **Step 4: Add failing tests for planner precedence and mapping trust**

  Cover threshold-free `PARTIAL`, mapping-independent missing evidence preceding questions, applicable approved mapping, sole current-head manifest hint without trust elevation, one eligible inferred mapping, ambiguity, and hostile provider provenance at confidence `1.0`. Add one parameterized test that independently breaks each of the seven auto-use conditions and expects `MAPPING_NEEDED` when clarification is useful.

  ```python
  outcome = plan_ephemeral_audit(request)
  assert outcome.state is PlanningState.MAPPING_NEEDED
  assert outcome.mapping_question is not None
  assert outcome.plan is None
  ```

- [ ] **Step 5: Run planner trust tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_planner_v03.py -k "mapping or threshold or precedence"`

  Expected: failures because planning outcomes and selection logic are absent.

- [ ] **Step 6: Implement mapping selection and missing-evidence analysis**

  Implement exact-one-state `PlanningOutcome`. Check threshold first. Resolve approved mapping, sole valid manifest hint, then inferred mapping. Build questions with two through eight literal mapping choices only when answering can produce a viable plan. Require one config/results binding and unambiguous train/eval dataset references per role; missing entire artifacts yield typed `MissingEvidence`.

- [ ] **Step 7: Add failing tests for stable plan identity**

  Assert the hand-derived plan ID equals `"plan-" + sha256(canonical_bytes).hexdigest()[:24]`, and remains unchanged when claim text, claimed values, provider detail, discovery reason, or tuple object identities change. Assert it changes for metric, direction, threshold, binding, selector, role, artifact path/hash, repository, or head changes.

- [ ] **Step 8: Run identity tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_planner_v03.py -k plan_id`

  Expected: failures because canonical identity derivation is absent.

- [ ] **Step 9: Implement canonical plan projection and READY construction**

  Use literal dictionaries, sorted bindings/evidence, `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`, and SHA-256. Exclude all fields forbidden by the spec. Produce `EphemeralAuditPlan` with its exact selected mapping and zero missing evidence.

- [ ] **Step 10: Run all Task 2 tests and verify GREEN**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_planner_v03.py`

  Expected: all tests pass.

- [ ] **Step 11: Commit Task 2**

  ```powershell
  git add claimci/analysis/planner.py claimci/analysis/__init__.py tests/test_ephemeral_planner_v03.py
  git commit -m "feat: plan validated ephemeral audits"
  ```

---

### Task 3: Single-capture confined materializer

**Files:**
- Create: `claimci/analysis/materialize.py`
- Modify: `claimci/analysis/__init__.py`
- Test: `tests/test_ephemeral_materialize_v03.py`

**Interfaces:**
- Consumes: a `READY` `EphemeralAuditPlan`, trusted `RuntimeExecutionContext`, selected normalized evidence, and existing `audit_research()`.
- Produces: `MaterializationLimits`, `RuntimeExecutionContext`, `MaterializationPartial`, `MaterializationUnavailable`, and `execute_ephemeral_audit(plan, runtime) -> AuditResult`.

- [ ] **Step 1: Add failing tests for runtime context and one-time artifact capture**

  Assert runtime repository/head equality, checkout/scratch separation, source/scratch non-overlap, regular-file and symlink rejection, exact bytes/hash/size validation, and bounded reads. Instrument the file-opening boundary so each selected customer path is opened exactly once and no path is reopened during writing or Audit.

- [ ] **Step 2: Run capture tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_materialize_v03.py -k "capture or runtime or symlink or hash"`

  Expected: import failure for the missing materializer.

- [ ] **Step 3: Implement trusted runtime validation and immutable capture**

  `RuntimeExecutionContext` contains exact `RepositoryIdentity`, checkout root, current `GitCommitSha`, scratch root, and limits. Use repository-relative paths, lexical component `lstat`, resolved confinement, `os.open` with `O_NOFOLLOW` where available, `fstat` regular-file identity checks, bounded reads, post-read path/identity checks, and SHA-256 over captured bytes. Build `PassiveArtifact` from those bytes. Never reopen accepted customer paths.

- [ ] **Step 4: Add failing tests for results/config/dataset conversion**

  Use literal normalized observations from registered JSON/CSV/YAML adapters
  where supported and bounded fakes only for unsupported passive dataset
  identity. Assert native results contain only ordered `runs`, metric values,
  and genuine integer seeds; no summaries. Assert absent seeds are omitted,
  string seeds cause `MaterializationPartial`, config dotted keys nest
  deterministically, conflicting config paths fail partial, partial config
  reaches real Audit policy, and dataset bytes are copied exactly.

- [ ] **Step 5: Run conversion tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_materialize_v03.py -k "results or seed or config or dataset"`

  Expected: failures because native conversion is absent.

- [ ] **Step 6: Implement deterministic native conversion**

  Create only the fixed tree and filenames. Serialize results with strict JSON and configs with `yaml.safe_dump(sort_keys=True)`. Copy only captured JSONL dataset bytes. Refuse missing whole artifact categories, unsupported seed representations, non-JSONL dataset representations, ambiguous splits, duplicate/conflicting fields, or a missing threshold. Generate `research.yaml` only after every required output is representable.

- [ ] **Step 7: Add failing lifecycle and fail-closed tests**

  Assert plan-tree exclusive creation, per-file exclusive creation, collision as `MaterializationUnavailable`, cleanup after success, representation failure, integrity failure, injected write failure, and injected Audit exception. Assert the caller scratch root and an unrelated sibling remain. Assert no subprocess/import/network API is invoked.

- [ ] **Step 8: Run lifecycle tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_materialize_v03.py -k "cleanup or collision or exclusive or exception"`

  Expected: failures because exclusive tree lifecycle is absent.

- [ ] **Step 9: Implement exclusive tree lifecycle and real Audit delegation**

  Resolve and validate roots, call `plan_root.mkdir(exist_ok=False)`, write each file with mode `"x"`, invoke `audit_research(plan_root / "research.yaml", artifact_root=plan_root)`, return that exact object, and remove only `plan_root` in an outer `finally` when this invocation created it.

- [ ] **Step 10: Run all Task 3 tests and verify GREEN**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_ephemeral_materialize_v03.py`

  Expected: all tests pass and no plan tree remains.

- [ ] **Step 11: Commit Task 3**

  ```powershell
  git add claimci/analysis/materialize.py claimci/analysis/__init__.py tests/test_ephemeral_materialize_v03.py
  git commit -m "feat: materialize confined ephemeral audits"
  ```

---

### Task 4: Validated claim conversion and one-call Research Review

**Files:**
- Create: `claimci/review/analysis_bridge.py`
- Modify: `claimci/review/__init__.py` only if consistent with existing exports
- Test: `tests/test_analysis_review_bridge_v03.py`

**Interfaces:**
- Consumes: exact `ScientificClaim`, issued `SourceBundle`, `AnalysisReviewContext`, existing `ReviewConfig`, `ReviewerProvider`, `StructuredRequest`, and `ProviderResponse`.
- Produces: `validate_scientific_claim_for_audit(...)`, `AnalysisReviewContext`, and `run_analysis_review(...) -> AdvisoryResearchInterpretation`.

- [ ] **Step 1: Add failing tests for ScientificClaim validation**

  Assert exact issued source ID/span/text, metric and direction anchoring, explicit bounded-threshold language, finite absolute magnitude/unit consistency, and no threshold from vague or from-to-only language. Assert unknown source, paraphrased text, unsupported claim type/direction/unit, and unanchored metric fail controlled validation. Assert provider provenance stays provider provenance.

- [ ] **Step 2: Run claim-bridge tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_analysis_review_bridge_v03.py -k claim`

  Expected: import failure for the missing Review bridge.

- [ ] **Step 3: Implement conservative source-anchored conversion**

  Reuse issued source lines to confirm exact source text and locations. Use narrow case-insensitive token boundaries for metric/direction and explicit bounded-language patterns. Accept only `ClaimType.METRIC_IMPROVEMENT`, deterministic higher/lower directions, and unambiguous absolute native-unit thresholds.

- [ ] **Step 4: Add failing tests for the one-call synthesis path**

  Provide a complete fake `ReviewerProvider`. Assert exactly one `synthesize_review` call, zero `extract_claims` calls, deterministic payload is read-only, missing evidence/provenance is present, strict output schema validation, and no call to `run_review`, `audit_research`, or materialization. Assert verdict-like provider fields are rejected and advisory text cannot modify the deterministic snapshot.

- [ ] **Step 5: Run synthesis tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_analysis_review_bridge_v03.py -k synthesis`

  Expected: failures because `AnalysisReviewContext` and `run_analysis_review` are absent.

- [ ] **Step 6: Implement one-call synthesis without fallback**

  Build a bounded structured request with literal advisory policy and strict schema. Use only `provider.synthesize_review`. Parse one JSON object into `AdvisoryResearchInterpretation`; reject unknown keys, verdict/authority fields, unknown missing evidence, excessive output, or malformed confidence. Do not call existing `run_review()`.

- [ ] **Step 7: Run Task 4 and existing Review regression tests**

  Run:

  ```text
  python -m pytest -q -p no:cacheprovider tests/test_analysis_review_bridge_v03.py
  python -m pytest -q -p no:cacheprovider tests/test_review_orchestrator_day3.py tests/test_review_tools_day3.py tests/test_review_workflow_day3.py
  ```

  Expected: all tests pass, demonstrating additive behavior.

- [ ] **Step 8: Commit Task 4**

  ```powershell
  git add claimci/review/analysis_bridge.py claimci/review/__init__.py tests/test_analysis_review_bridge_v03.py
  git commit -m "feat: add advisory analysis synthesis bridge"
  ```

---

### Task 5: Hosted unified integration and scenarios A-I

**Files:**
- Create: `claimci/analysis/integration.py`
- Modify: `claimci/analysis/__init__.py`
- Test: `tests/test_unified_analysis_v03.py`

**Interfaces:**
- Consumes: `PlanningRequest`, `RuntimeExecutionContext`, `ReviewConfig`, optional provider, planner, materializer, deterministic factory, and Review bridge.
- Produces: `run_unified_analysis(request, runtime, review_config, provider=None) -> UnifiedAnalysisResult`.

- [ ] **Step 1: Add failing tests for exact state resolution**

  Assert planning `MAPPING_NEEDED`, pre-Audit `PARTIAL`, runtime-integrity `UNAVAILABLE`, unexpected Audit/materializer `UNAVAILABLE`, successful Audit plus Review failure/disabled `PARTIAL` with authority retained, and Audit plus Review `COMPLETE`. Assert a real `INSUFFICIENT_EVIDENCE` Audit plus Review remains pipeline `COMPLETE` with that authoritative verdict.

- [ ] **Step 2: Run state tests and verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_unified_analysis_v03.py -k state`

  Expected: import failure for the missing integration entry point.

- [ ] **Step 3: Implement hosted orchestration and error classification**

  Call planner once, materializer/Audit at most once, `DeterministicAuditOutcome.from_audit_result()` exactly once on success, and Review at most once. Convert only documented operational exceptions to typed states. Never catch public programmer type errors. Sanitize error messages to bounded codes/descriptions without raw artifact/provider content.

- [ ] **Step 4: Add end-to-end acceptance tests A-I**

  Add exactly these behavior tests: `test_scenario_a_zero_manifest_normalized_evidence_runs_full_analysis`, `test_scenario_b_native_manifest_preserves_audit_bytes`, `test_scenario_c_ambiguous_candidate_mapping_asks_without_verdict`, `test_scenario_d_missing_dataset_is_pre_audit_partial`, `test_scenario_e_advisory_disagreement_cannot_override_audit`, `test_scenario_f_explicit_threshold_reaches_native_audit`, `test_scenario_g_vague_improvement_never_invents_threshold`, `test_scenario_h_identical_inputs_produce_stable_plan_and_result`, and `test_scenario_i_hostile_provider_mapping_never_executes`. Use real planner, real materializer, real existing Audit, strict test adapter outputs, and one fake provider. Scenario B invokes the unchanged legacy `audit_research`/renderer twice and compares literal bytes. Scenario H compares canonical plan/result fields and verdict/findings rather than host-specific temporary absolute paths.

- [ ] **Step 5: Run A-I tests and verify RED where integration is incomplete**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_unified_analysis_v03.py -k scenario_`

  Expected: one or more failures identify missing integration behavior rather than fixture errors.

- [ ] **Step 6: Complete the minimal integration for all scenarios**

  Wire typed missing evidence into final results, pass actual deterministic/provenance context to Review, and keep advisory output separate. Do not add an Audit-only `COMPLETE` branch.

- [ ] **Step 7: Run all v0.3 planner/integration tests and verify GREEN**

  Run:

  ```text
  python -m pytest -q -p no:cacheprovider tests/test_analysis_claims_v03.py tests/test_ephemeral_planner_v03.py tests/test_ephemeral_materialize_v03.py tests/test_analysis_review_bridge_v03.py tests/test_unified_analysis_v03.py
  ```

  Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 5**

  ```powershell
  git add claimci/analysis/integration.py claimci/analysis/__init__.py tests/test_unified_analysis_v03.py
  git commit -m "feat: integrate unified ephemeral analysis"
  ```

---

### Task 6: Compatibility, adversarial hardening, and final branch gate

**Files:**
- Modify only files implicated by a failing regression or adversarial test.
- Test: all existing and new tests.

**Interfaces:**
- Verifies every public interface and frozen design invariant; adds no new product behavior.

- [ ] **Step 1: Run focused legacy compatibility suites**

  Run:

  ```text
  python -m pytest -q -p no:cacheprovider tests/test_audit.py tests/test_manifest.py tests/test_day2_demo.py tests/test_direction_day2.py
  python -m pytest -q -p no:cacheprovider tests/test_review_orchestrator_day3.py tests/test_review_tools_day3.py tests/test_review_workflow_day3.py tests/test_review_workflow_isolation_day3.py
  ```

  Expected: all pass without golden-output changes.

- [ ] **Step 2: Add a failing adversarial regression for every defect found**

  Probe forged provider trust, subclass/factory bypasses, mixed role assignments, selector drift, digest swaps, same-content symlink swaps, scratch/root overlap, cleanup target substitution, repeated evidence IDs, non-finite/huge numerics, unsupported seeds, and Review payload authority fields. For any observed acceptance, first add one minimal test and verify it fails before changing production code.

- [ ] **Step 3: Implement only fixes proven by RED tests and rerun focused suites**

  Run all five new test files plus `tests/test_analysis_contracts_v03.py`; expect all pass.

- [ ] **Step 4: Run compile and diff checks**

  Run:

  ```text
  python -m compileall -q claimci
  git diff --check
  ```

  Expected: both exit zero with no output.

- [ ] **Step 5: Run the full offline suite**

  Run: `python -m pytest -q -p no:cacheprovider`

  Expected: all tests pass with only established platform skips.

- [ ] **Step 6: Review scope and public exports**

  Confirm `git diff --name-only origin/main...HEAD` contains only the integrated
  analysis/discovery/adapter packages, planner/review integration, tests, and
  approved docs. Confirm no workflow, CLI command, provider retry, external
  dependency, or customer artifact was changed.

- [ ] **Step 7: Commit final hardening if Task 6 changed files**

  ```powershell
  git add claimci/analysis/contracts.py claimci/analysis/claims.py claimci/analysis/planner.py claimci/analysis/materialize.py claimci/analysis/integration.py claimci/analysis/__init__.py claimci/review/analysis_bridge.py tests/test_analysis_contracts_v03.py tests/test_analysis_claims_v03.py tests/test_ephemeral_planner_v03.py tests/test_ephemeral_materialize_v03.py tests/test_analysis_review_bridge_v03.py tests/test_unified_analysis_v03.py
  git commit -m "test: harden ephemeral analysis boundaries"
  ```

- [ ] **Step 8: Invoke finishing-development-branch**

  Re-run the full verification evidence, review commit history/status, push `feat/ephemeral-plan-v03`, open a PR against `main`, and do not merge.

---

## Task 7: Concrete Discovery and Adapter pre-merge integration

- [x] Merge the exact Auto Discovery and Adapter branch heads into the feature
  branch without copying their implementations.
- [x] Require exact concrete discovery contracts and scope a repository-wide
  result, mappings, and questions to the selected claim ID.
- [x] Apply the same seven-condition hosted gate to `MANIFEST_HINT`; provider
  provenance never auto-executes.
- [x] Give a runtime-valid `RepoMapping.approve()` result unconditional
  precedence over unapproved candidates while failing closed on an invalid
  approval.
- [x] Exercise registered CSV/YAML adapters, generic `UNSPECIFIED` evidence,
  explicit role mapping, materialization, real Audit, and unified Review in one
  cross-package test.
- [x] Cover multiple claims, approved/provider/manifest regressions, and hosted
  scenarios A through I, then run the complete offline gate.
