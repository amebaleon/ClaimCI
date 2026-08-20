# Verification Input Snapshot and Replay Recipe v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build trusted pre-Audit verification snapshots and post-Audit replay recipes without changing deterministic authority or legacy Audit behavior.

**Architecture:** Exact-head materialization constructs factory-only snapshot identities from the selected plan, freshly captured artifacts, and fixed projectors before Audit. The native Audit runs once; only its actual `AuditResult` can create the replay commitment. The existing Evidence Trace is then reduced to a matching reference and combined into a non-executable recipe.

**Tech Stack:** Python 3.11 standard library, frozen dataclasses, existing ClaimCI analysis contracts, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-claimci-verification-input-replay-v1-design.md`

## Global Constraints

- Base exactly `122f6176388ceabbb792ff071f0e4e20f4b32bbe`.
- No new runtime dependency.
- `AuditResult`, CLI, and `render_json()` remain byte-compatible when companions are absent.
- Provider output cannot construct trusted snapshot or replay authority.
- Replay never executes customer code, authorizes experiment replay, or changes verdict.
- Training and Benchmark profiles both work, including selector-scoped same-table evidence.
- Do not implement Stale Evidence or Verdict Regression.
- Core representation bounds are not a Hosted transport allocation.

---

### Task 1: Pre-Audit verification contracts and golden digests

**Files:**
- Create: `claimci/analysis/verification.py`
- Create: `tests/test_verification_replay_v1.py`

**Interfaces:**
- Produces immutable `VerificationArtifactIdentity`, companion identity records, `AuditSemanticsCompatibility`, and `VerificationInputSnapshot`.
- Produces canonical JSON projection helpers used by replay and materialization.

- [ ] Write failing tests for constructor sealing, subclass rejection, malformed IDs/hashes/paths/selectors, deep immutability, and provider-shaped dictionaries being rejected.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/test_verification_replay_v1.py` and verify failure because the module/interfaces do not exist.
- [ ] Implement bounded canonical helpers and factory-only records.
- [ ] Add literal digest golden vectors for artifact, compatibility, captured input, comparison frame, and complete snapshot.
- [ ] Verify RED with hand-written expected SHA-256 values, then implement minimal canonical projections until GREEN.
- [ ] Add identity-only reduction tests with an explicit small complete-projection ceiling; assert the full snapshot digest is retained and `UNAVAILABLE` cannot execute.

### Task 2: Post-Audit replay contracts

**Files:**
- Create: `claimci/analysis/replay.py`
- Modify: `tests/test_verification_replay_v1.py`

**Interfaces:**
- `ReplayEngineProvenance.current(...) -> ReplayEngineProvenance`
- `ReplayAuditCommitment.from_audit_result(result) -> ReplayAuditCommitment`
- `ReplayTraceReference.from_trace(trace, commitment) -> ReplayTraceReference`
- `ReplayRecipe.from_execution(...) -> ReplayRecipe`

- [ ] Write failing tests proving direct/subclass construction is closed and a fake verdict/provider object cannot create an Audit commitment.
- [ ] Verify the test fails for missing interfaces.
- [ ] Implement commitment by delegating to the existing stable Audit projection and requiring exact `AuditResult`/`Verdict` types.
- [ ] Test trace/audit mismatch, engine-revision separation from compatibility, fixed `experiment_replay_supported=False`, and recipe golden digest.
- [ ] Implement the minimal factories and canonical recipe projection.

### Task 3: Exact-head artifact projectors and snapshot builder

**Files:**
- Modify: `claimci/analysis/verification.py`
- Modify: `claimci/analysis/materialize.py`
- Modify: `tests/test_verification_replay_v1.py`

**Interfaces:**
- Internal `build_verification_input_snapshot(plan, runtime, bound, captured, manifest, measurement_context) -> VerificationInputSnapshot`.
- Internal projectors produce extraction and Audit-semantic commitments without source-format bytes.

- [ ] Add RED tests for Training result/config/JSONL projections and formatting-only JSON/YAML/JSONL changes.
- [ ] Add RED Benchmark test where one physical table has distinct baseline/candidate selectors and values; assert equal source SHA but distinct verification evidence IDs and Audit-semantic hashes.
- [ ] Implement projectors using only freshly revalidated `NormalizedEvidence`, `ArtifactBinding`, and `PassiveArtifact` values.
- [ ] Add profile selection and Evidence Obligation identity tests, then implement their fixed semantic/full projections.
- [ ] Add measurement reference tests proving source-only changes alter only source snapshot and complete input snapshot identities.
- [ ] Add comparison-frame tests proving candidate result values are excluded while candidate measurement-protocol semantics remain included.

### Task 4: Canonical execution ordering and replay creation

**Files:**
- Modify: `claimci/analysis/trace.py`
- Modify: `claimci/analysis/materialize.py`
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/integration.py`
- Modify: `claimci/analysis/__init__.py`
- Modify: `tests/test_ephemeral_materialize_v03.py`
- Modify: `tests/test_benchmark_evidence_profile_v0.py`
- Modify: `tests/test_unified_analysis_v03.py`

**Interfaces:**
- `EphemeralAuditExecution` gains optional `input_snapshot` and `replay_recipe` companions.
- Canonical materialization populates both; compatibility execution still returns `AuditResult`.
- `UnifiedAnalysisResult` carries companions only with deterministic authority.

- [ ] Write a failing ordering test whose Audit fake asserts the snapshot factory has already completed before invocation.
- [ ] Implement pre-Audit snapshot construction immediately after exact capture/profile revalidation/native-tree preparation.
- [ ] Write failing tests for factory-created Audit commitment, Trace reference, and recipe after exactly one Audit call.
- [ ] Implement replay creation after Trace construction and bind Trace/Audit identities.
- [ ] Add state-matrix tests preventing replay companions without deterministic authority and proving advisory content cannot alter recipe or verdict.
- [ ] Export only the approved public contracts and serialization helpers.

### Task 5: Trust, bounds, compatibility, and focused gates

**Files:**
- Modify: `tests/test_verification_replay_v1.py`
- Modify focused existing tests only where additive fields require assertions.

- [ ] Add hostile provider verdict/identity injection and unapproved mapping regressions.
- [ ] Add approved-mapping test proving approval affects binding identity but not Audit commitment authority.
- [ ] Add source revision/distribution changes with equal Audit semantics compatibility.
- [ ] Add legacy direct Audit, manifest, report, and CLI byte-compatibility tests.
- [ ] Run focused Replay/snapshot tests.
- [ ] Run Benchmark Profile, Measurement Drift, adapters, obligations/planner/materializer, Trace/Audit/report/CLI suites.
- [ ] Run full `python -m pytest -q -p no:cacheprovider`.
- [ ] Run `python -m compileall -q claimci`.
- [ ] Run `git diff --check 122f6176388ceabbb792ff071f0e4e20f4b32bbe...HEAD`.

### Task 6: Security review and stacked Draft PR

**Files:**
- Review every changed file; modify only if a confirmed in-scope defect has a RED regression.

- [ ] Freeze the exact base/head diff and run the scoped security-diff workflow.
- [ ] Validate factory-only authority, canonical digest separation, path/selector confinement, no execution surface, no provider trust elevation, and bounded projections.
- [ ] Re-run the complete verification gate after any review fix.
- [ ] Inspect `git status`, `git diff`, and `git diff --check`.
- [ ] Commit focused implementation changes, push `codex/verification-input-replay-v1`, and open a Draft PR against `codex/benchmark-evidence-profile-v0`.
- [ ] Report exact base/head, files, tests, digest contracts, capabilities, security result, and blockers; stop before merge.
