# Evidence Obligations and Specific Missing Evidence v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the canonical deterministic obligations graph and use it to gate pre-Audit planning without changing Audit authority or legacy execution.

**Architecture:** A focused obligations module creates source-bound claim supports and exact validated artifact-binding supports, instantiates bounded policy templates, derives composites, and reduces them to one pre-Audit decision. Planner supplies trust-selected mapping facts and consumes that decision; plan/result envelopes carry the companion additively while plan identity and native Audit remain unchanged.

**Tech Stack:** Python 3.11+, frozen dataclasses, enums, SHA-256 canonical commitments, pytest, existing ClaimCI analysis/planner/Trace contracts.

**Spec:** `docs/superpowers/specs/2026-08-20-claimci-evidence-obligations-v1-design.md`

## Global Constraints

- Base exactly on Scientific Claim Type System PR #17 head `9c5cf5582945054ed75a1799bc4f60f3e0f1118e`.
- Do not modify PR #16 or #17 branches; target the stacked Draft PR at `codex/scientific-claim-types-v0`.
- Preserve `AuditResult`, CLI behavior, plan identity, optional `research.yaml`, LLM non-authority, and passive-only repository handling.
- Missing/unsupported dominates ambiguity; only a useful bounded question may yield `MAPPING_NEEDED`.
- Never turn claim fields into artifact evidence or approval into scientific/verdict authority.
- Do not implement Evidence Obligations UI or Measurement Drift.

---

### Task 1: Obligation contracts and support factories

**Files:**
- Create: `claimci/analysis/obligations.py`
- Modify: `claimci/analysis/__init__.py`
- Test: `tests/test_evidence_obligations_v1.py`

**Interfaces:**
- Produces the canonical enums, targets, support references, obligations, bundle, deterministic JSON projection, and legacy projection.
- `claim_field_support(claim, field, constraint_kind=None)` re-recovers source semantics and is the only supported `ClaimFieldSupport` factory.
- `validated_artifact_support(evidence, binding, mapping, metric=...)` revalidates exact evidence/binding trust and is the only supported `ArtifactEvidenceSupport` factory.

- [x] Write failing tests importing the missing contracts and asserting frozen/factory-only behavior, field recovery, HELD_OUT support, provider rejection, role/split/selector validation, and minimal serialization.
- [x] Run `python -m pytest -q -p no:cacheprovider tests/test_evidence_obligations_v1.py` and verify failure is caused by missing interfaces.
- [x] Implement the smallest immutable contracts and factories needed by the tests.
- [x] Rerun the focused test and keep it green while enforcing bounds and final classes.

### Task 2: Policy templates, graph derivation, and decision reduction

**Files:**
- Modify: `claimci/analysis/claim_types.py`
- Modify: `claimci/analysis/obligations.py`
- Modify: `tests/test_claim_types_v0.py`
- Modify: `tests/test_evidence_obligations_v1.py`

**Interfaces:**
- `ClaimEvidencePolicy.obligation_template_ids` is deterministic from canonical semantics.
- `assess_evidence_obligations(...)` returns a factory-created bundle with a validated graph and one READY/PARTIAL/MAPPING_NEEDED decision.

- [x] Write failing literal policy-template tests for metric improvement, HELD_OUT, and each unsupported primary.
- [x] Write failing graph tests for duplicate/dangling/cyclic IDs, child-derived composite state, deterministic ordering, reason/effect matrices, and bounded output.
- [x] Implement template generation and the graph/bundle factory.
- [x] Rerun claim-type and obligation tests to green.

### Task 3: Planner, mapping-question, plan, and result integration

**Files:**
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/planner.py`
- Modify: `claimci/analysis/integration.py`
- Modify: `claimci/analysis/materialize.py`
- Modify: `tests/test_analysis_contracts_v03.py`
- Modify: `tests/test_ephemeral_planner_v03.py`
- Modify: `tests/test_ephemeral_materialize_v03.py`
- Modify: `tests/test_unified_analysis_v03.py`

**Interfaces:**
- `MappingQuestion.blocking_obligation_ids` links a useful question to canonical blockers.
- `PlanningOutcome`, `EphemeralAuditPlan`, and `UnifiedAnalysisResult` carry `evidence_obligations` additively.
- Planner returns the obligation-derived lifecycle state and derives legacy missing artifacts without prose parsing.

- [x] Write failing planner tests for READY, missing, unsupported, ambiguity-only, missing-plus-ambiguity, unsupported-plus-ambiguity, unbounded/useless clarification, and stale/invalid approval.
- [x] Write failing result/materializer tests proving obligations survive the lifecycle, pre-Audit blockers cannot execute, runtime drift remains unavailable, and Audit-level insufficiency remains authoritative.
- [x] Add the additive contract fields and useful-question linkage validation.
- [x] Replace ad hoc missing-evidence reduction with canonical obligation assessment while preserving mapping trust selection.
- [x] Rerun the focused contract/planner/materializer/unified tests to green.

### Task 4: Compatibility and hostile authority coverage

**Files:**
- Modify: `tests/test_evidence_obligations_v1.py`
- Modify: `tests/test_analysis_cross_package_v03.py`
- Modify: `tests/test_hostile_zero_config_v03.py`
- Modify: `tests/test_claim_types_v0.py`

**Interfaces:**
- Existing executable metric fixtures produce the same `AuditClaimSpec`, plan ID, `AuditResult`, and verdict.
- Unsupported types never reach Audit; provider objects cannot change obligations, supports, effects, decisions, or authority.

- [x] Add RED regressions for provider-requiredness/state/reason/support injection, approval-as-verdict confusion, stale approval, unsupported compiler non-reachability, no-manifest execution, and passive-only behavior.
- [x] Make only the minimal production corrections needed for GREEN.
- [x] Run all focused analysis, hostile, Trace, CLI, and Audit suites.

### Task 5: Verification and stacked Draft PR

**Files:**
- Review every changed source, test, design, and plan file.

- [x] Run `python -m pytest -q -p no:cacheprovider`.
- [x] Run `python -m compileall -q claimci`.
- [ ] Run `git diff --check 9c5cf5582945054ed75a1799bc4f60f3e0f1118e...HEAD` after commit.
- [x] Run the scoped security review over the exact base-to-head diff.
- [x] Confirm exact ancestry, staged scope, and branch; confirm clean status after commit.
- [ ] Commit, push `codex/evidence-obligations-v1`, and open one Draft PR with base `codex/scientific-claim-types-v0`.
- [ ] Verify the remote PR head/base/draft/mergeability/checks without merging.
