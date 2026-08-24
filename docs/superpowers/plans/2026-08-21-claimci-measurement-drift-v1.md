# Evaluator and Measurement Drift v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and `superpowers:test-driven-development` task by task.

**Goal:** Add bounded semantic/source protocol identities, native drift reporting, and explicit upstream-procedure obligations without changing deterministic authority or legacy behavior.

**Architecture:** Exact-head materialization creates source commitments from validated bindings. Native Audit recovers semantic components from its confined passive inputs and compares them by fixed policy, referencing existing CONFIG/DATASET findings. Evidence Obligations alone gates explicitly required aggregation semantics before Audit.

**Base:** Exact Evidence Obligations PR #18 head `b4c053da81d119cb9324e56ddecdf49e178eee4b`.

**Spec:** `docs/superpowers/specs/2026-08-21-claimci-measurement-drift-v1-design.md`

## Global constraints

- Work only on `codex/measurement-drift-v1`; stack the Draft PR on `codex/evidence-obligations-v1`.
- No changes to PR #16/#17/#18 branches, ClaimCI-Web, deployment, customer execution, Replay, Staleness, Regression, or provider-call counts.
- Add no `MEASUREMENT.*` rule unless its Evidence Trace category is implemented in the same change.
- Preserve absent-extension `AuditResult`, JSON, plan ID, CLI, and manifest behavior.

### Task 1: Measurement contracts and pure comparison

**Files:**
- Create `claimci/measurement.py`
- Create `tests/test_measurement_drift_v1.py`
- Modify `claimci/models.py`, `claimci/report.py`

- [x] Add RED tests for enum exactness, immutability/finality, bounds, canonical ordering, semantic/source digest separation, direction/threshold exclusion, provider-shaped object rejection, and reduction separation.
- [x] Implement the minimal contracts and canonical factories.
- [x] Add RED comparator/Audit tests for verified, existing CONFIG/DATASET invalidation, missing semantics, byte-only warning, no duplicate findings, and absent-context byte compatibility.
- [x] Integrate the additive Audit companion and minimal JSON projection without adding native rule IDs.

### Task 2: Required upstream-procedure obligations

**Files:**
- Create `claimci/analysis/measurement.py`
- Modify `claimci/analysis/claim_types.py`
- Modify `claimci/analysis/obligations.py`
- Modify `claimci/analysis/contracts.py`, `claimci/analysis/__init__.py`
- Modify `tests/test_evidence_obligations_v1.py`, `tests/test_ephemeral_planner_v03.py`

- [x] Add RED source-recovery tests for supported arithmetic mean, missing support, unsupported procedures, ambiguity-only mapping, and missing/unsupported dominance.
- [x] Add RED hostile tests proving provider provenance cannot invent requirement, procedure, support, scope, semantic ID, drift state, or authority.
- [x] Implement fixed source recovery, target/support contracts, policy templates, and obligation assessment.
- [x] Keep legacy claims free of new required obligations and preserve existing plan IDs.

### Task 3: Exact-head source snapshots and native semantic recovery

**Files:**
- Modify `claimci/analysis/materialize.py`
- Modify `claimci/audit.py`
- Modify `tests/test_ephemeral_materialize_v03.py`, `tests/test_evidence_trace_v1.py`, `tests/test_analysis_cross_package_v03.py`

- [x] Add RED tests for exact binding source identity, fresh hash-bound materialization, semantic dataset commitment, metric/evaluation config recovery, source-only change, runtime drift, and passive-only behavior.
- [x] Build source snapshots only from runtime-revalidated plan bindings and captured artifacts.
- [x] Recover protocol semantics inside native Audit from confined parsed config/dataset inputs.
- [x] Attach bounded report, preserve existing finding/verdict reduction, and prove complete Trace classification with no new rule category.

### Task 4: Compatibility and hostile regression pass

**Files:**
- Modify only scoped tests needed for public/CLI/Hosted compatibility.

- [x] Prove legacy direct Audit and CLI output/exit codes remain unchanged without a measurement context.
- [x] Prove direction and claim threshold changes leave measurement IDs unchanged.
- [x] Prove explicit unsupported procedure never reaches Audit or Research Review.
- [x] Prove no `research.yaml` prerequisite, third LLM call, raw source persistence, or customer-code execution path is introduced.

### Task 5: Verification and stacked Draft PR

- [x] Run focused Measurement Drift, obligation, planner, materializer, Trace, Audit, report, CLI, cross-package, and hostile suites.
- [x] Run `python -m pytest -q -p no:cacheprovider`.
- [x] Run `python -m compileall -q claimci`.
- [x] Run `git diff --check b4c053da81d119cb9324e56ddecdf49e178eee4b...HEAD`.
- [x] Run scoped security review over the exact source diff.
- [ ] Commit, push, and open one Draft PR with base `codex/evidence-obligations-v1`; verify remote head/base/draft/mergeability/checks without merging.
