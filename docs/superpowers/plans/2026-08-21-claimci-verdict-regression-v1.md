# Verdict Regression v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, non-authoritative Core classifier for historical analysis transitions over exact Replay commitments.

**Architecture:** A new sealed `claimci.analysis.regression` module derives lifecycle classification from validated unified results and derives scientific comparability only from Replay verification snapshots. The output is contextual metadata and never mutates or replaces current Audit authority.

**Tech Stack:** Python 3.11+, frozen dataclasses, enums, pytest, existing ClaimCI Replay and verification contracts.

**Spec:** `docs/superpowers/specs/2026-08-21-claimci-verdict-regression-v1-design.md`

## Global Constraints

- Base exactly on Replay head `a358892bed036d6ff9439e845a63384fd00fc700`.
- Do not modify Replay construction or deterministic Audit semantics.
- A verdict/evidence regression requires actual Replay Audit commitments on both sides.
- Availability and mapping outcomes are never scientific/evidence regressions.
- Do not add D1, Web, deployment, history selection, or Stale Evidence behavior.
- Do not mutate `Finding`, `Impact`, `Verdict`, or current result authority.

---

### Task 1: Historical outcome contracts and execution classification

**Files:**
- Create: `claimci/analysis/regression.py`
- Test: `tests/test_verdict_regression_v1.py`

**Interfaces:**
- Produces `ExecutionClassification`, `OutcomeComparability`, `VerdictTransitionKind`, `RegressionClassification`.
- Produces `HistoricalAnalysisOutcome.from_unified_result(result, *, unavailable_classification=None)`.

- [x] Write RED tests for enum membership, all lifecycle mappings, typed unavailable classifications, exact-type rejection, constructor sealing, finality, and immutability.
- [x] Run `python -m pytest -q -p no:cacheprovider tests/test_verdict_regression_v1.py` and confirm failure because the module is absent.
- [x] Implement the four enums and sealed historical outcome factory with no comparison behavior.
- [x] Rerun the focused tests and keep them green.

### Task 2: Exact Replay comparability

**Files:**
- Modify: `claimci/analysis/regression.py`
- Modify: `tests/test_verdict_regression_v1.py`

**Interfaces:**
- Consumes exact `ReplayAuditCommitment`, `ReplayRecipe`, and `VerificationInputSnapshot` retained by the historical outcome.
- Produces internal deterministic comparability used by the final report.

- [x] Add RED tests for `EXACT`, `CAPTURED_SCOPE`, `CLAIM_CHANGED`, `MEASUREMENT_CHANGED`, `BASELINE_CHANGED`, `ENGINE_CHANGED`, `INDETERMINATE`, and `LEGACY_UNAVAILABLE`.
- [x] Add RED tests proving measurement source-only drift and exact engine build drift do not change comparability when semantic commitments match.
- [x] Implement ordered comparison using audit-spec, measurement semantic IDs, profile semantics, Audit semantics compatibility, comparison-frame digest, captured-input digest, and complete baseline/reference artifact semantics.
- [x] Rerun the focused tests and retain green behavior.

### Task 3: Full transition matrix and regression classification

**Files:**
- Modify: `claimci/analysis/regression.py`
- Modify: `tests/test_verdict_regression_v1.py`

**Interfaces:**
- Produces sealed `VerdictRegressionReport`.
- Produces `classify_verdict_regression(previous, current) -> VerdictRegressionReport`.

- [x] Add a table-driven RED test for every supported/not-supported/insufficient/no-verdict raw transition with hand-derived literal expected kinds.
- [x] Add RED classification tests for regression, evidence regression, coverage regression, resolution, unchanged, and not-comparable cases.
- [x] Add RED tests for completeness regression/recovery, mapping attention, all three unavailable kinds, legacy deterministic results, incompatible frames, and actual-commitment gating.
- [x] Implement minimal transition and classification tables. Override two-verdict incompatible transitions with `RAW_TRANSITION_NOT_COMPARABLE`.
- [x] Verify current `AuditResult`, findings, impacts, deterministic payload, Replay commitment, and authoritative verdict remain byte/value-identical after comparison.
- [x] Rerun the focused tests.

### Task 4: Public exports and compatibility

**Files:**
- Modify: `claimci/analysis/__init__.py`
- Modify: `tests/test_verdict_regression_v1.py`

**Interfaces:**
- Re-exports only the four enums, `HistoricalAnalysisOutcome`, `VerdictRegressionReport`, and `classify_verdict_regression`.

- [x] Add a RED public-import test.
- [x] Add the focused exports without modifying Replay or current result contracts.
- [x] Run the focused test file and adjacent Replay tests:
  `python -m pytest -q -p no:cacheprovider tests/test_verdict_regression_v1.py tests/test_verification_replay_v1.py tests/test_unified_analysis_v03.py`.

### Task 5: Verification and stacked PR

**Files:**
- Verify all scoped files and documentation.

- [x] Run `python -m pytest -q -p no:cacheprovider`.
- [x] Run `python -m compileall -q claimci`.
- [x] Run `git diff --check a358892bed036d6ff9439e845a63384fd00fc700...HEAD` after commit.
- [x] Inspect exact Stale Evidence branch/PR state and overlap; rebase/retarget only if canonical integration benefits.
- [x] Review `git status`, staged diff, and staged diff-check; stage only scoped paths.
- [x] Create one focused commit, push `codex/verdict-regression-v1`, and open a Draft PR against `codex/verification-input-replay-v1`.
- [x] Report exact head, files, tests, PR URL/base, Stale overlap result, and stop without merge.
