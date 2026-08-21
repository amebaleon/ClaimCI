# Stale Evidence Detection v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded deterministic comparator for previous Replay verification inputs versus a current exact-head snapshot without changing Audit execution or authority.

**Architecture:** A new pure `claimci.analysis.staleness` module validates exact existing Replay/snapshot contracts, compares semantic and source identities independently, and returns immutable metadata. It consumes optional exact issued artifact identities only for relocation; it has no provider, Audit, Trace, or Web integration path.

**Tech Stack:** Python 3.11 standard library, frozen dataclasses, existing ClaimCI verification/replay contracts, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-claimci-stale-evidence-v1-design.md`

## Global Constraints

- Base exactly `a358892bed036d6ff9439e845a63384fd00fc700`.
- No new runtime dependency.
- Do not modify parent branches or implement Web/D1 history.
- Never create a Finding, Impact, Verdict, or Audit-skip authority.
- Provider/LLM output cannot establish identity equivalence or relocation.
- Existing scheduled Audit always continues.
- Benchmark table selectors remain exact and selector-scoped.

---

### Task 1: Immutable states and assessment contract

**Files:**
- Create: `claimci/analysis/staleness.py`
- Create: `tests/test_stale_evidence_v1.py`

**Interfaces:**
- Produces the three canonical enums, fixed reason codes, `ArtifactRelocation`, and constructor-sealed `StaleEvidenceAssessment`.
- Produces `assess_stale_evidence(previous_recipe, current_snapshot, *, issued_candidates=())`.

- [ ] Write failing import and constructor-sealing tests with literal expected enum values.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/test_stale_evidence_v1.py` and confirm failure because the module is absent.
- [ ] Implement only immutable enums/records, exact-type validation, canonical digest helpers, and the legacy/unavailable outcomes.
- [ ] Re-run the focused tests and keep them green.

### Task 2: Semantic and source comparison

**Files:**
- Modify: `claimci/analysis/staleness.py`
- Modify: `tests/test_stale_evidence_v1.py`

**Interfaces:**
- Internal comparison classifies claim/profile/obligation/measurement/compatibility and artifact changes without reading repository files.

- [ ] Add RED tests for unrelated head/base changes, claim statement/spec changes, profile/obligation semantic changes, measurement semantic/source separation, and Core build provenance changes.
- [ ] Implement minimal root comparison and state precedence.
- [ ] Add RED tests for source-byte-only change, order-insensitive row reordering, extraction-only uncertainty, Audit-semantic change, unversioned projector coverage, and identity-only snapshots.
- [ ] Implement exact artifact pairing and conservative material/indeterminate/byte-only rules.
- [ ] Run focused tests after each RED/GREEN cycle.

### Task 3: Selector-scoped comparison and relocation

**Files:**
- Modify: `claimci/analysis/staleness.py`
- Modify: `tests/test_stale_evidence_v1.py`

**Interfaces:**
- Relocation accepts only exact `VerificationArtifactIdentity` values and returns bounded `ArtifactRelocation` records.

- [ ] Add RED Benchmark tests proving changed baseline/candidate table selectors are material even with identical physical bytes.
- [ ] Implement exact selector/binding comparison.
- [ ] Add RED tests for unique relocation, ambiguous relocation, absent equivalent evidence, and provider-shaped candidate rejection.
- [ ] Implement kind/role/split/Audit-semantic candidate search, compatibility checks, and required actions.
- [ ] Run the focused staleness and Replay/Benchmark suites.

### Task 4: Public exports and compatibility

**Files:**
- Modify: `claimci/analysis/__init__.py`
- Modify: `tests/test_stale_evidence_v1.py`

**Interfaces:**
- Publicly exports only approved metadata contracts and `assess_stale_evidence`.

- [ ] Add a RED public-import test and assertions that no finding/verdict/skip fields exist.
- [ ] Add focused exports in `claimci.analysis.__init__`.
- [ ] Verify legacy Replay, Trace, Measurement, Profile, Obligation, Audit, report, and CLI tests remain unchanged.

### Task 5: Full verification and stacked Draft PR

**Files:**
- Review only the files above plus this spec/plan.

- [ ] Run `python -m pytest -q -p no:cacheprovider`.
- [ ] Run `python -m compileall -q claimci`.
- [ ] Run `git diff --check a358892bed036d6ff9439e845a63384fd00fc700...HEAD`.
- [ ] Review `git status`, staged diff, and exact base ancestry.
- [ ] Commit focused changes, push `codex/stale-evidence-v1`, and open a Draft PR with base `codex/verification-input-replay-v1`.
- [ ] Report exact head/base, tests, changed files, and stop before merge.
