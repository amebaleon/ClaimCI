# Scientific Claim Type System v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical source-bound claim taxonomy and deterministic compiler seam that executes only metric-improvement claims and returns an exact pre-Audit partial for recognized unsupported claims.

**Architecture:** A new focused claim-types module recovers a canonical primary claim and orthogonal constraints from `ClaimReference.text`. Discovery carries that semantic object, planning compiles only metric improvement, and Evidence Trace records the semantic projection without giving it authority or changing plan identity.

**Tech Stack:** Python 3.11+, frozen dataclasses, enums, pytest, existing ClaimCI analysis and Evidence Trace contracts.

**Spec:** `docs/superpowers/specs/2026-08-20-claimci-scientific-claim-types-v0-design.md`

## Global Constraints

- Base exactly on Evidence Trace PR #16 head `2c4a1c329bfe75a189af38537d7da3d877545cfb`.
- Preserve `AuditResult`, CLI behavior, existing metric-improvement execution, optional `research.yaml`, LLM non-authority, and passive-only repository handling.
- Do not implement Evidence Obligations or Measurement Drift.
- Unsupported deterministic claim compilation returns exact reason `unsupported_deterministic_claim_compiler` before mapping or Audit.
- `HELD_OUT` does not change `AuditClaimSpec` or plan identity in v0.

---

### Task 1: Canonical claim contracts and classifier/compiler

**Files:**
- Create: `claimci/analysis/claim_types.py`
- Modify: `claimci/analysis/__init__.py`
- Modify: `claimci/review/analysis_bridge.py`
- Test: `tests/test_claim_types_v0.py`
- Test: `tests/test_analysis_review_bridge_v03.py`

**Interfaces:**
- Produces immutable `PrimaryClaimKind`, concrete primary records, `EvaluationConstraint`, `CanonicalScientificClaim`, and `ClaimEvidencePolicy`.
- Produces `recover_scientific_claim(reference)`, `claim_evidence_policy(claim)`, `compile_audit_claim(claim)`, and a stable semantic trace projection.

- [x] Write contract, precedence, hostile provider-provenance, ambiguity, compiler, and serialization tests.
- [x] Run the focused test and confirm failures are caused by missing interfaces.
- [x] Implement the minimal frozen contracts, bounded recognizer, policy seam, and source-recovering compiler.
- [x] Run the focused test to green and refactor without changing behavior.

### Task 2: Discovery and deterministic planning integration

**Files:**
- Modify: `claimci/analysis/discovery/models.py`
- Modify: `claimci/analysis/discovery/claims.py`
- Modify: `claimci/analysis/planner.py`
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/integration.py`
- Test: `tests/test_auto_discovery_v03.py`
- Test: `tests/test_ephemeral_planner_v03.py`
- Test: `tests/test_unified_analysis_v03.py`

**Interfaces:**
- Discovery carries source-recovered canonical semantics independently of provider fields.
- `PlanningRequest` carries canonical claim/policy and an optional compiled audit claim.
- Unsupported primaries return exact pre-Audit partial before evidence or mapping selection.
- Ready `EphemeralAuditPlan` carries canonical semantics while plan identity remains audit-relevant only.

- [x] Write failing discovery tests for held-out precedence, absolute/generalization/generic recognition, and provider field invention.
- [x] Implement discovery integration and rerun focused tests to green.
- [x] Write failing planner/unified tests for unsupported exact partial, no mapping/Audit, metric compatibility, and constraint-insensitive plan identity.
- [x] Implement optional compiler handling and exact partial propagation.
- [x] Run all discovery/planner/unified focused tests to green.

### Task 3: Evidence Trace semantic provenance

**Files:**
- Modify: `claimci/analysis/materialize.py`
- Test: `tests/test_ephemeral_materialize_v03.py`
- Test: `tests/test_evidence_trace_v1.py`

**Interfaces:**
- Natural-language claim trace carries a bounded canonical semantic projection.
- `HELD_OUT` remains visible as a constraint while record authority remains non-authoritative.

- [x] Write failing trace tests for HELD_OUT retention, unchanged plan identity, and absence of verdict authority.
- [x] Add the canonical projection to the existing natural-language claim trace entry.
- [x] Run trace/materializer tests to green.

### Task 4: Full verification and stacked Draft PR

**Files:**
- Review every changed source, test, specification, and plan file.

- [ ] Run `python -m pytest -q -p no:cacheprovider`.
- [ ] Run `python -m compileall -q claimci`.
- [ ] Run `git diff --check 2c4a1c329bfe75a189af38537d7da3d877545cfb...HEAD` after commit.
- [ ] Review `git status`, staged diff, and exact base ancestry.
- [ ] Commit the focused change and push `codex/scientific-claim-types-v0`.
- [ ] Open one Draft PR against `feat/evidence-trace-v1` and verify its base/head/mergeability without merging.
