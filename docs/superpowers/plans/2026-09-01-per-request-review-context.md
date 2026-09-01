# Per-Request Research Review Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce `max_context_chars` independently for extraction and synthesis so two individually bounded provider requests may run even when their combined logical sizes exceed 60,000 characters.

**Architecture:** Keep the existing two-call advisory state machine and deterministic validators unchanged. Replace only the synthesis preflight's cumulative comparison with a per-request comparison, then document that the existing limit applies to each provider request.

**Tech Stack:** Python 3.11+, dataclasses, pytest, strict structured provider requests.

**Spec:** Current user request for PR #33 after passive Langfuse #16712 calibration run `33486871699` exposed the cumulative preflight.

## Global Constraints

- Keep `max_calls=2` and `max_context_chars=60000`.
- Keep the 16-claim cap, 5,000 extraction output tokens, 4,000 synthesis output tokens, output-character limits, and provider timeout design.
- Do not reduce retained claims or evidence and do not weaken citation ownership, malformed-output validation, or deterministic Audit authority.
- Rerun the passive public-only Langfuse calibration exactly once after verification; do not modify or execute Langfuse.
- Do not merge PR #33.

---

### Task 1: Enforce context independently per provider request

**Files:**
- Modify: `tests/test_review_orchestrator_day3.py`
- Modify: `claimci/review/orchestrator.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-15-claimci-day3-design.md`

**Interfaces:**
- Consumes: `_request_chars(task, payload, schema) -> int` and `ReviewLimits.max_context_chars`.
- Produces: unchanged `ResearchReview` states, with extraction and synthesis each checked independently against the same configured limit.

- [ ] **Step 1: Add the failing cumulative-boundary regression**

  Use the real `FakeProvider` fixture with `max_context_chars=4000`. Assert both completed `ProviderCallRecord.input_chars` values are individually at most 4,000, their sum is greater than 4,000, synthesis is dispatched, and the review is complete.

- [ ] **Step 2: Add fail-closed per-request characterization tests**

  Add a 60,000-character PR description test that proves extraction alone exceeds 60,000 and no provider call occurs. Add four route-valid 16,000-character evidence files so synthesis alone exceeds 60,000; assert `PARTIAL`, `CONTEXT_LIMIT`, and exactly one extraction call.

- [ ] **Step 3: Run the focused tests and verify RED**

  Run:

  ```powershell
  python -m pytest -q tests/test_review_orchestrator_day3.py -k "context_limit or context_is_bounded_per_provider_request"
  ```

  Expected: the cumulative-boundary regression fails because current code returns `PARTIAL` after one call; the two fail-closed cases pass.

- [ ] **Step 4: Implement the minimal production change**

  Change only the synthesis guard:

  ```python
  if synthesis_chars > config.limits.max_context_chars:
  ```

  Keep the existing `PARTIAL`, `CONTEXT_LIMIT`, claims, evidence, audits, and call ledger return unchanged. Clarify `_request_chars`, README, and the Day 3 design wording as per-provider-request logical context.

- [ ] **Step 5: Verify GREEN and frozen synthesis boundaries**

  Run the three context regressions plus `tests/test_review_synthesis_contract_v1.py`. Confirm exact claim coverage, dynamic per-claim schema, citation ownership, and invalid-synthesis `PARTIAL` behavior remain green.

- [ ] **Step 6: Run complete verification and publish**

  Run the affected Research Review suite, full project suite, `python -m compileall -q claimci tests`, and `git diff --check`. Independently review the diff, commit and push PR #33 without merging, then rerun passive calibration PR #35 exactly once and report its exact outcome without automatically patching any new failure.
