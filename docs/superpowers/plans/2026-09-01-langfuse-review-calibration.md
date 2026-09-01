# Langfuse Research Review Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before reporting success.

**Goal:** Make the passive Langfuse #16712 Research Review cite reported Markdown benchmark measurements, produce one valid interpretation per retained claim, and describe bounded-snapshot evidence gaps without asserting repository-wide absence.

**Architecture:** Keep `claimci.review` as a fixed two-provider-call advisory layer around the unchanged deterministic ClaimCI Audit. Add provenance and Markdown-table selection only at the advisory evidence boundary, specialize the existing synthesis response schema from trusted issued IDs, and retain the current fail-closed validator and PARTIAL state.

**Tech Stack:** Python 3.11+, dataclasses, strict JSON Schema structured responses, pytest.

**Spec:** Current-task requirements continuing PR #33 and calibrating draft PR #35 against public Langfuse PR #16712.

## Global Constraints

- The deterministic ClaimCI Audit remains authoritative; Research Review remains advisory and non-blocking.
- Preserve the 16-claim cap, 5,000-token extraction budget, 4,000-token synthesis budget, and current provider timeout design.
- Preserve path confinement, bounded source/evidence selection, claim IDs, evidence IDs, citation ownership, authority separation, malformed-output rejection, and atomic PARTIAL handling for invalid synthesis.
- Treat Markdown measurements as reported summary evidence, never as proof of execution or as raw executed results.
- Analyze only passive public Langfuse evidence; do not execute or modify the Langfuse repository.
- Do not merge or deploy.

---

### Task 1: Reported benchmark evidence and snapshot-safe gaps

**Files:**
- Create: `tests/test_review_benchmark_markdown_v1.py`
- Modify: `claimci/review/evidence.py`
- Modify: `claimci/review/orchestrator.py`
- Modify: `claimci/review/report.py`
- Test: `tests/test_review_benchmark_markdown_v1.py`
- Test: `tests/test_review_scope_v1.py`
- Test: `tests/test_review_report_day3.py`

**Interfaces:**
- Consumes: validated `ScientificClaim.source`, bounded changed paths, confined file captures, and existing content-addressed `EvidenceReference` IDs.
- Produces: provenance-labeled references for `reported_measurement`, `executable_benchmark_definition`, `executed_result_artifact`, or ordinary supporting context, plus an explicit independent-verification gap when only reported measurements/definitions are available.

- [ ] **Step 1: Write failing evidence tests**

  Add a resource-reduction claim and a changed `README.md` Markdown table containing a `4.0x` row-fold measurement. Assert that the emitted reference cites the table span, has `reported_measurement` provenance, and records that independent verification still needs an executed result artifact. Add SQL and structured-result fixtures and assert their provenance remains distinct. Add an unchanged-README case and assert PR-scoped heuristics do not issue it.

- [ ] **Step 2: Verify the evidence tests fail for the missing behavior**

  Run `python -m pytest -q tests/test_review_benchmark_markdown_v1.py` and confirm failure because Markdown tables are not routed and `EvidenceReference` has no provenance field.

- [ ] **Step 3: Implement the minimal advisory evidence change**

  Parse only bounded changed Markdown documents already eligible for source selection; recognize conventional header/separator/data-row tables; match a table to an exact source span or a claim magnitude plus metric/subject token; emit an exact-span, content-addressed reported-measurement reference. Classify SQL/source benchmark definitions and structured result artifacts separately without changing deterministic adapters or Audit inputs.

- [ ] **Step 4: Add and verify snapshot-safe wording tests**

  Assert deterministic gap descriptions say `Evidence not available in the analyzed snapshot` or `This claim could not be independently verified from the available artifacts`, the Markdown report explains the bounded-input scope, and the synthesis contract forbids repository-wide absence claims.

- [ ] **Step 5: Run focused green verification**

  Run `python -m pytest -q tests/test_review_benchmark_markdown_v1.py tests/test_review_scope_v1.py tests/test_review_report_day3.py tests/test_review_evidence_day3.py tests/test_review_orchestrator_day3.py`.

### Task 2: Synthesis schema/validator alignment

**Files:**
- Create: `tests/test_review_synthesis_contract_v1.py`
- Modify: `claimci/review/orchestrator.py`
- Test: `tests/test_review_orchestrator_day3.py`
- Test: `tests/test_review_synthesis_partial_v1.py`

**Interfaces:**
- Consumes: trusted retained claim IDs and the trusted `evidence_id -> claim_ids` ownership relation.
- Produces: a per-request strict schema whose interpretation count and allowed identifiers mirror the unchanged deterministic validator, plus safe rejection-category wording without raw provider output.

- [ ] **Step 1: Write failing boundary tests**

  Capture a two-claim synthesis request with asymmetric citation ownership. Assert `minItems == maxItems == 2`, each schema alternative fixes one issued claim ID and permits only that claim's evidence IDs, the payload exposes the same ownership directly, and an evidence-free claim can express only `citations: []`.

- [ ] **Step 2: Verify the boundary tests fail for the static schema**

  Run `python -m pytest -q tests/test_review_synthesis_contract_v1.py` and confirm the current static schema accepts arbitrary IDs/counts and has no direct ownership map.

- [ ] **Step 3: Implement the per-request schema**

  Deep-copy the existing interpretation-row schema for each trusted claim, constrain `claim_id` and citation items from the issued ownership relation, set exact row-count bounds, and pass that schema both to context accounting and `StructuredRequest`. Keep `_parse_interpretations` as the final authority and retain `SYNTHESIS_INVALID`/PARTIAL for any invalid response.

- [ ] **Step 4: Preserve safe failure diagnostics**

  Map trusted validator messages to fixed categories such as claim coverage, citation ownership, or malformed output without including provider prose or unissued attacker-controlled values.

- [ ] **Step 5: Run focused green verification**

  Run `python -m pytest -q tests/test_review_synthesis_contract_v1.py tests/test_review_orchestrator_day3.py tests/test_review_synthesis_partial_v1.py tests/test_review_hardening_day3.py`.

### Task 3: Full verification, review, PR update, and passive calibration

**Files:**
- Modify only if required by a verified regression or reviewer finding.

**Interfaces:**
- Consumes: Tasks 1-2 at the PR #33 head and the passive public files in draft PR #35.
- Produces: fresh test/compile/diff evidence, a reviewable pushed PR #33, and one final passive Langfuse calibration record.

- [ ] **Step 1: Verify all affected review behavior**

  Run `python -m pytest -q tests/test_review_*.py tests/test_analysis_review_bridge_v03.py` with PowerShell-expanded test paths, then `python -m compileall -q claimci tests` and `git diff --check origin/main...HEAD`.

- [ ] **Step 2: Independently review the complete diff**

  Review for authority separation, provenance honesty, prompt-injection/path-confinement regressions, exact ID ownership, invalid-synthesis PARTIAL behavior, and the frozen budgets/timeouts. Fix every Critical or Important finding and re-run its covering tests.

- [ ] **Step 3: Commit and push PR #33**

  Commit only the scoped implementation, tests, and this plan on `fix/review-output-budgets`; push without merging or deploying.

- [ ] **Step 4: Rerun the passive calibration**

  Use the trusted PR #33 implementation against only the passive public Langfuse #16712 snapshot. Do not execute Langfuse code. Record status, extracted/retained claims, evidence references, interpretations, missing-evidence findings, provider calls/tokens/cost, and any remaining failure.

- [ ] **Step 5: Report merge readiness**

  Report the exact requested calibration fields and whether PR #33 is ready to merge; keep PR #35 as a draft calibration fixture.
