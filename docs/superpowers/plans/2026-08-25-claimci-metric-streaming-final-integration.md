# ClaimCI Metric + Streaming Final Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay Metric Identity V1 and Streaming Artifact V1 onto the security-remediated Core main, prove the cross-role table-selector remediation remains authoritative, publish one Draft Core integration PR, and repin the existing Hosted Draft PR to the new exact Core head.

**Architecture:** The Core integration branch starts at exact current `origin/main` commit `b7d1771bad6c3b699994e41b3da23640887c7b66`. It replays only the Metric delta `5068fb528b1af88393f110946ff13434c3800a15..381d2237e14e3f0ab997e11988fa747c777d955d` and Streaming delta `381d2237e14e3f0ab997e11988fa747c777d955d..e3dcb9ba8c8e4a06d3a32b22a00aefe692ea1a43`, retaining current-main conflict semantics. Hosted logic remains intact and receives only the new Core pin/gitlink plus any strictly necessary API compatibility correction.

**Tech Stack:** Python 3.11–3.13 production contract, pytest, Ruff, Git worktrees, GitHub CLI, Docker Linux/amd64, Node.js 22, Vitest.

**Spec:** `docs/superpowers/specs/2026-08-24-claimci-metric-identity-v1-design.md`; `docs/superpowers/specs/2026-08-25-claimci-streaming-artifact-v1-design.md`; `docs/superpowers/specs/2026-08-21-claimci-benchmark-evidence-profile-v0-design.md`.

## Global Constraints

- Preserve Core PR #27 remediation commit `33ab96c00e1136d6ee86cdd8602e5a832e3f330b` and merge commit `b7d1771bad6c3b699994e41b3da23640887c7b66`.
- Do not rewrite or merge Core PR #28 or #29.
- Do not mutate Core `main`, ClaimCI-Web `main`, staging, production, D1, Web UI, Cloud API, or deployment resources.
- Cross-role baseline/candidate table bindings may share a physical path only when target columns differ or shared typed predicates prove row disjointness.
- Boolean/String and Number/String lexical aliases, subset/superset predicates, and overlapping streaming selectors fail closed.
- `AuditResult` remains the sole verdict authority; provider output cannot create source, selector, candidate, binding, completeness, or verdict authority.
- Run the complete Core suite exactly once after focused integration is green.
- Keep both pull requests Draft and stop before merge.

---

### Task 1: Replay Metric Identity V1 on the Remediated Base

**Files:**
- Replay: `claimci/analysis/__init__.py`
- Replay and resolve: `claimci/analysis/contracts.py`
- Replay: `claimci/analysis/materialize.py`
- Create from feature delta: `claimci/analysis/metric_identity.py`
- Replay: `claimci/analysis/planner.py`, `claimci/audit.py`
- Create from feature delta: `tests/test_metric_identity_v1.py`

**Interfaces:**
- Consumes: current-main `_validate_bindings()` and `_table_selectors_are_row_disjoint()` security guard.
- Produces: `MetricCandidate`, `MetricIdentity`, `MetricBinding`, `resolve_metric_binding()`, `integrate_metric_binding()`, and exact-head materialization revalidation.

- [ ] **Step 1: Verify immutable ancestry before replay**

  Run:

  ```powershell
  git rev-parse HEAD
  git merge-base --is-ancestor 33ab96c00e1136d6ee86cdd8602e5a832e3f330b HEAD
  git status --short
  ```

  Expected: head `b7d1771bad6c3b699994e41b3da23640887c7b66`, ancestry exit `0`, clean worktree except this committed plan.

- [ ] **Step 2: Replay the exact Metric commit**

  Run:

  ```powershell
  git cherry-pick 381d2237e14e3f0ab997e11988fa747c777d955d
  ```

  If `contracts.py` conflicts, retain `_TABLE_NUMBER`, `_table_number_text()`, `_table_predicates_may_match_same_cell()`, `_table_selectors_are_row_disjoint()`, and the `overlapping_physical_cells` rejection while adding Metric types/imports. Never select the old feature side wholesale.

- [ ] **Step 3: Verify Metric plus remediation together**

  Run:

  ```powershell
  python -m pytest -q tests/test_metric_identity_v1.py tests/test_benchmark_table_adapter_v0.py -p no:cacheprovider
  ```

  Expected: all Metric Identity and all 24 remediation tests pass.

---

### Task 2: Replay Streaming Artifact V1 Without Weakening Selector Authority

**Files:**
- Create: `claimci/analysis/artifact_source.py`, `claimci/analysis/adapters/streaming.py`, `claimci/analysis/streaming_dataset.py`
- Replay and resolve: `claimci/analysis/contracts.py`, `claimci/analysis/materialize.py`, `claimci/analysis/metric_identity.py`, `claimci/analysis/planner.py`, `claimci/audit.py`
- Replay: structured, tabular, dataset, registry, discovery, measurement, trace, passive-file, and dataset-check modules named by the Streaming delta.
- Create: `tests/test_streaming_source_v1.py`, `tests/test_streaming_jsonl_v1.py`, `tests/test_streaming_tabular_v1.py`, `tests/test_streaming_dataset_v1.py`, `tests/test_streaming_materialize_v1.py`, `tests/test_streaming_security_v1.py`.

**Interfaces:**
- Consumes: Task 1 exact `MetricBinding` and current-main cross-role table-selector validation.
- Produces: factory-only `ArtifactSource`, immutable `ScanCompleteness`, bounded JSONL/CSV/TSV scans, scratch-confined dataset spill, and streaming-aware materialization.

- [ ] **Step 1: Replay the exact Streaming commits in order**

  Run:

  ```powershell
  git cherry-pick 9fc62456a71f3e4b0a7515ab8c4c891e4e74c766 faf7ad50ec74abf441d2bdc033f509157a86901f 00bdaa9ed66cca143ff7b4dadb1fc0fb523d985c 51cd8a3f3c640e1a66e74925e6cface4b49b3bfb 8c03f7ab86c9dca0c2a8b34da0198f09bc9e2a12 6280ea5351fa213ec98dfa527ed73d8438be91fe 369fd308558d9b069f95fb3d1690d3488483f8e4 e3dcb9ba8c8e4a06d3a32b22a00aefe692ea1a43
  ```

  Resolve conflicts by preserving the Task 1 Metric APIs and current-main table overlap checks, then adding only Streaming source/completeness behavior.

- [ ] **Step 2: Run the focused Streaming and compatibility matrix**

  Run:

  ```powershell
  python -m pytest -q tests/test_streaming_source_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_tabular_v1.py tests/test_streaming_dataset_v1.py tests/test_streaming_materialize_v1.py tests/test_streaming_security_v1.py tests/test_metric_identity_v1.py tests/test_benchmark_table_adapter_v0.py -p no:cacheprovider
  ```

  Expected: all source, integrity, completeness, spill, binding, Metric, and remediation cases pass.

---

### Task 3: Add the Combined Metric/Streaming Security Regression Matrix

**Files:**
- Create: `tests/test_metric_streaming_security_integration.py`
- Modify only if a new test exposes a real gap: `claimci/analysis/contracts.py` or `claimci/analysis/metric_identity.py`.

**Interfaces:**
- Consumes: `artifact_source_from_snapshot()`, `extract_metric_candidate_scan()`, `resolve_metric_binding()`, `approve_metric_binding()`, `integrate_metric_binding()`, `TableSelector`, and typed `TablePredicate`.
- Produces: direct release regression coverage showing Metric and Streaming cannot bypass `_validate_bindings()`.

- [ ] **Step 1: Add exact combined tests**

  Create helpers that issue an exact `ArtifactSource` from a temporary CSV/TSV, construct fixed adapter `TableSelector` matches, extract one baseline and one candidate `MetricCandidate`, approve the one `[acc]` pair, and integrate it into a same-path `MappingCandidate`.

  Add these tests:

  ```python
  def test_same_path_overlapping_metric_binding_fails_closed(): ...
  @pytest.mark.parametrize("suffix", ["csv", "tsv"])
  def test_streaming_metric_binding_cannot_reintroduce_cross_role_overlap(...): ...
  @pytest.mark.parametrize("typed_value,string_value", [(True, "true"), (1, "1"), (1, "1.0"), (1, "1e0")])
  def test_typed_lexical_aliases_remain_overlapping(...): ...
  def test_subset_superset_predicates_remain_overlapping(): ...
  def test_disjoint_role_predicates_stream_and_bind_successfully(): ...
  def test_atomic_acc_pair_keeps_two_exact_streaming_endpoints(): ...
  ```

  Each rejection must assert `AnalysisContractError` with `conflicting baseline`; the success cases must assert distinct candidate IDs, exact role-specific predicates, `[acc]`, and one integrated baseline plus candidate binding.

- [ ] **Step 2: Run the new tests before any corrective production edit**

  Run:

  ```powershell
  python -m pytest -q tests/test_metric_streaming_security_integration.py -p no:cacheprovider
  ```

  If a test exposes a bypass, preserve the failing output, make the smallest contracts-layer correction, and rerun. Do not add a Streaming special case.

- [ ] **Step 3: Run the combined focused gate and commit**

  Run:

  ```powershell
  python -m pytest -q tests/test_metric_streaming_security_integration.py tests/test_benchmark_table_adapter_v0.py tests/test_metric_identity_v1.py tests/test_streaming_tabular_v1.py tests/test_streaming_security_v1.py -p no:cacheprovider
  git add tests/test_metric_streaming_security_integration.py claimci/analysis/contracts.py claimci/analysis/metric_identity.py
  git commit -m "test: preserve table selector security across streaming metrics"
  ```

  Stage production files only when the tests required a real correction.

---

### Task 4: Recertify Core and Create One Draft Integration PR

**Files:**
- Verify every changed Core source/test/doc file.
- No deployment or release configuration changes.

**Interfaces:**
- Consumes: completed integrated Core tree.
- Produces: one immutable integrated head and one Draft Core PR against `main`.

- [ ] **Step 1: Run the one complete Core suite and static gates**

  Run once:

  ```powershell
  python -m pytest -q -p no:cacheprovider
  python -m compileall -q claimci
  python -m ruff check claimci tests
  git diff --check b7d1771bad6c3b699994e41b3da23640887c7b66..HEAD
  ```

- [ ] **Step 2: Run the immutable security diff review**

  Scan exact range `b7d1771bad6c3b699994e41b3da23640887c7b66..HEAD`, review every changed production file, validate any candidates, and require zero unresolved reportable findings.

- [ ] **Step 3: Push and create exactly one Draft Core PR**

  Push `codex/metric-streaming-release-integration`, create a Draft PR against `main`, reference PRs #27/#28/#29, and verify the PR head equals `git rev-parse HEAD`.

---

### Task 5: Repin and Recertify the Existing Hosted Draft PR

**Files:**
- Modify: `C:/ClaimCI-Web-worktrees/hosted-streaming-companion-v1/runner/pyproject.toml`
- Update gitlink: `C:/ClaimCI-Web-worktrees/hosted-streaming-companion-v1/runner/vendor/ClaimCI`
- Modify only if the integrated Core API requires it: existing runner source/tests already changed by PR #45.
- Read, and modify only if factually required: currently merged Pilot Data Handling Notice.

**Interfaces:**
- Consumes: exact integrated Core head fixed by Task 4.
- Produces: PR #45 at a new exact head with unchanged Hosted authority semantics.

- [ ] **Step 1: Update only the exact Core pin and vendored gitlink**

  Set `$integratedHead=(git -C C:/ClaimCI-worktrees/metric-streaming-release-integration rev-parse HEAD).Trim()`, replace the old `e3dcb9...` dependency pin with `$integratedHead`, fetch the vendored repository, and checkout `$integratedHead` in the gitlink without rewriting Hosted logic.

- [ ] **Step 2: Run Hosted under the production-supported runtime**

  Build/use the repository production Linux/amd64 Python 3.13 container path. Inside it run the complete runner suite including `RuntimeCaTrustTest.test_ssl_cert_file_preserves_verification_for_urllib_and_openai`. A failure in that runtime is `RELEASE_BLOCKER`.

- [ ] **Step 3: Run Node 22 cross-runtime verification**

  Use Node 22, set `CLAIMCI_PINNED_CORE_PATH` to `runner/vendor/ClaimCI`, and run:

  ```powershell
  npx.cmd vitest run --config src/usage/vitest.config.ts src/analysis/cross-runtime.integration.test.ts
  ```

- [ ] **Step 4: Run focused Hosted gates and inspect the Data Notice**

  Run Metric mapping, large JSONL, large CSV/TSV, stale binding, incomplete scan, integrity failure, compileall, changed-package lint/typecheck, and `git diff --check`. Confirm the notice still truthfully says snapshots/scratch are temporary and full large artifacts are not sent to OpenAI; leave it unchanged if truthful.

- [ ] **Step 5: Security-review the minimal old-PR-head to new-PR-head diff**

  Review exact range `c787acbd96d8caf304740c3c4d800bccb6cc8d31..HEAD`; require zero unresolved reportable findings.

- [ ] **Step 6: Push PR #45 and keep it Draft**

  Push only `codex/hosted-streaming-companion-v1`, update the PR body with the exact Core pin, production-runtime runner result, Node 22 result, security result, and no-deployment statement, then verify PR #45 remains Draft.

---

### Task 6: Final Provenance Check and Stop

**Files:**
- No further source mutation.

**Interfaces:**
- Consumes: both Draft PR heads and sealed verification evidence.
- Produces: final report; no merge or deployment.

- [ ] **Step 1: Verify exact remote heads and lifecycle state**

  Confirm the Core Draft PR and ClaimCI-Web PR #45 are open Drafts, both local branches are clean and match their remote heads, and neither repository main branch was mutated locally.

- [ ] **Step 2: Report readiness truthfully**

  Report Core base/head, #27 preservation, Metric/Streaming results, full Core count, security result, Hosted old/new head, Core pin, Linux/Python 3.13 runner/TLS result, Node 22 cross-runtime result, Data Notice decision, and both Draft PR URLs. State `METRIC + STREAMING INTEGRATION = READY FOR MERGE` only when every required gate is green.
