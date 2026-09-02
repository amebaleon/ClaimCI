# Bounded External Review Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make realistic large external pull requests reach ClaimCI's existing two-call advisory Research Review through a provider-free, coordinate-bound, deterministic, bounded preflight, while preserving all existing trust, evidence, locality, cost, and Audit boundaries.

**Architecture:** Bind requested-base, explicit comparison-base, and head roots to exact Git commits; derive a complete NUL-safe metadata-only change inventory from the trusted local Git object graph; classify and rank every changed entry without scanning or comparing the unrelated tree; materialize only the bounded selected scope; prove three free gates before provider construction; then feed the resulting `SourceBundle` into the existing extraction, evidence-discovery, deterministic-audit, and synthesis flow. Keep the legacy pairwise collector unchanged for compatibility callers, and keep scope completeness separate from provider-route completeness.

**Tech Stack:** Python 3.11+, frozen dataclasses and enums, local Git plumbing invoked without a shell, bounded UTF-8/JSON handling, pytest, GitHub Actions YAML.

**Spec:** `C:\ClaimCI-External-Study-2026-09-02\report\core-remediation-design-gate.md`

## Global Constraints

- Implement on exact base `7d64a3957a3d0b0a6712f0d89d1ba6497208822c`; do not rebase this branch.
- Do not increase `MAX_REPOSITORY_PATHS=2048`, `MAX_CHANGE_COMPARISON_FILES=512`, `MAX_CHANGE_COMPARISON_FILE_BYTES=1_048_576`, the 1 MiB metadata ceiling, the 60,000-character request ceiling, 24-file/16,000-character materialization limits, 16-claim limit, 24,000-character aggregate output limit, two provider calls, or zero retries.
- Do not add a provider ranker, embedding lookup, third LLM/provider call, retry, endpoint, tool, or response schema.
- Preserve advisory-only Review, authoritative deterministic Audit, PR #36/#38 behavior, path confinement, exact claim ownership, exact-hint locality, `routing_incomplete`, `unlocalized_prefix` non-citability, Java source classification, and exact Java cardinality.
- Do not add `.sh` to `SOURCE_SUFFIXES`. Only changed shell files matching the fixed submission/job/launch/eval/benchmark runner policy may be classified as passive `SUBMISSION_CONFIG`; route them as `EvidenceKind.CONFIG` with `EvidenceProvenance.SUPPORTING_ARTIFACT`, never as source, executed output, result, or scientific proof.
- Treat inventory metadata and material seeds as routing inputs only. Neither can become a `ScientificClaim`, citation, verdict, deterministic fact, or evidence of execution.
- A provider hint cannot add a path to the inventory or issued scope. Deleted entries remain in inventory but are not head evidence.
- `COMPLETE` requires three `PASS_COMPLETE` gates, `scope.complete=True`, two valid bounded provider responses, deterministic output validation, and `routing_incomplete=False`. Any known safe omission caps the final status at `PARTIAL`. An untrustworthy or unusable scope is `UNAVAILABLE` with zero provider calls.
- Every test and replay preflight must use zero provider calls. Do not mutate external checkouts, execute repository code/workloads/benchmarks, contact maintainers, deploy, or change ClaimCI-Web.
- Write every production behavior from a failing test. Commit focused green increments. The final branch must contain one Draft PR only and must not be marked ready, merged, or deployed.

## Task interfaces

| Task | Consumes | Produces | Must not change |
|---|---|---|---|
| 1. Contracts and Git inventory | explicit roots/SHAs/basis; local Git object graph | immutable validated `ChangeInventory` and exact reason-bearing failure | legacy collector limits/behavior; repository content |
| 2. Scope planner and OpenASR taxonomy | complete inventory; PR metadata; bounded changed documents | deterministic seeds, ranked material, bounded `ReviewScope`/`SourceBundle` | conventional source suffixes; evidence authority |
| 3. Three-gate preflight | validated scope; shared request serializers; `ReviewLimits` | `ReviewPreflight`, exact projections, status ceiling | provider construction/calls; runtime envelopes |
| 4. Orchestrator and evidence routing | passed preflight and issued scope | existing two-call review with honest scope/routing state | Audit, ownership, locality, call count/retries |
| 5. Reports, CLI, and workflows | preflight result; explicit coordinates | stable schema-v2 JSON/Markdown; provider-free CLI mode; trusted CI inputs | advisory policy; external credentials before gates |
| 6. Synthetic corpus and five replays | frozen coordinate manifests and read-only checkouts | deterministic A-E regressions and replay ledger | network/provider/external repository state |
| 7. Full verification and review | complete diff and all tests | verification evidence and independent findings resolved | base commit/history |
| 8. Draft PR and founder gate | verified branch and replay result | one Draft PR; stop at approval gate | ready/merge/deploy/paid run |

---

### Task 1: Add exact coordinate contracts and a bounded Git metadata inventory

**Files:**

- Create: `claimci/review/inventory.py`
- Modify: `claimci/review/models.py`
- Modify: `claimci/review/__init__.py`
- Create: `tests/test_review_inventory_v1.py`

**Interfaces:**

- `SnapshotIdentity(role, root, sha)` binds one resolved regular directory to one full lowercase Git commit SHA.
- `build_git_change_inventory(requested_base, comparison_base, head, comparison_basis) -> ChangeInventory` performs trusted local Git plumbing only.
- `ChangeInventory.entries` is complete, canonical, strictly POSIX-sorted metadata and contains `ADDED`, `MODIFIED`, and `DELETED` entries; rename detection is disabled so a rename is delete plus add.
- Git failures are converted to structured Gate-1 issues by Task 3; exception text must not contain customer content or unbounded Git output.

- [ ] **Step 1: Add RED immutable-contract tests**

  In `tests/test_review_inventory_v1.py`, construct the public models and assert strict enum/type/value checks, immutable tuples, full lowercase 40/64-hex SHA validation, exact role validation, strict sorted unique paths, declared count equality, and `complete=True` for `trusted_git_object_graph`. Reject absolute paths, `..`, `\\`, drives, UNC paths, controls, empty components, duplicate/case-preserving aliases, unknown status, unknown basis, wrong schema, booleans-as-integers, inconsistent roots, and an incomplete trusted inventory.

  Include this exact contract shape:

  ```python
  inventory = ChangeInventory(
      schema_version=1,
      requested_base_sha=BASE,
      comparison_base_sha=MERGE_BASE,
      head_sha=HEAD,
      comparison_basis=ComparisonBasis.MERGE_BASE,
      source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
      declared_entry_count=3,
      complete=True,
      entries=(
          ChangeEntry("deleted.txt", ChangeStatus.DELETED),
          ChangeEntry("src/model.py", ChangeStatus.MODIFIED),
          ChangeEntry("tests/test_model.py", ChangeStatus.ADDED),
      ),
  )
  ```

- [ ] **Step 2: Run the contract tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_inventory_v1.py -k "contract or invalid or duplicate or ordering"
  ```

  Expected: collection fails because the new public models/module do not exist.

- [ ] **Step 3: Implement the frozen contracts**

  Add the approved `ComparisonBasis`, `ChangeStatus`, `ChangeInventorySource`, `SnapshotRole`, `SnapshotIdentity`, `ChangeEntry`, and `ChangeInventory` enums/dataclasses. Reuse the repository-path safety semantics already enforced by review models; do not normalize unsafe strings into acceptable paths. Export only the intended models and inventory builder from `claimci.review`.

- [ ] **Step 4: Add RED real-Git adapter tests**

  Create tiny local repositories in pytest using `git init`, deterministic commits, a divergent base/head graph, additions/modifications/deletions, and a rename. Assert:

  - the requested base root, comparison root, and head root each resolve to their declared `HEAD`;
  - `DIRECT_BASE` requires `comparison_base_sha == requested_base_sha`;
  - `MERGE_BASE` requires the declared comparison SHA to be the unique `git merge-base --all requested head` result;
  - two requests sharing the same head but using different requested/comparison-base coordinates produce distinct inventory identities and changed entries; full scope/materialization isolation is exercised after the planner exists in Task 6;
  - a wrong/abbreviated/missing SHA, dirty head worktree, symlinked root, missing object, multiple merge bases, or undeclared/mismatched comparison basis fails closed;
  - output is NUL-safe and filenames containing tabs/newlines are rejected by the portable path contract, not split incorrectly;
  - a rename is emitted as one deletion and one addition;
  - encoded metadata above 1 MiB or more than 8,192 entries fails before any path content is opened;
  - an external diff driver, textconv, hook, or executable file in the checkout is never invoked;
  - repeated calls return byte-for-byte equal inventory serialization.

  Monkeypatch `capture_confined_regular_file` and `Path.read_*` for worktree files to raise during inventory construction, proving the adapter uses Git object metadata and never reads repository content.

- [ ] **Step 5: Run the Git adapter tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_inventory_v1.py -k "git or merge or rename or metadata or never_reads"
  ```

  Expected: failures identify the missing Git adapter and bounds.

- [ ] **Step 6: Implement bounded no-shell Git plumbing**

  Use an argv tuple with `shell=False`, a fixed environment that disables prompts and optional locks, a short fixed timeout, and bounded stdout/stderr capture. Verify exact `HEAD` and clean index/worktree for each declared root. Use plumbing equivalent to:

  ```text
  git -c core.quotepath=false rev-parse --verify HEAD^{commit}
  git -c core.quotepath=false merge-base --all <requested-base> <head>
  git -c core.quotepath=false diff-tree -r --no-commit-id --name-status -z --no-renames --no-ext-diff --ignore-submodules=all <comparison-base> <head>
  ```

  Read at most 1 MiB plus one sentinel byte from the metadata stream, enforce 8,192 entries, parse only `A`, `M`, and `D`, require a unique merge base, and sort normalized `ChangeEntry` values by `(path, status.value)`. Do not invoke the shell and do not use `git diff` content, patches, attributes, textconv, submodules, hooks, or worktree file reads.

- [ ] **Step 7: Verify Task 1 and commit**

  ```powershell
  python -m pytest -q tests/test_review_inventory_v1.py tests/test_review_models_day3.py tests/test_review_hardening_day3.py -k "inventory or model or path or symlink or source_comparison"
  python -m compileall -q claimci tests
  git diff --check
  ```

  Commit only Task 1 files with message `Add coordinate-bound Git change inventory`.

---

### Task 2: Build deterministic bounded material scope and narrow submission-config support

**Files:**

- Create: `claimci/review/preflight.py`
- Modify: `claimci/review/models.py`
- Modify: `claimci/review/path_policy.py`
- Modify: `claimci/review/sources.py`
- Create: `tests/test_review_scope_preflight_v1.py`
- Modify: `tests/test_review_sources_day3.py`

**Interfaces:**

- `classify_review_material(path) -> ReviewMaterialKind` classifies metadata without reading content.
- `build_review_scope(inputs, config, inventory) -> ReviewScope` retains the complete inventory internally while issuing/materializing only a deterministic bounded subset.
- Existing `collect_review_sources()` keeps its signature and observable 2,048/512/1 MiB legacy behavior for auto-discovery and existing programmatic callers.
- `ReviewScope.sources` adapts exactly into the existing `SourceBundle` type; `issued_changed_paths` excludes deletions and is a subset of `issued_paths`.

- [ ] **Step 1: Add RED classification and OpenASR tests**

  Assert the fixed taxonomy for document, source, test, config, result, manifest, benchmark, submission-config, and other. Specifically assert:

  ```python
  assert classify_review_material("transformers/submit_jobs_qwen3asr.sh") is ReviewMaterialKind.SUBMISSION_CONFIG
  assert classify_review_material("scripts/launch_eval.bash") is ReviewMaterialKind.SUBMISSION_CONFIG
  assert classify_review_material("scripts/install.sh") is ReviewMaterialKind.OTHER
  assert not is_source_file("transformers/submit_jobs_qwen3asr.sh")
  ```

  Use fixed case-insensitive basename tokens `submit`, `submission`, `job`, `launch`, `eval`, `evaluation`, and `benchmark`, applied only to `.sh`/`.bash` changed paths. Arbitrary shell utilities remain `OTHER`.

- [ ] **Step 2: Add RED deterministic seed/ranking tests**

  Test bounded, normalized PR metadata and changed-document seeds with quantitative (`%`, units, ratios, durations, counts), comparative/causal, benchmark, reproducibility, and model/dataset terms. Assert prose-only seeds cannot route without a validated local changed candidate. Ranking order must be:

  1. exact safe path mention;
  2. fixed material-kind/category agreement;
  3. normalized seed/path token overlap;
  4. fixed material-kind priority;
  5. canonical POSIX path.

  Repeat with shuffled inventory input and assert identical `issued_paths`, `selected_paths`, seeds, omissions, source IDs, and total characters. Seeds remain `MaterialClaimSeed` objects and never become `ScientificClaim` or evidence references.

- [ ] **Step 3: Add RED large-repository regressions A-E**

  Build synthetic fixtures that prove:

  - **A / comparison explosion:** 513 equal-sized irrelevant eligible files plus `zz_target/benchmark.py`; only declared changed metadata is considered, and no unrelated file is opened or hashed.
  - **B / lexical disappearance:** 2,049 lexical paths precede two changed `src/models/glm/*.py` entries; both changed entries are issued deterministically.
  - **C / huge unrelated file:** an unchanged 2,187,560-byte YAML exists in the checkout; monkeypatch its open/stat/capture path to raise and prove it is never touched.
  - **D / OpenASR:** the one changed submission script routes an evaluation/config seed, but is represented only as passive supporting configuration; missing external revisions/results stay explicit.
  - **E / selected huge file:** a required changed material file above 16 MiB is recorded as `PREFLIGHT_G2_CANDIDATE_TOO_LARGE`, never fully retained, and prevents `COMPLETE` while another safe route may permit `PARTIAL`.

  Also assert deleted-only, external-only, unsafe, unreadable, undecodable, and non-localizable required routes cannot produce a false complete scope.

- [ ] **Step 4: Run scope tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_scope_preflight_v1.py tests/test_review_sources_day3.py -k "material or seed or deterministic or irrelevant or lexical or huge or openasr or deleted"
  ```

- [ ] **Step 5: Implement metadata classification and bounded selection**

  Add the approved `ReviewMaterialKind`, `MaterialClaimSeed`, `ScopeIssue`, and `ReviewScope` immutable contracts. Keep all entries in `scope.inventory`; record every omitted candidate with a stable reason/code/count. Select at most `limits.max_files`, materialize at most `limits.max_file_chars` per file and within the allocated aggregate context, and use `capture_confined_regular_file` for every content read. Do not enumerate the head tree or call legacy `_changed()` on the declared-inventory path.

  Preserve changed Markdown/`paper.tex` as extraction prose. Source/config/result/manifest/benchmark/submission-config paths are issued for routing but are not inserted into extraction as prose. Exact local path literals may add deterministic supplements only when confined, regular, bounded, and already authorized by the declared scope; they cannot expand raw inventory based on provider text.

- [ ] **Step 6: Keep legacy collector compatibility explicit**

  Do not replace `collect_review_sources()`. Add a separate scoped helper that produces the same `SourceRecord`/`SourceBundle` objects from `ReviewScope`. Run `test_auto_discovery_v03.py` and assert its first-2,048 behavior remains frozen. Where a legacy preflight adapter is used, record traversal/comparison exhaustion as explicit incomplete/failure codes instead of claiming complete scope.

- [ ] **Step 7: Verify Task 2 and commit**

  ```powershell
  python -m pytest -q tests/test_review_scope_preflight_v1.py tests/test_review_sources_day3.py tests/test_review_hardening_day3.py tests/test_auto_discovery_v03.py
  python -m compileall -q claimci tests
  git diff --check
  ```

  Commit Task 2 with message `Plan bounded material scope before review`.

---

### Task 3: Implement the three provider-free gates with shared request accounting

**Files:**

- Modify: `claimci/review/preflight.py`
- Create: `claimci/review/request_budget.py`
- Modify: `claimci/review/orchestrator.py`
- Modify: `claimci/review/models.py`
- Create: `tests/test_review_preflight_v1.py`
- Modify: `tests/test_review_output_budget_v1.py`
- Modify: `tests/test_review_orchestrator_day3.py`

**Interfaces:**

- `preflight_review(inputs, config) -> ReviewPreflight` has no provider parameter/import and never constructs or calls one.
- `ReviewPreflight.gates` always contains Gates 1, 2, and 3 in order. Upstream failure yields downstream `NOT_EVALUATED` with `PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE`.
- Shared pure builders/serializers calculate the exact runtime extraction size and a conservative worst-valid synthesis reserve under the unchanged contracts.
- `ready_for_provider=True` only when all gates are `PASS_COMPLETE` or `PASS_PARTIAL`.

- [ ] **Step 1: Add RED gate-state and reason-code tests**

  Exercise every approved Gate-1, Gate-2, and Gate-3 reason from the design gate. Assert each issue serializes as `{code, path?, observed?, limit?}`, generic `PREFLIGHT_FAILED` never appears, metrics have deterministic key ordering, and the status ceiling is exactly:

  ```text
  any FAIL          -> UNAVAILABLE / ready_for_provider false
  any PASS_PARTIAL  -> PARTIAL / ready_for_provider true
  all PASS_COMPLETE -> COMPLETE / ready_for_provider true
  ```

  Include wrong requested/comparison/head identity, direct/merge-base mismatch, absent inventory, invalid path/status/schema, metadata/count bound, no seed, no route, only deleted routes, unsupported type, selection truncation, unreadable/oversized/locality failure, call limit below two, extraction context overflow, synthesis reserve overflow, file/source-char overflow, and output reserve failure.

- [ ] **Step 2: Add RED zero-provider sentinels**

  Monkeypatch `OpenAIReviewerProvider` with a constructor that raises `AssertionError`, pass no injected provider, and call `run_review()` for representative failures of all three gates. Assert `UNAVAILABLE`, empty calls/usage, structured preflight output, and zero factory constructions. Call `preflight_review()` directly for the same boundaries while guarding the provider module/factory with import and construction sentinels; because the preflight API has no provider parameter or import, any provider access is itself a test failure.

- [ ] **Step 3: Add RED request-projection boundaries**

  Compare Gate 3's extraction projection to the exact `StructuredRequest` logical character count built by runtime for the same scope. Test 59,999/60,000/60,001 boundaries, 23/24/25 files, 15/16/17 claims, 15,999/16,000/16,001 source characters, and the 24,000 aggregate output reserve. Construct a case where extraction fits but the worst-valid synthesis envelope does not; assert call 1 never happens.

  The synthesis reserve must account for the largest valid bounded serialization of:

  - `max_claims` accepted claims;
  - per-claim dynamic schema/ownership constraints;
  - citable selected evidence and missing-evidence records;
  - deterministic audit snapshots/bundles;
  - policy/contract/schema/wrapper overhead.

- [ ] **Step 4: Run Gate tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_preflight_v1.py tests/test_review_output_budget_v1.py tests/test_review_orchestrator_day3.py -k "preflight or projection or context or provider_factory"
  ```

- [ ] **Step 5: Implement gates and shared request builders**

  Implement immutable `GateDisposition`, `PreflightGateResult`, and `ReviewPreflight`. Refactor extraction/synthesis payload/schema construction only enough that preflight and runtime call the same pure serializer and overhead builders. Planner trimming must be deterministic and must add omissions before final Gate-3 evaluation; never trim a sole route and then pass. Keep runtime context checks as defense in depth.

- [ ] **Step 6: Verify Task 3 and commit**

  ```powershell
  python -m pytest -q tests/test_review_preflight_v1.py tests/test_review_output_budget_v1.py tests/test_review_synthesis_contract_v1.py tests/test_review_orchestrator_day3.py
  python -m compileall -q claimci tests
  git diff --check
  ```

  Commit Task 3 with message `Gate providers behind bounded preflight`.

---

### Task 4: Integrate preflight scope with existing orchestration and evidence locality

**Files:**

- Modify: `claimci/review/orchestrator.py`
- Modify: `claimci/review/evidence.py`
- Modify: `claimci/review/models.py`
- Modify: `tests/test_review_orchestrator_day3.py`
- Modify: `tests/test_review_evidence_day3.py`
- Modify: `tests/test_review_exact_hint_locality_v1.py`
- Modify: `tests/test_review_java_routing_v1.py`
- Modify: `tests/test_review_cardinality_guard_v1.py`

**Interfaces:**

- `ResearchReview.preflight` carries immutable gate/scope data through every post-preflight return.
- `run_review()` evaluates preflight before default provider construction and consumes only `preflight.scope`/its bounded `SourceBundle` on the coordinate-bound path.
- `discover_evidence()` receives issued paths, selected paths, priority paths, changed paths, and the explicit comparison root without broadening its trust boundary.

- [ ] **Step 1: Add RED state-machine tests**

  Assert ordering with trace hooks:

  ```text
  preflight -> provider factory -> extraction -> evidence -> synthesis
  ```

  No provider object may be constructed for a failing preflight. Passing complete scope preserves the existing two calls and can return `COMPLETE`; passing partial scope executes at most two calls but must end `PARTIAL` even when provider output and routing are otherwise perfect. Disabled review remains `DISABLED` without preflight or provider work. Legacy inputs remain compatible and are explicitly marked `legacy_pairwise_v1`.

- [ ] **Step 2: Add RED evidence-routing regressions**

  Prove:

  - an exact provider hint outside `scope.issued_paths` is never opened and keeps `routing_incomplete=True`;
  - exact hints to changed, issued CONFIG/RESULT/MANIFEST/BENCHMARK/SUBMISSION_CONFIG artifacts may use the narrow existing exact-path route after all membership/confinement checks;
  - the OpenASR submission script yields only `EvidenceKind.CONFIG` plus `SUPPORTING_ARTIFACT`;
  - arbitrary `.sh` hints fail route matching;
  - failed changed-region localization stays `UNLOCALIZED_PREFIX`, is excluded from citable synthesis evidence, and cannot satisfy required scope;
  - evidence IDs stay claim-owned and synthesis cannot cite another claim's evidence;
  - `.java` remains SOURCE and exact cardinality still requires complete, claim-owned, complete-file Java test evidence.

- [ ] **Step 3: Run integration tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_orchestrator_day3.py tests/test_review_evidence_day3.py tests/test_review_exact_hint_locality_v1.py tests/test_review_java_routing_v1.py tests/test_review_cardinality_guard_v1.py -k "preflight or scope or exact or shell or locality or cardinality or java"
  ```

- [ ] **Step 4: Implement the narrow integration**

  Add preflight to `_result()` and all returns after preflight. Move only the default provider factory below a ready preflight. Keep the existing provider schemas and call sequence. Feed manifest reservation and audit plans through the issued scope without exceeding its bounds. Generalize the exact changed-artifact bypass to the approved supporting-artifact kinds while retaining membership, confinement, changed-only, ownership, and locality validation. Keep `EvidenceBundle.routing_incomplete` independent from `ReviewScope.complete`.

- [ ] **Step 5: Verify Task 4 and commit**

  ```powershell
  python -m pytest -q tests/test_review_orchestrator_day3.py tests/test_review_evidence_day3.py tests/test_review_scope_v1.py tests/test_review_exact_hint_locality_v1.py tests/test_review_java_routing_v1.py tests/test_review_cardinality_guard_v1.py tests/test_review_synthesis_contract_v1.py tests/test_review_synthesis_partial_v1.py
  python -m compileall -q claimci tests
  git diff --check
  ```

  Commit Task 4 with message `Integrate bounded scope with evidence review`.

---

### Task 5: Expose stable preflight reports, CLI controls, and trusted workflow coordinates

**Files:**

- Modify: `claimci/review/report.py`
- Modify: `claimci/cli.py`
- Modify: `claimci/review/__init__.py`
- Modify: `.github/workflows/claimci.yml`
- Modify: `.github/workflows/claimci-external.yml`
- Modify: `templates/claimci-external.yml.tmpl`
- Modify: `docs/external-install-v0.md`
- Modify: `tests/test_review_report_day3.py`
- Modify: `tests/test_review_cli_day3.py`
- Modify: `tests/test_review_workflow_day3.py`
- Modify: `tests/test_external_install_v0.py`

**Interfaces:**

- Additive CLI inputs: `--requested-base-root`, `--comparison-base-root`, `--requested-base-sha`, `--comparison-base-sha`, `--comparison-basis`, and `--head-sha`.
- `--preflight-only` renders the same stable preflight-bearing JSON/Markdown without loading credentials, constructing a provider, or entering extraction.
- Review report schema becomes version 2 and includes coordinates, inventory source/count/completeness, each gate, scope mode/completeness/selected/issued/omitted metrics, projection metrics, and final status ceiling. Provider usage remains zero for preflight-only/failure.

- [ ] **Step 1: Add RED report-schema tests**

  Assert deterministic schema-v2 JSON and Markdown for complete, partial, failed, and not-evaluated gates. Sort coordinates/issues/metrics explicitly. The Markdown section must distinguish `scope_complete` from `routing_incomplete` and label submission scripts as passive supporting configuration. Update fallback JSON literals to schema 2. Existing review status/policy/blocking keys remain present and advisory.

- [ ] **Step 2: Add RED CLI tests**

  Assert all six coordinate arguments are required as a group on the official declared path, direct/merge-base semantics are explicit, `--preflight-only` never invokes `run_review` provider work, event JSON remains capped at 1 MiB, and JSON plus Markdown rendering does not rerun preflight. Invalid coordinate/inventory states render structured unavailable preflight instead of leaking Git output. Legacy CLI use without the coordinate group remains supported but explicitly uses the legacy mode.

- [ ] **Step 3: Add RED workflow tests**

  Assert both workflow copies/template:

  - use exact `github.event.pull_request.base.sha` as requested base and exact `github.event.pull_request.head.sha` as head;
  - fetch enough trusted Git history to derive a unique merge base;
  - record/pass the explicit comparison SHA/root and direct-versus-merge basis;
  - create a separate comparison checkout when it differs from requested base;
  - invoke `claimci review --preflight-only` before the paid review step;
  - expose `OPENAI_API_KEY` only to the later paid step;
  - do not execute PR code or install dependencies from the PR checkout;
  - retain `continue-on-error`, advisory reporting, trusted config sourcing, passive head checkout, and Audit/Review separation.

- [ ] **Step 4: Run presentation tests and verify RED**

  ```powershell
  python -m pytest -q tests/test_review_report_day3.py tests/test_review_cli_day3.py tests/test_review_workflow_day3.py tests/test_external_install_v0.py
  ```

- [ ] **Step 5: Implement schema-v2 reports and CLI**

  Keep config schema version 1. Bump only the review output schema to 2, update every checked-in fallback, and document the additive contract. Make `--preflight-only` call `preflight_review()` directly and adapt it to a zero-call `ResearchReview` for shared rendering. Never require or read an API key in this mode.

- [ ] **Step 6: Implement trusted workflow sequencing**

  Generate the merge base from the local object graph, validate the full SHA, determine basis explicitly, materialize the comparison checkout, run the free preflight artifact step, and condition the paid step on a ready preflight result. Do not cache or pass untrusted executable output as inventory. The paid invocation repeats/consumes the same bound coordinates and cannot silently substitute `github.sha` for the requested base.

- [ ] **Step 7: Verify Task 5 and commit**

  ```powershell
  python -m pytest -q tests/test_review_report_day3.py tests/test_review_cli_day3.py tests/test_review_workflow_day3.py tests/test_external_install_v0.py tests/test_hostile_zero_config_v03.py
  python -m compileall -q claimci tests
  git diff --check
  ```

  Commit Task 5 with message `Expose free preflight in reports and CI`.

---

### Task 6: Freeze A-E regression manifests and replay the five external checkouts

**Files:**

- Create: `tests/fixtures/review-preflight-v1/A-InferenceX-2630.json`
- Create: `tests/fixtures/review-preflight-v1/B-Transformers-48455.json`
- Create: `tests/fixtures/review-preflight-v1/C-Weave-7801.json`
- Create: `tests/fixtures/review-preflight-v1/D-OpenASR-205.json`
- Create: `tests/fixtures/review-preflight-v1/E-NVCF-1425.json`
- Create: `tests/test_review_external_regressions_v1.py`
- Create: `scripts/replay_review_preflight.py`

**Interfaces:**

- Checked-in fixtures contain compact immutable coordinate/change metadata only, never full external content.
- The replay script accepts `C:\ClaimCI-External-Study-2026-09-02`, reads the frozen local inputs/checkouts, creates verified disposable snapshots outside that tree, calls `preflight_review()` only, and writes/prints a deterministic local result without provider/network activity.

- [ ] **Step 1: Add the compact frozen manifests**

  Record the exact requested base, comparison base, head, basis, and complete A/M/D path list for:

  ```text
  A bdca939fe5ca04cf23bb06cbc83cbdfe95030203 -> bdca939fe5ca04cf23bb06cbc83cbdfe95030203 -> c38fb02378565de853fed5860c7459473f655a83
  B 8e35c981dbdb2caf75b341023e243d834f829b2b -> 8e35c981dbdb2caf75b341023e243d834f829b2b -> 7d6d2ca28ac3be64204d49b89cf0651bd8ed923f
  C 625a8241cd46898f86b8290a937f3f1d3ef71729 -> c31ce92835cd2415bbb3c7361e756b12ce2133f9 -> a1c2bc8e82f96980651de1c6eb8b331cb804a144
  D d1e99b25524814332d6868a5645e568670834cfb -> d1e99b25524814332d6868a5645e568670834cfb -> 9e7a2dfeda0eb3284a87a0105e8dedd45f7967d6
  E 72485b3e08a08461474fef5d0977456a6544e73a -> 812a162ad964c077357a86a69f8f08ba11f87d9b -> c2a1dd54782e17a2eb443b428cd01eb9964abdd1
  ```

  Derive these fixtures only from the frozen local Git object graphs and cross-check counts against `preflight-results.json`: 58, 2, 12, 1, and 10 PR/effective changed files respectively.

- [ ] **Step 2: Add RED regression-corpus tests**

  Load each manifest into a synthetic minimal tree and assert deterministic coordinate isolation, no path disappearance, no naive comparison reads, no unrelated huge-file access, OpenASR supporting-config classification, required-large-file incompleteness, explicit merge-base mismatch failures for C/E, maximum two provider calls after a passed gate, and zero provider calls during every preflight. Add a dedicated same-head regression: build two histories with one identical head object but different requested/comparison-base tuples, run them consecutively, and assert different inventory identities, changed entries, selected/materialized paths, serialized preflights, and cache keys (or assert the implementation has no cross-call cache).

- [ ] **Step 3: Implement the provider-free replay tool**

  The script must refuse network URLs and dirty/mismatched roots, use the exact frozen input descriptions/capture metadata, report all three gates and issues per candidate, count only `PASS_COMPLETE`/`PASS_PARTIAL` as ready, and exit nonzero when fewer than three of five are ready. It must not import a provider class, read `OPENAI_API_KEY`, write into the external-study tree, execute repository files, or start workloads.

  For every candidate, create a `tempfile.TemporaryDirectory` outside the study tree. From the frozen local object source (`paid/<candidate>/head` for B/C/E and `phase1/<candidate>/repo` for A/D), run a local-only `git clone --no-local --no-hardlinks --no-checkout` into the temporary directory, then create detached requested-base, explicit comparison-base, and head worktrees inside that temporary clone. C and E therefore receive real comparison roots at `c31ce92835cd2415bbb3c7361e756b12ce2133f9` and `812a162ad964c077357a86a69f8f08ba11f87d9b` without changing the frozen source repository's `.git`. Reject any source that is not a local filesystem path. Before and after replay, record and compare the frozen source's `HEAD`, porcelain status, tracked-file identity summary, and `.git` metadata status; tests must prove only the temporary clone changes and all three disposable roots resolve to the declared SHAs. Cleanup is confined to the resolved temporary directory.

- [ ] **Step 4: Verify fixtures and replay logic**

  ```powershell
  python -m pytest -q tests/test_review_external_regressions_v1.py tests/test_review_preflight_v1.py tests/test_review_scope_preflight_v1.py
  python scripts/replay_review_preflight.py --study-root 'C:\ClaimCI-External-Study-2026-09-02' --output "$env:TEMP\claimci-preflight-replay.json"
  ```

  Inspect the output and verify `provider_calls=0` for A-E. If fewer than three candidates are ready, stop implementation acceptance and report the exact gates; do not start any paid run or create the PR. If at least three are ready, continue verification but still do not start a paid run.

- [ ] **Step 5: Commit Task 6**

  ```powershell
  git diff --check
  git status --short
  ```

  Commit Task 6 with message `Add frozen external preflight regressions`.

---

### Task 7: Run exhaustive verification and independent review

**Files:**

- Review: every changed file against the approved design and frozen base
- Update only implementation/tests/docs when a verified finding requires it

- [ ] **Step 1: Run focused review/preflight suites**

  ```powershell
  python -m pytest -q tests/test_review_inventory_v1.py tests/test_review_scope_preflight_v1.py tests/test_review_preflight_v1.py tests/test_review_external_regressions_v1.py
  python -m pytest -q tests/test_review_sources_day3.py tests/test_review_hardening_day3.py tests/test_review_scope_v1.py tests/test_review_evidence_day3.py tests/test_review_exact_hint_locality_v1.py tests/test_review_java_routing_v1.py tests/test_review_cardinality_guard_v1.py
  python -m pytest -q tests/test_review_orchestrator_day3.py tests/test_review_synthesis_contract_v1.py tests/test_review_synthesis_partial_v1.py tests/test_review_output_budget_v1.py tests/test_review_report_day3.py tests/test_review_cli_day3.py
  python -m pytest -q tests/test_auto_discovery_v03.py tests/test_hostile_zero_config_v03.py tests/test_review_workflow_day3.py tests/test_external_install_v0.py
  ```

- [ ] **Step 2: Run full repository verification**

  ```powershell
  python -m pytest -q
  python -m compileall -q claimci tests scripts
  git diff --check 7d64a3957a3d0b0a6712f0d89d1ba6497208822c..HEAD
  git status --short
  git rev-parse HEAD
  git merge-base HEAD 7d64a3957a3d0b0a6712f0d89d1ba6497208822c
  ```

  Require the full test suite to pass, compileall to succeed, no diff whitespace errors, a clean worktree, and merge base exactly `7d64a3957a3d0b0a6712f0d89d1ba6497208822c`.

- [ ] **Step 3: Run independent code/protocol review**

  Give the reviewer the exact design gate, frozen base, full diff, test evidence, and replay ledger. Require explicit findings for scope/provenance/lifecycle behavior, provider-zero ordering, inventory completeness, selection determinism, context reservation, report contract, OpenASR authority, PR #36/#38, Java/cardinality, workflow safety, and unchanged caps. Resolve every blocker/high-confidence issue through the same red-green discipline, rerun the affected focused tests, then rerun the full suite.

- [ ] **Step 4: Record final verification commit if needed**

  Commit only actual review fixes or generated fixture corrections with a descriptive message. Do not create an empty cleanup commit.

---

### Task 8: Publish one Draft PR and stop at the founder gate

**Files:**

- No additional source changes expected
- PR body must link the plan/spec and summarize exact verification/replay evidence

- [ ] **Step 1: Audit branch history and remote state**

  ```powershell
  git status --short
  git log --oneline --decorate 7d64a3957a3d0b0a6712f0d89d1ba6497208822c..HEAD
  git diff --stat 7d64a3957a3d0b0a6712f0d89d1ba6497208822c..HEAD
  gh pr list --head codex/bounded-external-review-scope --state all
  ```

  Require a clean branch, focused commits, no unexpected file, no pre-existing PR for this head, and no history rewrite/rebase.

- [ ] **Step 2: Push and create exactly one Draft PR**

  Push `codex/bounded-external-review-scope`, then create one Draft PR against the repository's frozen-base line with title `Bound external Research Review scope before providers`. The body must include the blocker reconstruction, architecture, unchanged caps/invariants, tests, exact five-candidate preflight table, zero-provider attestation, report/API change, and separately approved ClaimCI-Web follow-up.

- [ ] **Step 3: Verify Draft state and stop**

  Read back the PR URL/state and verify it is Draft. Do not mark it ready, merge it, deploy it, start a paid external run, contact maintainers, or mutate the frozen study checkouts.

  If at least three of five replays are ready, report that the implementation acceptance threshold is met and stop for explicit founder approval before any paid run. If fewer than three are ready, report the blocker table and stop without a PR, as required by Task 6.
