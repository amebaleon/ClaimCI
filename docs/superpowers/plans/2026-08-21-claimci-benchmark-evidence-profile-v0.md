# Benchmark Evidence Profile v0 Implementation Plan

> **Execution:** use test-driven development for each task and verification-before-completion before publishing.

**Goal:** Add a bounded Benchmark evidence-obligation family and native Audit path without changing Training behavior or deterministic authority.

**Base:** exact Benchmark Table Adapter head `7f8a65041809612578a4c41c6e0522ab046229fb`.

**Stack base for PR:** `codex/benchmark-table-adapter-v0`.

**Spec:** `docs/superpowers/specs/2026-08-21-claimci-benchmark-evidence-profile-v0-design.md`.

## Task 1: Profile contracts and deterministic selection

Files:
- Create `claimci/analysis/profiles.py`
- Create `tests/test_benchmark_evidence_profile_v0.py`
- Modify `claimci/analysis/__init__.py`

1. Add RED tests for fixed enums, registry immutability, final/factory-only records, provider-shaped construction rejection, metric routing, positive Benchmark facts, legacy Training selection, missing-data non-inference, profile ambiguity, and approval-not-profile-authority.
2. Implement the fixed definitions, bounded selection facts/outcome, profiled policy factory, canonical IDs, and deterministic selection.
3. Prove performance metrics with no Benchmark facts select incomplete Benchmark rather than Training.

## Task 2: Additive profile obligations

Files:
- Modify `claimci/analysis/obligations.py`
- Modify `claimci/analysis/contracts.py`
- Modify `tests/test_evidence_obligations_v1.py`
- Modify `tests/test_benchmark_evidence_profile_v0.py`

1. Add RED tests preserving exact `ArtifactEvidenceSlot` validation and legacy bundle bytes.
2. Add `ProfileEvidenceTarget` and factory-only `ProfileEvidenceSupport` with fixed role/support kinds and exact selected-binding provenance.
3. Add profiled assessment while retaining the existing Training entrypoint unchanged.
4. Parameterize all variants/result forms and prove the maximum activated bundle is 13, never above the existing 16 cap.

## Task 3: CSV/TSV profile component projection

Files:
- Modify `claimci/analysis/adapters/tabular.py`
- Modify `tests/test_benchmark_table_adapter_v0.py`

1. Add RED tests for exact-row selected profile scalar cells, no row-position/first-match/type inference, selector conflicts, and no raw-row persistence.
2. Permit only fixed profile component target fields, preserve scalar text, and bind every selected field to the same exact table predicates.
3. Prove one physical table safely yields distinct baseline/candidate selector-scoped evidence variants.

## Task 4: Profile-aware planning

Files:
- Modify `claimci/analysis/planner.py`
- Modify `claimci/analysis/contracts.py`
- Modify `tests/test_ephemeral_planner_v03.py`
- Modify `tests/test_benchmark_evidence_profile_v0.py`

1. Add RED lifecycle tests for Benchmark partial/ready, semantic profile ambiguity, indirect mapping clarification, provider-only approval, reference evidence, aggregate missing metadata, and unsupported formulas.
2. Select mappings against the fixed selected profile; rerun profile selection after approval.
3. Build Benchmark plans with profiled policy and reference evidence. Keep the exact Training plan-ID path unchanged; commit profile identity only for Benchmark.

## Task 5: Exact-head materialization, Measurement Drift, and native Audit

Files:
- Create `claimci/benchmark_audit.py`
- Modify `claimci/audit.py`
- Modify `claimci/measurement.py`
- Modify `claimci/analysis/measurement.py`
- Modify `claimci/analysis/materialize.py`
- Modify `tests/test_ephemeral_materialize_v03.py`
- Create bounded synthetic fixtures/tests in `tests/test_benchmark_evidence_profile_v0.py`

1. Capture RED end-to-end tests for raw runs, reported aggregate, fixed cost derivation, metric mismatch, workload/config mismatch, insufficient procedure, source drift, selector/hash drift, and passive-only execution.
2. Implement one dispatcher: Training delegates to unchanged `audit_research`; Benchmark consumes fresh normalized evidence and returns `AuditResult` through `determine_verdict()`.
3. Reuse Measurement Drift with profile policy identity and fixed subject/procedure scope. Never put claim direction/threshold or comparison subject into semantic protocol equality.
4. Model VESSL runtime/cost, Unsloth throughput/VRAM, Cerebrium cost derivation, kernel timing, and model-quality quality fixtures without network access.

## Task 6: Trace and compatibility

Files:
- Modify `claimci/analysis/materialize.py`
- Modify `claimci/analysis/trace.py` only if public classification contracts require it
- Modify `tests/test_evidence_trace_v1.py`, `tests/test_audit.py`, `tests/test_report.py`, `tests/test_cli.py`, `tests/test_analysis_cross_package_v03.py`

1. Add RED classification tests for every new `BENCHMARK.*` rule and public rule-fact compatibility.
2. Emit bounded passive and derived trace entries without complete table rows, raw source, or provider prose.
3. Prove existing Training Audit, manifest, CLI, report, plan identity, and obligation serialization remain unchanged.

## Task 7: Hostile verification and stacked Draft PR

1. Run focused profile, table adapter, Measurement Drift, obligation, planner, materializer, Audit, Trace, report, CLI, and hostile suites.
2. Run `python -m pytest -q -p no:cacheprovider`.
3. Run `python -m compileall -q claimci`.
4. Run `git diff --check 7f8a65041809612578a4c41c6e0522ab046229fb...HEAD`.
5. Perform a scoped security review of authority, selectors, passive parsing, formula handling, exact-head revalidation, and resource bounds.
6. Commit, push `codex/benchmark-evidence-profile-v0`, and open one Draft PR based on `codex/benchmark-table-adapter-v0`. Verify remote base/head/draft/mergeability/checks. Do not merge or deploy.
