# ClaimCI Evidence Trace and Provenance/Authority v1 Implementation Plan

**Goal:** Add a Core-owned immutable evidence trace companion to the unchanged
deterministic `AuditResult`, then carry a bounded additive projection through
the Hosted runner, Worker codec, and existing D1 persistence.

**Core base:** `9140d73f2eeeb9d396382e39bd036e8877c2a66f`

**Web base:** `ad4ff4cd2b7f23e5bcd03780781c98bcd1356079`

**Branches:** `feat/evidence-trace-v1` in ClaimCI and ClaimCI-Web

## Locked invariants

- `AuditResult`, `Verdict`, `render_json()`, Audit rules, CLI output/exit
  behavior, and deterministic authority remain unchanged.
- Provider output is a non-authoritative proposal and cannot construct passive
  evidence, deterministic derivation, mapping approval, or verdict authority.
- Only exact-head, confined, hash-bound, freshly adapter-validated evidence can
  enter passive trace records.
- `RepoMapping.approve()` remains the only explicit mapping trust transition
  and never becomes scientific/verdict authority.
- No arbitrary source dump, dataset row persistence, repository code execution,
  network expansion, workflow change, deployment, staging mutation, or secret
  change.
- GitHub Check and Dashboard public projections remain unchanged.

## Task 1: Core immutable trace contracts

1. Add RED tests for record-kind/authority compatibility, deep immutability,
   finite JSON safety, canonical commitments, structural limits, and the
   16,384-byte bundle ceiling.
2. Add `claimci/analysis/trace.py` with the frozen value objects and strict
   serialization helpers.
3. Export the additive public contracts from `claimci.analysis`.
4. Add hostile tests proving provider provenance cannot instantiate passive or
   deterministic trace records and approval cannot instantiate verdict
   authority.
5. Run focused contract tests and commit.

## Task 2: Exact materialization trace

1. Add RED tests around the existing no-manifest materializer for exact head,
   path, hash, adapter/version, selectors, roles/splits, scalar commitments,
   derived evidence, and actual rule IDs.
2. Refactor the existing execution body without changing its ordering or native
   files so it returns `EphemeralAuditExecution(AuditResult, trace)`.
3. Build trace only from `_evidence_by_binding()` output, freshly captured
   `PassiveArtifact` values, the selected mapping, and the actual Audit result.
4. Keep `execute_ephemeral_audit()` as a compatibility wrapper returning only
   the same Audit result.
5. Represent over-bound/detail failures with explicit completeness rather than
   changing verdict authority.
6. Run materialization, planner, Audit, report, CLI, and compatibility tests;
   commit an exact tested Core SHA.

## Task 3: Unified Core orchestration

1. Add RED tests proving deterministic COMPLETE and deterministic PARTIAL paths
   carry trace while mapping-needed/unavailable paths never fabricate one.
2. Add the optional final trace field to `UnifiedAnalysisResult` and preserve
   positional/source compatibility.
3. Attach deterministic authority only from the actual Audit result and attach
   Research Review only as an advisory trace record.
4. Assert existing `render_json()` fixture bytes, verdicts, and exit semantics
   are unchanged.
5. Run the full Core suite and compileall; commit and push only after green.

## Task 4: Stack ClaimCI-Web on the exact Core commit

1. Update the runner Core git pin, pin tests, cross-runtime harness, and
   verification documentation to the exact tested Core commit.
2. Capture RED runner contract/projection tests before production edits.
3. Add strict Python parsing/projection for the optional trace with the same
   authority matrix, structural limits, and 16,384-byte ceiling.
4. Dynamically verify the full 65,536-byte Hosted result. Emit a bounded trace
   projection when needed; preserve a deterministic outcome in typed PARTIAL if
   even the minimal trace representation cannot fit.
5. Add real no-`research.yaml` cross-runtime tests proving complete trace and
   unchanged deterministic verdict.

## Task 5: Worker validation and D1 round-trip

1. Add RED TypeScript contract tests for complete/bounded traces, authority
   confusion, duplicate/unknown keys, invalid hashes/selectors/rules, non-finite
   values, and UTF-8 over-bound payloads.
2. Add strict trace types/parser to `cloud/src/analysis/contracts.ts`.
3. Persist the trace inside existing `deterministic_json`; add repository
   round-trip and corruption regressions without a migration.
4. Prove old trace-absent result rows remain valid.
5. Prove public API, Check renderer, Check reconciliation, and Dashboard
   projection omit trace and remain byte/semantically unchanged where expected.

## Task 6: Full verification and Draft PRs

Run, at minimum:

```text
# Core
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check origin/main...HEAD

# Hosted runner, pinned to exact Core commit
python -m pytest -q runner/tests -p no:cacheprovider
python -m compileall -q runner

# ClaimCI-Web Cloud
npm test
npm run lint
npm run typecheck
npm run build

# ClaimCI-Web root
npm run verify

git diff --check origin/main...HEAD
```

Also run focused real cross-runtime no-manifest tests, hostile provider-authority
tests, traced-versus-compatibility Audit byte comparisons, and an independent
security/trust-boundary review. Push and open linked Draft PRs only after every
required gate is green. Do not merge or deploy.
