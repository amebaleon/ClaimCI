# ClaimCI v0.3 Passive Dataset Identity Implementation Plan

**Goal:** Replace the test-only dataset identity stub with a fixed passive JSONL
adapter and split-bearing repository mappings that feed exact bytes to the
existing deterministic Audit.

**Base:** `4095970b528a1deaeaa7b4a3c201ac2e9f8ee6a9` (PR #13)

**Branch:** `feat/dataset-identity-v03`

**PR base:** `feat/ephemeral-plan-v03` while PR #13 is unmerged.

## Locked invariants

- No repository code execution, network access, dataset download, row
  reinterpretation, new dataset policy, provider authority, or workflow/CLI
  change.
- `DatasetSplit` is mapping metadata, never a synthetic content selector.
- Dataset adapter evidence remains role/split-agnostic and passive.
- Only clear deterministic inference or explicit `RepoMapping.approve()` can
  supply executable role/split identity.
- Captured JSONL bytes reach the existing Audit unchanged.

## Task 1: Split-bearing immutable contracts

1. Add failing tests for `DatasetSplit`, dataset-only split validation,
   immutability/serialization, approval scoping, and split-sensitive identity.
2. Add `DatasetSplit` and an optional final `dataset_split` field to
   `ArtifactBinding`, preserving positional compatibility.
3. Include split in mapping signatures and exact approved runtime scoping.
4. Run focused contract/planner tests and commit.

## Task 2: Fixed passive dataset adapter

1. Add failing tests for exact `.jsonl`/`DATASET` gating, size/hash/path
   revalidation, identity-only output, empty mappings, unspecified role/split,
   and absence of parsing/execution/network behavior.
2. Implement `claimci/analysis/adapters/dataset.py` and export/register
   `claimci-jsonl-dataset-v1` in fixed order before the generic JSONL adapter.
3. Add a fixed-registry `extract_registered_artifact()` helper with controlled
   first-match behavior.
4. Update registry/security contracts without weakening existing checks.
5. Run adapter-focused tests and commit.

## Task 3: Preserve split through Discovery and planning

1. Add failing tests proving clear four-slot dataset mapping, manifest split
   preservation, provider split remaining untrusted, and one same-slot dataset
   question with role/split-bearing literal choices.
2. Preserve `DatasetSplit` in inferred, manifest, provider, choice, scope,
   signature, and plan-ID paths.
3. Make executable mapping coverage require exactly one train and one eval
   binding per baseline/candidate role.
4. Keep absent datasets typed `PARTIAL`; return a resolvable upstream dataset
   question as `MAPPING_NEEDED` without creating authority.
5. Run discovery/planner-focused tests and commit.

## Task 4: Runtime revalidation and exact materialization

1. Add failing tests for split/path/kind/adapter/hash drift, fresh registered
   adapter revalidation, unchanged byte copying, malformed JSONL reaching the
   existing Audit, and non-JSONL `PARTIAL` behavior.
2. Pair each selected dataset binding with its captured passive artifact.
3. Re-run the fixed dataset adapter over captured bytes and compare the exact
   planned evidence before fixed-name copy.
4. Select `train.jsonl`/`eval.jsonl` only from the validated binding split.
5. Run materializer/Audit-focused tests and commit.

## Task 5: Remove the fake and verify the hosted path

1. Replace `_dataset_evidence` in the cross-package test with the registered
   adapter/helper.
2. Exercise Discovery -> registered adapters -> planning -> exact materializer
   -> existing Audit -> one-call advisory Review with no `research.yaml`.
3. Add hostile/provider and ambiguity regressions needed by the design.
4. Run focused cross-package and unified tests.
5. Run the full gate:

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

6. Independently review scope, deterministic authority, passive-data handling,
   mapping approval, lifecycle cleanup, and public interfaces.
7. Push `feat/dataset-identity-v03` and open a draft PR against
   `feat/ephemeral-plan-v03`. Do not merge.
