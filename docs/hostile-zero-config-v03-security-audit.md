# ClaimCI v0.3 zero-config hostile security audit

Date: 2026-08-18

Base: `origin/main` at `c54beaf5cd06d57425eeed383942bf44392ede6f`

Scope: Discovery → Adapter → Mapping → Planner → Materializer → `run_unified_analysis()`

## Result

The audit validated two fail-closed defects and fixed both without changing the
deterministic Audit, Research Review orchestration, CLI, workflows, or provider
limits.

1. **Passive repository reads had a check/open race.** Source collection,
   evidence excerpts, and discovery artifact hashing checked a pathname for
   confinement and then reopened it by name. A controlled open-time redirection
   demonstrated that outside-root text could enter claim sources/evidence and
   that artifact metadata could be pinned to redirected bytes. The remediation
   uses one bounded descriptor capture, verifies regular-file identity before
   and after reading, rejects symlinked components, rechecks parent/root
   identities, and derives content, size, and SHA-256 from the same capture.
2. **Materialization did not independently apply the 0.90 confidence floor to
   `MANIFEST_HINT`.** The planner rejected a low-confidence hint, but a
   correctly re-identified hand-built plan could reach the deterministic Audit.
   Materialization now applies the confidence floor to every unapproved
   `MappingCandidate`; explicit `RepoMapping.approve()` remains the only trust
   transition that bypasses automatic inference rules.

Regression coverage is in `tests/test_hostile_zero_config_v03.py` and
`tests/test_ephemeral_materialize_v03.py`. Existing bounded-I/O tests were
retargeted to instrument the new descriptor capture instead of the removed
pathname digest helpers.

## Hostile coverage

| Attack surface | Result |
| --- | --- |
| Path traversal and non-portable paths | Rejected by `RepositoryPath`, evidence path validation, issued-path membership, and runtime confinement. |
| Static symlinks, link swaps, and TOCTOU | Static links remain excluded. Privilege-independent open-redirection regressions now verify source/evidence/discovery fail closed; materialization descriptor identity also rejects the substitution. |
| Artifact size/SHA drift | Discovery derives size/hash from one capture. Materialization rechecks path identity, size, SHA-256, per-file bounds, and aggregate bounds. |
| Oversized passive artifacts | Repository indexing, evidence, adapters, and materialization retain fixed file/context/aggregate limits; sparse oversized fixtures are rejected before full hashing. |
| Malformed JSON/JSONL/YAML/TOML and duplicate keys/headers | Adapter and deterministic-Audit regressions remain strict. Dataset JSONL stays byte-preserving until the existing deterministic dataset check. |
| Prompt injection in README/results/config | Repository text remains passive data. Provider output cannot create deterministic authority, and the Review remains advisory. No customer code is executed. |
| Provider mapping/split tampering | Provider proposals remain `INFERRED` with `PROVIDER_PROPOSAL` provenance and cannot auto-execute. Dataset splits require an explicit `RepoMapping.approve()` transition. |
| `MANIFEST_HINT` elevation | Hints remain unapproved inputs, require the 0.90 automatic gate, and are now independently rejected below that floor by materialization. |
| Approval bypass | `RepoMapping` is sealed, accepts exact validated types through `approve()`, and runtime bindings must remain within the approved path/kind/role/split slots. |
| Multi-claim contamination | Planning scopes artifacts, evidence, candidates, questions, and approved bindings to one selected claim; real cross-package multi-claim tests pass. |
| Dataset role/split manipulation | Split is mapping-owned, participates in mapping/plan identity, and is revalidated against the fixed passive JSONL adapter before the existing Audit. |
| Stale repository/head identity | Plan/runtime repository and head SHA mismatches fail before materialization. The hosting layer remains responsible for supplying the trusted checkout-to-head association. |
| Advisory overwrite of deterministic authority | `authoritative_verdict` remains derived only from an actual `AuditResult` snapshot; provider/advisory failures cannot replace it. |

## Verification

- Baseline before changes: `952 passed, 13 skipped`.
- Focused contracts/adapters: `272 passed`.
- Focused planner/materializer/cross-package/unified: `97 passed, 2 skipped`.
- Focused discovery/review/materialization cluster: `144 passed, 11 skipped`.
- Final full suite: `957 passed, 13 skipped`.
- `python -m compileall -q claimci`: pass.
- `git diff --check origin/main...HEAD`: pass.

## Review limitations and assumptions

- The audit was performed sequentially because this session prohibited
  subagents; scope was not reduced.
- Windows symlink creation was unavailable to the test user. The new
  open-redirection tests model the same check/use substitution directly at both
  `Path.open` and `os.open`, so they do not skip for missing symlink privilege.
- `RuntimeExecutionContext.repository`, `head_sha`, and checkout acquisition are
  trusted host inputs. The core verifies plan/runtime agreement and artifact
  bytes, but intentionally does not execute Git or customer repository code to
  rediscover the checkout commit.
- Deliberate trusted Python code using low-level reflection such as
  `object.__new__`/`object.__setattr__` is outside the passive provider/PR-data
  threat model.

ClaimCI-Web was outside scope and was not touched.
