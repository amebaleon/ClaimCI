# ClaimCI Day 3 Implementation Plan

Date: 2026-08-15

## Goal

Ship a strictly two-call, opt-in, evidence-grounded LLM research reviewer while
preserving the deterministic ClaimCI auditor as the sole blocking authority.

## Wave 1: Claim and source contracts, test-first

1. Add failing tests for review-only enums/dataclasses, configuration defaults,
   strict opt-in, supported claim types, exact source quotes/locations,
   confidence/magnitude validation, and deterministic stable IDs.
2. Add failing source-collection tests for PR title/body, changed README and
   Markdown notes, `paper.md`, `paper.tex`, base/head hashing, selection limits,
   decoding failures, and ignored unchanged/unrelated files.
3. Implement `claimci.review.models`, `config`, and `sources` without importing
   deterministic authority types into the review model layer.
4. Run the focused tests, then the complete existing suite.

## Wave 2: Evidence discovery and deterministic tools, test-first

1. Add failing tests for claim-type routing, path metadata heuristics,
   deterministic ordering, bounded excerpts/hashes, and selective content.
2. Add adversarial path tests for traversal, absolute/alternate separators,
   NUL, missing, directory, special file, symlink escape, and nonexistent
   provider hints.
3. Add failing tests for manifest discovery and the one-way adapter from actual
   `AuditResult` objects to immutable deterministic tool snapshots.
4. Assert that review/provider response models cannot construct or mutate
   `Verdict`, `Finding`, `Severity`, `Impact`, thresholds, or deterministic
   evidence.
5. Implement `claimci.review.evidence` and `tools` using only trusted Python and
   existing ClaimCI functions.
6. Run the focused tests, then the complete suite.

## Wave 3: Provider and two-call orchestration, test-first

1. Add a fake provider and failing tests proving the exact sequence:
   extraction call -> deterministic discovery/tools -> synthesis call.
2. Assert no third call, no retries, no filesystem/provider tools, and that
   lowering a call budget stops safely.
3. Add strict response tests for malformed JSON, duplicate keys, non-finite
   values, deep structures, wrong/extra fields, hallucinated quotes, invented
   authority fields, unknown claim/evidence IDs, and oversized output.
4. Add call/context/output/file/timeout boundaries and per-call/aggregate
   token/cost observability tests. Assert usage cannot affect review status or
   deterministic results.
5. Implement the provider-neutral protocol, strict parsers, fixed orchestrator,
   and lazy OpenAI Responses API adapter with injectable mock client, zero
   retries, no tools, and environment-only credentials.
6. Run the focused tests, then the complete suite. Do not make a live API call.

## Wave 4: Review views, CLI, and GitHub opt-in, test-first

1. Add failing JSON/Markdown tests for advisory labeling, claim-by-claim
   sections, exact deterministic snapshots, evidence citations, missing and
   unsupported inference, usage fields, escaping, bounds, ordering, and parity.
2. Add failing CLI tests for default disabled, enabled mocked review, one
   invocation producing both views, missing key/SDK/provider errors, controlled
   streams, and unchanged `audit` behavior.
3. Add workflow/static tests proving trusted-base configuration, explicit
   opt-in, provider-key isolation from Check publication, passive PR data,
   exactly one review invocation/two provider calls, separate neutral advisory
   Check, and untouched deterministic gate enforcement.
4. Implement renderers, CLI dispatch, advisory GitHub payload, optional workflow
   step, configuration example, and user documentation including private-data
   egress.
5. Run the focused tests, then the complete suite.

## Wave 5: Adversarial hardening

Add a failing regression before every real fix found in:

- prompt injection and policy override text in PR metadata, Markdown, TeX,
  configs, results, paths, and provider output;
- hallucinated claims, citations, metrics, magnitudes, and finding IDs;
- malformed manifests and missing artifacts reached through review routing;
- weird metrics, Unicode/console behavior, Markdown/JSON consistency, duplicate
  files/content, case collisions, path separator differences, and symlinks;
- flat/deep/large repository trees and budget boundary off-by-one cases;
- provider timeout/refusal/quota/missing usage/inconsistent totals/cost overflow;
- missing configuration/key/SDK and GitHub advisory publication failures;
- unchanged deterministic mode, fixtures, exits, reports, and Check mapping.

Run the complete suite after every fix wave. Do not add unrelated product
features.

## Wave 6: Independent review and release evidence

1. Ask an independent read-only reviewer to audit scope, deterministic
   authority, type separation, path/data-egress security, prompt injection,
   provider budgets, GitHub isolation, reporting accuracy, and product-goal
   alignment.
2. Reproduce each real finding with a failing test, implement the smallest fix,
   and rerun the complete suite after each review-fix wave.
3. Run the final full suite, compile the package, exercise the disabled/no-key
   deterministic path, exercise review end-to-end with a fake provider, verify
   all existing fixtures, and statically validate the workflow.
4. Do not run a live paid API smoke test unless the user explicitly requests
   that separate action.
5. Report total passing tests, architecture, supported claim types, authority
   separation, API configuration, estimated and observed mocked token envelope,
   remaining limitations, and the next highest-value feature.
