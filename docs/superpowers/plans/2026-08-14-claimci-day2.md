# ClaimCI Day 2 Implementation Plan

Date: 2026-08-14

## Goal

Ship pull-request integration, higher/lower metric direction, and a polished rejected-improvement demo while preserving every Day 1 contract and test.

## Wave 1: Direction, test-first

1. Add failing model/manifest tests for `Direction`, omitted-default behavior, lower parsing, and invalid values.
2. Add failing result tests for higher/lower exact boundaries, declines, relative values, zero/negative baselines, overflow, and the unchanged four-argument API.
3. Add failing audit/report tests proving direction propagation, lower wording, deterministic JSON, and native units.
4. Implement the smallest changes in `models.py`, `audit.py`, `result_check.py`, and `report.py`.
5. Run the focused tests, then the complete suite.

## Wave 2: Markdown and GitHub adapter, test-first

1. Add failing tests for `render_markdown`, semantic parity with JSON, escaping, verdict drivers, and deterministic bytes.
2. Add failing CLI tests for `--markdown`, mutual exclusion, output streams, and unchanged exit codes.
3. Add failing adapter tests for `success`, `failure`, `action_required`, malformed/unknown input, summary bounds, head SHA, and JSON safety.
4. Implement `render_markdown`, CLI selection, and `claimci.github` payload generation.
5. Add a static workflow contract test, then create `.github/workflows/claimci.yml` with minimal permissions, a trusted-base auditor, a passive and path-confined PR checkout, job summary, Check publication for every PR, and explicit verdict enforcement.
6. Run the focused tests, then the complete suite.

## Wave 3: Demo, test-first

1. Add a failing integration test for the exact demo metrics, finding IDs/evidence, verdict, exit code, JSON, and Markdown.
2. Create the self-contained `examples/day2_demo` manifest and artifacts.
3. Add a root `research.yaml` that makes this repository’s own workflow auditable without changing its default configuration.
4. Document exact 60-second commands and screenshot views.
5. Run the focused tests, every named fixture, then the complete suite.

## Wave 4: Adversarial hardening

Add focused regressions before each fix for real gaps found in:

- malformed and unusual direction values;
- huge, tiny, negative, zero, non-finite, and bool metrics;
- missing, unreadable, directory, NUL, and outside-working-directory artifacts;
- malformed YAML/JSON/JSONL and duplicate-aware data overlap;
- Markdown table/code injection and JSON/Markdown numeric consistency;
- unknown verdicts, truncated summaries, missing output files, and Check publication prerequisites;
- default-higher and existing fixture compatibility;
- false positives from cross-experiment shared training data or permitted model/learning-rate differences.

After each fix wave, run the full suite. Do not add unrelated product features.

## Wave 5: Independent review and release evidence

1. Ask an independent read-only reviewer to audit scope, behavior, provenance boundaries, GitHub security, and product-goal alignment.
2. Reproduce and fix any real issue with a failing regression first.
3. Run `python -m pytest -q -p no:cacheprovider`, compile the package, run all fixtures, run the demo in human/Markdown/JSON modes, and validate the workflow YAML/payload locally without a network mutation.
4. Report the final test count, integration status, exact demo steps, screenshot views, remaining limitations, and the single next highest-value feature.
