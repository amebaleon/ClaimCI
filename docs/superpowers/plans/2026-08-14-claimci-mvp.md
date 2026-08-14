# ClaimCI MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a deterministic `claimci audit research.yaml` CLI that audits experiment fairness, raw metrics, seed evidence, and exact train/eval leakage.

**Architecture:** Focused pure check modules return typed findings; one orchestrator loads the manifest and combines checks; separate renderers provide human and JSON output behind a thin `argparse` CLI. Verdicts derive from explicit finding impacts rather than presentation severity.

**Tech Stack:** Python 3.11+, PyYAML, pytest, standard-library `argparse`, `dataclasses`, `statistics`, `json`, `hashlib`, and `pathlib`.

## Global Constraints

- The public command is exactly `claimci audit path/to/research.yaml`, with optional `--json`.
- Use Python 3.11+ and keep runtime dependencies to PyYAML.
- Resolve artifact paths relative to `research.yaml`.
- Recompute metrics from raw runs; never trust a claimed summary blindly.
- Compare absolute improvement to `minimum_improvement`; report relative improvement separately.
- Detect exact canonical JSON overlap only; no embeddings or semantic similarity.
- Findings require stable rule ID, severity, title, explanation, and evidence.
- No web frontend, database, external API, Docker, CI service, experiment tracker, authentication, or unnecessary framework.

---

### Task 1: Package contract, models, and manifest loading

**Files:**
- Create: `pyproject.toml`
- Create: `claimci/__init__.py`
- Create: `claimci/models.py`
- Create: `claimci/audit.py`
- Test: `tests/test_manifest.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Severity`, `Impact`, `Verdict`, `Finding`, `MetricSummary`, `OverlapSummary`, `AuditResult`, `ExperimentPaths`, `ResearchSpec`, `ClaimCIError`, and `load_research_spec(path: Path) -> ResearchSpec`.
- `ResearchSpec` retains an absolute manifest path and resolved baseline/candidate artifact paths.

- [ ] Write tests that load a literal valid manifest from a temporary directory, prove paths resolve from the manifest directory, and reject a non-mapping root, missing required key, invalid metric, and negative threshold.
- [ ] Run `python -m pytest tests/test_manifest.py tests/test_models.py -q` and verify collection/import fails because the package is absent.
- [ ] Add the package metadata, typed models, and strict manifest loader with concise `ClaimCIError` messages.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Raw result and seed evidence checks

**Files:**
- Create: `claimci/result_check.py`
- Test: `tests/test_result_check.py`

**Interfaces:**
- Produces: `load_results(path, metric) -> LoadedResults`, `summarize_runs(values) -> MetricSummary`, and `check_results(baseline, candidate, metric, threshold) -> ResultCheck`.
- `ResultCheck` carries both summaries, signed absolute/relative improvement, and findings.

- [ ] Write literal tests for three-run mean/sample standard deviation, exact-threshold support, below-threshold failure despite a passing relative percentage, zero-baseline relative `None`, raw-summary mismatch, nonfinite/missing metric values, missing and duplicate seeds, one-run evidence, and 5-to-1 imbalance.
- [ ] Run `python -m pytest tests/test_result_check.py -q` and confirm expected import/attribute failures.
- [ ] Implement finite-number parsing, sample statistics using `statistics.stdev`, raw summary verification, seed findings, and absolute threshold logic.
- [ ] Re-run the focused tests and then all tests.

### Task 3: Config fairness checks

**Files:**
- Create: `claimci/config_check.py`
- Test: `tests/test_config_check.py`

**Interfaces:**
- Produces: `load_config(path) -> Mapping[str, object]` and `check_configs(baseline, candidate) -> list[Finding]`.
- Supported paths are `training_steps`, `epochs`, `batch_size`, `dataset.identifier`, `dataset.version`, the complete `evaluation` mapping, `model`, and `learning_rate`.

- [ ] Write parameterized tests proving evaluation mismatch and a 3.0x compute advantage invalidate, while model, training-dataset, learning-rate, and small training-parameter differences are reported without blanket invalidation; include matching-config verification.
- [ ] Run `python -m pytest tests/test_config_check.py -q` and confirm failure before implementation.
- [ ] Implement explicit field comparison, deterministic compute proxy/ratio evidence, and stable findings.
- [ ] Re-run focused and full tests.

### Task 4: Exact JSONL leakage checks

**Files:**
- Create: `claimci/dataset_check.py`
- Test: `tests/test_dataset_check.py`

**Interfaces:**
- Produces: `canonicalize_sample(value) -> bytes`, `hash_sample(value) -> str`, `load_dataset(path) -> DatasetHashes`, and `check_dataset_leakage(label, train, eval) -> tuple[OverlapSummary, list[Finding]]`.

- [ ] Write tests using the literal known SHA-256 for `{"id":"a","value":1}`, reordered keys/whitespace, no overlap, duplicate-aware overlap count/rate, blank lines, and malformed JSON with line-number context.
- [ ] Run `python -m pytest tests/test_dataset_check.py -q` and confirm failure before implementation.
- [ ] Implement strict JSONL parsing, canonical UTF-8 JSON hashing, `Counter` intersection, and exact leakage findings.
- [ ] Re-run focused and full tests.

### Task 5: Audit orchestration, reporting, and CLI

**Files:**
- Modify: `claimci/audit.py`
- Create: `claimci/report.py`
- Create: `claimci/cli.py`
- Test: `tests/test_audit.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `audit_research(path: Path) -> AuditResult`, `render_human(result) -> str`, `render_json(result) -> str`, and `main(argv: Sequence[str] | None = None) -> int`.
- Verdict precedence is `INVALIDATES`, then `INSUFFICIENT`, then `SUPPORTED`; exits map to `1`, `2`, and `0`.

- [ ] Write integration tests that build complete temporary artifacts and assert aggregation, no short-circuiting after an artifact error, verdict precedence, severity group rendering, JSON-native statistics/evidence, stderr errors, and CLI exits.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement orchestration that converts artifact failures into findings, deterministic renderers, CLI parsing, and the console entry point.
- [ ] Re-run focused and full tests.

### Task 6: Runnable fixtures and documentation

**Files:**
- Create: `examples/<fixture>/research.yaml` and referenced artifacts for all eight required fixtures
- Create: `README.md`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Each fixture is runnable as `claimci audit examples/<fixture>/research.yaml`.
- Fixture verdicts: valid supported; compute mismatch, data leak, false summary/unsupported raw claim, different eval dataset, and multiple failures not supported; seed mismatch and missing result insufficient evidence.

- [ ] Write a parameterized subprocess test over all eight fixture manifests, including expected exit code/verdict and a JSON parse check from a working directory outside the repository.
- [ ] Run `python -m pytest tests/test_fixtures.py -q` and confirm fixtures are missing.
- [ ] Add compact artifact fixtures and a README with install, schema, command, output, checks, exit codes, and Day 1 limitations.
- [ ] Re-run the fixture test and full suite.

### Task 7: End-to-end verification and independent review

**Files:**
- Verify all project files; change only files needed for fixes found by verification/review.

- [ ] Create a fresh virtual environment, install with `python -m pip install -e ".[dev]"`, and run `python -m pytest -q`.
- [ ] Run `claimci audit` and `claimci audit --json` against every fixture, recording exit and verdict.
- [ ] Run a packaging smoke check from outside the repository to prove the console script and relative paths work.
- [ ] Request an independent code/spec review, fix all critical or important findings with regression tests, and re-run the full verification commands.
