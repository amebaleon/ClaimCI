# ClaimCI External-Repository Installation v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a same-owner/private, full-SHA-pinned reusable GitHub workflow
that runs ClaimCI from trusted external source while keeping consumer PR code
passive.

**Architecture:** A root composite action carries and installs the immutable
ClaimCI package. A reusable multi-job workflow preserves deterministic Audit,
optional two-call Review, and isolated neutral publication. A deterministic
release renderer stamps the installer SHA, while a demo materializer produces
a clean consumer repository fixture.

**Tech Stack:** GitHub Actions YAML, Python 3.11, PyYAML, argparse, pytest.

## Global Constraints

- Pilot consumers are private repositories under the same GitHub owner as the
  private ClaimCI repository.
- Every external ClaimCI workflow/action reference is a full immutable
  40-character commit SHA; no branch, tag, short SHA, or runtime ref input.
- Consumer PR head content is never installed, imported, built, sourced, or
  executed.
- `ClaimCI Audit` remains the sole blocking scientific authority.
- `ClaimCI Research Review` remains completed/neutral and advisory.
- Review configuration comes only from the consumer base SHA.
- Review makes at most two provider calls with zero retries.
- `OPENAI_API_KEY` is optional, exists only in the read-only review job, and is
  never logged, rendered, serialized, committed, or shared with PR code.
- Deterministic Audit works without the optional LLM dependency or API key.
- Do not add a public automation surface, GitHub App, server, database,
  dashboard, billing, organization-sharing layer, or runtime updater.
- Automated tests and implementation must make no OpenAI/provider calls.

---

### Task 1: Immutable installer carrier and release renderer

**Files:**
- Create: `action.yml`
- Create: `templates/claimci-external.yml.tmpl`
- Create: `scripts/render_external_workflow.py`
- Test: `tests/test_external_install_v0.py`

**Interfaces:**
- Composite input: `install-review-dependencies`, default `false`, accepted
  values exactly `true` or `false`.
- Renderer CLI:
  `python scripts/render_external_workflow.py --installer-sha <40-hex> --output <path>`.
- Template marker: `__CLAIMCI_INSTALLER_SHA__`, occurring exactly in immutable
  ClaimCI action references.

- [ ] **Step 1: Write failing installer and renderer tests**

  Assert that the root action installs from `github.action_path`, never `.`,
  `consumer-base`, or `pull-request`; validate the strict boolean input and
  optional `llm` extra. Invoke the renderer with malformed and valid literal
  SHAs and assert controlled failure or deterministic YAML output.

- [ ] **Step 2: Run the focused test and confirm RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_external_install_v0.py`

  Expected: failures because the action, template, and renderer do not exist.

- [ ] **Step 3: Implement the minimal carrier and renderer**

  The renderer validates `[0-9a-f]{40}`, rejects forty zeroes, requires at least
  one marker, replaces every marker, safe-loads the result, and writes UTF-8
  only after all validation succeeds.

- [ ] **Step 4: Run focused and complete suites**

  Run the focused command above, followed by
  `python -m pytest -q -p no:cacheprovider`.

### Task 2: Reusable workflow security contract

**Files:**
- Generate: `.github/workflows/claimci-external.yml`
- Modify: `tests/test_external_install_v0.py`

**Interfaces:**
- `workflow_call.inputs.manifest-path`: optional string, default
  `research.yaml`.
- `workflow_call.inputs.review-config-path`: optional string, default
  `.claimci/review.yaml`.
- `workflow_call.secrets.OPENAI_API_KEY`: optional and explicitly named.
- Workflow jobs: deterministic Audit, read-only optional Review, and isolated
  neutral Review publisher.

- [ ] **Step 1: Commit the verified carrier when explicitly authorized**

  Record its full commit SHA. This is the immutable installer revision used by
  the generated workflow; do not use a synthetic, short, tag, or branch ref.

- [ ] **Step 2: Generate the active workflow**

  Run the renderer with the carrier commit SHA and output
  `.github/workflows/claimci-external.yml`.

- [ ] **Step 3: Add failing security-contract assertions before behavior edits**

  Parse the generated YAML and assert the exact inputs/secret, full-SHA action
  pins, explicit base/head SHAs, `persist-credentials: false`, quoted manifest
  input with `--artifact-root pull-request`, base-root/config-root
  `consumer-base`, one review invocation producing both views, per-job
  permissions, secret separation, deterministic Check mapping, neutral Review,
  bounded handoff, and absence of any install/import/execute path from consumer
  checkouts.

- [ ] **Step 4: Implement only the reusable workflow needed by those tests**

  Adapt the proven repository-local workflow without changing deterministic or
  review Python code. Use the same immutable checkout/setup-python action SHAs.

- [ ] **Step 5: Run focused and complete suites**

  Run the external-install tests, all Day 2/Day 3 workflow tests, then the
  complete offline suite.

### Task 3: Clean consumer fixture and customer documentation

**Files:**
- Create: `scripts/materialize_external_demo.py`
- Create: `examples/external_install/README.md`
- Create: `examples/external_install/customer-workflow.yml.template`
- Create: `examples/external_install/research.yaml`
- Create: `examples/external_install/baseline-config.yaml`
- Create: `examples/external_install/baseline-results.json`
- Create: `examples/external_install/baseline-train.jsonl`
- Create: `examples/external_install/baseline-eval.jsonl`
- Create: `examples/external_install/candidate-config.yaml`
- Create: `examples/external_install/candidate-results.json`
- Create: `examples/external_install/candidate-train.jsonl`
- Create: `examples/external_install/candidate-eval.jsonl`
- Create: `examples/external_install/review-disabled.yaml`
- Create: `docs/external-install-v0.md`
- Modify: `README.md`
- Modify: `tests/test_external_install_v0.py`

**Interfaces:**
- Materializer CLI:
  `python scripts/materialize_external_demo.py --workflow-sha <40-hex> --output <empty-directory> [--enable-review]`.
- Generated consumer files include `.github/workflows/claimci.yml`,
  `research.yaml`, declared artifacts, and optionally
  `.claimci/review.yaml`; they never include `claimci/`, `pyproject.toml`, or
  installer code.

- [ ] **Step 1: Add failing clean-repository fixture tests**

  Generate into a temporary empty directory. Assert the tiny caller has one
  reusable-workflow job pinned to the supplied SHA, explicit permissions and
  named secret, the consumer contains no ClaimCI source/package, all manifest
  paths are relative/confined, and `audit_research` yields the known
  NOT_SUPPORTED fixture with compute mismatch, exact leakage, and supported
  submitted result findings.

- [ ] **Step 2: Run the focused test and confirm RED**

  Expected: failure because the materializer and copyable fixture do not exist.

- [ ] **Step 3: Implement fixture generation and documentation**

  Validate output is absent or empty and the workflow SHA is a nonzero full
  hex commit. Copy only the allowlisted fixture files and write the rendered
  caller atomically. Document `research.yaml`, optional trusted review config,
  optional `OPENAI_API_KEY`, Actions access settings, branch protection, fork
  data egress, immutable updates, troubleshooting, and exact ClaimCI-Demo live
  pilot steps.

- [ ] **Step 4: Run focused and complete suites**

  Run external-install tests, fixture/audit tests, and the complete suite.

### Task 4: Adversarial contract verification and independent review

**Files:**
- Modify: `tests/test_external_install_v0.py`
- Modify only if a real defect is found: files introduced by Tasks 1-3

**Interfaces:**
- No new runtime interface.

- [ ] **Step 1: Add regressions for actual boundary gaps**

  Cover malformed SHA, all-zero SHA, duplicate/missing template marker,
  traversal/NUL/occupied output directory, workflow tampering, `secrets:
  inherit`, PR-controlled ref/config, install from consumer paths, permission
  escalation, provider key in audit/publisher, multiple review invocations,
  retry loops, and non-neutral advisory publication.

- [ ] **Step 2: Run complete offline verification**

  Run `python -m pytest -q -p no:cacheprovider`,
  `python -m compileall -q claimci scripts`, and `git diff --check`.

- [ ] **Step 3: Request independent read-only final review**

  Review the complete diff against the design, GitHub private-sharing
  constraints, passive-data boundary, immutable installation, authority
  separation, optional/no-key path, and fixture instructions. Make no network
  or provider calls.

- [ ] **Step 4: Fix only confirmed findings test-first and reverify**

  Add a focused failing regression for each confirmed defect, apply the minimum
  fix, rerun focused tests, then rerun the complete suite once.
