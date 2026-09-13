# ClaimCI

Experimental CI for verifying quantitative, experimental, and behavioral claims in AI/ML pull requests.

**Status: Maintenance mode**

Commercial development has ended.
The project remains available as open-source software.

## What ClaimCI does

ClaimCI checks whether a pull request's claims are supported by its available
code, experiment configuration, datasets, and result artifacts. Its local CLI
provides deterministic auditing; optional model-assisted review is advisory.
Hosted accounts, login, dashboard, billing, and managed analysis are deprecated.
There is no hosted-service availability or commercial support commitment.
Automated model review is disabled in this repository by default.

- [Architecture](docs/architecture.md)
- [Case studies and attribution limits](docs/case-studies/README.md)
- [Apache-2.0 license](LICENSE)

## Local CLI quick start

Requires Python 3.11 or newer. Deterministic audits need no API key or backend.

```sh
python -m pip install -e .
claimci audit examples/valid/research.yaml --json
```

## GitHub Actions example

This read-only example runs a pinned version of ClaimCI on passive PR artifacts.
Review and update the immutable installer revision deliberately.

```yaml
name: ClaimCI local audit
on: [pull_request, workflow_dispatch]
permissions:
  contents: read
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
          path: artifacts
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
        with:
          python-version: '3.11'
      - uses: amebaleon/ClaimCI@7d64a3957a3d0b0a6712f0d89d1ba6497208822c
      - run: claimci audit artifacts/research.yaml --artifact-root artifacts --markdown
```

## Limitations and maintenance status

Evidence support is limited to the supplied artifacts and supported adapters.
A finding about missing evidence is not proof that a claim is false.
Model-assisted review may miss issues or produce incorrect interpretations and
requires human review. Optional API use is billed by the user's provider.
ClaimCI does not reproduce arbitrary GPU experiments or guarantee scientific
correctness, security, or merge readiness. No new commercial features or
response-time guarantees are planned; fixes are best effort.

## Detailed usage

> ClaimCI independently verifies whether scientific claims are supported by the underlying experiments.

ClaimCI is a deterministic command-line auditor for experiment configs, raw run results, seed evidence, and exact train/eval data leakage.

## Install

ClaimCI requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

## Research manifest

Artifact paths are resolved relative to the `research.yaml` file. Trusted CI
invocations also pass `--artifact-root`; in that mode declarations must be
relative and cannot traverse or resolve (including through symlinks) outside
the checked-out artifact tree.

```yaml
claim:
  metric: accuracy
  minimum_improvement: 0.05
  direction: higher  # optional; higher (default) or lower

baseline:
  config: base/config.yaml
  results: base/results.json
  train_dataset: base/train.jsonl
  eval_dataset: base/eval.jsonl

candidate:
  config: candidate/config.yaml
  results: candidate/results.json
  train_dataset: candidate/train.jsonl
  eval_dataset: candidate/eval.jsonl
```

`minimum_improvement` is an absolute metric-unit threshold. For accuracy stored from 0 to 1, `0.05` means five percentage points. Relative improvement is reported separately and is never used in place of this threshold.

`direction` declares which way is favorable. It accepts `higher` or `lower` and defaults to `higher`, so Day 1 manifests remain valid. Directional improvement is candidate minus baseline for `higher`, and baseline minus candidate for `lower`.

Human reports render accuracy thresholds and absolute improvements as percentage points. Other metrics, such as loss, remain in metric units. JSON reports normalize nested evidence to JSON-safe values (including string mapping keys and deterministic strings for non-finite numbers), so they can be parsed by standard JSON tooling.

Configs use this Day 1 vocabulary:

```yaml
training_steps: 100000
epochs: 10
batch_size: 32
dataset:
  identifier: example-train
  version: v1
evaluation:
  dataset_identifier: example-eval
  dataset_version: v1
  split: test
model: model-name
learning_rate: 0.001
```

Results contain raw runs and may contain an optional claimed summary:

```json
{
  "runs": [
    {"seed": 1, "accuracy": 0.71},
    {"seed": 2, "accuracy": 0.73}
  ],
  "summary": {"accuracy": 0.72}
}
```

## Run

```bash
claimci audit examples/valid/research.yaml
```

For stable machine-readable output:

```bash
claimci audit examples/valid/research.yaml --json
```

For GitHub PR reviews and Actions job summaries:

```bash
claimci audit examples/valid/research.yaml --markdown
```

Example output (abridged):

```text
ClaimCI Audit

Claim:
candidate improves accuracy by >= 5.00 percentage points

Verdict:
SUPPORTED

Metrics:
- Baseline: count=3, mean=0.600000, sample stddev=0.020000
- Candidate: count=3, mean=0.700000, sample stddev=0.020000
- Absolute improvement: 0.100000 (10.00 percentage points)
- Relative improvement: 16.67%

VERIFIED
- [RESULT.RECOMPUTED] Raw accuracy recomputed successfully: ...
```

Exit codes are `0` for `SUPPORTED`, `1` for `NOT_SUPPORTED`, and `2` for `INSUFFICIENT_EVIDENCE` or invalid input.

## GitHub pull-request integration (self-managed)

The project owner's hosted workflows are disabled in maintenance mode. The
source remains available for users to configure in their own repositories.

The checked-in [ClaimCI workflow](.github/workflows/claimci.yml) runs automatically when a pull request is opened, updated, reopened, or marked ready for review. It audits the root `research.yaml`, appends the Markdown report to the Actions job summary, and creates a `ClaimCI Audit` Check Run on the pull-request head commit, including for fork pull requests.

| ClaimCI verdict | Check conclusion |
|---|---|
| `SUPPORTED` | `success` |
| `NOT_SUPPORTED` | `failure` |
| `INSUFFICIENT_EVIDENCE` | `action_required` |

`action_required` is intentionally non-passing and distinct from a rejected claim. Configure `ClaimCI Audit` as a required check to gate merges. To use another manifest path, change `CLAIMCI_MANIFEST` in the workflow.

To make the authoritative quality gate required, open **Settings → Branches →
Branch protection rules**, enable **Require status checks to pass before
merging**, and select the exact status name **`ClaimCI Audit`**. Do not require
the Actions job named `ClaimCI audit`: that job reports workflow/integration
health and remains successful for a correctly published scientific
`NOT_SUPPORTED` or `INSUFFICIENT_EVIDENCE` result.

The workflow uses `pull_request_target` for a writable Check token, but it does
not execute pull-request code. Both GitHub actions are pinned to immutable
commits; ClaimCI is installed from the trusted base commit; and the PR checkout
is treated only as passive, `--artifact-root`-confined scientific input. The PR
tree is never imported, installed, or used as a build source. This repository-
local model requires ClaimCI itself to be present on the base branch; a
published reusable action or GitHub App is still needed for centralized
versioning across unrelated customer repositories.

## Optional ClaimCI Research Review (Day 3)

The same workflow can add an advisory, LLM-powered **ClaimCI Research Review**
for ordinary research pull requests. It reads the pull-request title and
description plus selected changed Markdown/TeX research notes, then uses
bounded deterministic artifact discovery and the existing ClaimCI audit tools
to ground the review. Supported review claim categories include metric
improvements, compute equivalence, held-out evaluation, resource reductions,
component causality, no-external-reward statements, implementation claims, and
other scientific statements.

Research review is explicitly **opt in**. The trusted base checkout must carry
`.claimci/review.yaml`; a pull request cannot enable review or change its
provider policy. A minimal configuration is:

```yaml
schema_version: 1
enabled: true
policy: advisory
provider: openai
model: gpt-5.6-terra
limits:
  max_calls: 2
  max_context_chars: 60000
  max_output_chars: 24000
  max_files: 24
  max_file_chars: 16000
  max_claims: 16
  extraction_max_output_tokens: 5000
  synthesis_max_output_tokens: 4000
  timeout_seconds: 90
```

The workflow supplies `OPENAI_API_KEY` only to the trusted review step. ClaimCI
reads it at runtime through the provider SDK; it is never printed, rendered,
serialized, or committed. The default provider is OpenAI's `gpt-5.6-terra`;
trusted runtime configuration may set `CLAIMCI_OPENAI_MODEL` to a compatible
model. Automated tests use mocked providers and never make paid API calls.
Each review uses at most two provider calls (claim extraction and evidence-
grounded synthesis), with a 60,000-character context budget applied
independently to each provider request, a 16-claim materiality cap, task-specific
output budgets of 5,000 and 4,000 tokens, and a 24,000-character audit-wide
output budget. Legacy trusted configurations using
`max_output_tokens_per_call` remain readable and map that value to both calls.
Provider input/output/total tokens and estimated cost, when supplied, are
recorded for observability only. The checked-in configuration allows 90 seconds per provider call; trusted configurations may choose any finite timeout up to 120 seconds.

Selected private repository content may therefore be sent to the configured
external provider when an owner enables this configuration. With a trusted
base checkout, heuristic evidence discovery is confined to files changed by
the pull request. Unchanged files enter review only through a relevant
`research.yaml` whose manifest or declared artifact changed, or through an
exact indexed SOURCE/TEST provider hint. Those hints remain untrusted selectors:
ClaimCI validates path identity and confinement, assigns supporting-artifact
provenance, and retains claim-scoped citation ownership. Complete files are
marked `complete_file`; oversized changed files may use a bounded, contiguous
`changed_region`. An oversized unchanged file without deterministic locality is
marked `unlocalized_prefix`, cannot be cited, and keeps evidence routing
incomplete unless other citable evidence covers the claim. Its bytes remain in
the local review record but are omitted from provider synthesis evidence.
Structured research
artifacts such as SQL, JSON, YAML, CSV, and TOML participate in the
bounded base/head change index. This boundary is especially important for
`pull_request_target` and fork pull requests: the head checkout is passive
data, while configuration and code come from the trusted base branch. Disable
or omit `.claimci/review.yaml` when external data egress is not acceptable.

The review check is always `neutral` and is never a quality gate. When one
structured extraction response mixes valid claims with isolated malformed
candidates, ClaimCI excludes only the invalid candidates, reviews every
retained claim, and reports an explicit `PARTIAL` result. A uniquely
recoverable exact full-line quote may have its line location repaired
deterministically; paraphrases and ambiguous occurrences remain invalid, and
all-invalid extraction still fails closed. Once extraction and deterministic
evidence discovery have succeeded, malformed or semantically invalid synthesis
is reported as `PARTIAL` with no invalid interpretation accepted. Its Markdown
labels distinguish
deterministic evidence, LLM interpretation, missing evidence, and unsupported
inference. Only the separate deterministic
**ClaimCI Audit** Check can block a PR; enabling or disabling review does not
change deterministic audit bytes, verdicts, or exit codes. A future strict
organization policy may make unresolved review findings non-passing on this
separate check, but Day 3 intentionally remains advisory.

## Current checks

- Compares training steps, epochs, batch size, model, learning rate, training dataset identity/version, and the full evaluation mapping.
- Calculates a deterministic training-compute proxy and invalidates a candidate advantage of at least `1.5x`.
- Recomputes count, mean, and sample standard deviation from raw run values.
- Keeps absolute metric-unit improvement separate from relative percentage improvement (accuracy is rendered in percentage points; other metrics stay in native units).
- Supports deterministic `higher`-is-better and `lower`-is-better claims, defaulting to `higher`.
- Detects missing, duplicate, single-run, and severely imbalanced seed evidence.
- Checks optional claimed summaries against recomputed means.
- Parses JSONL samples, canonicalizes JSON, hashes it with SHA-256, and performs duplicate-aware exact train/eval overlap checks independently for each experiment.
- Verifies that baseline and candidate evaluation artifacts contain the same duplicate-aware canonical record multiset.
- Emits stable rule IDs, severities, evidence, impacts, and deterministic verdicts.

The verdict policy is explicit: an invalidating fairness, leakage, integrity, or threshold finding yields `NOT_SUPPORTED`; otherwise missing or too-thin evidence yields `INSUFFICIENT_EVIDENCE`; otherwise the claim is `SUPPORTED`. In particular, `CONFIG.MISSING_FIELDS` yields `INSUFFICIENT_EVIDENCE` unless another invalidating finding takes precedence. Presentation severity alone does not decide the verdict.

## Example fixtures

Every manifest below is runnable:

| Fixture | Expected verdict |
|---|---|
| `valid` | `SUPPORTED` |
| `compute_mismatch` | `NOT_SUPPORTED` |
| `seed_mismatch` | `INSUFFICIENT_EVIDENCE` |
| `data_leak` | `NOT_SUPPORTED` |
| `fake_or_unsupported_result` | `NOT_SUPPORTED` |
| `different_eval_dataset` | `NOT_SUPPORTED` |
| `missing_result` | `INSUFFICIENT_EVIDENCE` |
| `multiple_failures` | `NOT_SUPPORTED` |
| `day2_demo` | `NOT_SUPPORTED` |

The fixture manifests share small common artifacts under `examples/shared` and override only the artifact needed to demonstrate each rule.

## 60-second rejected-improvement demo

The self-contained `day2_demo` candidate appears dramatically better on accuracy: `0.90` versus a `0.60` baseline. Its raw +30 percentage-point result clears the five-point threshold, but it used a `3.0x` compute proxy, has only one candidate seed, and leaked one of two evaluation records into training. ClaimCI therefore returns `NOT_SUPPORTED`.

```powershell
python -m pip install -e ".[dev]"
claimci audit examples/day2_demo/research.yaml
claimci audit examples/day2_demo/research.yaml --markdown > day2-demo.md
claimci audit examples/day2_demo/research.yaml --json > day2-demo.json
```

The command intentionally exits `1`; open `day2-demo.md` in a Markdown preview to show the PR-ready view. See [the demo guide](examples/day2_demo/README.md) for the exact story and limitations.

## Historical external private-repository pilot

This describes the former private pilot. Core is now public; the old access
sharing setup is not required for the public action.

ClaimCI v0 can be consumed from a separate private repository owned by the same
GitHub user through a full-SHA-pinned reusable workflow. The consumer contains
only a tiny caller workflow and its declared research artifacts; it does not
copy ClaimCI source or installer code. GitHub's private Actions sharing boundary
does not support arbitrary external repositories, so configure the ClaimCI
repository's Actions access for same-owner private repositories before testing.

Generate the clean PR#1-style consumer fixture with a reviewed workflow commit:

```powershell
python scripts/materialize_external_demo.py `
  --workflow-sha <reviewed-workflow-commit-sha> `
  --output C:\path\to\ClaimCI-Demo
```

Add `--enable-review` only when the trusted base should carry the disabled
advisory review configuration. The deterministic Audit needs no API key and is
the only blocking Check; Review is neutral and optional. See the
[external-install v0 guide](docs/external-install-v0.md) for the exact manual
pilot steps, branch protection, fork/passive-data boundary, and troubleshooting.
When review is enabled, pull_request_target fork-authored head content may be sent to the configured provider; keep review disabled when that egress is not approved.

## Current limitations

ClaimCI does not establish statistical significance, causal validity, artifact
authenticity/provenance, semantic or group leakage, or equal wall-clock/FLOP
compute. It does not rerun experiments. JSON number spellings such as `1` and
`1.0` remain distinct after canonical serialization, and leakage detection
covers exact within-experiment train/eval matches only. The workflow trusts the
base-branch ClaimCI implementation and local artifacts; it is not yet a hosted,
centrally versioned verifier. Artifact size limits and cryptographic
attestations are not yet implemented.
