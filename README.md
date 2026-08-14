# ClaimCI

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

## GitHub pull-request integration

The checked-in [ClaimCI workflow](.github/workflows/claimci.yml) runs automatically when a pull request is opened, updated, reopened, or marked ready for review. It audits the root `research.yaml`, appends the Markdown report to the Actions job summary, and creates a `ClaimCI Audit` Check Run on the pull-request head commit, including for fork pull requests.

| ClaimCI verdict | Check conclusion |
|---|---|
| `SUPPORTED` | `success` |
| `NOT_SUPPORTED` | `failure` |
| `INSUFFICIENT_EVIDENCE` | `action_required` |

`action_required` is intentionally non-passing and distinct from a rejected claim. Configure `ClaimCI Audit` as a required check to gate merges. To use another manifest path, change `CLAIMCI_MANIFEST` in the workflow.

The workflow uses `pull_request_target` for a writable Check token, but it does
not execute pull-request code. Both GitHub actions are pinned to immutable
commits; ClaimCI is installed from the trusted base commit; and the PR checkout
is treated only as passive, `--artifact-root`-confined scientific input. The PR
tree is never imported, installed, or used as a build source. This repository-
local model requires ClaimCI itself to be present on the base branch; a
published reusable action or GitHub App is still needed for centralized
versioning across unrelated customer repositories.

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

## Current limitations

ClaimCI does not establish statistical significance, causal validity, artifact
authenticity/provenance, semantic or group leakage, or equal wall-clock/FLOP
compute. It does not rerun experiments. JSON number spellings such as `1` and
`1.0` remain distinct after canonical serialization, and leakage detection
covers exact within-experiment train/eval matches only. The workflow trusts the
base-branch ClaimCI implementation and local artifacts; it is not yet a hosted,
centrally versioned verifier. Artifact size limits and cryptographic
attestations are not yet implemented.
