# ClaimCI MVP Design

## Purpose and scope

ClaimCI is a deterministic Python 3.11+ command-line tool that audits whether a candidate experiment supports a declared improvement over a baseline. Day 1 reads local YAML, JSON, and JSONL artifacts only. It never trains a model, calls a service, performs semantic similarity, or trusts a reported summary instead of the raw run values.

The accepted command is:

```text
claimci audit path/to/research.yaml [--json]
```

## Considered approaches

1. **Focused library with a thin CLI (selected).** Pure check modules return typed findings, `audit.py` orchestrates them, and `report.py` owns presentation. This is small enough for an MVP while keeping tomorrow's GitHub-comment output separate from scientific rules.
2. **Single-file CLI.** This would be faster to scaffold but would mix parsing, policy, calculations, and formatting, making stable findings and unit tests harder to maintain.
3. **Rule/plugin framework.** Extensible, but unnecessary for eight fixed Day 1 rules and likely to obscure deterministic verdict logic.

## Inputs

`research.yaml` has a `claim` mapping and two experiment mappings. Its artifact paths may be relative; relative paths are resolved against the manifest's directory, never the process working directory.

```yaml
claim:
  metric: accuracy
  minimum_improvement: 0.05
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

The Day 1 config vocabulary is deliberately small:

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

Results contain a non-empty `runs` list. Every run must contain the claim metric as a finite number. A run should contain an integer `seed`; missing seed metadata is reported rather than silently invented. An optional `summary` mapping may be supplied, but it is checked against and never substituted for the recomputed raw mean.

## Components

- `models.py`: frozen enums/dataclasses for severity, impact, verdict, findings, metric summaries, overlap summaries, and audit results.
- `config_check.py`: YAML config loading and explicit comparison of the supported fairness fields.
- `result_check.py`: result parsing, seed checks, sample statistics, summary verification, and improvement calculations.
- `dataset_check.py`: strict JSONL parsing, canonical JSON serialization, SHA-256 hashing, and duplicate-aware within-experiment train/eval overlap.
- `audit.py`: manifest validation, path resolution, error-to-finding conversion, check orchestration, and verdict evaluation.
- `report.py`: stable human and JSON renderers.
- `cli.py`: `argparse` entry point and verdict-based exit codes.

## Findings and policy

Every finding has a stable `rule_id`, severity, title, explanation, evidence mapping, and impact. Severity controls presentation. Impact controls the verdict so a warning does not accidentally invalidate a claim.

| Condition | Severity | Impact |
|---|---|---|
| Candidate compute proxy is at least 1.5x baseline | CRITICAL | INVALIDATES |
| Evaluation mapping differs | CRITICAL | INVALIDATES |
| Exact train/eval overlap in either experiment | CRITICAL | INVALIDATES |
| Recomputed improvement is below the absolute threshold | CRITICAL | INVALIDATES |
| Claimed summary disagrees with raw mean | CRITICAL | INVALIDATES |
| Required artifact is missing/malformed or raw runs are unavailable | CRITICAL | INSUFFICIENT |
| Either experiment has fewer than two runs or missing/duplicate seed metadata | WARNING | INSUFFICIENT |
| Seed counts are severely imbalanced (at least 3x and two runs apart) | WARNING | NONE unless another evidence rule is insufficient |
| Training data, epochs, batch size, steps, or learning rate differ without reaching the compute threshold | WARNING | NONE |
| Model differs | INFO | NONE |
| Raw metrics/configs/datasets successfully verify | VERIFIED | NONE |

The compute proxy is `training_steps * batch_size` when all four values are positive numbers. Otherwise ClaimCI falls back to the available positive `training_steps`, then `epochs * batch_size`, then `epochs`. It reports the proxy and ratio. Only a candidate advantage of at least `1.5x` invalidates; other explicit differences remain visible without a blanket failure.

`minimum_improvement` is an **absolute** metric-unit threshold. For accuracy expressed from 0 to 1, `0.05` means five percentage points. ClaimCI also reports relative improvement as `(candidate_mean - baseline_mean) / abs(baseline_mean)` when the baseline is nonzero, but never compares that relative value to the absolute threshold.

Verdict precedence is deterministic:

1. Any `INVALIDATES` finding produces `NOT_SUPPORTED`.
2. Otherwise, any `INSUFFICIENT` finding produces `INSUFFICIENT_EVIDENCE`.
3. Otherwise the claim is `SUPPORTED`.

CLI exits are `0` for `SUPPORTED`, `1` for `NOT_SUPPORTED`, and `2` for `INSUFFICIENT_EVIDENCE` or invalid CLI/input usage.

## Dataset canonicalization

Each nonblank JSONL line must parse as JSON. The parsed value is serialized with sorted object keys, compact separators, UTF-8, and non-ASCII preservation, then hashed with SHA-256. `Counter` intersection is used, so repeated overlapping records count deterministically. The overlap rate is `overlap_count / evaluation_record_count`. Only baseline train-to-baseline eval and candidate train-to-candidate eval are checked; sharing training data across experiments is expected and is not leakage by itself.

## Error handling and output

Malformed top-level manifests produce a concise `ClaimCI error:` message with no traceback. Artifact failures inside a valid manifest become findings so the report can show all discoverable issues and return `INSUFFICIENT_EVIDENCE`. Findings are emitted in check order and rendered in fixed severity order (`CRITICAL`, `WARNING`, `VERIFIED`, `INFO`). JSON output uses enum values and JSON-native nulls and is serialized with sorted keys.

## Testing and fixtures

Unit tests cover literal statistics, threshold boundaries, every config category, seed evidence, canonical hashes, duplicate-aware leakage, verdict precedence, missing/malformed inputs, report rendering, and CLI exits. Integration tests run all required fixtures: `valid`, `compute_mismatch`, `seed_mismatch`, `data_leak`, `fake_or_unsupported_result`, `different_eval_dataset`, `missing_result`, and `multiple_failures`.

Expected fixture verdicts are respectively: `SUPPORTED`, `NOT_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `NOT_SUPPORTED`, `NOT_SUPPORTED`, `NOT_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, and `NOT_SUPPORTED`.

## Day 1 limitations

ClaimCI checks artifact consistency, exact overlap, and arithmetic support. It does not establish statistical significance, causal validity, provenance/authenticity, semantic or group leakage, equal FLOPs/wall-clock cost, or correctness of the experiment runner. Metrics are treated as higher-is-better and dataset samples are compared only for exact canonical JSON equality.
