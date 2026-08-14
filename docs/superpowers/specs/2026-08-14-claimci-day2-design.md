# ClaimCI Day 2 Design

Date: 2026-08-14

## Objective

Extend the deterministic Day 1 auditor into a pull-request quality gate without weakening its scientific boundary:

> ClaimCI independently verifies whether scientific claims are supported by the underlying experiments.

Day 2 adds GitHub Check Run integration, explicit metric direction, and one polished adversarial demo. It does not add experiment execution, statistical significance, artifact signing, semantic leakage, or a hosted service.

## Frozen public contracts

### Direction

The optional manifest field is:

```yaml
claim:
  metric: loss
  minimum_improvement: 0.05
  direction: lower
```

Accepted values are the lowercase strings `higher` and `lower`. Surrounding whitespace is ignored; any other explicit value is an input error. Omitting the field defaults to `higher`, preserving every Day 1 manifest and the four-argument `check_results(...)` API.

Improvement is a signed value in the declared favorable direction:

- `higher`: `candidate_mean - baseline_mean`
- `lower`: `baseline_mean - candidate_mean`

Relative improvement is the same directional difference divided by `abs(baseline_mean)`. It is unavailable when the baseline is zero or the derived ratio is non-finite. The absolute metric-unit threshold remains the only threshold used for the claim decision.

Human output for the default higher direction remains compatible with Day 1. Lower-direction output explicitly says that lower is better. To keep the Day 1 JSON shape compatible, the default higher direction is implicit in JSON; lower-direction reports add `claim.direction: lower`. Finding evidence always records the normalized direction for new claim-decision findings.

### GitHub Markdown

`claimci audit research.yaml --markdown` emits deterministic GitHub-flavored Markdown. `--json` and `--markdown` are mutually exclusive. The Markdown contains:

- the verdict and a one-sentence interpretation;
- the declared metric, direction, and threshold;
- recomputed baseline/candidate summaries and directional improvement;
- the findings that drive the verdict first;
- all findings with stable rule IDs and JSON-safe evidence.

Markdown table cells and evidence are escaped so artifact-controlled metric names or values cannot corrupt the report. The values in Markdown and JSON come from the same `AuditResult`; neither renderer recomputes scientific quantities.

### GitHub Check Run

The repository includes `.github/workflows/claimci.yml`, triggered by
`pull_request_target`. It installs ClaimCI from the trusted base commit, checks
out the pull-request head into a separate passive-data directory, audits its
`research.yaml` with explicit artifact-root confinement, appends Markdown to
the job summary, and creates an explicit `ClaimCI Audit` Check Run on the PR
head SHA.

Check conclusions are:

| ClaimCI verdict | GitHub conclusion | Meaning |
|---|---|---|
| `SUPPORTED` | `success` | The declared claim passed every required audit. |
| `NOT_SUPPORTED` | `failure` | The claim or a fairness/integrity invariant failed. |
| `INSUFFICIENT_EVIDENCE` | `action_required` | More or valid evidence is required; this is deliberately non-passing and distinct from failure. |
| malformed input/integration error | `failure` | ClaimCI could not produce a trustworthy verdict. |

`neutral` is not used because GitHub branch protection treats neutral checks as successful. A small Python adapter builds the API payload from the JSON report and Markdown. The workflow sends that file with `gh api --input`, avoiding shell interpolation of report content.

The target-context token is permitted to create Checks for same-repository and
fork pull requests, but pull-request code is never executed with it. Both
checkout/setup actions are pinned to immutable commits; the PR checkout is not
imported, installed, or built; and absolute paths, parent traversal, and
symlink escapes are rejected by `--artifact-root`. The base branch must contain
the trusted ClaimCI package and workflow. A published reusable action or
GitHub App remains the path to centralized versions and policy across unrelated
repositories.

The default manifest is `research.yaml`. Repositories adopting ClaimCI either place their manifest there or edit the single `CLAIMCI_MANIFEST` workflow environment value.

### Demo

`examples/day2_demo` is self-contained. It presents an apparently strong accuracy result:

- baseline mean `0.60` from three seeds;
- candidate mean `0.90` from one seed;
- reported directional improvement `+0.30` (30 percentage points, 50% relative);
- a `0.05` claim threshold that the raw score alone passes.

ClaimCI nevertheless returns `NOT_SUPPORTED` because candidate compute proxy is exactly `3.0x` and one candidate evaluation record occurs canonically in its training set. The one-seed candidate independently yields insufficient-evidence findings. Non-target config and data properties remain matched so the story is attributable to those three facts.

The auditor also verifies the actual baseline/candidate evaluation-record
multisets, not only their declared config identities. The demo therefore uses
the same two evaluation records in both arms; leakage is confined to the
candidate training split.

## Alternatives considered

### Rely only on the Actions job conclusion

Rejected. A job can naturally show success or failure, but it cannot represent `INSUFFICIENT_EVIDENCE` as both distinct and non-passing. An explicit Check Run supports `action_required`.

### Map insufficient evidence to `neutral`

Rejected. It looks distinct in the UI but GitHub treats a neutral required check as successful, contradicting “non-passing.”

### Build a GitHub App now

Deferred. It is the durable answer for centralized policy, installation,
versioning, and richer annotations across customer repositories, but adds
authentication, hosting, and lifecycle scope unrelated to the requested Day 2
core.

## Determinism and failure policy

- All renderers are pure functions of `AuditResult`.
- Report ordering follows the existing finding order and fixed severity order.
- GitHub payload JSON is sorted and standards-compliant (`allow_nan=False`).
- Missing, malformed, or inconsistent adapter inputs produce a failure payload rather than a neutral/insufficient result.
- The payload summary is bounded to the GitHub API limit with a deterministic truncation notice.
- A Check API publication failure fails the workflow rather than silently claiming integration success.
- Scientific audit exit codes remain `0`, `1`, and `2`; GitHub conclusion mapping uses the parsed verdict, not exit code alone, because exit `2` also covers invalid input.

## Verification

Testing covers direct direction arithmetic, manifest propagation, report parity, adapter payloads, workflow security/static structure, all Day 1 fixtures, the Day 2 demo, malformed inputs, numeric boundaries, JSONL duplicates, path failures, and renderer escaping. The complete suite runs after each implementation or hardening wave. A separate read-only final review checks behavior against the product objective and reports remaining boundaries honestly.
