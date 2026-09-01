# ClaimCI Day 3 Design

Date: 2026-08-15

## Objective

Add an opt-in, LLM-powered research reviewer around the existing deterministic
ClaimCI core so a normal research pull request can be understood without
encoding every natural-language claim in `research.yaml`.

The reviewer extracts scientific claims from pull-request metadata and changed
research documents, discovers a bounded set of likely evidence, invokes
trusted deterministic ClaimCI audits where possible, and renders an
evidence-grounded advisory review.

The deterministic product boundary is unchanged:

> ClaimCI's existing deterministic findings are the sole authority for
> blocking scientific conclusions and the existing `ClaimCI Audit` quality
> gate.

LLM extraction, interpretation, missing evidence, unsupported inference,
provider failure, or malformed LLM output cannot by themselves block a pull
request in Day 3.

## Explicit non-goals

Day 3 does not add a website, database, dashboard, billing, semantic leakage
detection, full experiment reproduction, cryptographic provenance, arbitrary
model tools, filesystem access for the model, or a strict review gate.

## Compatibility contracts

- `claimci audit` remains available without an API key and retains its
  arguments, serialized models, renderers, exit codes, and GitHub conclusion
  mapping.
- Existing `Verdict`, `Finding`, `Severity`, `Impact`, `ResearchSpec`, and
  `AuditResult` types are not extended with LLM-produced state.
- Existing deterministic report bytes are not changed by enabling or disabling
  research review.
- Research review is a separate `claimci review` mode and a separate advisory
  GitHub surface.
- A future organization policy may map unresolved review statuses to a
  non-passing *separate* review check. Day 3 implements only advisory policy.

## Two-call architecture

A complete review uses exactly two provider calls and never a third call:

1. **Claim extraction** receives bounded PR title/description, a bounded
   repository path index, and changed research-document excerpts. It returns
   structured claim candidates.
2. **Evidence-grounded synthesis** receives only validated claims, trusted
   evidence references, missing-evidence records, and read-only snapshots of
   deterministic tool output. It returns interpretations and citations.

Between those calls, evidence discovery is trusted deterministic Python code.
It uses claim-type routing, repository metadata and path heuristics, existing
manifests, and bounded artifact inspection. The provider has no filesystem,
network, shell, code-interpreter, file-search, web-search, or function-tool
access.

The configurable call limit can be lower than two for cost control. If it
prevents the second call, the review is incomplete/unavailable; the
orchestrator never compensates with another provider call. Provider retries are
disabled.

## Package structure

Day 3 is additive under `claimci.review`:

- `models.py`: LLM-safe claim, source, interpretation, status, limit, and usage
  models. It does not import or construct deterministic ClaimCI authority
  types.
- `config.py`: strict trusted review configuration and default-disabled policy.
- `sources.py`: PR metadata parsing, base/head comparison, bounded document
  selection, exact line locations, and repository path indexing.
- `evidence.py`: claim-type routing, path confinement, selective artifact
  excerpts, hashes, and missing-evidence records.
- `tools.py`: the only bridge to deterministic ClaimCI. It converts existing
  `AuditResult` objects into immutable, read-only snapshots for review input.
- `provider.py`: provider-neutral two-operation protocol and raw structured
  response envelope.
- `openai_provider.py`: optional, lazy OpenAI Responses API adapter.
- `orchestrator.py`: the fixed extract -> discover/tools -> synthesize state
  machine and budget accounting.
- `report.py`: strict JSON and safe GitHub Markdown review renderers.
- `github.py`: a separate advisory Check payload builder whose conclusion is
  always neutral in Day 3.

The existing `claimci.models`, `claimci.audit`, `claimci.report`, and
`claimci.github` remain the deterministic authority.

## Review configuration and opt-in

Review is disabled unless a trusted repository configuration explicitly opts
in. The default path is `.claimci/review.yaml` in the trusted base checkout:

```yaml
schema_version: 1
enabled: true
policy: advisory
provider: openai
model: gpt-5.6-terra
limits:
  max_calls: 2
  max_context_chars: 60000
  max_output_chars: 12000
  max_files: 24
  max_file_chars: 16000
  max_output_tokens_per_call: 2000
  timeout_seconds: 30
```

The configuration parser rejects unknown fields, invalid types, unsafe limits,
unsupported policy values, and `max_calls > 2`. A missing configuration or
`enabled: false` performs no provider calls. An API key alone never enables
review.

In GitHub, the configuration is read from the trusted base checkout, not from
the pull-request tree. This is the repository owner's explicit consent for
selected private repository content to be sent to the configured external
provider. The README must state this data-egress boundary prominently.

## Source collection

Eligible claim sources are:

- pull-request title;
- pull-request description;
- changed `README` Markdown files;
- changed Markdown research notes;
- changed `paper.md`;
- changed `paper.tex`.

Trusted code parses `GITHUB_EVENT_PATH` or explicit local metadata files. PR
strings are data and are never interpolated into shell commands. Base/head
documents are compared by content hash; no PR code or build hook is executed.

Only regular files confined beneath the checked-out repository may be read.
Selection applies deterministic ordering, extension/name allowlists, file
count, per-file character, per-provider-request logical context, nesting, and
decoding limits before provider construction. A bounded relative-path repository
index may be sent; file content is not sent until selected.

## Structured claims

Normalized claim types are:

- `metric_improvement`
- `compute_equivalence`
- `held_out_evaluation`
- `resource_reduction`
- `component_causality`
- `no_external_reward`
- `implementation_claim`
- `other_scientific`

Every accepted claim contains:

- stable claim ID assigned by trusted code;
- original source text;
- normalized claim type;
- subject;
- optional metric;
- optional review-layer direction (`higher`, `lower`, or `not_applicable`);
- optional magnitude with raw text, numeric value when parseable, unit, and
  absolute/relative kind;
- qualifiers/constraints;
- source kind, repository-relative path when applicable, and exact line span;
- confidence from 0 through 1.

The provider proposes candidates, but trusted code accepts a claim only when
the quote exactly matches the declared bounded source location. Unknown fields,
invalid enums/types, duplicate IDs, non-finite numbers, invalid ranges,
oversized strings, or hallucinated quotes are rejected. Trusted code assigns
the final stable claim IDs after validation.

## Evidence discovery

Discovery is deterministic and claim-type routed:

- metric improvement -> manifests, results, configs;
- compute equivalence -> manifests, training configs, run metadata;
- held-out evaluation -> manifests, train/eval declarations and artifacts;
- cost/latency/memory reduction -> benchmark results, configs, and relevant
  changed implementation files;
- component causality -> ablation/result manifests and relevant changed source;
- no external reward -> reward/config declarations and relevant changed source;
- implementation claims -> relevant changed source and tests.

Routing uses bounded path/name heuristics and literal claim terms. It does not
perform semantic leakage analysis or execute source. Suggested provider paths,
if present in a response, are only untrusted hints: trusted code resolves them
beneath the repository, enforces the same budgets, and records nonexistent or
unsafe hints as missing evidence.

Each repository evidence reference contains a trusted ID, relative path, line
span, SHA-256 digest, size, and bounded excerpt. Citations in synthesis output
must name an issued evidence ID; unknown or mismatched citations are rejected.

## Deterministic tool boundary

Existing manifests discovered within the repository may be passed to
`audit_research(..., artifact_root=repository_root)`. This is the preferred
deterministic tool path because it preserves the existing schema, fairness
policy, and rule IDs.

Only `claimci.review.tools` may import deterministic authority models. It
creates read-only snapshots directly from an actual `AuditResult`. The provider
may read those snapshots, but provider response schemas contain no fields for
`Verdict`, `Finding`, `Severity`, `Impact`, thresholds, or deterministic
evidence construction.

The review renderer copies deterministic sections only from trusted tool
snapshots. It never parses deterministic findings from model prose. If a
review-triggered deterministic audit produces a real ClaimCI failure, that
unaltered audit result may be routed through the existing quality-gate adapter.
The advisory layer cannot upgrade, downgrade, suppress, or invent that result.

## Provider abstraction

`ReviewerProvider` has exactly two operations:

- `extract_claims(request) -> ProviderResponse`
- `synthesize_review(request) -> ProviderResponse`

`ProviderResponse` contains raw structured output plus provider name, model,
request ID when available, and usage metadata. Parsing and scientific
validation remain provider-neutral trusted code.

The OpenAI adapter is an optional dependency and is imported only when enabled.
It reads `OPENAI_API_KEY` from the runtime environment through the SDK default;
ClaimCI never prints, serializes, logs, reads back, or writes the key. Tests
inject mock providers/clients and never make paid calls. Real calls occur only
for an explicit manual/live smoke test.

The OpenAI adapter uses the Responses API with strict JSON Schema structured
output, no tools, fixed endpoint behavior, low reasoning effort, zero retries,
and the configured timeout/output-token limit. The default model is
`gpt-5.6-terra`; `CLAIMCI_OPENAI_MODEL` may override it in trusted runtime
configuration.

## Cost and context controls

Default hard limits are:

- maximum provider calls: 2;
- maximum serialized logical context per provider request: 60,000 characters;
- maximum provider output across the audit: 24,000 characters;
- maximum selected files: 24;
- maximum characters from one file: 16,000;
- maximum extraction output tokens: 5,000;
- maximum synthesis output tokens: 4,000;
- timeout per call: 30 seconds;
- retries: 0.

The orchestrator accounts for each provider request's serialized input and raw
output before accepting responses. Limit exhaustion produces an advisory
unavailable/partial status and cannot affect deterministic findings.

Every live provider response records, when supplied by the provider:

- input tokens;
- output tokens;
- total tokens;
- estimated USD cost.

Usage is stored per call and totaled in the review. Cost is calculated only
when a trusted model price is configured/known; otherwise it is `null`.
Usage/cost is observability metadata only and is excluded from all scientific
status and gate decisions.

## Synthesis and presentation

The second response may contain only:

- interpretation text for an accepted claim ID;
- citations to issued repository-evidence IDs;
- missing-evidence notes;
- unsupported-inference notes;
- an advisory confidence.

Trusted code derives presentation labels such as
`DETERMINISTIC_EVIDENCE`, `LLM_INTERPRETATION`, `MISSING_EVIDENCE`, and
`UNSUPPORTED_INFERENCE`. The model does not return a ClaimCI verdict.

GitHub Markdown begins with `ClaimCI Research Review (Advisory)` and states that
only the separate deterministic ClaimCI Audit can block. Each claim shows the
exact quote/location, deterministic evidence copied from actual audit results,
bounded repository evidence, explicitly labeled LLM interpretation, missing
evidence, and unsupported inference. Artifact/model text is escaped with the
same defensive standards as Day 2 reports.

The JSON form is strict, sorted, finite, schema-versioned, and produced from the
same immutable review result. It includes provider/call usage but never secrets
or full prompts.

## GitHub integration

The existing `ClaimCI Audit` workflow and Check remain authoritative. An
optional review step is enabled only by the trusted base
`.claimci/review.yaml` configuration.

The provider step receives `OPENAI_API_KEY` but not `GH_TOKEN` or Checks-write
credentials. A later trusted publish step may create a separate
`ClaimCI Research Review` Check whose Day 3 conclusion is always `neutral`.
The review is also appended to the job summary. One CLI invocation writes both
JSON and Markdown so the provider is called exactly twice, not once per view.

Disabled, unavailable, malformed, timed-out, or over-budget reviews remain
advisory and never change the existing job/check conclusion. The workflow must
document that opt-in can send selected private repository content to an
external provider, including for fork pull requests.

## Failure policy

- Malformed provider output is never repaired by another LLM call.
- Unknown claim IDs, evidence IDs, paths, hashes, or extra schema fields are
  rejected.
- Prompt-injection and policy-override strings remain quoted untrusted data.
- Provider refusal, timeout, quota error, missing SDK/key, or invalid output
  yields `REVIEW_UNAVAILABLE` with a fixed safe explanation.
- Raw malformed provider output and secrets are not rendered or logged.
- Advisory CLI/provider failures do not alter deterministic audit exit codes.

## Verification requirements

Automated tests use only fake providers and mocked OpenAI clients. Coverage
must include:

- every supported claim type and exact source-location validation;
- hallucinated/misquoted claims;
- malformed, duplicate-key, non-finite, deeply nested, unknown-field, and
  oversized provider output;
- selective evidence routing and false-positive cases;
- absolute, traversal, NUL, missing, directory, special-file, and symlink paths;
- nonexistent model-suggested evidence paths;
- prompt injection in every source class and attempts to override ClaimCI
  policy;
- invented findings, verdicts, severities, impacts, thresholds, and citations;
- call, context, file, output, timeout, and usage accounting boundaries;
- missing key/SDK, provider errors, and zero paid/network calls in tests;
- Markdown/JSON parity, escaping, deterministic ordering, and review size;
- default-disabled configuration and no-key deterministic compatibility;
- unchanged existing audit outputs, exits, and GitHub Check conclusions.

The complete existing suite runs after every implementation/fix wave. A final
independent read-only reviewer assesses product-goal alignment, authority
separation, data egress, prompt injection, cost controls, and regressions before
completion.
