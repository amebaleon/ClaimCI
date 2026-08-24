# Metric Identity v1 Design

## Goal and authority

Metric Identity v1 lets ClaimCI select one measured metric from bounded passive
multi-metric result artifacts without treating a lexical alias as global
authority. The authority sequence is:

```text
bounded passive MetricCandidate extraction
  -> conservative MetricIdentity comparison
  -> exact match, or one explicit pair approval
  -> exact selector integration into the repository mapping
  -> exact-head pair revalidation
  -> native Audit
```

Provider output cannot construct a candidate, identity, binding, approval, or
Audit finding. Customer repository content remains passive and is never
executed.

## Candidate identity

`MetricCandidate` is factory-created only by fixed ClaimCI adapters. Each
candidate commits independently to:

- candidate ID;
- experiment role;
- artifact path;
- artifact SHA-256;
- fixed adapter ID;
- complete selector identity;
- raw metric name.

Candidate extraction is limited to `RESULTS` and `BENCHMARK` artifacts already
accepted by the fixed JSON, JSONL, native-results, CSV, or TSV adapters. It
reuses their parse, graph, record, column, and logical-record bounds and adds a
32-candidate-per-artifact limit. Code and Markdown indexing, arbitrary parser
plugins, and large-artifact streaming are excluded.

## Conservative lexical identity

`MetricIdentity` compares one candidate's raw name with one canonical claim
metric. Exact case-folded text is an exact identity. V1 has one fixed lexical
alias required by the Pilot: `acc` for `accuracy`. It does not perform fuzzy,
substring, embedding, provider, or repository-learned matching. An identity is
candidate-scoped and carries no approval authority.

The lexicon is fixed ClaimCI policy. User approval never mutates it and is
never stored as a repository-wide `acc -> accuracy` rule.

## Atomic pair approval

`MetricBinding` always contains exactly two independent candidate identities:
one baseline and one candidate. A unique exact/exact pair can be bound
deterministically. A lexical alias, or more than one viable pair, produces one
bounded `MetricBindingQuestion` whose proposals enumerate the exact two
candidates. Approving one proposal creates one binding over the whole pair.

When both roles use `acc`, the single proposal label is `[acc]`. That one user
action approves both enumerated candidates atomically; it is not two partial
approvals. A binding records repository, exact head, canonical metric,
authority, approver when applicable, and both complete candidate identities.
There is no partial binding and no factory that approves one role.

An approved binding is reusable only when repository, head, canonical metric,
and both exact candidates are unchanged. A stale supplied approval blocks
automatic fallback and produces a new approval requirement.

## Mapping and normalized evidence

Applying a binding adds only its two exact `metric_value` selectors to the
corresponding baseline and candidate result bindings. Existing path, kind,
role, dataset split, provenance, trust, and unrelated selectors remain intact.
Repository mapping approval and metric-pair approval remain separate
authorities.

Bound extraction reruns the fixed adapter with the selected selector. The raw
metric name must still equal the candidate identity before the observation is
projected to the canonical claim metric. The raw name remains in the binding
and Audit evidence; only the confined native Audit payload uses the canonical
name.

## Exact-head materialization

An executable plan may carry an additive `MetricBinding`. Its plan identity
commits the complete pair. At materialization ClaimCI captures the selected
bytes, then independently re-extracts both candidate sets and the selected
observations with the fixed adapters. It revalidates repository, head, role,
path, hash, adapter, selector, raw name, candidate ID, and canonical projection.

Both endpoints are revalidated as one operation. If either endpoint is absent,
changed, forged, or stale, materialization invalidates the entire binding and
fails closed with re-approval required. It never executes Audit with one old
and one new endpoint.

## Focused Audit integration

After successful exact-head pair revalidation, native Audit receives a trusted
bounded context and emits `RESULT.METRIC_IDENTITY_VERIFIED` with `Impact.NONE`.
The evidence enumerates both exact candidate identities and the pair authority.
The existing result comparison and `determine_verdict()` remain the only
numeric claim/verdict path.

## Compatibility and exclusions

Direct `audit_research`, manifests, CLI output, current adapters, current
repository mappings, historical plan IDs, and zero-configuration single-metric
flows remain byte- and behavior-compatible when no Metric Binding is supplied.
The new fields and arguments are optional and default to absent.

V1 does not add large artifact streaming, Code/Markdown indexing, fuzzy metric
matching, global alias persistence, ClaimCI-Web behavior, deployment, release
certification, or changes to the certified release PR.

## Acceptance

Focused tests cover bounded JSON/JSONL/CSV/native multi-metric extraction,
factory-only contracts, conservative `acc` identity, the single `[acc]` atomic
approval, no future candidate inheritance, exact mapping integration, bound
normalization, whole-pair stale invalidation at materialization, focused Audit
evidence, provider-authority rejection, and unchanged zero-config behavior.
