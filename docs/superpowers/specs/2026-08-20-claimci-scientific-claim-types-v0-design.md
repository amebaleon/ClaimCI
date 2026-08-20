# Scientific Claim Type System v0 Design

## Goal

Introduce a canonical, source-bound scientific claim taxonomy between discovery and deterministic planning without changing Audit authority, legacy CLI behavior, or the Evidence Obligations and Measurement Drift systems.

The canonical flow is:

```text
ClaimReference
+ one PrimaryScientificClaim
+ zero or more orthogonal ClaimConstraints
→ ClaimEvidencePolicy seam
→ deterministic compiler when supported
→ AuditClaimSpec
```

Recognition is not deterministic verification. Only `MetricImprovementClaim` has a v0 compiler. Recognized unsupported types stop before evidence/mapping selection with the exact partial reason `unsupported_deterministic_claim_compiler`.

## Frozen scope

Primary types:

- `MetricImprovementClaim`: recognized and deterministically executable.
- `AbsoluteMetricClaim`: recognized, not executable in v0.
- `GeneralizationClaim`: recognized, not executable in v0.
- `GenericQuantitativeClaim`: recognized, not executable in v0.

Orthogonal constraint:

- `EvaluationConstraint(HELD_OUT)`.

Precedence:

1. Executable metric improvement, including held-out wording.
2. Exact single-role metric bound.
3. Irreducible broad generalization.
4. Other unambiguous finite bounded quantitative language.

Malformed, non-finite, or ambiguous quantitative language does not become a canonical claim.

## Trust boundary

Canonical fields are recovered from `ClaimReference.text` and its validated source provenance. Provider claim type, metric, direction, values, thresholds, qualifiers, and constraints are proposals only. They never choose the canonical primary type or create `AuditClaimSpec` fields.

The deterministic compiler re-recovers the canonical claim from the reference before compiling. Only an exact `MetricImprovementClaim` whose threshold semantics are representable by the native Audit can produce `AuditClaimSpec`. Unit-bearing minimum thresholds are recognized but compiler-unsupported rather than silently converted or discarded. All other primaries have no compiler in v0.

The existing advisory Review bridge uses this same recovery/compiler boundary. Provider fields must agree with the recovered primary, metric, direction, and any explicit threshold; they cannot upgrade or rewrite source semantics.

The canonical claim and constraint contracts contain no verdict, severity, impact, or authority field. `AuditResult` remains the sole verdict authority.

## Contracts

`CanonicalScientificClaim` contains:

- the existing `ClaimReference`;
- exactly one concrete primary claim;
- a unique tuple of supported constraints.

The primary records carry only bounded source-derived fields and their `FieldProvenance`. Metric improvement carries metric, direction, optional source-quoted baseline/candidate values, and optional explicit minimum improvement. Absolute metric carries one experiment role, comparator, and bound. Generalization and generic quantitative records preserve only their bounded source-derived semantic fields.

`ClaimEvidencePolicy` is an immutable handoff seam containing a stable policy ID, primary kind, constraint kinds, and optional deterministic compiler ID. It does not contain or evaluate evidence obligations in this branch.

## Planning behavior

`PlanningRequest.audit_claim` becomes optional only for a recognized canonical claim without a v0 compiler. The request carries its canonical claim and policy seam. It rejects any attempt to pair an unsupported primary with an `AuditClaimSpec`.

`plan_ephemeral_audit()` checks compiler support before missing-evidence and mapping logic. Unsupported canonical claims return:

```text
state = PARTIAL
reason = unsupported_deterministic_claim_compiler
```

They cannot return `MAPPING_NEEDED` and cannot reach materialization or Audit.

For metric improvement, the compiler produces the same audit-relevant projection as the existing planner. `HELD_OUT` is semantic provenance only in v0: it does not alter `AuditClaimSpec`, mapping selection, or plan identity.

## Evidence Trace

Executable plans carry the canonical claim in addition to the existing `ClaimReference`. The existing natural-language claim trace entry adds a bounded normalized semantic projection with primary kind/fields and constraint kinds. `HELD_OUT` therefore survives into trace provenance.

The trace record remains `NON_AUTHORITATIVE_INPUT`; constraints have no consumer rule or verdict authority. Provider proposal trace remains separately non-authoritative.

## Compatibility

- Existing `AuditResult`, `AuditClaimSpec`, and native CLI contracts are unchanged.
- Existing manifest execution is unchanged.
- Hosted analysis does not require `research.yaml`.
- Existing metric-improvement claims retain deterministic execution and plan identity.
- Existing advisory Review claim categories remain available; the new canonical taxonomy owns deterministic compilation.
- Customer repository content remains passive and is never executed.
- Evidence Obligations and Measurement Drift are explicitly out of scope.

## Verification

Tests must cover immutable contracts, precedence, held-out constraints, unsupported compiler partials, provider type/field invention, malformed quantitative text, Audit non-reachability, metric plan identity compatibility, trace preservation, and absence of verdict authority.
