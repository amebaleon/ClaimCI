# Verdict Regression v1 Design

## Goal and authority

Verdict Regression v1 classifies transitions between two already-produced
ClaimCI analysis outcomes. It is historical, contextual metadata only. It does
not select history, change an existing `AuditResult`, create a verdict, alter a
`Finding` or `Impact`, reinterpret current authority, or classify operational
availability as scientific regression.

A verdict regression requires an actual `ReplayAuditCommitment` and compatible
`VerificationInputSnapshot` on both sides. A deterministic payload or verdict
string without Replay authority is legacy and not comparable.

## Public contracts

`claimci.analysis.regression` owns four fixed enums:

- `ExecutionClassification`: `DETERMINISTIC_COMPLETE`,
  `PRE_AUDIT_EVIDENCE_PARTIAL`, `MAPPING_NEEDED`,
  `POST_AUDIT_ADVISORY_PARTIAL`, `OPERATIONAL_UNAVAILABLE`,
  `PROVIDER_UNAVAILABLE`, and `RUNNER_SNAPSHOT_UNAVAILABLE`.
- `OutcomeComparability`: `EXACT`, `CAPTURED_SCOPE`, `CLAIM_CHANGED`,
  `MEASUREMENT_CHANGED`, `BASELINE_CHANGED`, `ENGINE_CHANGED`,
  `INDETERMINATE`, and `LEGACY_UNAVAILABLE`.
- `VerdictTransitionKind`: the complete supported/not-supported/insufficient/
  no-verdict transition matrix, plus `UNCHANGED`,
  `COMPLETENESS_REGRESSION`, `COMPLETENESS_RECOVERY`, and
  `RAW_TRANSITION_NOT_COMPARABLE`.
- `RegressionClassification`: `REGRESSION`, `EVIDENCE_REGRESSION`,
  `COVERAGE_REGRESSION`, `RESOLUTION`, `UNCHANGED`, and `NOT_COMPARABLE`.

`HistoricalAnalysisOutcome` is immutable, final, and constructor-sealed. Its
factory accepts only an exact validated `UnifiedAnalysisResult`. It records:

- the derived execution classification;
- the current deterministic verdict, if one exists;
- the exact Replay Audit commitment, recipe, and verification snapshot, when
  the unified result carries them.

For `UNAVAILABLE`, the trusted host must supply one of the three typed
unavailable classifications. Free-form reason text is never parsed. Supplying
an unavailable classification for another lifecycle state is rejected.

`VerdictRegressionReport` is immutable and final. It records previous/current
execution classifications and verdicts, comparability, transition kind,
regression classification, stable Audit digests, and frame/input digests when
available. It has no severity-bearing `Finding`, `Impact`, or authoritative
verdict field. `classify_verdict_regression(previous, current)` is the only
comparison entry point.

## Execution classification

The factory maps validated lifecycle state as follows:

| Unified outcome | Classification |
| --- | --- |
| `COMPLETE` with deterministic authority | `DETERMINISTIC_COMPLETE` |
| `PARTIAL` without deterministic authority | `PRE_AUDIT_EVIDENCE_PARTIAL` |
| `MAPPING_NEEDED` | `MAPPING_NEEDED` |
| `PARTIAL` with deterministic authority | `POST_AUDIT_ADVISORY_PARTIAL` |
| `UNAVAILABLE` plus trusted operational kind | the exact supplied unavailable classification |

Advisory content never supplies a verdict or changes this mapping.

## Comparability

Comparability is derived only when both sides have exact Replay commitments and
non-unavailable verification snapshots.

Checks run in this order:

1. Missing Replay commitment or snapshot for a deterministic side yields
   `LEGACY_UNAVAILABLE`.
2. Different `VerificationClaimIdentity.audit_spec_sha256` yields
   `CLAIM_CHANGED`. Claim prose alone is not sufficient.
3. Different baseline or candidate measurement `semantic_protocol_id` yields
   `MEASUREMENT_CHANGED`. Measurement `source_snapshot_id` is provenance only
   and does not affect comparability.
4. Different `EvidenceProfileIdentity.profile_semantics_sha256` yields
   `INDETERMINATE`; profile selection IDs alone do not.
5. Different `AuditSemanticsCompatibility.audit_semantics_sha256` yields
   `ENGINE_CHANGED`. Exact build revision, package provenance, and distribution
   digest do not.
6. Equal `comparison_frame_sha256` and equal
   `captured_audit_input_sha256` yield `EXACT`.
7. Equal comparison frames with different captured Audit inputs yield
   `CAPTURED_SCOPE`; this is the normal comparable case for changed candidate
   result values under the same claim, ruler, baseline, and engine semantics.
8. If comparison frames differ and both complete snapshots prove different
   baseline/reference Audit-semantic evidence, the result is
   `BASELINE_CHANGED`.
9. Any remaining frame difference is `INDETERMINATE`.

Only `EXACT` and `CAPTURED_SCOPE` are compatible scientific comparison frames.
Source-only or formatting-only changes may change full snapshot identity while
leaving both narrower digests comparable.

## Transition and regression rules

Raw transition kind preserves the observed verdict movement. When two verdicts
exist but the frame is not `EXACT` or `CAPTURED_SCOPE`, the kind is
`RAW_TRANSITION_NOT_COMPARABLE` and classification is `NOT_COMPARABLE`.

For compatible outcomes with two actual Audit commitments:

| Previous | Current | Classification |
| --- | --- | --- |
| `SUPPORTED` | `NOT_SUPPORTED` | `REGRESSION` |
| decisive verdict | `INSUFFICIENT_EVIDENCE` | `EVIDENCE_REGRESSION` |
| `NOT_SUPPORTED` | `SUPPORTED` | `RESOLUTION` |
| `INSUFFICIENT_EVIDENCE` | decisive verdict | `RESOLUTION` |
| same verdict | same completeness | `UNCHANGED` |
| complete | post-Audit advisory partial, same verdict | transition `COMPLETENESS_REGRESSION`, classification `UNCHANGED` |
| post-Audit advisory partial | complete, same verdict | transition `COMPLETENESS_RECOVERY`, classification `RESOLUTION` |

For outcomes without two Audit commitments:

- a previous actual Audit followed by `PRE_AUDIT_EVIDENCE_PARTIAL` is
  `COVERAGE_REGRESSION`; the current verdict is null and this is not a verdict
  regression;
- `MAPPING_NEEDED` is workflow attention and `NOT_COMPARABLE`;
- all operational/provider/runner-snapshot unavailable transitions are
  `NOT_COMPARABLE`;
- a pre-Audit evidence partial followed by an actual deterministic Audit is
  `COMPLETENESS_RECOVERY` and contextual `RESOLUTION`;
- any other no-verdict transition is `NOT_COMPARABLE`.

## Security and compatibility

- The classifier accepts no provider payload and has no generic deserializer.
- Provider prose, unavailable reason text, advisory review, mapping labels,
  and confidence do not affect scientific comparison.
- `RepoMapping.approve()` remains mapping authority only.
- Existing `AuditResult`, `Finding`, `Impact`, `Verdict`, Replay contracts,
  Evidence Trace, CLI, Hosted behavior, and provider limits remain unchanged.
- Exact Core build changes remain comparable when
  `AuditSemanticsCompatibility` and all frame commitments match.
- V1 adds no D1 schema, Web history selection, check rendering, deployment,
  or history lookup.

## Stale Evidence stacking

This branch is based exactly on Replay Draft PR #23 head
`a358892bed036d6ff9439e845a63384fd00fc700`. The completed Stale Evidence sibling
was inspected at exact head `5d1826e5f6f89535f38c41fdf505578e350f1bc5`.
Its assessment is intentionally asymmetric: it asks whether a prior Replay
input and a current exact-head snapshot require reverification. In particular,
changed candidate Audit-semantic evidence is a reverification trigger. Verdict
Regression instead compares two already-completed Audits and must retain
`CAPTURED_SCOPE` comparability when candidate values changed under the same
claim, measurement, baseline, profile, and Audit semantics. Reusing the Stale
assessment would conflate “rerun required” with “historically non-comparable.”
The modules therefore remain independent siblings over the same Replay and
verification contracts; their only patch overlap is additive package exports.

## Acceptance

Tests cover every raw verdict transition; all execution classifications; all
comparability states and precedence; actual-Audit commitment gating;
source-only measurement drift; exact build versus semantics compatibility;
coverage/completeness distinctions; mapping and unavailable cases; legacy
results; constructor sealing; immutability; advisory/provider non-authority;
and non-mutation of current findings, impacts, and verdicts.
