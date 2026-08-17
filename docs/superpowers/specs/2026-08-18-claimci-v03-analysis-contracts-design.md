# ClaimCI v0.3 Analysis Contracts Design

Date: 2026-08-18

## Objective

Add an immutable, validated, standard-library-only contract layer that later
Claim Discovery, Evidence Discovery, Adapter, Planner, and integration work can
share. The contracts describe a zero-configuration analysis path without
implementing discovery algorithms or changing the existing deterministic Audit
or advisory Research Review.

The future primary path is:

> GitHub App installation -> existing research pull request -> claim discovery
> -> evidence discovery -> adapter matching -> ephemeral audit plan ->
> deterministic verification -> advisory Research Review -> unified result.

Ordinary users are not required to add `research.yaml` or convert result files
to ClaimCI-specific JSON. A legacy manifest remains an optional,
high-confidence `MANIFEST` hint and advanced override.

## Non-goals

This branch does not implement discovery algorithms, concrete production
adapters, plan execution, a GitHub App, hosted services, workflow changes,
provider calls, or filesystem execution. It does not change existing CLI
behavior, deterministic audit semantics, Research Review orchestration, or the
two-call/zero-retry provider boundary.

## Package boundary

The additive package is:

```text
claimci/analysis/
    __init__.py
    confidence.py
    contracts.py
```

`confidence.py` owns the finite `[0, 1]` confidence value. `contracts.py` owns
portable values, enums, immutable records, the adapter protocol, and safe
serialization. `__init__.py` is the single documented import surface.

The package uses only the Python standard library and the existing
deterministic `AuditResult`, `Verdict`, and JSON renderer. It does not import a
provider SDK, the Research Review orchestrator, shell/process APIs, or workflow
code.

## Portable validated values

- `Confidence` rejects booleans, non-numeric values, NaN, infinity, and values
  outside `[0, 1]`.
- `RepositoryPath` is a canonical repository-relative POSIX file path. It
  rejects empty paths, surrounding whitespace, NUL, backslashes, absolute
  POSIX paths, Windows drives and UNC paths, `.` or `..` segments, repeated
  separators, and non-canonical spellings.
- `Sha256Digest` accepts exactly 64 lowercase hexadecimal characters.
- `GitCommitSha` accepts a non-zero lowercase 40- or 64-character hexadecimal
  commit identifier.

Path contracts do not replace runtime filesystem confinement. Later consumers
must still resolve paths under the issued checkout root, reject symlink
components and non-regular files, and verify expected size and digest before
deterministic use.

## Enumerations

The public vocabulary is:

- `ArtifactKind`: `RESULTS`, `CONFIG`, `DATASET`, `BENCHMARK`, `DOCUMENT`,
  `SOURCE`, `TEST`, `MANIFEST`.
- `ExperimentRole`: `BASELINE`, `CANDIDATE`, `REFERENCE`, `UNSPECIFIED`.
- `ProvenanceKind`: `DETERMINISTIC_DISCOVERY`, `ADAPTER_EXTRACTION`,
  `MANIFEST_HINT`, `PROVIDER_PROPOSAL`, `USER_APPROVED`.
- `MappingTrust`: `INFERRED`, `MANIFEST_HINT`, `USER_APPROVED`.
- `SelectorKind`: `JSON_POINTER`, `DOTTED_PATH`, `COLUMN`.
- `AnalysisState`: `COMPLETE`, `MAPPING_NEEDED`, `PARTIAL`, `UNAVAILABLE`.
- `AnalysisAuthority`: `DETERMINISTIC`, `ADVISORY`.

Wire values are stable lowercase strings except deterministic `Verdict`, whose
existing uppercase wire values remain unchanged.

## Provenance and declarative selectors

`FieldProvenance` records how a field was obtained, with a bounded explanation
and optional confined source path/source identifier. Inferred normalized fields
must carry provenance; provider proposals remain proposals and do not become
trusted by receiving a high confidence score.

`EvidenceSelector` supports only bounded declarative JSON-pointer,
dotted-key-path, or column-name selectors. `FieldMapping` binds one normalized
target field to one validated selector and its provenance. Selectors are data,
not Python expressions, callables, import paths, shell fragments, or database
queries. An `adapter_id` is an identifier for a later trusted registry; it is
never dynamically imported from pull-request content.

## Artifact and adapter contracts

`ArtifactCandidate` contains the confined path, `ArtifactKind`, SHA-256, size,
confidence, discovery reason, relevant claim IDs, and provenance.

`PassiveArtifact` combines one candidate with immutable `bytes`. It is the only
artifact content surface supplied to an adapter. The shared `Adapter` protocol
is:

```python
class Adapter(Protocol):
    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None: ...
    def extract(
        self,
        artifact: PassiveArtifact,
        match: AdapterMatch,
    ) -> NormalizedEvidence: ...
```

`AdapterMatch` contains a trusted-registry adapter ID, confined path,
confidence, typed field mappings, and match-evidence provenance. It cannot
contain a repository root, loader callback, executable path, or command.

This API preserves the passive PR boundary: contracts permit bounded parsing
of supplied bytes, never execution of repository content.

## Normalized evidence

`NormalizedObservation` represents optional metric name/value, run identifier,
seed, and experiment role. It can additionally carry typed tuples of
`ConfigValue`, `DatasetReference`, and `ComputeEvidence`. No observation is
forced to contain every category, but each observation must contain at least
one substantive value. Metric names and values appear together, all numeric
values are finite, and inferred fields carry provenance.

`NormalizedEvidence` ties one or more observations to the exact
`ArtifactCandidate` and `AdapterMatch` that produced them. Their paths must
match. Its tuple-based structure is deeply immutable.

## Mapping and explicit approval

`ArtifactBinding` assigns one confined artifact path and kind to an experiment
role, optional adapter ID, typed field mappings, and provenance.

`MappingCandidate` is always a proposal. It contains one or more bindings,
confidence, provenance, and `MappingTrust.INFERRED` or
`MappingTrust.MANIFEST_HINT`. The same path cannot be assigned conflicting
baseline and candidate roles.

`MappingChoice` and `MappingQuestion` describe the smallest required user
clarification. A question has two through eight unique candidate choices, a
bounded prompt, and no free-form executable data.

`RepoMapping` is a distinct trust transition. Its public constructor is
disabled; `RepoMapping.approve(...)` creates it from a validated
`MappingCandidate`, repository identity, and explicit approver identity. Its
trust is fixed to `USER_APPROVED`. Provider-output deserialization does not
construct `RepoMapping` values.

A legacy `research.yaml` can therefore be represented as:

1. an `ArtifactCandidate(kind=ArtifactKind.MANIFEST)`;
2. manifest-hint provenance and high confidence;
3. baseline/candidate `ArtifactBinding` values for config, results, train, and
   evaluation datasets;
4. a `MappingCandidate(trust=MappingTrust.MANIFEST_HINT)`.

This representation is optional and does not make the manifest a prerequisite
for discovery or planning.

## Ephemeral audit plan

`RepositoryIdentity` identifies repository owner/name. `ClaimReference`
contains a bounded claim ID/text, optional confined source path, confidence,
and provenance. `MissingEvidence` records a typed missing artifact/role and
explanation.

`EphemeralAuditPlan` is tied to repository identity, optional positive PR
number, head SHA, claim, baseline evidence, candidate evidence, mapping
provenance, missing evidence, and overall confidence. It serializes with
`"ephemeral": true`, exposes no commit/write method or committed repository
path, and never implies that a customer repository contains a generated plan.

## Unified result and authority boundary

`DeterministicAuditOutcome` has no public constructor. The only public creation
path is:

```python
DeterministicAuditOutcome.from_audit_result(result: AuditResult)
```

The factory rejects all other objects and deep-freezes a JSON-safe snapshot of
the existing deterministic result. It does not reinterpret or recalculate the
verdict.

`AdvisoryResearchInterpretation` has fixed
`AnalysisAuthority.ADVISORY` authority and intentionally has no verdict,
severity, impact, threshold, or blocking-conclusion field.

`UnifiedAnalysisResult` keeps `deterministic` and `research_interpretation` in
separate typed fields. Its `authoritative_verdict` property is exactly:

```python
return None if self.deterministic is None else self.deterministic.verdict
```

Advisory text, confidence, mapping state, or provider status has no code path
for setting or overriding that value.

State invariants are:

- `COMPLETE` requires a deterministic outcome.
- `MAPPING_NEEDED` requires a bounded `MappingQuestion` and has no
  deterministic outcome.
- `UNAVAILABLE` requires a non-empty reason and has no deterministic outcome.
- `PARTIAL` requires at least one available result, question, or explicit
  reason; it never implies success from absence of evidence.

## Immutability and serialization

All records use frozen, slotted dataclasses; collection fields are tuples; and
arbitrary deterministic payloads are recursively copied into immutable
mappings/tuples. `to_jsonable(value)` returns a detached JSON-compatible copy,
converts enums to wire values, preserves repository paths as strings, and
rejects unsupported or non-finite values. The package provides no generic
deserializer that could manufacture an approved mapping or deterministic
authority from provider JSON.

## Validation and test strategy

`tests/test_analysis_contracts_v03.py` will cover:

- every path escape and non-canonical path class;
- confidence, size, digest, commit SHA, numeric, and selector validation;
- enum stability and artifact construction;
- deep immutability and detached JSON-safe serialization;
- conflicting baseline/candidate bindings;
- mapping-question lower/upper bounds and duplicate choices;
- `RepoMapping.approve(...)` as the only approved trust transition;
- passive adapter protocol conformance using a trivial byte-parsing fake;
- ephemeral plan identity, evidence, and serialization invariants;
- rejection of provider/dictionary attempts to construct deterministic
  authority;
- an advisory interpretation containing verdict-like text cannot change
  `authoritative_verdict`;
- all four unified-result states;
- representation of the existing legacy manifest as a high-confidence hint;
- an import/API guard against provider SDKs and PR-code execution surfaces.

The final branch gate is:

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

An independent reviewer must additionally compare the implementation, tests,
and public exports against this design before publication.

## Consumer notes for follow-on branches

- Sidebar 1 (Auto Discovery) may create candidates, claims, and inferred
  provenance, but not approved mappings or deterministic outcomes.
- Sidebar 2 (Adapters) receives only passive bytes and validated metadata; it
  must return typed selectors and normalized evidence without executing PR
  content.
- Sidebar 3 (Planner/Integration) performs runtime path/index/hash checks,
  chooses inferred versus approved mappings, creates ephemeral plans, invokes
  the existing deterministic engine, and alone bridges a real `AuditResult`
  into the unified authoritative outcome.

These contracts intentionally do not decide discovery scoring thresholds,
adapter registry contents, clarification UX, persistence format, or hosted
service architecture.
