# ClaimCI v0.3 Auto Discovery Design

Date: 2026-08-18

## Objective

Add a bounded, passive, deterministic-first discovery layer that turns an
existing research pull request into typed claim, artifact, and mapping
proposals for later adapter and planner branches. The zero-configuration path
must work without `research.yaml`; a valid manifest remains an optional
high-confidence hint and advanced override.

The public entry point is:

```python
discover_repository(...) -> DiscoveryResult
```

Discovery never executes repository content, adapters, the deterministic
Audit, or Research Review. It never creates `RepoMapping`,
`EphemeralAuditPlan`, `DeterministicAuditOutcome`, or a verdict.

## Non-goals

This branch does not implement concrete adapters, normalized evidence
extraction, audit planning or execution, GitHub workflows, hosted services,
provider calls, or the unified-result bridge. It does not change the existing
Research Review state machine or its observable source/evidence behavior.

## Additive package boundary

The implementation is additive under:

```text
claimci/analysis/discovery/
    __init__.py       public discovery surface
    models.py         immutable discovery-specific records and limits
    repository.py     bounded passive file inspection and change metadata
    claims.py         deterministic claim extraction and provider validation
    artifacts.py      artifact classification, ranking, and manifest hints
    mappings.py       mapping proposals, precedence, and one minimal question
    service.py        discover_repository orchestration
```

Existing `claimci.review.sources.collect_review_sources`,
`claimci.review.sources.validate_claim_candidates`, and
`claimci.review.evidence.discover_evidence` are reused as public behavior.
Their signatures, limits, ordering, errors, and Research Review semantics are
not changed. Discovery adapts their outputs into the shared analysis contracts
rather than moving or duplicating their implementation.

## Public API

`claimci.analysis.discovery` exports:

```python
def discover_repository(
    head_root: Path,
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    pr_number: int | None = None,
    base_root: Path | None = None,
    pr_title: str = "",
    pr_description: str = "",
    approved_mapping: RepoMapping | None = None,
    provider_claim_payload: Mapping[str, object] | None = None,
    provider_mapping_payload: Mapping[str, object] | None = None,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> DiscoveryResult: ...
```

The immutable exported records are:

- `DiscoveryLimits`: downstream claim/artifact/mapping bounds that cannot
  weaken the existing Review repository, depth, size, or context bounds.
- `ClaimedValue`: one explicit finite numeric value, optional unit, and field
  provenance.
- `DiscoveredClaim`: a shared `ClaimReference`, existing `ClaimType` and
  `ClaimDirection`, subject/metric, optional explicit baseline/candidate values,
  optional explicitly stated minimum improvement, qualifiers, evidence hints,
  and the trusted source location.
- `ArtifactIssue`: one confined path and bounded reason explaining why a
  malformed, unreadable, or oversized artifact was isolated.
- `DiscoveryResult`: repository/PR/head identity, claims, ranked
  `ArtifactCandidate` values, mapping proposals, optional approved mapping, at
  most one `MappingQuestion`, skipped-artifact issues, and deterministic index
  metadata.

`DiscoveryResult.preferred_mapping` returns the first usable mapping under the
fixed precedence below. It returns a proposal or approved value; it never
performs a trust transition.

## Repository indexing and passive source collection

`discover_repository` calls `collect_review_sources` exactly once with the
head checkout, optional base checkout, PR metadata, and compatible
`ReviewLimits`. This retains the current bounds:

- at most 2,048 indexed regular paths;
- at most 8,192 visited repository entries;
- maximum repository depth 64;
- symlinked files and directories are not indexed;
- bounded base/head comparisons;
- bounded changed-document collection and UTF-8 context.

Discovery treats `SourceBundle.repository_paths` as the only issued path
index. Every later path must be a canonical `RepositoryPath`, must be present
in that index, and must pass a second lexical-component, confinement,
regular-file, size, and digest check before bytes are inspected.

Artifact change status is computed only for the bounded shortlist. Comparison
uses deterministic path order and a separate file/byte budget. Exhausting the
comparison budget yields unknown change status, not unbounded I/O.

Malformed, unreadable, non-UTF-8, or oversized individual artifacts produce a
bounded `ArtifactIssue` and are skipped when the repository scan can safely
continue. Repository-wide traversal/depth bounds remain controlled terminal
errors because continuing would make the index incomplete in an unsafe or
non-deterministic way.

Repository text is untrusted quoted data. No content is imported, evaluated,
compiled, built, executed, or treated as an instruction.

## Deterministic claim discovery

Deterministic heuristics run before optional provider proposals. They inspect
only the bounded PR title, PR description, and changed research documents
issued by `collect_review_sources`.

The supported categories are the existing `ClaimType` values:

- metric improvement;
- compute equivalence;
- held-out evaluation;
- resource reduction;
- component causality;
- no external reward;
- implementation claim;
- other scientific claim.

Patterns are deliberately narrow and evidence-preserving. Explicit text such
as `accuracy improved 71 -> 79`, `71 → 79`, or `71% to 79%` may populate
baseline and candidate `ClaimedValue` fields when the association is
unambiguous. An explicit phrase such as `at least 5 percentage points` may
populate `minimum_improvement`. The numeric difference between 71 and 79 is
not automatically converted into a minimum threshold.

`accuracy improved` remains a metric-improvement claim with no numeric pair
and no minimum threshold. The legacy `0.05` manifest default is never injected
into text-discovered claims.

Claim IDs, confidence, provenance, and output order are deterministic.
Explicit typed/value claims rank above vague claims. Claims are ordered by
trusted source order, line position, category, and canonical text; exact
duplicates are removed without provider preference.

## Provider proposal boundary

This branch does not call an LLM. A caller may optionally supply already
obtained structured proposal payloads.

Provider claim payloads first pass the existing strict
`validate_claim_candidates` source-quote validation. Discovery then validates
all evidence hints as canonical indexed repository paths, converts confidence
to `Confidence`, and assigns `ProvenanceKind.PROVIDER_PROPOSAL`. Explicit
numeric values are accepted only when they are unambiguously present in the
validated quoted source.

Provider mapping payloads use an exact bounded schema. Paths must be canonical
and indexed; kinds and roles must be known enum values; adapter IDs must pass
the shared trusted-registry identifier grammar; selectors must construct
validated `EvidenceSelector` and `FieldMapping` values; confidence must be
finite and bounded; and every field receives provider-proposal provenance.

Provider proposals can create only `DiscoveredClaim`, `ArtifactCandidate`, or
`MappingCandidate(trust=INFERRED)` proposals. They cannot create or approve a
`RepoMapping`, execute an adapter, create a plan, run Audit, set a verdict, or
construct deterministic authority. Prompt-injection-like repository text does
not alter this policy.

## Artifact discovery and ranking

Discovery calls `discover_evidence` with validated scientific claims,
repository paths, changed paths, and validated evidence hints. It converts the
bounded evidence references to shared `ArtifactCandidate` values and
supplements them with obvious, path-classifiable results/config/dataset/
benchmark/manifest candidates that are needed for mapping.

The score is an explicit deterministic sum of bounded signals:

- approved repository mapping membership;
- valid manifest declaration;
- changed status;
- exact artifact name and path tokens;
- artifact kind and claim-type routing;
- explicit claim literals such as metric/subject;
- baseline/candidate relationship tokens;
- validated provider hint membership.

Scores are clamped into `Confidence([0, 1])`. Ties use canonical path, kind,
and claim-ID ordering. No filesystem enumeration or hash-map order is exposed.

An `ArtifactCandidate` contains the actual SHA-256 and byte size observed from
the passive head file. A malformed artifact is isolated instead of becoming a
low-confidence candidate.

## Manifest hints and trust precedence

Valid `research.yaml` or `research.yml` content is parsed as passive YAML with
the existing unique-key loader. The manifest and every declared artifact path
must be indexed, confined, regular, and bounded. Relative declarations resolve
from the manifest directory.

A valid manifest may produce:

- `ArtifactCandidate(kind=MANIFEST, confidence=0.99)`;
- baseline/candidate config, results, train-dataset, and eval-dataset
  `ArtifactBinding` values;
- `MappingCandidate(confidence=0.99, trust=MANIFEST_HINT)`;
- `FieldProvenance(kind=MANIFEST_HINT)`.

Confidence is not trust. A manifest never becomes `USER_APPROVED`, including
when it is newly added or modified by the PR head. It cannot automatically
create a `RepoMapping`, plan, deterministic outcome, or verdict.

The immutable precedence is:

1. an explicitly approved `RepoMapping` for the same repository;
2. valid `MANIFEST_HINT` mapping candidates;
3. deterministic or provider-originated inferred candidates.

Within one tier, candidates sort by descending confidence and then stable
mapping ID. An approved mapping is passed through unchanged and is never
reconstructed from repository or provider data.

## Mapping inference and minimal clarification

Inference assigns path/kind pairs to baseline, candidate, reference, or
unspecified roles using explicit path segments and filenames. It proposes
`ArtifactBinding` and `MappingCandidate` values only; it does not invoke an
adapter or read selectors dynamically.

Obvious pairs such as `baseline_results.json` and
`candidate_results.json`, with corresponding config files, can produce a
high-confidence inferred mapping without a question. Missing evidence is not
guessed.

When an ambiguity blocks a useful mapping, Discovery returns at most one
`MappingQuestion`. Candidate ambiguities are grouped by mapping slot
(artifact kind and experiment role). The chosen group maximizes:

1. number of affected claims;
2. number of downstream required slots expected to become resolvable;
3. fixed slot priority, beginning with candidate results and baseline results;
4. stable canonical path order.

The question contains only the two through eight bounded choices for that one
slot. It does not ask about unrelated mappings. Approved mappings and an
unambiguous higher-precedence manifest hint suppress lower-tier questions.

## Determinism

For identical repository bytes, base bytes, PR metadata, repository identity,
head SHA, PR number, approved mapping, provider payloads, and limits, discovery
returns equal values and identical ordering. Determinism is enforced by:

- sorted repository/index paths;
- source-order plus line-order claim traversal;
- canonical hashes for IDs;
- explicit score tuples;
- stable tie breakers;
- immutable tuple outputs;
- no timestamps, randomness, network state, locale-dependent ordering, or
  mutable global caches.

## Error handling

`DiscoveryError` is a bounded controlled error for invalid inputs, repository
root failure, unsafe repository-wide bounds, or malformed provider payloads.
Individual file failures that are safe to isolate become `ArtifactIssue`
values. Safe messages never include file contents or provider output.

## Test strategy

`tests/test_auto_discovery_v03.py` covers real temporary repositories and the
public API:

- discovery with no `research.yaml`;
- obvious results/config baseline-candidate naming;
- ambiguous candidate results returning one minimal question;
- approved mapping precedence;
- valid manifest confidence/provenance and precedence without trust elevation;
- a newly changed head manifest remains passive `MANIFEST_HINT`;
- traversal, Windows/UNC, NUL, and unindexed provider paths;
- symlink exclusion and component confinement;
- repository entry/depth/file/byte bounds;
- oversized or malformed individual artifact isolation;
- prompt-injection-like README text cannot alter policy or select a path;
- explicit `71 -> 79`, `71 → 79`, and `71% to 79%` values;
- vague `accuracy improved` without a threshold;
- explicitly stated thresholds only;
- every supported claim category and multiple claims;
- strict provider claims, mappings, selectors, hints, confidence, and bounds;
- provider proposals cannot create approved or deterministic types;
- stable ordering under repeated runs;
- no adapter, Audit, import, build, or repository-code execution;
- unchanged Research Review source/evidence regression suites.

The final branch gate is:

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

## Sidebar 3 consumption

Sidebar 3 imports `discover_repository`, calls it once for the trusted head/base
checkout pair and PR metadata, and consumes:

- `result.claims` as typed claim proposals;
- `result.artifacts` as hash/size-bound passive artifact candidates;
- `result.preferred_mapping` under the fixed trust precedence;
- `result.mapping_question` when user clarification is required;
- `result.artifact_issues` as non-authoritative missing/invalid evidence.

Sidebar 3 must revalidate paths, symlinks, size, and digest at use time; run
trusted adapters over `PassiveArtifact` bytes; decide whether mapping evidence
is sufficient; construct the ephemeral plan; invoke the existing Audit; and
alone bridge a real `AuditResult` into deterministic authority. Discovery
outputs never authorize those actions by themselves.
