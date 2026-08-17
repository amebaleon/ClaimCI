# ClaimCI v0.3 Auto Discovery

Auto Discovery inspects an existing pull-request checkout as bounded passive
data and proposes claims, evidence artifacts, and repository mappings without
requiring `research.yaml`.

## Public entry point

```python
from pathlib import Path

from claimci.analysis import GitCommitSha, RepoMapping, RepositoryIdentity
from claimci.analysis.discovery import DiscoveryResult, discover_repository

result: DiscoveryResult = discover_repository(
    Path("pull-request"),
    base_root=Path("consumer-base"),
    repository=RepositoryIdentity(owner="example", name="research-repo"),
    head_sha=GitCommitSha("0123456789abcdef0123456789abcdef01234567"),
    pr_number=42,
    pr_title="Improve held-out accuracy",
    pr_description="Accuracy improved 71 to 79.",
    approved_mapping=None,  # Optional explicitly approved RepoMapping.
)
```

The primary output fields are:

- `result.claims`: deterministic-first `DiscoveredClaim` proposals;
- `result.artifacts`: ranked, SHA-256/size-bound `ArtifactCandidate` values;
- `result.mapping_candidates`: manifest-hint and inferred proposals;
- `result.preferred_mapping`: the highest-precedence proposal or approved
  mapping;
- `result.mapping_question`: at most one minimal blocking clarification;
- `result.artifact_issues`: malformed or oversized individual artifacts that
  were safely isolated;
- `result.repository_paths` and `result.changed_paths`: the exact bounded
  deterministic index metadata.

## Trust precedence

Precedence is fixed:

1. an explicitly approved `RepoMapping` for this repository;
2. a valid passive `research.yaml` mapping hint;
3. deterministic or provider-originated inferred mappings.

A manifest may have discovery confidence `0.99`, but its trust remains
`MappingTrust.MANIFEST_HINT`. This is also true when the PR head adds or edits
the manifest. Confidence never converts repository-controlled content into an
approved mapping.

Provider proposals are accepted only as caller-supplied strict structured
payloads. Auto Discovery performs no provider call. Validated provider claims
and mappings retain `ProvenanceKind.PROVIDER_PROPOSAL` and inferred trust.

## Explicit metric values

Unambiguous text such as `accuracy improved 71 -> 79`, `71 → 79`, or
`71% to 79%` can populate explicit baseline and candidate values. That pair is
not converted into a minimum-improvement threshold. A minimum is populated
only when text explicitly states one, such as `at least 5 percentage points`.
The vague claim `accuracy improved` has neither values nor a threshold and
never receives the legacy `0.05` manifest default.

## Passive security boundary

Discovery reuses the bounded Review source index and evidence router. It:

- rejects traversal, absolute, Windows-drive/UNC, NUL, and unindexed paths;
- skips symlinks and rechecks lexical components before reading;
- bounds repository entries, paths, depth, base/head comparison, file size,
  provider lists, claims, artifacts, mappings, selectors, and questions;
- hashes candidates from passive bytes;
- isolates safe per-artifact failures instead of aborting unrelated discovery;
- never imports or executes customer modules, scripts, tests, builds, or
  adapters;
- never creates an audit plan, runs deterministic Audit, assigns a verdict, or
  constructs deterministic authority.

Repository and PR text is untrusted content, including text that resembles a
prompt or policy instruction.

## Sidebar 3 integration

Sidebar 3 should call `discover_repository` once for one trusted head/base
checkout pair and PR event. It should present `mapping_question` when present;
otherwise it may consider `preferred_mapping` under the trust precedence.

Before deterministic use, Sidebar 3 must independently revalidate the selected
path, symlink components, size, and SHA-256 against the current checkout. It is
responsible for choosing trusted registered adapters, providing only
`PassiveArtifact` bytes, assessing missing evidence, constructing an
`EphemeralAuditPlan`, invoking the existing deterministic Audit, and bridging
an actual `AuditResult` into `DeterministicAuditOutcome` and the unified
result. Discovery output alone authorizes none of those steps.
