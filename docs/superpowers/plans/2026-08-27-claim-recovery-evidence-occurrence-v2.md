# Claim Recovery and Evidence Occurrence Identity V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover bounded subject-first metric-improvement claims through one canonical parser and issue collision-safe evidence-v2 IDs from factory-trusted physical artifact occurrences.

**Architecture:** Extend only `recover_scientific_claim()`, then project its typed output into deterministic discovery. Add a factory-issued occurrence companion to passive and streaming artifact wrappers; every built-in adapter hashes that physical scope plus a fixed adapter version and canonical selector material into `evidence-v2-*`, while planner duplicate checks continue to prevent duplicate authority.

**Tech Stack:** Python 3.11+, frozen dataclasses/enums, `hashlib`, canonical JSON, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-claim-recovery-evidence-occurrence-v2-design.md`

## Global Constraints

- `recover_scientific_claim()` is the sole natural-language metric-improvement grammar authority.
- The grammar extension is bounded to the approved subject/verb forms and never derives thresholds arithmetically.
- Physical occurrence identity is separate from `ExperimentRole`; scientific role labels are excluded from evidence ID material.
- `ArtifactCandidate`, provider payload schemas, mapping payload schemas, `NormalizedEvidence`, and completed-result JSON shapes remain unchanged.
- Occurrence scope is factory-issued from repository, physical snapshot role, exact commit, and candidate; provider/mapping data cannot supply it.
- Evidence-v2 canonical material is exactly schema version, repository owner/name, snapshot role/commit, artifact path/kind/full SHA/size, adapter ID/version, and provenance-free canonical selector material.
- Evidence ID output is `evidence-v2-` plus the full lowercase SHA-256 digest of canonical JSON.
- No D1 migration, Audit/verdict change, provider retry, deployment, production run, merge, or `/app` gate change.
- If a migration becomes necessary, stop with `EVIDENCE_ID_SCHEMA_GATE`.
- Use strict RED → GREEN → REFACTOR for every production behavior. Record the failing and passing commands/output in the task report.
- Run only each task's focused tests during implementation. The controller runs the full Core suite exactly once after all tasks and reviews are complete.

---

### Task 1: Canonical subject-first recovery and discovery projection

**Files:**
- Modify: `claimci/analysis/claim_types.py`
- Modify: `claimci/analysis/discovery/claims.py`
- Test: `tests/test_claim_types_v0.py`
- Test: `tests/test_auto_discovery_v03.py`
- Test: `tests/test_analysis_cross_package_v03.py`

**Interfaces:**
- Consumes: existing `recover_scientific_claim(reference: ClaimReference) -> CanonicalScientificClaim | None`.
- Produces: the same public signature with bounded subject-first support; deterministic discovery projects `MetricImprovementClaim` and `ClaimQuantity` values without regex reparsing.

- [ ] **Step 1: Add canonical RED tests for every accepted form and exact values**

Add a literal table to `tests/test_claim_types_v0.py`:

```python
@pytest.mark.parametrize(
    ("text", "metric", "direction", "baseline", "candidate"),
    (
        ("accuracy improved from 0.60 to 0.70", "accuracy", Direction.HIGHER, 0.60, 0.70),
        ("candidate improves accuracy from 0.60 to 0.70", "accuracy", Direction.HIGHER, 0.60, 0.70),
        ("the candidate improves accuracy from 0.60 to 0.70", "accuracy", Direction.HIGHER, 0.60, 0.70),
        ("model improves accuracy from 0.60 to 0.70", "accuracy", Direction.HIGHER, 0.60, 0.70),
        ("candidate increases accuracy from 0.60 to 0.70", "accuracy", Direction.HIGHER, 0.60, 0.70),
        ("candidate reduces loss from 0.40 to 0.30", "loss", Direction.LOWER, 0.40, 0.30),
    ),
)
def test_metric_improvement_recovery_uses_one_bounded_grammar(
    text: str,
    metric: str,
    direction: Direction,
    baseline: float,
    candidate: float,
) -> None:
    recovered = recover_scientific_claim(_reference(text))
    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.metric == metric
    assert recovered.primary.direction is direction
    assert recovered.primary.baseline_value is not None
    assert recovered.primary.candidate_value is not None
    assert recovered.primary.baseline_value.value == baseline
    assert recovered.primary.candidate_value.value == candidate
```

- [ ] **Step 2: Add RED locality, ambiguity, and no-arithmetic-threshold tests**

Add literal assertions covering:

```python
def test_subject_first_threshold_uses_existing_explicit_continuation() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate improves accuracy from 0.60 to 0.70; "
            "improved by at least 0.05"
        )
    )
    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.minimum_improvement is not None
    assert recovered.primary.minimum_improvement.value == 0.05


def test_subject_first_lower_threshold_uses_existing_explicit_continuation() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate reduces loss from 0.40 to 0.30; "
            "reduced by at least 0.05"
        )
    )
    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.direction is Direction.LOWER
    assert recovered.primary.minimum_improvement is not None
    assert recovered.primary.minimum_improvement.value == 0.05


def test_subject_first_does_not_derive_threshold_from_arithmetic() -> None:
    recovered = recover_scientific_claim(
        _reference("candidate improves accuracy from 0.60 to 0.70; 0.70 - 0.60 = 0.10")
    )
    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.minimum_improvement is None


@pytest.mark.parametrize(
    "text",
    (
        "candidate improves accuracy from 0.60 to 0.70 and loss from 0.40 to 0.30",
        "candidate improves accuracy from 0.60 to 0.70; improved by at least 0.05; improved by at least 0.06",
        "candidate says model improves accuracy from 0.60 to 0.70 while candidate reduces loss from 0.40 to 0.30",
    ),
)
def test_subject_first_rejects_ambiguous_metric_improvement(text: str) -> None:
    assert recover_scientific_claim(_reference(text)) is None
```

- [ ] **Step 3: Add RED deterministic/provider discovery projections using the exact canonical source**

In `tests/test_auto_discovery_v03.py`, issue the exact PR description and assert one `DiscoveredClaim` has canonical metric/direction/value fields and `scientific_claim`:

```python
claim_text = (
    "The candidate improves accuracy from 0.60 to 0.70 under the same "
    "configuration and evaluation dataset."
)
result = discover_repository(
    head,
    repository=RepositoryIdentity("amebaleon", "claimci-production-smoke-fixture"),
    head_sha=GitCommitSha("d" * 40),
    pr_number=1,
    pr_title="Production smoke: verify supported result",
    pr_description=claim_text,
)
assert len(result.claims) == 1
assert result.claims[0].reference.text == claim_text
assert result.claims[0].metric == "accuracy"
assert result.claims[0].direction is ClaimDirection.HIGHER
assert result.claims[0].baseline_value is not None
assert result.claims[0].baseline_value.value == 0.60
assert result.claims[0].candidate_value is not None
assert result.claims[0].candidate_value.value == 0.70
assert result.claims[0].minimum_improvement is None
```

In `tests/test_analysis_cross_package_v03.py`, add provider-projection tests for the exact subject-first claim, an ambiguous two-metric claim, and the arithmetic suffix. Assert the validated exact source span produces the same baseline/candidate values as deterministic discovery, ambiguity is rejected through the existing provider validation boundary, and the arithmetic suffix leaves `minimum_improvement is None`.

- [ ] **Step 4: Run the focused RED tests and record the expected failures**

Run:

```powershell
python -m pytest tests/test_claim_types_v0.py tests/test_auto_discovery_v03.py tests/test_analysis_cross_package_v03.py -q
```

Expected: new subject-first recovery and discovery assertions fail because the canonical parser currently recognizes only metric-first grammar.

- [ ] **Step 5: Implement the bounded canonical grammar**

In `claim_types.py`, add one explicit subject-first regex and normalize both past/present verbs through `_verb_direction`:

```python
_SUBJECT_METRIC_IMPROVEMENT = re.compile(
    r"\b(?:the\s+)?(?:candidate|model)\s+"
    r"(?P<verb>improves|increases|raises|grows|reduces|decreases|lowers|drops)\s+"
    r"(?P<metric>[A-Za-z][A-Za-z0-9_.-]{0,63})\b",
    flags=re.IGNORECASE,
)

_LOWER_VERBS = frozenset(
    {"decreased", "fell", "dropped", "reduced", "reduces", "decreases", "lowers", "drops"}
)
```

Change `_verb_direction()` to return `Direction.LOWER` exactly when `verb.casefold() in _LOWER_VERBS`. Refactor `_recover_primary()` to collect matches from `_METRIC_IMPROVEMENT` and `_SUBJECT_METRIC_IMPROVEMENT`, require exactly one total match, and reuse the existing tail/value-pair/threshold locality logic unchanged. Do not add subtraction or generic-number logic.

- [ ] **Step 6: Remove metric grammar from discovery and project canonical values**

Delete `_METRIC_PAIR`, `_METRIC_VAGUE`, `_VALUE_PAIR`, `_MINIMUM`, `_explicit_pair`, `_explicit_minimum`, and `_claimed_values` from `discovery/claims.py`, including the metric-improvement branch in `_classify_line()`. Use a bounded provisional `ClaimReference` only to ask `recover_scientific_claim()` for the canonical primary kind needed by `_claim_id`; legacy classifiers run only when canonical recovery returns `None` and may not emit `ClaimType.METRIC_IMPROVEMENT`. Then create the final exact-ID `ClaimReference`, call `recover_scientific_claim(final_reference)`, and project values only from that final recovery. `DiscoveredClaim.__post_init__` independently enforces the same final-reference recovery. For a final `MetricImprovementClaim`, convert its typed quantities with:

```python
def _claimed_value(quantity: ClaimQuantity | None) -> ClaimedValue | None:
    if quantity is None:
        return None
    return ClaimedValue(quantity.value, quantity.unit, quantity.provenance)
```

In `_provider_claims`, create the final `ClaimReference` before numeric projection, call `recover_scientific_claim(reference)`, and use the same `_claimed_value` projection when the recovered primary is a `MetricImprovementClaim`; otherwise set baseline/candidate/minimum to `None`. Provider `ScientificClaim.claimed_magnitude` is not reparsed. Keep legacy non-metric classifiers only after canonical recovery returns `None`. They must not recognize `ClaimType.METRIC_IMPROVEMENT`.

- [ ] **Step 7: Run GREEN tests and refactor without changing behavior**

Run:

```powershell
python -m pytest tests/test_claim_types_v0.py tests/test_auto_discovery_v03.py tests/test_analysis_cross_package_v03.py -q
```

Expected: all focused tests pass with no new warnings.

- [ ] **Step 8: Commit Task 1**

```powershell
git add claimci/analysis/claim_types.py claimci/analysis/discovery/claims.py tests/test_claim_types_v0.py tests/test_auto_discovery_v03.py tests/test_analysis_cross_package_v03.py
git commit -m "feat: broaden canonical metric claim recovery"
```

### Task 2: Factory-issued physical occurrence wrappers

**Files:**
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/artifact_source.py`
- Modify: `claimci/analysis/materialize.py`
- Modify: `claimci/analysis/__init__.py`
- Create: `tests/analysis_occurrence_support.py`
- Test: `tests/test_analysis_contracts_v03.py`
- Test: `tests/test_streaming_source_v1.py`
- Test: `tests/test_streaming_materialize_v1.py`
- Modify: `tests/test_adapters_config_v03.py`
- Modify: `tests/test_adapters_core_v03.py`
- Modify: `tests/test_adapters_csv_v03.py`
- Modify: `tests/test_adapters_native_v03.py`
- Modify: `tests/test_adapters_security_v03.py`
- Modify: `tests/test_adapters_structured_v03.py`
- Modify: `tests/test_analysis_cross_package_v03.py`
- Modify: `tests/test_benchmark_evidence_profile_v0.py`
- Modify: `tests/test_benchmark_table_adapter_v0.py`
- Modify: `tests/test_dataset_identity_v03.py`
- Modify: `tests/test_ephemeral_materialize_v03.py`
- Modify: `tests/test_ephemeral_planner_v03.py`
- Modify: `tests/test_metric_identity_v1.py`
- Modify: `tests/test_streaming_jsonl_v1.py`
- Modify: `tests/test_streaming_tabular_v1.py`
- Modify: `tests/test_unified_analysis_v03.py`

**Interfaces:**
- Produces: `ArtifactSnapshotRole`, factory-only `ArtifactOccurrence`, `artifact_occurrence_from_snapshot`, `passive_artifact_from_snapshot`, and occurrence-scoped `ArtifactSource`.
- Preserves: `PassiveArtifact.candidate`, `ArtifactSource.repository`, `ArtifactSource.head_sha`, and `ArtifactSource.candidate` as read-only properties.

- [ ] **Step 1: Add RED contract tests for trusted issuance and immutable scope**

Add tests that assert direct occurrence construction fails, factories bind exact repository/role/commit/candidate, bytes still validate size/SHA, and changing role or commit creates unequal occurrence values:

```python
def test_artifact_occurrence_requires_trusted_factory() -> None:
    with pytest.raises(TypeError, match="trusted snapshot factory"):
        ArtifactOccurrence()


def test_passive_artifact_carries_factory_issued_occurrence() -> None:
    artifact = passive_artifact_from_snapshot(
        RepositoryIdentity("owner", "repo"),
        ArtifactSnapshotRole.HEAD,
        GitCommitSha("a" * 40),
        _candidate(),
        RAW_RESULTS,
    )
    assert artifact.repository == RepositoryIdentity("owner", "repo")
    assert artifact.snapshot_role is ArtifactSnapshotRole.HEAD
    assert artifact.commit == GitCommitSha("a" * 40)
    assert artifact.candidate == _candidate()
```

- [ ] **Step 2: Add RED streaming-source tests for explicit base/head identity**

Issue two sources for the same candidate/root/commit with `HEAD` and `BASE`; assert their occurrences differ while existing compatibility properties remain exact. Also assert the legacy factory call without a role produces `HEAD`.

- [ ] **Step 3: Run focused RED tests**

Run:

```powershell
python -m pytest tests/test_analysis_contracts_v03.py tests/test_streaming_source_v1.py -q
```

Expected: imports/factory assertions fail because occurrence contracts do not exist.

- [ ] **Step 4: Implement occurrence contracts and factories**

In `contracts.py`, define:

```python
class ArtifactSnapshotRole(str, Enum):
    HEAD = "head"
    BASE = "base"


@dataclass(frozen=True, slots=True, init=False)
class ArtifactOccurrence:
    repository: RepositoryIdentity
    snapshot_role: ArtifactSnapshotRole
    commit: GitCommitSha
    candidate: ArtifactCandidate

    def __init__(self) -> None:
        raise TypeError("ArtifactOccurrence values require the trusted snapshot factory")
```

Add a private `_from_snapshot` classmethod that assigns only validated concrete values. Make `PassiveArtifact` hold `occurrence` and `content`, validate content against `occurrence.candidate`, and expose repository/snapshot role/commit/candidate properties. Define the two public factories after `RepositoryIdentity` so all runtime types are available.

- [ ] **Step 5: Attach the same occurrence to `ArtifactSource`**

Change `_from_snapshot` to accept an issued occurrence. Keep compatibility properties and extend the public source factory with keyword-only `snapshot_role=ArtifactSnapshotRole.HEAD`. The factory issues the occurrence internally; callers cannot pass provider dictionaries or provenance as scope.

- [ ] **Step 6: Move Core production capture to the factory**

In `materialize._capture_artifact()`, use `passive_artifact_from_snapshot(runtime.repository, ArtifactSnapshotRole.HEAD, runtime.head_sha, candidate, content)`. Pass `snapshot_role=ArtifactSnapshotRole.HEAD` at both production `artifact_source_from_snapshot` call sites: the large-file branch in `_capture_artifact()` and the passive-to-streaming dataset reissuance in `_streaming_dataset_sources()`.

- [ ] **Step 7: Migrate test construction through one explicit test helper**

Create `tests/analysis_occurrence_support.py`:

```python
from claimci.analysis import (
    ArtifactCandidate,
    ArtifactSnapshotRole,
    GitCommitSha,
    PassiveArtifact,
    RepositoryIdentity,
    passive_artifact_from_snapshot,
)

TEST_REPOSITORY = RepositoryIdentity("claimci-tests", "fixture")
TEST_HEAD = GitCommitSha("a" * 40)


def passive_artifact(
    candidate: ArtifactCandidate,
    content: bytes,
    *,
    repository: RepositoryIdentity = TEST_REPOSITORY,
    snapshot_role: ArtifactSnapshotRole = ArtifactSnapshotRole.HEAD,
    commit: GitCommitSha = TEST_HEAD,
) -> PassiveArtifact:
    return passive_artifact_from_snapshot(
        repository,
        snapshot_role,
        commit,
        candidate,
        content,
    )
```

Replace every test-only direct `PassiveArtifact(...)` construction with this helper or an explicit factory call when the test needs different scope. Do not change provider fixtures or `ArtifactCandidate` construction.

- [ ] **Step 8: Run focused GREEN compatibility tests**

Run:

```powershell
python -m pytest tests/test_adapters_config_v03.py tests/test_adapters_core_v03.py tests/test_adapters_csv_v03.py tests/test_adapters_native_v03.py tests/test_adapters_security_v03.py tests/test_adapters_structured_v03.py tests/test_analysis_contracts_v03.py tests/test_analysis_cross_package_v03.py tests/test_benchmark_evidence_profile_v0.py tests/test_benchmark_table_adapter_v0.py tests/test_dataset_identity_v03.py tests/test_ephemeral_materialize_v03.py tests/test_ephemeral_planner_v03.py tests/test_metric_identity_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_materialize_v1.py tests/test_streaming_source_v1.py tests/test_streaming_tabular_v1.py tests/test_unified_analysis_v03.py -q
```

Expected: all selected contracts/adapters/source tests pass with factory-issued synthetic scope.

- [ ] **Step 9: Commit Task 2**

```powershell
git add claimci/analysis/contracts.py claimci/analysis/artifact_source.py claimci/analysis/materialize.py claimci/analysis/__init__.py tests/analysis_occurrence_support.py tests/test_adapters_config_v03.py tests/test_adapters_core_v03.py tests/test_adapters_csv_v03.py tests/test_adapters_native_v03.py tests/test_adapters_security_v03.py tests/test_adapters_structured_v03.py tests/test_analysis_contracts_v03.py tests/test_analysis_cross_package_v03.py tests/test_benchmark_evidence_profile_v0.py tests/test_benchmark_table_adapter_v0.py tests/test_dataset_identity_v03.py tests/test_ephemeral_materialize_v03.py tests/test_ephemeral_planner_v03.py tests/test_metric_identity_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_materialize_v1.py tests/test_streaming_source_v1.py tests/test_streaming_tabular_v1.py tests/test_unified_analysis_v03.py
git commit -m "feat: bind artifacts to trusted snapshot occurrences"
```

### Task 3: Evidence ID v2 for whole-artifact adapters

**Files:**
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/evidence_identity.py`
- Modify: `claimci/analysis/adapters/core.py`
- Modify: `claimci/analysis/adapters/config.py`
- Modify: `claimci/analysis/adapters/native.py`
- Modify: `claimci/analysis/adapters/structured.py`
- Modify: `claimci/analysis/adapters/dataset.py`
- Modify: `claimci/analysis/adapters/tabular.py`
- Test: `tests/test_adapters_core_v03.py`
- Test: `tests/test_adapters_registry_v03.py`
- Test: `tests/test_analysis_contracts_v03.py`
- Test: `tests/test_adapters_config_v03.py`
- Test: `tests/test_adapters_native_v03.py`
- Test: `tests/test_adapters_structured_v03.py`
- Test: `tests/test_dataset_identity_v03.py`
- Create: `tests/test_evidence_identity_v2.py`

**Interfaces:**
- Produces: fixed `Adapter.semantic_version`, provenance-free `canonical_selector_material`, and `_evidence_id(adapter_id, adapter_semantic_version, artifact, mappings)`.
- Consumes: Task 2 occurrence-scoped `PassiveArtifact`/`ArtifactSource`.

- [ ] **Step 1: Add RED canonical-material and whole-artifact identity matrix tests**

Use literal expected IDs computed independently from literal dictionaries, not the production helper. Cover same occurrence stability and separation by repository, snapshot role, commit, path, kind, full SHA/size, adapter ID/version, and mapping selector. Assert claim IDs, provenance details, and `ExperimentRole` changes do not affect the ID.

The expected helper in the test is limited to canonical JSON and hashlib:

```python
def _expected_v2(material: dict[str, object]) -> str:
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "evidence-v2-" + hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 2: Add RED collision tests to config/native/structured/dataset adapters**

For byte-identical candidates at `baseline-config.yaml` and `candidate-config.yaml`, issue HEAD passives under the same repository/commit and assert extracted IDs differ. Re-read one occurrence and assert its ID is identical. For dataset extraction assert the same rules with empty selector material.

- [ ] **Step 3: Run focused RED tests**

Run:

```powershell
python -m pytest tests/test_evidence_identity_v2.py tests/test_adapters_config_v03.py tests/test_adapters_native_v03.py tests/test_adapters_structured_v03.py tests/test_dataset_identity_v03.py -q
```

Expected: new prefix/material assertions fail because adapters still emit SHA-truncated legacy IDs.

- [ ] **Step 4: Implement canonical selector material**

In `evidence_identity.py`, add:

```python
def canonical_selector_material(
    mappings: tuple[FieldMapping, ...],
) -> tuple[dict[str, object], ...]:
    if not isinstance(mappings, tuple) or not all(type(item) is FieldMapping for item in mappings):
        raise TypeError("selector material requires a tuple of FieldMapping values")
    return tuple(
        field_mapping_material(item)
        for item in sorted(mappings, key=field_mapping_identity)
    )
```

Do not include `FieldProvenance` or its detail/source ID.

- [ ] **Step 5: Replace legacy evidence hashing with exact v2 material**

Require a fixed non-empty semantic version and build exactly:

```python
material = {
    "schema_version": 2,
    "repository": {
        "owner": artifact.occurrence.repository.owner,
        "name": artifact.occurrence.repository.name,
    },
    "snapshot": {
        "role": artifact.occurrence.snapshot_role.value,
        "commit": str(artifact.occurrence.commit),
    },
    "artifact": {
        "path": str(artifact.candidate.path),
        "kind": artifact.candidate.kind.value,
        "sha256": str(artifact.candidate.sha256),
        "size": artifact.candidate.size,
    },
    "adapter": {"id": adapter_id, "version": adapter_semantic_version},
    "selector": list(canonical_selector_material(mappings)),
}
```

Return `evidence-v2-` plus the full digest. Validate that the trusted adapter ID matches `AdapterMatch.adapter_id` at each call site.

Extend the private `_ArtifactEnvelope` protocol with `occurrence: ArtifactOccurrence` as well as `candidate: ArtifactCandidate`; the v2 helper must reject envelopes without a concrete issued occurrence instead of falling back to a legacy or synthetic scope.

- [ ] **Step 6: Give every whole-artifact adapter a trusted fixed version and route all IDs through v2**

Add `semantic_version = "1"` to every fixed built-in adapter: YAML, TOML, native manifest/results/config, JSON, JSONL, passive dataset, CSV, and TSV. During this task, retain `_dataset_evidence_id(adapter_id, artifact)` as a private compatibility delegator whose entire body is `return _evidence_id(adapter_id, "1", artifact, ())`; this keeps the streaming dataset import green until Task 4. The passive dataset adapter may call `_evidence_id(self.adapter_id, self.semantic_version, artifact, match.mappings)` directly. Update `Adapter` protocol with the read-only semantic-version attribute, update the contract-test fake adapter, and assert the fixed registry exposes version `"1"` for every entry. Do not change CSV/TSV evidence hashing until Task 4.

- [ ] **Step 7: Run focused GREEN tests**

Run:

```powershell
python -m pytest tests/test_evidence_identity_v2.py tests/test_adapters_core_v03.py tests/test_adapters_registry_v03.py tests/test_analysis_contracts_v03.py tests/test_adapters_config_v03.py tests/test_adapters_native_v03.py tests/test_adapters_structured_v03.py tests/test_dataset_identity_v03.py -q
```

Expected: all focused tests pass; every new extraction ID is a 76-character `evidence-v2-*` value.

- [ ] **Step 8: Commit Task 3**

```powershell
git add claimci/analysis/contracts.py claimci/analysis/evidence_identity.py claimci/analysis/adapters/core.py claimci/analysis/adapters/config.py claimci/analysis/adapters/native.py claimci/analysis/adapters/structured.py claimci/analysis/adapters/dataset.py claimci/analysis/adapters/tabular.py tests/test_evidence_identity_v2.py tests/test_adapters_core_v03.py tests/test_adapters_registry_v03.py tests/test_analysis_contracts_v03.py tests/test_adapters_config_v03.py tests/test_adapters_native_v03.py tests/test_adapters_structured_v03.py tests/test_dataset_identity_v03.py
git commit -m "feat: issue evidence v2 identities"
```

### Task 4: Selector-scoped and streaming evidence v2

**Files:**
- Modify: `claimci/analysis/adapters/tabular.py`
- Modify: `claimci/analysis/adapters/streaming.py`
- Modify: `claimci/analysis/adapters/dataset.py`
- Test: `tests/test_adapters_csv_v03.py`
- Test: `tests/test_benchmark_table_adapter_v0.py`
- Test: `tests/test_streaming_tabular_v1.py`
- Test: `tests/test_streaming_jsonl_v1.py`
- Test: `tests/test_streaming_dataset_v1.py`
- Test: `tests/test_metric_streaming_security_integration.py`

**Interfaces:**
- Consumes: Task 3 `_evidence_id` and canonical selector material.
- Produces: identical v2 IDs for passive/streaming extraction of one occurrence+selector, and different IDs for different canonical selectors.

- [ ] **Step 1: Add RED selector and passive/streaming parity tests**

For one CSV occurrence, extract with two validated table selectors and assert distinct IDs. Extract the same selector through passive and streaming paths with identical occurrence scope and assert identical IDs. Repeat whole-artifact stability for JSONL and dataset streaming.

- [ ] **Step 2: Run focused RED tests**

Run:

```powershell
python -m pytest tests/test_adapters_csv_v03.py tests/test_benchmark_table_adapter_v0.py tests/test_streaming_tabular_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_dataset_v1.py -q
```

Expected: streaming/selector paths either call the old signature or produce selector-scoped-v1/legacy IDs.

- [ ] **Step 3: Route tabular and streaming paths through the same v2 helper**

Use the Task 3 fixed CSV/TSV `semantic_version`. Pass the full `canonical_match.mappings` to `_evidence_id`; remove the selector-scoped-v1 special branch. Streaming CSV/TSV uses the same adapter ID/version and mappings. Streaming JSONL and dataset use their matching registered adapter ID/version and empty or canonical mappings as appropriate. Replace the streaming dataset `_dataset_evidence_id` call with `_evidence_id(_DATASET_ADAPTER_ID, "1", source, match.mappings)`, remove the streaming import, and then delete the now-unused compatibility delegator from `adapters/dataset.py`.

- [ ] **Step 4: Preserve streaming integrity fields outside identity**

Keep `scan_completeness` and `source_trace_sha256` unchanged on `NormalizedEvidence`; they are validation/audit fields, not physical occurrence/selector identity fields. Confirm tests mutate scan integrity without altering the canonical evidence ID and still fail through existing integrity gates.

- [ ] **Step 5: Run focused GREEN tests**

Run:

```powershell
python -m pytest tests/test_adapters_csv_v03.py tests/test_benchmark_table_adapter_v0.py tests/test_streaming_tabular_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_dataset_v1.py tests/test_metric_streaming_security_integration.py -q
```

Expected: all focused tests pass and passive/streaming selector identity is identical.

- [ ] **Step 6: Commit Task 4**

```powershell
git add claimci/analysis/adapters/tabular.py claimci/analysis/adapters/streaming.py claimci/analysis/adapters/dataset.py tests/test_adapters_csv_v03.py tests/test_benchmark_table_adapter_v0.py tests/test_streaming_tabular_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_dataset_v1.py tests/test_metric_streaming_security_integration.py
git commit -m "feat: unify selector scoped evidence v2"
```

### Task 5: Planner collision and exact controlled-fixture acceptance

**Files:**
- Create: `tests/fixtures/production_smoke/README.md`
- Create: `tests/fixtures/production_smoke/research.yaml`
- Create: `tests/fixtures/production_smoke/baseline-config.yaml`
- Create: `tests/fixtures/production_smoke/baseline-eval.jsonl`
- Create: `tests/fixtures/production_smoke/baseline-results.json`
- Create: `tests/fixtures/production_smoke/baseline-train.jsonl`
- Create: `tests/fixtures/production_smoke/candidate-config.yaml`
- Create: `tests/fixtures/production_smoke/candidate-eval.jsonl`
- Create: `tests/fixtures/production_smoke/candidate-results.json`
- Create: `tests/fixtures/production_smoke/candidate-results-base.json`
- Create: `tests/fixtures/production_smoke/candidate-train.jsonl`
- Test: `tests/test_ephemeral_planner_v03.py`
- Create: `tests/test_production_smoke_remediation.py`
- Modify: `tests/test_analysis_contracts_v03.py`
- Modify: `tests/test_analysis_cross_package_v03.py`
- Modify: `tests/test_auto_discovery_v03.py`
- Modify only if a RED test proves a remaining defect: `claimci/analysis/planner.py`

**Interfaces:**
- Consumes: Tasks 1–4 canonical discovery and v2 extraction.
- Produces: exact fixture claim count `1` followed by the honest deterministic `PARTIAL / required_threshold_not_recovered` boundary; planner still rejects a true repeated identity.

- [ ] **Step 1: Copy the controlled fixture byte-for-byte and pin its hashes**

Copy the ten files from controlled tree `90266b8ad6b63cda4f766e31039c75e0967f9b1c`. Store the base candidate result separately. In the test, hard-code the approved SHA-256 map:

```python
EXPECTED_HEAD_SHA256 = {
    "README.md": "245906ae491e68f661e4b4571122bb3a9372211708af56bb07f28db87981e1bb",
    "research.yaml": "8c7121eeb7603b816de26699640cd28be1ca2192ecfc057a1a8a0daf4cdd3d16",
    "baseline-config.yaml": "e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d",
    "candidate-config.yaml": "e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d",
    "baseline-eval.jsonl": "62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600",
    "candidate-eval.jsonl": "62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600",
    "baseline-results.json": "8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493",
    "candidate-results.json": "7ea7c722aaae5e2739ae13800daa5f5ea12bd1675fda6e04994c95e22ce54dc3",
    "baseline-train.jsonl": "7ace1dd7aa40731aaee3c6e40d1d5d67f053e654488361e849b3d2b065626417",
    "candidate-train.jsonl": "1381bb732892ff2d772a5d06a0b7da2f3f43336195102073c7029799f390dc44",
}
```

The base candidate result SHA is `8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493`.

- [ ] **Step 2: Add the RED planner collision matrix**

Assert different paths/same bytes are accepted by `PlanningRequest`, HEAD versus BASE same path/bytes produce distinct IDs, same occurrence re-read is stable, different selectors differ, and passing the same `NormalizedEvidence` twice still raises `AnalysisContractError` rather than multiplying authority.

- [ ] **Step 3: Add the exact fixture end-to-end planning test**

Reconstruct base/head temp trees, call `discover_repository()` with exact metadata, extract every supported HEAD artifact through `passive_artifact_from_snapshot()` and the registered adapter, select the sole claim, then call `planning_request_from_discovery()` and `plan_ephemeral_audit()`.

Assert:

```python
assert len(discovery.claims) == 1
assert discovery.claims[0].reference.text == EXACT_CLAIM
assert discovery.claims[0].metric == "accuracy"
assert discovery.claims[0].direction is ClaimDirection.HIGHER
assert discovery.claims[0].baseline_value is not None
assert discovery.claims[0].baseline_value.value == 0.60
assert discovery.claims[0].candidate_value is not None
assert discovery.claims[0].candidate_value.value == 0.70
assert discovery.claims[0].minimum_improvement is None
assert evidence
assert all(re.fullmatch(r"evidence-v2-[0-9a-f]{64}", item.evidence_id) for item in evidence)
assert config_ids["baseline-config.yaml"] != config_ids["candidate-config.yaml"]
assert outcome.state is PlanningState.PARTIAL
assert outcome.reason == "required_threshold_not_recovered"
assert outcome.plan is None
assert outcome.mapping_question is None
```

Also assert the blocking obligation is exactly `claim.threshold`, with state `MISSING` and reason `REQUIRED_THRESHOLD_NOT_RECOVERED`. The checked-in manifest validly supplies all eight artifact bindings, but its `minimum_improvement` hint is mapping metadata and must not become canonical claim-field authority. The exact approved prose contains no threshold, so synthesizing `0.10` from arithmetic or importing manifest `0.05` into the claim is forbidden. If implementation exposes an independently real `READY` or `MAPPING_NEEDED` authority path, stop the task and report it for design re-review rather than weakening or relabeling the assertion.

- [ ] **Step 4: Run Stage B/C RED tests before any planner edit**

Run:

```powershell
python -m pytest tests/test_evidence_identity_v2.py tests/test_ephemeral_planner_v03.py tests/test_production_smoke_remediation.py -q
```

Expected: if Tasks 1–4 are complete, the fixture should already pass. A failure here is evidence of a remaining collision/planner integration defect; do not edit planner unless the failure proves it.

- [ ] **Step 5: Make only the minimal proven planner correction, if required**

Preserve both existing duplicate-ID checks. Correct only a demonstrated stale assumption about legacy ID shape or wrapper construction. Do not add random salts, scientific roles, provider data, or silent deduplication.

- [ ] **Step 6: Run Stage B/C GREEN tests**

Run:

```powershell
python -m pytest tests/test_evidence_identity_v2.py tests/test_ephemeral_planner_v03.py tests/test_ephemeral_materialize_v03.py tests/test_streaming_materialize_v1.py tests/test_production_smoke_remediation.py -q
```

Expected: all tests pass; exact fixture claim count is 1 and planning terminates honestly as `PARTIAL / required_threshold_not_recovered`, never `scientific_routing_uncertain`.

- [ ] **Step 7: Run the focused historical-decoding/provider-schema regression**

Add an immutable legacy `NormalizedEvidence` value containing a bounded `evidence-...` identifier and construct/serialize it through the current Core contract path, proving no v2 prefix requirement was added to historical values. Core intentionally has no completed-result JSON decoder. The companion plan owns the executable immutable completed-result regressions through `runner/claimci_hosted_runner/contracts.py` and `cloud/src/analysis/contracts.ts` plus the repository loader. In provider/discovery schema tests, attempt to supply `occurrence`, `repository`, `snapshot_role`, and `commit` fields in claim and mapping objects. Assert exact-schema validation rejects the hostile fields (or the existing decoder omits them before trusted code), no trusted occurrence factory is called, and no provider value can become repository/snapshot/commit scope.

The authoritative cross-run mapping-response verifier is Hosted `_public_mapping_ids`, not Core's in-memory mapping IDs. Its exact repository/head/artifact/adapter/selector v2 binding and stale-response mutation matrix are owned by the companion plan. Core must keep provider/mapping schemas occurrence-free and must not add a second approval authority.

Run:

```powershell
python -m pytest tests/test_analysis_contracts_v03.py tests/test_analysis_cross_package_v03.py tests/test_auto_discovery_v03.py -q
```

Expected: legacy bounded evidence IDs still decode and provider/discovery schemas have no occurrence fields.

- [ ] **Step 8: Commit Task 5**

```powershell
git add tests/fixtures/production_smoke tests/test_ephemeral_planner_v03.py tests/test_production_smoke_remediation.py tests/test_analysis_contracts_v03.py tests/test_analysis_cross_package_v03.py tests/test_auto_discovery_v03.py
# Add claimci/analysis/planner.py only if the RED test proved and required an edit.
git commit -m "test: prove production smoke deterministic planning"
```

If the RED test proves a planner edit is necessary, add that one path explicitly before committing. In all cases verify `git status --short` and `git diff --cached --name-status` so only intended files enter the commit.

## Branch completion gate

After all five tasks pass their independent reviews, the controller performs one whole-branch review, one scoped security review, and the single final Core verification pass. Only then may it push this feature branch and create one Draft PR. It must not merge or deploy the PR.

Create `docs/superpowers/reports/2026-08-27-claim-recovery-evidence-occurrence-v2-verification.md` with the immutable base/head commits, exact command transcripts and exit codes, fixture/source/config/design/plan SHA-256 values, reviewed diff allowlist, security-review result, and an explicit deferred compatibility obligation naming both Hosted/Cloud completed-result decoder tests required in the companion PR. Run the whole Core suite exactly once at this gate:

```powershell
python -m pytest -q
python -m compileall claimci tests
python -m pip check
git diff --check origin/main...HEAD
git status --short
git diff --name-status origin/main...HEAD
git diff --stat origin/main...HEAD
```

Before running, inspect `pyproject.toml` and repository automation for configured lint/type commands. Execute each configured repository-supported lint/type command once and record it; if none exists, record the negative configuration search rather than inventing a checker. The scoped security review must cover provider/mapping authority, factory-only snapshot issuance, path/commit trust, selector canonicalization, collision behavior, legacy decoding, and fail-closed materialization. Stage and commit only the named report after its contents and hashes are independently reviewed.
