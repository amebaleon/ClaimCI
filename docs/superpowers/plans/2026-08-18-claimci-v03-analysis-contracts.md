# ClaimCI v0.3 Analysis Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable, validated shared contracts for future zero-config discovery, adapters, planning, deterministic verification, and unified reporting without changing existing runtime behavior.

**Architecture:** A new standard-library-only `claimci.analysis` package exposes validated scalar value objects, frozen typed records, a passive-byte adapter protocol, explicit mapping trust transitions, an ephemeral plan, and a unified result whose authoritative verdict can only come from a factory-created snapshot of an actual `AuditResult`. Existing Audit and Research Review modules remain unchanged.

**Tech Stack:** Python 3.11+, standard-library dataclasses/enums/protocols, existing `claimci.models.AuditResult` and `claimci.report.render_json`, pytest.

## Global Constraints

- Add no runtime dependency.
- Do not implement discovery algorithms or production adapters.
- Do not change workflows, CLI behavior, deterministic audit semantics, Research Review orchestration, provider limits, or provider call behavior.
- Repository content is passive bytes and is never executed.
- All repository paths are canonical, relative POSIX paths and still require later runtime root/symlink/index verification.
- A legacy `research.yaml` is an optional high-confidence `MANIFEST` hint, not a prerequisite.
- Provider-originated data can propose claims, selectors, and mappings but cannot construct deterministic authority or an approved reusable mapping.
- All production changes follow red-green-refactor; every behavioral test must fail for the expected missing-contract reason before implementation.

---

### Task 1: Validated primitives, artifacts, and passive adapter protocol

**Files:**
- Create: `claimci/analysis/confidence.py`
- Create: `claimci/analysis/contracts.py`
- Create: `claimci/analysis/__init__.py`
- Create: `tests/test_analysis_contracts_v03.py`

**Interfaces:**
- Produces: `Confidence`, `AnalysisContractError`, `RepositoryPath`, `Sha256Digest`, `GitCommitSha`, `ArtifactKind`, `ExperimentRole`, `ProvenanceKind`, `MappingTrust`, `SelectorKind`, `AnalysisState`, `AnalysisAuthority`, `FieldProvenance`, `EvidenceSelector`, `FieldMapping`, `ArtifactCandidate`, `PassiveArtifact`, `AdapterMatch`, and `Adapter`.
- Consumes: only the Python standard library.

- [ ] **Step 1: Write failing primitive and artifact tests**

Add tests that name the concrete contract failures:

```python
@pytest.mark.parametrize("value", [True, math.nan, math.inf, -0.01, 1.01])
def test_confidence_rejects_nonfinite_boolean_and_out_of_range_values(value):
    with pytest.raises((TypeError, ValueError)):
        Confidence(value)


@pytest.mark.parametrize(
    "value",
    ["", ".", "../result.json", "/result.json", "C:/result.json",
     "//server/share/result.json", "data\\result.json", "data//result.json",
     "data/./result.json", "data/\x00result.json"],
)
def test_repository_path_rejects_nonportable_or_unconfined_values(value):
    with pytest.raises((TypeError, ValueError)):
        RepositoryPath(value)


@pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_artifact_candidate_rejects_invalid_sha256(digest):
    with pytest.raises((TypeError, ValueError)):
        _artifact(sha256=digest)
```

Assert exact `ArtifactKind` members, negative/bool sizes, duplicate/empty claim
IDs, blank reasons, and wrong field types are rejected. Assert frozen attribute
assignment fails.

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_analysis_contracts_v03.py
```

Expected: collection fails because `claimci.analysis` does not exist.

- [ ] **Step 3: Implement validated primitives and public enums**

Implement `Confidence` as:

```python
@dataclass(frozen=True, slots=True, order=True)
class Confidence:
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("confidence must be a number")
        normalized = float(self.value)
        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError("confidence must be finite from 0 through 1")
        object.__setattr__(self, "value", normalized)

    def __float__(self) -> float:
        return self.value
```

Implement `RepositoryPath(str)`, `Sha256Digest(str)`, and `GitCommitSha(str)`
with strict construction-time validation from the design. Implement all public
string enums with stable lowercase values.

- [ ] **Step 4: Implement provenance, selectors, and passive artifacts**

Use frozen, slotted records with these signatures:

```python
@dataclass(frozen=True, slots=True)
class FieldProvenance:
    kind: ProvenanceKind
    detail: str
    source_path: RepositoryPath | None = None
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSelector:
    kind: SelectorKind
    expression: str
    provenance: FieldProvenance


@dataclass(frozen=True, slots=True)
class FieldMapping:
    target_field: str
    selector: EvidenceSelector
    provenance: FieldProvenance


@dataclass(frozen=True, slots=True)
class ArtifactCandidate:
    path: RepositoryPath
    kind: ArtifactKind
    sha256: Sha256Digest
    size: int
    confidence: Confidence
    discovery_reason: str
    relevant_claim_ids: tuple[str, ...]
    provenance: FieldProvenance


@dataclass(frozen=True, slots=True)
class PassiveArtifact:
    candidate: ArtifactCandidate
    content: bytes


@dataclass(frozen=True, slots=True)
class AdapterMatch:
    adapter_id: str
    path: RepositoryPath
    confidence: Confidence
    mappings: tuple[FieldMapping, ...]
    match_evidence: tuple[FieldProvenance, ...]
```

`PassiveArtifact` must recompute and validate both byte length and SHA-256.
Selector validation is syntax-specific, bounded, and declarative. Reject
duplicate target fields in one `AdapterMatch`.

Define the protocol without roots, callables, or execution surfaces:

```python
class Adapter(Protocol):
    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None: ...
    def extract(
        self,
        artifact: PassiveArtifact,
        match: AdapterMatch,
    ) -> NormalizedEvidence: ...
```

Use a forward reference for `NormalizedEvidence` until Task 2 defines it.

- [ ] **Step 5: Verify Task 1 GREEN and commit**

Run the focused test file. Then stage only the four Task 1 paths and commit:

```text
git commit -m "feat: add analysis artifact contracts"
```

---

### Task 2: Normalized evidence, mapping proposals, approval, and ephemeral plans

**Files:**
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/__init__.py`
- Modify: `tests/test_analysis_contracts_v03.py`

**Interfaces:**
- Consumes: Task 1 values, provenance, selectors, artifacts, and adapter match.
- Produces: `ConfigValue`, `DatasetReference`, `ComputeEvidence`, `NormalizedObservation`, `NormalizedEvidence`, `ArtifactBinding`, `MappingCandidate`, `MappingChoice`, `MappingQuestion`, `RepositoryIdentity`, `RepoMapping`, `ClaimReference`, `MissingEvidence`, and `EphemeralAuditPlan`.

- [ ] **Step 1: Write failing normalized-evidence tests**

Test finite metric/config/compute values, paired metric name/value, optional
run/seed/role, typed dataset references, at least one substantive field, exact
artifact/match path agreement, and tuple/frozen behavior.

```python
def test_normalized_observation_represents_partial_typed_evidence():
    observation = NormalizedObservation(
        metric_name="accuracy",
        metric_value=0.9,
        run_id="candidate-run-1",
        seed=7,
        experiment_role=ExperimentRole.CANDIDATE,
        config_values=(ConfigValue("batch_size", 32, PROVENANCE),),
        dataset_references=(
            DatasetReference(RepositoryPath("data/eval.jsonl"), "eval", PROVENANCE),
        ),
        compute_evidence=(ComputeEvidence("training_steps", 1000, "steps", PROVENANCE),),
        provenance=PROVENANCE,
    )
    assert observation.metric_value == 0.9
```

- [ ] **Step 2: Write failing mapping and approval tests**

Cover same-path baseline/candidate conflicts, duplicate bindings, trust/provenance
consistency, two-to-eight question choices, duplicate choice IDs, blank prompts,
and deep immutability.

Prove the explicit approval boundary:

```python
def test_repo_mapping_requires_explicit_approval_factory():
    with pytest.raises(TypeError):
        RepoMapping(repository=REPOSITORY, bindings=MAPPING.bindings)  # type: ignore[call-arg]

    approved = RepoMapping.approve(
        repository=REPOSITORY,
        candidate=MAPPING,
        approved_by="owner:amebaleon",
    )
    assert approved.trust is MappingTrust.USER_APPROVED
    assert approved.source_trust is MAPPING.trust
```

- [ ] **Step 3: Write failing plan tests and verify RED**

Test repository identity, PR-number and head-SHA validation, role-consistent
baseline/candidate evidence, mapping provenance, missing evidence, and serialized
`ephemeral: true`. Assert the type exposes no committed path/write method.

Run the focused file and confirm failures are missing types/behavior rather than
test errors.

- [ ] **Step 4: Implement normalized evidence and mapping records**

Use these principal signatures:

```python
@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    provenance: FieldProvenance
    metric_name: str | None = None
    metric_value: float | None = None
    run_id: str | None = None
    seed: str | int | None = None
    experiment_role: ExperimentRole = ExperimentRole.UNSPECIFIED
    config_values: tuple[ConfigValue, ...] = ()
    dataset_references: tuple[DatasetReference, ...] = ()
    compute_evidence: tuple[ComputeEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedEvidence:
    evidence_id: str
    artifact: ArtifactCandidate
    adapter_match: AdapterMatch
    observations: tuple[NormalizedObservation, ...]
```

Require provenance when any field is inferred/extracted. Implement
`ArtifactBinding`, `MappingCandidate`, `MappingChoice`, and `MappingQuestion`
as described by the design. `MappingCandidate` permits only inferred or
manifest-hint trust.

Implement `RepoMapping` with `@dataclass(..., init=False)` and an `approve`
classmethod that validates a real `MappingCandidate` and records its source
trust.

- [ ] **Step 5: Implement the ephemeral plan**

```python
@dataclass(frozen=True, slots=True)
class EphemeralAuditPlan:
    plan_id: str
    repository: RepositoryIdentity
    pr_number: int | None
    head_sha: GitCommitSha
    claim: ClaimReference
    baseline_evidence: tuple[NormalizedEvidence, ...]
    candidate_evidence: tuple[NormalizedEvidence, ...]
    mapping_provenance: tuple[FieldProvenance, ...]
    missing_evidence: tuple[MissingEvidence, ...]
    confidence: Confidence
    ephemeral: bool = field(default=True, init=False)
```

Validate all nested types, collection uniqueness, role consistency, bounded
text, and positive PR numbers. Do not add persistence behavior.

- [ ] **Step 6: Verify Task 2 GREEN and commit**

Run the focused file, then the existing model/manifest suites:

```text
python -m pytest -q -p no:cacheprovider tests/test_analysis_contracts_v03.py tests/test_models.py tests/test_manifest.py tests/test_review_models_day3.py
```

Stage only the Task 2 paths and commit:

```text
git commit -m "feat: add analysis mapping and plan contracts"
```

---

### Task 3: Deterministic authority bridge, unified state, serialization, and legacy manifest

**Files:**
- Modify: `claimci/analysis/contracts.py`
- Modify: `claimci/analysis/__init__.py`
- Modify: `tests/test_analysis_contracts_v03.py`

**Interfaces:**
- Consumes: existing `claimci.models.AuditResult`, `claimci.models.Verdict`, and `claimci.report.render_json`; Task 1/2 contracts.
- Produces: `DeterministicAuditOutcome`, `AdvisoryResearchInterpretation`, `UnifiedAnalysisResult`, and `to_jsonable`.

- [ ] **Step 1: Write failing authority-separation tests**

Construct a real `AuditResult(Verdict.NOT_SUPPORTED, ...)`, then prove:

```python
def test_only_actual_audit_result_can_create_deterministic_authority():
    with pytest.raises(TypeError):
        DeterministicAuditOutcome.from_audit_result(
            {"verdict": "SUPPORTED"}  # type: ignore[arg-type]
        )

    outcome = DeterministicAuditOutcome.from_audit_result(AUDIT_RESULT)
    assert outcome.verdict is Verdict.NOT_SUPPORTED


def test_advisory_text_cannot_override_authoritative_verdict():
    advisory = AdvisoryResearchInterpretation(
        summary="Provider says SUPPORTED, but this remains advisory.",
        interpretations=("NOT_SUPPORTED", "SUPPORTED"),
        missing_evidence=(),
        confidence=Confidence(1.0),
    )
    unified = UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=DeterministicAuditOutcome.from_audit_result(AUDIT_RESULT),
        research_interpretation=advisory,
    )
    assert unified.authoritative_verdict is Verdict.NOT_SUPPORTED
```

Assert the deterministic constructor is disabled and advisory objects have no
verdict/impact/severity field.

- [ ] **Step 2: Write failing unified-state and serialization tests**

Cover all four state invariants, detached/deeply immutable deterministic
payloads, enum/path/confidence conversion, JSON dumping with `allow_nan=False`,
and caller mutation of a serialized dictionary not affecting the contract.

- [ ] **Step 3: Write the legacy manifest representation test**

Build a `MANIFEST` candidate and eight artifact bindings corresponding to the
existing `research.yaml` baseline/candidate config, results, train dataset, and
evaluation dataset. Assert:

- trust is `MANIFEST_HINT` and confidence is high;
- the manifest is optional to `EphemeralAuditPlan` itself;
- every legacy artifact role/kind/path is representable;
- no new parser or audit-engine behavior is introduced.

- [ ] **Step 4: Verify RED and implement the one-way deterministic factory**

Use a disabled constructor:

```python
@dataclass(frozen=True, slots=True, init=False)
class DeterministicAuditOutcome:
    verdict: Verdict
    payload: Mapping[str, object]
    authority: AnalysisAuthority

    @classmethod
    def from_audit_result(cls, result: AuditResult) -> DeterministicAuditOutcome:
        if not isinstance(result, AuditResult):
            raise TypeError("deterministic authority requires an actual AuditResult")
        payload = json.loads(render_json(result))
        instance = object.__new__(cls)
        object.__setattr__(instance, "verdict", result.verdict)
        object.__setattr__(instance, "payload", _deep_freeze(payload))
        object.__setattr__(instance, "authority", AnalysisAuthority.DETERMINISTIC)
        return instance
```

`_deep_freeze` accepts only JSON-compatible finite values and recursively copies
mappings/lists. The factory must not accept dictionaries, provider response
objects, advisory records, or duck-typed values.

- [ ] **Step 5: Implement advisory and unified-result contracts**

`AdvisoryResearchInterpretation.authority` is fixed with `init=False` and the
class has no deterministic conclusion fields. Implement strict state checks in
`UnifiedAnalysisResult.__post_init__` and derive `authoritative_verdict` only
from `self.deterministic.verdict`.

- [ ] **Step 6: Implement safe serialization and public exports**

`to_jsonable(value)` recursively handles approved contract dataclasses,
`Mapping`, tuples, enums, `Confidence`, and string value objects. It returns new
plain dict/list values and rejects unsupported/non-finite values. Export the
complete documented interface from `claimci.analysis.__init__` using `__all__`.

Add an AST/import contract test proving production analysis modules do not
import provider SDKs, review orchestration, `subprocess`, or execution APIs.

- [ ] **Step 7: Verify Task 3 GREEN and commit**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_analysis_contracts_v03.py tests/test_models.py tests/test_manifest.py tests/test_review_models_day3.py tests/test_review_tools_day3.py
python -m compileall -q claimci
git diff --check
```

Stage only Task 3 paths and commit:

```text
git commit -m "feat: add unified analysis authority contract"
```

---

### Task 4: Complete verification, independent review, and publication

**Files:**
- Modify only if a reviewer finds a requirement-backed defect; every fix begins with a failing regression test.

**Interfaces:**
- Consumes: the frozen design and Tasks 1-3 implementation.
- Produces: verified branch `feat/v03-analysis-contracts` and one pull request against `main`.

- [ ] **Step 1: Audit the final diff and public API**

Compare every requirement in the design against `claimci.analysis.__all__`,
constructor invariants, and focused tests. Confirm no workflow, CLI, Audit,
Review, provider, or existing test file changed.

- [ ] **Step 2: Run the full offline gate fresh**

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

Record exact pass/skip counts and exit codes.

- [ ] **Step 3: Request independent read-only review**

Give the reviewer the base SHA, current HEAD, design path, user requirements,
and exact test output. Fix every critical/important finding test-first and rerun
the full gate.

- [ ] **Step 4: Verify publication scope**

Run `git status`, staged/unstaged diffs, commit list, and compare the branch to
`origin/main`. Confirm the branch contains only design, plan, new analysis
package, and focused tests.

- [ ] **Step 5: Push and open one PR**

Push `feat/v03-analysis-contracts` to `origin`, then create one non-draft PR
against `main` because the user explicitly requested a completed PR. Include
the design summary, authority boundary, tests, and consumer notes. Do not merge.
