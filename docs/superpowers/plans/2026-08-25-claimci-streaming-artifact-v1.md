# ClaimCI Streaming Artifact v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream large JSONL, CSV/TSV, and JSONL dataset artifacts through a trusted exact-head source with explicit integrity/completeness authority and bounded scratch, while preserving small `PassiveArtifact` behavior and Metric Identity v1.

**Architecture:** Add a factory-only confined `ArtifactSource` and immutable scan contracts, then implement fixed-registry streaming scanners for supported line-oriented formats. Materialization selects bytes or streams, performs exact-head revalidation, and gives native Audit a temporary fixed-schema dataset index; only Audit creates findings and verdicts.

**Tech Stack:** Python 3.11+, standard-library `os`, `hashlib`, `json`, `csv`, incremental UTF-8 decoding, `sqlite3`, immutable dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-claimci-streaming-artifact-v1-design.md`

## Global Constraints

- Base exactly `381d2237e14e3f0ab997e11988fa747c777d955d`; stack the Draft PR on `codex/metric-identity-v1`.
- Core only: do not touch ClaimCI-Web, main, PR #27, PR #28, the release conductor, deployment, Cloudflare, staging, or production.
- Keep `PassiveArtifact` and all existing bytes APIs compatible for small files.
- No customer code/import/shell/network/SQL execution; SQLite schema and statements are fixed and parameterized by Core.
- No Code/Markdown indexing, distributed processing, vector database, or LLM context expansion.
- A source must have verified EOF size/SHA and stable descriptor/path identity before any evidence is eligible.
- Semantic incompleteness is never upgraded by a later integrity drain.
- Only confirmed exact dataset leakage may survive an integrity-verified partial semantic scan; universal/aggregate evidence requires `COMPLETE`.
- Existing `AuditResult` and `determine_verdict()` remain the sole finding/verdict path.
- Generate large fixtures in pytest temporary directories; do not commit large data files.
- Follow RED → verify RED → minimal GREEN → verify GREEN for every production behavior.

---

### Task 1: Trusted source, scan authority, and streaming discovery hash

**Files:**
- Create: `claimci/analysis/artifact_source.py`
- Modify: `claimci/passive_files.py`
- Modify: `claimci/analysis/discovery/models.py`
- Modify: `claimci/analysis/discovery/repository.py`
- Modify: `claimci/analysis/discovery/artifacts.py`
- Modify: `claimci/analysis/__init__.py`
- Create: `tests/test_streaming_source_v1.py`

**Interfaces:**
- Consumes: `RepositoryIdentity`, `GitCommitSha`, `ArtifactCandidate`, `ArtifactKind`, `RepositoryPath`, `Sha256Digest`.
- Produces: `ScanPurpose`, `ScanState`, `ScanReason`, `StreamingLimits`, `ScanCompleteness`, `ArtifactSource`, `ArtifactScan`, `artifact_source_from_snapshot()`, and `inspect_confined_regular_file()`.

- [ ] **Step 1: Write failing source-contract and integrity tests**

Add real temporary-file tests with the desired public surface:

```python
def test_stream_source_verifies_incremental_size_sha_and_never_reads_unbounded(tmp_path):
    content = (b'{"run":1,"acc":0.9}\n' * 500_000)
    candidate = issued_candidate("results/eval.jsonl", content)
    source = artifact_source_from_snapshot(REPOSITORY, HEAD, tmp_path, candidate)
    with source.open_scan(ScanPurpose.INTEGRITY_ONLY, LIMITS) as scan:
        while scan.read_chunk(64 * 1024):
            pass
        report = scan.finish(records_scanned=0, semantic_complete=True)
    assert report.state is ScanState.COMPLETE
    assert report.integrity_verified is True
    assert report.integrity_bytes == len(content)
    assert scan.maximum_read_request <= 64 * 1024

def test_stream_source_rejects_public_construction_and_source_substitution(tmp_path):
    with pytest.raises(TypeError):
        ArtifactSource()
    source = source_fixture(tmp_path, b"old")
    replace_confined_file(tmp_path, b"new")
    with pytest.raises(StreamingIntegrityError):
        verify_source(source)
```

Also cover strict root/path confinement, parent/final symlinks, size mismatch,
SHA mismatch, truncated reads, integrity-I/O exhaustion, exact head/repository
fields, and `ScanCompleteness` invariant rejection.

- [ ] **Step 2: Run the source tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_streaming_source_v1.py -p no:cacheprovider
```

Expected: collection fails because `claimci.analysis.artifact_source` and the
streaming inspection function do not exist.

- [ ] **Step 3: Implement immutable contracts and descriptor-bound scanning**

Create exact enums and dataclasses:

```python
class ScanPurpose(str, Enum):
    INTEGRITY_ONLY = "integrity_only"
    SCHEMA = "schema"
    SELECTED_OBSERVATIONS = "selected_observations"
    DATASET_RECORDS = "dataset_records"

class ScanState(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE_BUDGET = "incomplete_budget"
    INCOMPLETE_LIMIT = "incomplete_limit"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class ScanCompleteness:
    purpose: ScanPurpose
    state: ScanState
    reason: ScanReason
    records_scanned: int
    semantic_bytes: int
    integrity_bytes: int
    expected_bytes: int
    integrity_verified: bool
    peak_buffer_bytes: int
    scratch_bytes: int = 0
```

`ArtifactSource` is `init=False`, stores its root in a private non-repr field,
and exposes only exact immutable metadata. `artifact_source_from_snapshot()`
validates the root, path components, issued candidate, repository, and head.
`open_scan()` repeats confinement, opens with `O_NOFOLLOW` where available,
and returns an `ArtifactScan` that reads at most 64 KiB per raw operation,
updates raw/domain-separated hashes, can stop semantic delivery, drains for
integrity, and rechecks handle/path/root identities in `finish()`/`__exit__()`.

- [ ] **Step 4: Add non-materializing discovery inspection**

In `passive_files.py`, add:

```python
@dataclass(frozen=True, slots=True)
class PassiveFileInspection:
    size: int
    sha256: str

def inspect_confined_regular_file(
    root: Path,
    relative: str,
    *,
    max_bytes: int,
) -> PassiveFileInspection: ...
```

Share confinement helpers with byte capture. Stream/hash without retaining
chunks. Extend `DiscoveryLimits` with 1 GiB per-file and 4 GiB aggregate
stream-integrity bounds while retaining the 16 MiB bytes threshold.
`inspect_artifact()` uses byte capture only below the threshold and streaming
inspection above it. `discover_artifacts()` decrements one aggregate integrity
budget; manifest text remains on the bounded bytes path.

- [ ] **Step 5: Run focused discovery/source tests to GREEN**

Run:

```powershell
python -m pytest -q tests/test_streaming_source_v1.py tests/test_auto_discovery_v03.py tests/test_analysis_contracts_v03.py -p no:cacheprovider
```

Expected: all pass, including a generated artifact larger than 16 MiB being
hash-inspected without materialized content.

- [ ] **Step 6: Commit Task 1**

```powershell
git add claimci/passive_files.py claimci/analysis/artifact_source.py claimci/analysis/discovery/models.py claimci/analysis/discovery/repository.py claimci/analysis/discovery/artifacts.py claimci/analysis/__init__.py tests/test_streaming_source_v1.py
git commit -m "feat: add trusted streaming artifact sources"
```

### Task 2: Streaming JSONL and schema-level MetricCandidates

**Files:**
- Create: `claimci/analysis/adapters/streaming.py`
- Modify: `claimci/analysis/adapters/structured.py`
- Modify: `claimci/analysis/adapters/dataset.py`
- Modify: `claimci/analysis/adapters/registry.py`
- Modify: `claimci/analysis/adapters/__init__.py`
- Modify: `claimci/analysis/metric_identity.py`
- Modify: `claimci/analysis/contracts.py`
- Create: `tests/test_streaming_jsonl_v1.py`

**Interfaces:**
- Consumes: Task 1 `ArtifactSource`, `ArtifactScan`, `StreamingLimits`, and scan enums; existing JSON pointer/mapping/provenance contracts.
- Produces: `StreamingExtraction`, `extract_registered_source()`, `scan_jsonl_schema()`, `scan_jsonl_observations()`, and source support in `extract_metric_candidates()` / `extract_bound_metric_evidence()`.

- [ ] **Step 1: Write failing JSONL streaming tests**

Generate files rather than fixtures. Required assertions include:

```python
def test_large_jsonl_discovers_schema_candidates_without_retaining_records(tmp_path):
    source = large_jsonl_source(tmp_path, records=12_500, padding_bytes=700)
    result = extract_metric_candidate_scan(source, role=ExperimentRole.BASELINE)
    assert result.completeness.state is ScanState.COMPLETE
    assert [item.raw_metric_name for item in result.candidates] == ["acc", "loss"]
    assert result.completeness.peak_buffer_bytes <= MAX_LOGICAL_RECORD_BYTES
    assert result.retained_raw_records == 0

def test_incomplete_jsonl_never_issues_complete_candidates(tmp_path):
    source = jsonl_source(tmp_path, records=20)
    outcome = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
        limits=StreamingLimits(max_semantic_records=10),
    )
    assert outcome.candidates == ()
    assert outcome.completeness.state is ScanState.INCOMPLETE_BUDGET
    assert outcome.completeness.integrity_verified is True
```

Add >8 MiB, >10,000 records, malformed middle record, strict UTF-8, blank
result line, duplicate keys, non-finite value, per-record depth/nodes, 1 MiB
logical line, candidate explosion, and selected observation budget tests.
Compare candidates from equivalent bytes/source artifacts and assert identical
candidate IDs, adapter IDs, selectors, raw names, paths, SHA, and roles.

- [ ] **Step 2: Run JSONL tests and verify RED**

```powershell
python -m pytest -q tests/test_streaming_jsonl_v1.py -p no:cacheprovider
```

Expected: failures for missing streaming registry/scanners and source-aware
Metric Identity APIs.

- [ ] **Step 3: Implement bounded JSONL scanner**

`scan_jsonl_schema()` reads one bounded binary line at a time, decodes strict
UTF-8, parses with `unique_json_object`, rejects constants, calls existing
graph validation per record, and maintains only bounded common-leaf/type state.
It never calls `read()`, `read_text()`, `decode()` on the complete artifact,
`splitlines()`, or builds a record tuple.

`scan_jsonl_observations()` validates an exact issued match and constructs only
selected normalized metric/run/seed observations up to 100,000. If the scan is
not complete, return no `NormalizedEvidence`. `StreamingExtraction` enforces:

```python
if completeness.state is ScanState.COMPLETE:
    assert completeness.integrity_verified and evidence is not None
else:
    assert evidence is None
```

The dataset identity adapter performs an `INTEGRITY_ONLY` scan and returns its
identity evidence with that purpose; it does not claim dataset-record semantic
completeness.

- [ ] **Step 4: Integrate fixed registry and Metric Identity**

Add `extract_registered_source(source, *, limits)` that dispatches only fixed
source-capable adapter IDs. Add `extract_metric_candidate_scan()` returning
candidates plus completeness; keep existing `extract_metric_candidates()` for
bytes, and allow a source only when its scan is complete. Source candidate
construction calls the same `_from_extraction()` factory and keeps adapter IDs
unchanged. Source-aware bound extraction canonicalizes only after a complete
selected-observation scan.

- [ ] **Step 5: Run JSONL, adapters, and Metric Identity to GREEN**

```powershell
python -m pytest -q tests/test_streaming_jsonl_v1.py tests/test_adapters_structured_v03.py tests/test_adapters_registry_v03.py tests/test_metric_identity_v1.py -p no:cacheprovider
```

Expected: all pass; source/bytes candidate identity parity is exact.

- [ ] **Step 6: Commit Task 2**

```powershell
git add claimci/analysis/adapters/streaming.py claimci/analysis/adapters/structured.py claimci/analysis/adapters/dataset.py claimci/analysis/adapters/registry.py claimci/analysis/adapters/__init__.py claimci/analysis/metric_identity.py claimci/analysis/contracts.py tests/test_streaming_jsonl_v1.py
git commit -m "feat: stream JSONL evidence and metric candidates"
```

### Task 3: Streaming CSV/TSV and exact TableSelector scans

**Files:**
- Modify: `claimci/analysis/adapters/streaming.py`
- Modify: `claimci/analysis/adapters/tabular.py`
- Modify: `claimci/analysis/metric_identity.py`
- Create: `tests/test_streaming_tabular_v1.py`

**Interfaces:**
- Consumes: Task 2 streaming outcomes and existing `TableSelector`, predicates, adapter validation, and normalized observations.
- Produces: `scan_delimited_schema()` and `scan_delimited_observations()` for fixed CSV/TSV adapters.

- [ ] **Step 1: Write failing delimited streaming tests**

Cover CSV and TSV parametrically:

```python
@pytest.mark.parametrize("suffix,delimiter", [("csv", ","), ("tsv", "\t")])
def test_large_table_discovers_numeric_columns_without_row_retention(...):
    source = generated_table_source(rows=12_500, bytes_over=8 * 1024 * 1024)
    outcome = extract_metric_candidate_scan(source, role=ExperimentRole.CANDIDATE)
    assert outcome.completeness.state is ScanState.COMPLETE
    assert raw_names(outcome) == ("accuracy", "f1", "loss")
    assert outcome.retained_raw_rows == 0

def test_table_selector_requires_complete_exact_cardinality(...):
    outcome = extract_selected_source(source, table_match(expected_cardinality=2), tiny_budget)
    assert outcome.evidence is None
    assert outcome.completeness.state is ScanState.INCOMPLETE_BUDGET
```

Also test exactly 256 columns, 257 rejection, duplicate/empty headers,
malformed row field counts, strict UTF-8, NUL, quoted newlines, cumulative
logical-record limit, numeric discovery, nonnumeric elimination, exact selected
rows, and >10,000 rows.

- [ ] **Step 2: Run table tests and verify RED**

```powershell
python -m pytest -q tests/test_streaming_tabular_v1.py -p no:cacheprovider
```

Expected: failures for missing delimited streaming functions/source dispatch.

- [ ] **Step 3: Implement incremental CSV/TSV parsing**

Feed `csv.reader(strict=True)` from an iterator of individually strict-decoded
physical lines supplied by `ArtifactScan`. Track cumulative physical bytes
until one logical row returns; raise the typed logical-record limit before
more than 1 MiB can be supplied. Restore `csv.field_size_limit()` in `finally`.

Schema mode retains only the header and numeric eligibility flags. Observation
mode retains only selected normalized cells. Full-column selections enforce
the observation bound. Table selections count every match through EOF, retain
only the bounded expected rows, and issue evidence only when the exact final
count equals `expected_cardinality` under `COMPLETE`.

- [ ] **Step 4: Integrate source candidates and rerun focused suites**

Use the same `claimci-csv-v1`/`claimci-tsv-v1` IDs and existing provenance/
selector canonicalization. Run:

```powershell
python -m pytest -q tests/test_streaming_tabular_v1.py tests/test_adapters_csv_v03.py tests/test_benchmark_table_adapter_v0.py tests/test_metric_identity_v1.py -p no:cacheprovider
```

Expected: all pass, including exact source/bytes candidate parity and no first
match from an incomplete table.

- [ ] **Step 5: Commit Task 3**

```powershell
git add claimci/analysis/adapters/streaming.py claimci/analysis/adapters/tabular.py claimci/analysis/metric_identity.py tests/test_streaming_tabular_v1.py
git commit -m "feat: stream delimited evidence with exact selectors"
```

### Task 4: Bounded dataset spill and partial-scan Audit evidence

**Files:**
- Create: `claimci/analysis/streaming_dataset.py`
- Modify: `claimci/dataset_check.py`
- Modify: `claimci/models.py`
- Modify: `claimci/audit.py`
- Modify: `claimci/measurement.py`
- Create: `tests/test_streaming_dataset_v1.py`

**Interfaces:**
- Consumes: dataset-capable `ArtifactSource`, scan contracts, runtime scratch root, existing canonical sample hash and dataset findings.
- Produces: factory-only `DatasetScanAuditContext`, `DatasetMultisetIdentity`, `stream_dataset_audit_context()`, and Audit consumption of exact/partial scan facts.

- [ ] **Step 1: Write failing complete/partial/cleanup tests**

Encode all three required authority cases:

```python
def test_complete_large_pair_may_emit_no_leakage(tmp_path):
    context = scan_dataset_context(full_nonoverlapping_sources(tmp_path))
    result = audit_research(manifest, dataset_scan_context=context)
    assert rule(result, "DATASET.NO_LEAKAGE")
    assert not rule(result, "DATASET.SCAN_INCOMPLETE")

def test_integrity_verified_partial_without_overlap_cannot_emit_no_leakage(tmp_path):
    context = scan_dataset_context(nonoverlap_sources(tmp_path), semantic_records=10)
    result = audit_research(manifest, dataset_scan_context=context)
    assert not rule(result, "DATASET.NO_LEAKAGE")
    assert rule(result, "DATASET.SCAN_INCOMPLETE").impact is Impact.INSUFFICIENT

def test_integrity_verified_partial_exact_overlap_remains_invalidating(tmp_path):
    context = scan_dataset_context(overlap_in_inspected_prefix(tmp_path), semantic_records=10)
    result = audit_research(manifest, dataset_scan_context=context)
    assert rule(result, "DATASET.EXACT_LEAKAGE").impact is Impact.INVALIDATES
    assert result.verdict is Verdict.NOT_SUPPORTED
```

Add integrity mismatch (no context/evidence), blank-line compatibility,
malformed/duplicate-key records, scratch exhaustion, bounded 128-hash preview,
fixed schema inspection, parameterized-statement inspection, permissions where
supported, and cleanup after success, parser failure, budget failure, and
Audit exception.

- [ ] **Step 2: Run dataset tests and verify RED**

```powershell
python -m pytest -q tests/test_streaming_dataset_v1.py -p no:cacheprovider
```

Expected: missing streaming dataset context/store APIs.

- [ ] **Step 3: Implement fixed SQLite spill session**

Create a context-managed randomized `*.sqlite3` file under the supplied
resolved scratch root. Use a fixed table:

```sql
CREATE TABLE dataset_hash_counts (
  experiment TEXT NOT NULL CHECK (experiment IN ('baseline','candidate')),
  split TEXT NOT NULL CHECK (split IN ('train','eval')),
  digest TEXT NOT NULL,
  multiplicity INTEGER NOT NULL,
  PRIMARY KEY (experiment, split, digest)
)
```

Use only constant SQL and parameters. Insert/update in bounded batches, check
database and sidecar sizes against `max_scratch_bytes`, and record peak use.
Always close and remove database/sidecars in `finally`.

- [ ] **Step 4: Implement dataset scan facts and Audit rules**

Stream strict JSONL records, preserve existing blank-line handling, canonicalize
with `canonicalize_sample()`, and upsert digest counts. SQL derives exact
intersection totals and sorted bounded previews. Complete pairs call existing
no-leakage/evaluation-alignment semantics. Incomplete pairs may create only:

```text
DATASET.EXACT_LEAKAGE       Impact.INVALIDATES  (confirmed overlap only)
DATASET.SCAN_INCOMPLETE     Impact.INSUFFICIENT
```

No overlap on an incomplete pair never calls `check_dataset_leakage()`'s
no-leakage branch. Evaluation alignment requires both eval scans complete.
`DatasetMultisetIdentity(record_count, canonical_sha256)` is computed by
streaming sorted `(digest, multiplicity)` rows and supplied to Measurement
Drift without materializing every hash.

- [ ] **Step 5: Run dataset/Audit/measurement tests to GREEN**

```powershell
python -m pytest -q tests/test_streaming_dataset_v1.py tests/test_dataset_check.py tests/test_audit.py tests/test_measurement_drift_v1.py -p no:cacheprovider
```

Expected: all pass and the scratch root is empty after every case.

- [ ] **Step 6: Commit Task 4**

```powershell
git add claimci/analysis/streaming_dataset.py claimci/dataset_check.py claimci/models.py claimci/audit.py claimci/measurement.py tests/test_streaming_dataset_v1.py
git commit -m "feat: add bounded streaming dataset evidence"
```

### Task 5: Exact-head materialization, Trace, and atomic Metric Binding

**Files:**
- Modify: `claimci/analysis/materialize.py`
- Modify: `claimci/analysis/measurement.py`
- Modify: `claimci/analysis/planner.py`
- Modify: `claimci/analysis/trace.py`
- Modify: `claimci/analysis/metric_identity.py`
- Modify: `claimci/audit.py`
- Create: `tests/test_streaming_materialize_v1.py`

**Interfaces:**
- Consumes: source-capable adapters, complete streaming normalized evidence, dataset Audit context, existing `EphemeralAuditPlan` and `MetricBinding`.
- Produces: bytes/source capture selection, streaming plan commitment, exact-head revalidation, streamed Trace commitments, and end-to-end Audit execution.

- [ ] **Step 1: Write failing materialization and Metric Binding tests**

Build actual plans from generated large baseline/candidate JSONL or CSV result
sources containing raw `acc`, approve the existing one-pair `[acc]` question,
then execute exact-head materialization:

```python
def test_large_acc_pair_streams_through_exact_binding_and_real_audit(...):
    result = execute_ephemeral_audit(plan_with_large_atomic_acc_pair, runtime)
    assert result.audit_result.verdict is Verdict.SUPPORTED
    assert rule(result.audit_result, "RESULT.METRIC_IDENTITY_VERIFIED")
    assert all(report.integrity_verified for report in result.scan_completeness)

@pytest.mark.parametrize("role", [ExperimentRole.BASELINE, ExperimentRole.CANDIDATE])
def test_changed_stream_endpoint_invalidates_entire_metric_pair(role, ...):
    mutate_exact_head_artifact(role)
    with pytest.raises(MaterializationUnavailable, match="entire metric pair"):
        execute_ephemeral_audit(plan, runtime)
```

Also cover head/repository/path/kind/size/SHA/adapter/selector/raw-name drift,
mixed small/large artifacts, a large TableSelector, dataset partial-positive
and partial-negative Audit outcomes, trace scan material, and no large dataset
copy in the native tree.

- [ ] **Step 2: Run materialization tests and verify RED**

```powershell
python -m pytest -q tests/test_streaming_materialize_v1.py -p no:cacheprovider
```

Expected: large candidates still fail the materialization file bound or source
objects are rejected where `PassiveArtifact` is required.

- [ ] **Step 3: Select bytes or source under runtime budgets**

Extend `MaterializationLimits` with streamed per-file, aggregate integrity,
semantic, record, observation, and scratch limits. `_capture_artifact()` keeps
returning `PassiveArtifact` at/below `max_file_bytes`; above it returns an
`ArtifactSource` created from the runtime repository/head/root. The captured
mapping becomes `PassiveArtifact | ArtifactSource` and requires exact candidate
equality in measurement/source binding code.

- [ ] **Step 4: Revalidate adapters, plans, and Metric Binding**

Freshly rerun every selected streaming adapter. Complete results/tables must
equal planned selectors and normalized observations. Add streaming
completeness material to plan identity only when present, keeping legacy plan
IDs unchanged. `_revalidate_metric_binding()` invokes source candidate and
bound-evidence scans independently for both roles; any failure/incomplete scan
raises the existing whole-pair reapproval error.

- [ ] **Step 5: Integrate dataset context and streamed Trace**

Before Audit, build one dataset scratch context for selected dataset bindings.
When it is present, `_write_native_tree()` does not copy dataset source bytes;
the manifest retains confined relative paths and `audit_research()` consumes
only the trusted context for datasets. Add a trusted constructor to
`BoundedValueRepresentation` that accepts the domain-separated digest and size
computed by `ArtifactScan`; never request source bytes for Trace.

Ensure the dataset scratch context remains open through Audit and Trace and is
cleaned in the outer `finally`, including Audit/Trace exceptions.

- [ ] **Step 6: Run focused end-to-end tests to GREEN**

```powershell
python -m pytest -q tests/test_streaming_materialize_v1.py tests/test_ephemeral_materialize_v03.py tests/test_metric_identity_v1.py tests/test_evidence_trace_v1.py tests/test_analysis_cross_package_v03.py tests/test_benchmark_evidence_profile_v0.py -p no:cacheprovider
```

Expected: all pass and the real Audit result—not scan code—sets every verdict.

- [ ] **Step 7: Commit Task 5**

```powershell
git add claimci/analysis/materialize.py claimci/analysis/measurement.py claimci/analysis/planner.py claimci/analysis/trace.py claimci/analysis/metric_identity.py claimci/audit.py tests/test_streaming_materialize_v1.py
git commit -m "feat: materialize exact-head streaming evidence"
```

### Task 6: Focused hostile review, performance proof, and compatibility

**Files:**
- Create: `tests/test_streaming_security_v1.py`
- Modify only for proven defects: streaming/source/dataset/materialization files from Tasks 1–5
- Modify: `docs/superpowers/specs/2026-08-25-claimci-streaming-artifact-v1-design.md` only if implemented names materially differ

**Interfaces:**
- Consumes: every new public/trusted streaming surface.
- Produces: adversarial regression coverage and final scope/security evidence.

- [ ] **Step 1: Add adversarial tests before any fixes**

Test real failures for traversal, root/parent/final symlink swaps, source
replacement between reopen and scan, descriptor mutation, provider-created
source/context attempts, huge logical records, candidate explosion, CSV quoted
record abuse, JSON depth/nodes, scratch exhaustion/injection strings, cleanup,
stale Metric Binding, truncated stream, and incomplete negative claims.

Add an AST/structural test that rejects unbounded `read()`, `read_bytes()`,
`read_text()`, whole-source `decode()`, `splitlines()`, and raw row/record list
accumulation in the new streaming path. Assert completeness peak buffer is at
most the logical-record limit plus one 64 KiB chunk for a generated >8 MiB
source. Do not use timing or exact RSS.

- [ ] **Step 2: Run security tests and verify any new test fails for its intended reason**

```powershell
python -m pytest -q tests/test_streaming_security_v1.py -p no:cacheprovider
```

Expected: each newly exposed weakness fails at the relevant assertion; tests
of already-correct behavior may be grouped only after one deliberately absent
security assertion establishes RED.

- [ ] **Step 3: Fix proven failures minimally and rerun focused security**

Keep fixes inside the approved streaming source/parser/spill/materialization
scope. Rerun the single failing test, then the security file, until green.

- [ ] **Step 4: Run the one final focused verification matrix**

Run once after feature code is complete:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_streaming_source_v1.py tests/test_streaming_jsonl_v1.py tests/test_streaming_tabular_v1.py tests/test_streaming_dataset_v1.py tests/test_streaming_materialize_v1.py tests/test_streaming_security_v1.py tests/test_adapters_core_v03.py tests/test_adapters_structured_v03.py tests/test_adapters_csv_v03.py tests/test_adapters_registry_v03.py tests/test_dataset_check.py tests/test_metric_identity_v1.py tests/test_ephemeral_planner_v03.py tests/test_ephemeral_materialize_v03.py tests/test_benchmark_table_adapter_v0.py tests/test_benchmark_evidence_profile_v0.py tests/test_audit.py tests/test_evidence_trace_v1.py tests/test_analysis_cross_package_v03.py
```

Expected: all pass with only established skips.

- [ ] **Step 5: Run final static and complete Core gates exactly once**

```powershell
python -m compileall -q claimci
python -m pytest -q -p no:cacheprovider
git diff --check 381d2237e14e3f0ab997e11988fa747c777d955d..HEAD
```

Run changed-file lint only if the repository adds a configured linter. Do not
run Web, Node, Cloud, OAuth, D1, staging, or production tests.

- [ ] **Step 6: Commit final security/compatibility changes**

```powershell
git add tests/test_streaming_security_v1.py claimci docs/superpowers/specs/2026-08-25-claimci-streaming-artifact-v1-design.md
git commit -m "test: harden streaming artifact boundaries"
```

### Task 7: Push one stacked Draft PR and stop

**Files:**
- No source changes.

**Interfaces:**
- Consumes: verified clean branch and frozen parent.
- Produces: exactly one Draft Core PR based on `codex/metric-identity-v1`.

- [ ] **Step 1: Verify frozen ancestry, scope, and parent movement**

```powershell
git status --short
git merge-base HEAD 381d2237e14e3f0ab997e11988fa747c777d955d
git ls-remote --heads origin refs/heads/codex/metric-identity-v1
git diff --name-only 381d2237e14e3f0ab997e11988fa747c777d955d..HEAD
```

The merge base must be the frozen SHA. If the remote parent moved, do not
rebase; report `PARENT_MOVED_REBASE_REQUIRED` while keeping the frozen branch.
The diff must contain only Core streaming code/tests/docs.

- [ ] **Step 2: Push only the streaming branch**

```powershell
git push -u origin codex/streaming-artifact-v1
```

- [ ] **Step 3: Create and verify exactly one Draft PR**

Check for an existing head PR first. If absent, create one with base
`codex/metric-identity-v1`, head `codex/streaming-artifact-v1`, the exact test
results, authority rules, security review, and Hosted companion statement.
Verify `isDraft=true`, exact base/head refs and head SHA. Do not merge.

- [ ] **Step 4: Report the requested 25 fields and stop**

Report exact base/head, branch/worktree, old/new boundaries, compatibility,
integrity/completeness/budgets, format behavior, dataset partial rules, spill
and cleanup, Metric Identity/Binding behavior, changed files, test counts,
memory/security evidence, `HOSTED_COMPANION_REQUIRED` if applicable, and the
Draft PR URL. Perform no deployment or release-conductor action.
