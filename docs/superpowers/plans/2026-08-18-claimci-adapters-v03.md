# ClaimCI v0.3 Adapter Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, deterministic, zero-configuration adapter package that translates passive JSON, JSONL, CSV, YAML, TOML, and native ClaimCI artifacts into the existing immutable analysis contracts without assigning scientific authority or inferred experiment roles.

**Architecture:** A shared `core.py` enforces bytes, record, column, depth, node, UTF-8, finite-number, selector, provenance, and integrity rules. Explicit format modules implement the eight approved adapters, and `registry.py` exposes only a fixed trusted tuple and exact-ID lookup. Generic adapters always emit `UNSPECIFIED`; only native `research.yaml` preserves explicit manifest roles as `MANIFEST_HINT` data.

**Tech Stack:** Python 3.11+, standard-library `json`, `csv`, `tomllib`, `hashlib`, `pathlib`; existing pinned `PyYAML==6.0.3`; existing `claimci.analysis` frozen contracts; pytest.

## Global Constraints

- Branch is exactly `feat/adapters-v03`, based on merged `origin/main` commit `cdf523542c66241b858b33f30362e601af15ffe7`.
- Artifact limit is 8 MiB; record limit 10,000; CSV column limit 256; nesting depth 64; structured nodes 100,000; logical record 1 MiB.
- Generic JSON/JSONL/CSV/YAML/TOML roles are always `ExperimentRole.UNSPECIFIED`.
- Native manifest roles are explicit `MANIFEST_HINT` provenance only; adapters never create `RepoMapping`, plans, verdicts, Audit results, or authority.
- Extraction recomputes SHA-256 immediately before parsing.
- No truncation, partial parsing at limits, unsafe YAML, pickle, dynamic import, plugins, eval, customer code import/execution, network access, dataset download, or large dataset parsing.
- Existing CLI, Audit, Research Review, workflow, and provider behavior must remain unchanged.
- Production code follows strict RED/GREEN TDD; stage only the files owned by each task.

---

### Task 1: Shared bounded parsing and selector core

**Files:**
- Create: `claimci/analysis/adapters/core.py`
- Create: `tests/test_adapters_core_v03.py`

**Interfaces:**
- Consumes: `PassiveArtifact`, `AdapterMatch`, `ArtifactKind`, `EvidenceSelector`, `FieldMapping`, `FieldProvenance`, `ProvenanceKind`, `RepositoryPath`, and `SelectorKind` from `claimci.analysis`; `load_unique_yaml` and `unique_json_object` from `claimci.parsing`.
- Produces public errors/constants: `AdapterError`, `AdapterParseError`, `AdapterLimitError`, `AdapterSelectorError`, `AdapterIntegrityError`, `MAX_ARTIFACT_BYTES`, `MAX_RECORDS`, `MAX_COLUMNS`, `MAX_DEPTH`, `MAX_NODES`, `MAX_LOGICAL_RECORD_BYTES`.
- Produces internal helpers used by later tasks: `_supports`, `_verify_integrity`, `_decode_utf8`, `_parse_json`, `_validate_graph`, `_resolve_json_pointer`, `_resolve_dotted_path`, `_scalar_leaves`, `_adapter_provenance`, `_mapping`, `_validate_match`, `_evidence_id`, `_config_target`, `_finite_number`.

- [ ] **Step 1: Write failing core tests**

Add behavior tests that construct real `ArtifactCandidate`/`PassiveArtifact` values and prove:

```python
def test_integrity_is_recomputed_before_extraction_helper():
    artifact = passive(b'{"accuracy":0.9}')
    object.__setattr__(artifact, "content", b'{"accuracy":0.1}')
    with pytest.raises(AdapterIntegrityError):
        _verify_integrity(artifact)


def test_json_pointer_decodes_only_rfc6901_tokens():
    value = {"a/b": {"~metric": [0.9]}}
    assert _resolve_json_pointer(value, "/a~1b/~0metric/0") == 0.9
    with pytest.raises(AdapterSelectorError):
        _resolve_json_pointer(value, "/a~2b")
```

Also test strict UTF-8, duplicate JSON keys, NaN/Infinity, exact limit acceptance and over-limit rejection, depth 64/65, node 100,000/100,001, canonical array indices, missing tokens, dotted segments, finite numeric conversion, stable config target, source path/selector/hash provenance, and recognized-kind/suffix checks.

- [ ] **Step 2: Run core tests to verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py
```

Expected: collection fails because `claimci.analysis.adapters.core` does not exist.

- [ ] **Step 3: Implement the minimal shared core**

Implement the approved errors and limits. `_verify_integrity` checks exact
`PassiveArtifact`, length, 8 MiB limit, and SHA-256. `_parse_json` uses
`object_pairs_hook=unique_json_object` and a `parse_constant` callback that
raises `AdapterParseError`. `_validate_graph` iteratively counts every mapping,
sequence, key, and scalar while enforcing finite numbers, depth, nodes, string
mapping keys, and no recursive/repeated container identities.

`_resolve_json_pointer` validates RFC 6901 escapes and canonical array indices.
`_resolve_dotted_path` validates every `[A-Za-z0-9_-]+` segment and walks
mappings only. `_scalar_leaves` emits sorted `(selector, scalar)` pairs and
reports unrepresentable/non-scalar leaves without rewriting them.

`_validate_match` requires exact adapter ID/path, allowlisted targets and
selector kind, source-path provenance matching the artifact, and no unused
target. `_adapter_provenance` includes adapter ID, selector, artifact path, and
full SHA-256. `_config_target` is `config.k` plus the first 16 lowercase hex
characters of SHA-256 over the UTF-8 dotted selector.

- [ ] **Step 4: Run Task 1 GREEN checks**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_analysis_contracts_v03.py tests/test_parsing_hardening_day2.py
python -m compileall -q claimci
git diff --check
```

- [ ] **Step 5: Commit Task 1**

Stage only the two Task 1 files and commit:

```text
feat: add bounded adapter parsing core
```

---

### Task 2: Generic JSON and per-record JSONL adapters

**Files:**
- Create: `claimci/analysis/adapters/structured.py`
- Create: `tests/test_adapters_structured_v03.py`
- Create: `tests/fixtures/adapters/nested_metric.json`
- Create: `tests/fixtures/adapters/seed_runs.jsonl`
- Create: `tests/fixtures/adapters/ambiguous_metrics.json`

**Interfaces:**
- Consumes: Task 1 helpers and existing `AdapterMatch`, `Confidence`, `ExperimentRole`, `NormalizedEvidence`, `NormalizedObservation`.
- Produces: `JsonAdapter` with ID `claimci-json-v1`; `JsonLinesAdapter` with ID `claimci-jsonl-v1`.
- Mapping targets are exactly `metric_value`, optional `metric_name`, `run_id`, and `seed`; JSON config artifacts additionally use generated `config.k<digest>` targets.

- [ ] **Step 1: Add realistic fixtures and failing JSON tests**

`nested_metric.json` contains a nested finite accuracy value plus run and seed.
`seed_runs.jsonl` contains three JSON objects with repeated accuracy/seed/run
fields. `ambiguous_metrics.json` contains both accuracy and loss.

Tests prove JSON Pointer nested extraction, optional metric-name derivation from
the terminal token, exact provenance/path/hash, and `UNSPECIFIED` role. They
also prove an ambiguous JSON match has no `metric_value` mapping and that
`extract()` raises `AdapterSelectorError` until passed a separately constructed,
validated `AdapterMatch` with an explicit pointer.

JSONL tests prove `/metrics/accuracy` resolves independently within each line,
source order is preserved, three observations retain run/seed, and no selector
can address the file as a global array.

Hostile cases cover blank/non-object JSONL records, duplicate keys, malformed
lines, nonfinite numbers, more than 10,000 records, a record over 1 MiB, wrong
selector kinds, path mismatch, adapter-ID mismatch, and generic role invariance.

- [ ] **Step 2: Run Task 2 tests to verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_structured_v03.py
```

Expected: collection fails because `structured.py` and adapters are absent.

- [ ] **Step 3: Implement JSON adapters**

`JsonAdapter.probe` accepts `.json` with `RESULTS`, `BENCHMARK`, `DOCUMENT`, or
`CONFIG`. For non-config artifacts it scans sorted scalar pointers, recognizes
one unique run/seed terminal, and maps a metric only when exactly one remaining
finite numeric pointer exists. For `CONFIG`, it emits stable config mappings and
never infers a metric.

`JsonLinesAdapter` accepts `.jsonl` with `RESULTS`, `BENCHMARK`, or `DOCUMENT`.
It validates every non-empty logical line independently, intersects scalar
pointers across records, and infers mappings only when the same unambiguous
pointer is valid in every record.

Both `extract` methods call `_verify_integrity` first, reparse, validate the
provided match, require `metric_value` for results evidence, never fill a missing
mapping, and create only `ExperimentRole.UNSPECIFIED` observations.

- [ ] **Step 4: Run Task 2 GREEN checks**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_adapters_structured_v03.py tests/test_analysis_contracts_v03.py
python -m compileall -q claimci
git diff --check
```

- [ ] **Step 5: Commit Task 2**

Stage only Task 2 production, tests, and three fixtures. Commit:

```text
feat: add bounded JSON adapters
```

---

### Task 3: Deterministic CSV adapter

**Files:**
- Create: `claimci/analysis/adapters/tabular.py`
- Create: `tests/test_adapters_csv_v03.py`
- Create: `tests/fixtures/adapters/seed_runs.csv`
- Create: `tests/fixtures/adapters/ambiguous_metrics.csv`

**Interfaces:**
- Consumes Task 1 core and existing normalized contracts.
- Produces `CsvAdapter` with ID `claimci-csv-v1` and `SelectorKind.COLUMN` mappings for `metric_value`, optional `run_id`, and optional `seed`.

- [ ] **Step 1: Add CSV fixtures and failing behavior tests**

`seed_runs.csv` has headers `run_id,seed,accuracy` and three finite rows.
`ambiguous_metrics.csv` adds a second numeric metric column.

Tests prove unambiguous automatic mappings, exact column matching, record-order
observations, run/seed preservation, finite accuracy extraction, provenance,
and `UNSPECIFIED` roles. Ambiguity tests prove no metric mapping and extraction
refusal until a validated external column mapping is supplied.

Hostile tests cover duplicate/empty headers, inconsistent field counts, invalid
UTF-8, malformed quoting, nonfinite selected values, 257 columns, 10,001 rows,
a logical row over 1 MiB, locale-style decimal commas not being reinterpreted,
wrong selector kind, path/adapter mismatch, and extraction-time SHA mutation.

- [ ] **Step 2: Run CSV tests to verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_csv_v03.py
```

Expected: collection fails because `CsvAdapter` is absent.

- [ ] **Step 3: Implement the CSV adapter**

Use `csv.reader(..., dialect="excel", strict=True)` with comma delimiter and no
dialect sniffing. Validate header and every full row before creating a match.
Compute logical-record byte size from decoded cell contents and separators.
Recognize run/seed only from the approved case-insensitive exact names. Infer a
metric only when exactly one remaining column parses to finite floats in every
row; otherwise omit `metric_value`. Extraction validates explicit mappings and
never guesses.

- [ ] **Step 4: Run Task 3 GREEN checks**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_adapters_csv_v03.py tests/test_analysis_contracts_v03.py
python -m compileall -q claimci
git diff --check
```

- [ ] **Step 5: Commit Task 3**

Stage the Task 3 files only. Commit:

```text
feat: add deterministic CSV adapter
```

---

### Task 4: Safe YAML and TOML configuration adapters

**Files:**
- Create: `claimci/analysis/adapters/config.py`
- Create: `tests/test_adapters_config_v03.py`
- Create: `tests/fixtures/adapters/config.yaml`
- Create: `tests/fixtures/adapters/config.toml`

**Interfaces:**
- Produces `YamlConfigAdapter` ID `claimci-yaml-config-v1` and `TomlConfigAdapter` ID `claimci-toml-config-v1`.
- Both accept `CONFIG` only, create dotted-path mappings with stable hashed target IDs, and emit one `UNSPECIFIED` config observation.

- [ ] **Step 1: Add config fixtures and failing tests**

The fixtures contain nested model, training, evaluation, string, integer,
float, boolean, and null values representable by dotted paths.

Tests prove sorted structural dotted mapping generation, exact `ConfigValue`
keys/values, `UNSPECIFIED` roles, provenance, and extraction of an externally
selected subset. Tests prove keys containing literal dots are not split or
renamed and cannot be selected in v1.

Hostile cases cover duplicate YAML/TOML keys, YAML Python tags, aliases/cycles,
lists/arrays, non-string YAML keys, depth/node limits, NaN/Infinity, malformed
syntax/UTF-8, wrong suffix/kind, unsupported selector kinds, adapter/path/SHA
mismatch, and deterministic ordering.

- [ ] **Step 2: Run config tests to verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_config_v03.py
```

Expected: collection fails because config adapters are absent.

- [ ] **Step 3: Implement safe config adapters**

YAML uses only `load_unique_yaml`; TOML uses only `tomllib.loads`. Wrap parser
errors as `AdapterParseError`, then apply Task 1 depth/node/finite checks.
Reject sequence values rather than partially omitting them. Flatten only
structural string-key mappings and JSON-safe scalar leaves. Generate one
collision-checked mapping per leaf with the hashed target convention. Extraction
resolves only mappings supplied in the match and emits them in selector order.

- [ ] **Step 4: Run Task 4 GREEN checks**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_adapters_config_v03.py tests/test_parsing_hardening_day2.py tests/test_analysis_contracts_v03.py
python -m compileall -q claimci
git diff --check
```

- [ ] **Step 5: Commit Task 4**

Stage Task 4 files only. Commit:

```text
feat: add safe configuration adapters
```

---

### Task 5: Native ClaimCI compatibility and fixed registry

**Files:**
- Create: `claimci/analysis/adapters/native.py`
- Create: `claimci/analysis/adapters/registry.py`
- Create: `claimci/analysis/adapters/__init__.py`
- Create: `tests/test_adapters_native_v03.py`
- Create: `tests/test_adapters_registry_v03.py`

**Interfaces:**
- Produces `NativeManifestAdapter` (`claimci-native-manifest-v1`), `NativeResultsAdapter` (`claimci-native-results-v1`), and `NativeConfigAdapter` (`claimci-native-config-v1`).
- Produces public `ADAPTERS: tuple[Adapter, ...]` and `get_adapter(adapter_id: str) -> Adapter`.
- Exports all eight adapter classes, five error classes, six limit constants, registry tuple, and lookup from `claimci.analysis.adapters`.

- [ ] **Step 1: Write failing native compatibility tests**

Use the repository's actual `research.yaml`,
`examples/shared/base/results.json`, and
`examples/shared/base/config.yaml` as passive bytes.

Manifest tests assert one claim observation plus explicit baseline and candidate
observations; exact resolved config/results/train/eval paths; `MANIFEST_HINT`
provenance; and no `RepoMapping`, `USER_APPROVED`, plan, verdict, or authority
surface. Add nested-manifest tests for safe parent-relative `../` resolution and
repository-escape/nonportable-path rejection.

Results tests assert three repeated run observations, seeds 1/2/3, accuracy
values 0.58/0.60/0.62, per-run JSON Pointers, optional summary metadata, and
`UNSPECIFIED` role. Multiple metric fields require an explicit pointer mapping.

Config tests assert sorted scalar values plus raw finite compute evidence for
`training_steps`, `epochs`, and `batch_size`, with no comparison or verdict and
`UNSPECIFIED` role.

- [ ] **Step 2: Write failing registry/security tests**

Assert exact adapter order and IDs, exact lookup, unknown/malformed ID rejection,
immutability of `ADAPTERS`, and no dynamic registry extension. Parse every
production adapter module with `ast` and reject imports/calls for `subprocess`,
`socket`, HTTP clients, pickle, importlib, Audit, Review orchestration/provider,
workflow modules, `eval`, `exec`, `compile`, `__import__`, `system`, `popen`, and
unsafe `yaml.load` use. Assert no adapter returns or imports verdict/authority
types and no source package outside `claimci/analysis/adapters` changed.

- [ ] **Step 3: Run native/registry tests to verify RED**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_native_v03.py tests/test_adapters_registry_v03.py
```

Expected: collection fails because native adapters and registry are absent.

- [ ] **Step 4: Implement native manifest/results/config adapters**

Manifest parsing reuses safe YAML and Task 1 graph validation. Resolve declared
paths lexically relative to the manifest parent with an explicit segment stack;
allow `..` only while the stack remains inside repository root, then construct
`RepositoryPath`. Emit manifest-role provenance only from explicit `baseline`
and `candidate` keys.

Native results resolves mappings relative to each object in `runs`, like JSONL
record-local pointers. It never treats the full file as one mapped array and
requires explicit mapping if multiple numeric metric keys exist. Native config
reuses config mapping/extraction and additionally emits only the raw declared
compute inputs; it does not calculate a ratio or compare experiments.

- [ ] **Step 5: Implement fixed registry and exports**

Instantiate exactly one of each approved adapter in fixed native-first order.
`get_adapter` requires a canonical string ID and performs only tuple lookup.
Unknown IDs raise `KeyError`; non-string/control-containing IDs raise
`AdapterSelectorError`. No module path or callable is accepted.

- [ ] **Step 6: Run Task 5 GREEN checks**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_adapters_structured_v03.py tests/test_adapters_csv_v03.py tests/test_adapters_config_v03.py tests/test_adapters_native_v03.py tests/test_adapters_registry_v03.py tests/test_analysis_contracts_v03.py tests/test_manifest.py tests/test_result_check.py tests/test_config_check.py
python -m compileall -q claimci
git diff --check
```

- [ ] **Step 7: Commit Task 5**

Stage only Task 5 files. Commit:

```text
feat: add native adapters and trusted registry
```

---

### Task 6: Full hostile-fixture and compatibility gate

**Files:**
- Create: `tests/test_adapters_security_v03.py`
- Modify only if a proven adapter bug requires it: files under `claimci/analysis/adapters/`

**Interfaces:**
- Consumes every public adapter interface.
- Produces no new production interface; this is the final adversarial and integration gate.

- [ ] **Step 1: Add cross-format hostile tests before fixes**

Add table-driven real-adapter tests for oversized bytes; record/depth/node bounds;
duplicate keys/headers; malformed UTF-8/syntax; JSON/YAML/CSV NaN and infinity;
path traversal; adapter/path/SHA substitution; ambiguous mapping refusal; stable
ordering; full source/selector/hash provenance; all generic roles remaining
`UNSPECIFIED`; manifest roles remaining `MANIFEST_HINT`; and one failing
artifact being catchable as `AdapterError` while the next candidate still
probes successfully.

Before changing production for any discovered bug, run the specific test and
record the expected RED failure.

- [ ] **Step 2: Apply only regression-driven fixes**

Fix each proven failure minimally inside the adapter package, rerunning its
specific RED/GREEN cycle. Do not change common contracts, workflows, CLI, Audit,
Review, or provider code.

- [ ] **Step 3: Run the complete adapter gate**

Run:

```text
python -m pytest -q -p no:cacheprovider tests/test_adapters_core_v03.py tests/test_adapters_structured_v03.py tests/test_adapters_csv_v03.py tests/test_adapters_config_v03.py tests/test_adapters_native_v03.py tests/test_adapters_registry_v03.py tests/test_adapters_security_v03.py
```

- [ ] **Step 4: Run the complete ClaimCI gate**

Run:

```text
python -m pytest -q -p no:cacheprovider
python -m compileall -q claimci
git diff --check
```

Inspect `git diff origin/main...HEAD` and confirm only the design/plan, adapter
package, adapter tests, and approved fixtures changed. Confirm no existing test
was weakened.

- [ ] **Step 5: Commit final hostile tests/fixes**

Stage only the final test and any regression-driven adapter files. Commit:

```text
test: harden v0.3 adapter boundaries
```

---

### Task 7: Publish the reviewed branch

**Files:** No source changes expected.

- [ ] **Step 1: Re-run fresh pre-publish verification**

Run the full pytest, compileall, and diff-check commands from Task 6 again on
the exact committed tree. Confirm the worktree is clean and the branch is based
on current `origin/main`.

- [ ] **Step 2: Review commit and public-interface inventory**

Record adapter IDs, supported selector kinds, exact unsupported formats/syntax,
limits, test results, files changed, and follow-on discovery/planner obligations.

- [ ] **Step 3: Push and open one PR**

Push exactly `feat/adapters-v03`, verify no matching PR already exists, and open
one non-draft PR against `main`. Do not merge it. Verify remote head SHA, PR
state, base/head, CI status, and final clean local status.
