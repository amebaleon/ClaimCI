# ClaimCI v0.3 Adapter Layer Design

## Status and scope

This design defines the first zero-configuration artifact Adapter Layer for
ClaimCI v0.3. It builds on the immutable contracts in `claimci.analysis` and
adds bounded deterministic parsing for JSON, CSV, JSONL, YAML, TOML, and the
existing ClaimCI manifest/results/config shapes.

This branch does not implement repository-wide discovery, baseline/candidate
inference, planning, deterministic audit changes, Research Review changes,
workflow changes, provider changes, dataset downloading, or hosted services.
Adapters inspect passive bytes and report structured evidence only.

## Architecture

The implementation lives in a fixed package:

```text
claimci/analysis/adapters/
    __init__.py
    core.py
    structured.py
    tabular.py
    config.py
    native.py
    registry.py
```

- `core.py` owns controlled errors, approved v1 limits, strict UTF-8 decoding,
  finite-value and structural graph validation, artifact integrity checks,
  selector resolution, stable provenance construction, and shared extraction
  helpers.
- `structured.py` owns generic JSON and JSONL adapters.
- `tabular.py` owns the CSV adapter.
- `config.py` owns generic YAML and TOML configuration adapters.
- `native.py` owns compatibility adapters for `research.yaml`, ClaimCI results,
  and ClaimCI configuration shapes.
- `registry.py` owns the fixed tuple of trusted adapter instances and lookup by
  identifier. Registry identifiers are data, never import paths.
- `__init__.py` is the documented adapter import surface.

All adapters implement the existing protocol exactly:

```python
probe(artifact: PassiveArtifact) -> AdapterMatch | None
extract(
    artifact: PassiveArtifact,
    match: AdapterMatch,
) -> NormalizedEvidence
```

The artifact already carries a confined `RepositoryPath`, declared kind,
length, and SHA-256. Adapters never receive a repository checkout root, native
filesystem path, loader callback, module name, or execution capability.

## Adapter identifiers and registry order

The fixed v1 identifiers are:

1. `claimci-native-manifest-v1`
2. `claimci-native-results-v1`
3. `claimci-native-config-v1`
4. `claimci-json-v1`
5. `claimci-jsonl-v1`
6. `claimci-csv-v1`
7. `claimci-yaml-config-v1`
8. `claimci-toml-config-v1`

Native adapters precede generic adapters so an explicit legacy ClaimCI shape
can receive its more precise compatibility match. Registry lookup is an exact
identifier comparison over this fixed collection. There is no plugin loading,
entry-point discovery, dynamic import, reflection-based construction, or
customer-controlled registry extension.

## Bounded parsing contract

The approved v1 limits are constants in `core.py`:

- maximum artifact bytes: 8 MiB (`8 * 1024 * 1024`);
- maximum CSV or JSONL logical records: 10,000;
- maximum CSV columns: 256;
- maximum container nesting depth: 64;
- maximum total structured nodes: 100,000;
- maximum one CSV or JSONL logical record: 1 MiB (`1024 * 1024`).

A value equal to a maximum is accepted. A value above it raises a controlled
`AdapterLimitError`. No parser truncates input, rows, records, columns, nodes,
or nested values. Hitting any bound is a whole-artifact unsupported condition,
never a partial successful parse.

`AdapterError` extends `AnalysisContractError`. `AdapterParseError`,
`AdapterLimitError`, `AdapterSelectorError`, and `AdapterIntegrityError` give
callers stable categories. A recognized malformed or oversized artifact may
raise one of these errors from `probe()`. Repository-wide discovery is expected
to catch `AdapterError` per candidate, record the artifact-local failure, and
continue examining other candidates. Errors from one artifact do not define a
repository-wide fatal state.

## Integrity, passive data, and extraction

`PassiveArtifact` validates size and SHA-256 when constructed. Every adapter's
`extract()` additionally recomputes SHA-256 over `artifact.content` immediately
before parsing and compares it with `artifact.candidate.sha256`. It also checks
the content length, exact adapter identifier, and equality of match path and
artifact path. This prevents discovery-time bytes or mappings from being
silently substituted before extraction.

Selectors and mappings are data. The implementation never evaluates them and
never uses them as Python, SQL, shell, regular-expression, template, import, or
filesystem syntax. Adapters do not import or execute customer code. Pickle and
other object deserialization formats are unsupported.

## Determinism and common data rules

All text formats require strict UTF-8. Locale-sensitive numeric parsing is not
used. JSON and JSONL use the standard JSON number grammar. CSV selected metric
values use Python's locale-independent decimal conversion after trimming ASCII
surrounding whitespace. YAML/TOML numeric values come from their safe standard
parsers. Booleans are never treated as numbers.

NaN, positive infinity, and negative infinity are rejected in every format.
JSON/JSONL reject `NaN`, `Infinity`, and `-Infinity` tokens during decoding.
YAML/TOML graphs are recursively checked for finite numeric values. CSV rejects
nonfinite values when a column is selected as numeric evidence.

JSON and YAML duplicate keys are rejected at every nesting level. TOML's
standard parser rejects duplicate keys. CSV duplicate or empty headers and
rows with a field count different from the header are rejected.

ClaimCI-generated ordering is stable:

- object/config leaves are emitted in lexicographically sorted selector order;
- CSV and JSONL observations preserve source record order;
- mapping candidates are ordered by stable target-field then selector order;
- registry order is the fixed order above.

Every extracted field has `ADAPTER_EXTRACTION` or `MANIFEST_HINT` provenance
that records the source repository path, exact selector/key/column, adapter ID,
artifact SHA-256, and whether the mapping was explicit or inferred. Run and
seed values are copied into `NormalizedObservation` when present.

## Selector model

### JSON and JSONL

JSON and JSONL support only validated RFC 6901 JSON Pointer selectors. `/`
separates path tokens, `~0` represents `~`, and `~1` represents `/`. Array
indices are canonical base-10 non-negative integers with no leading zero unless
the index is exactly `0`. The `-` append token is unsupported because selectors
are read-only. Missing tokens, wrong container types, invalid escapes,
out-of-range indices, and non-scalar selected metric/run/seed values raise
`AdapterSelectorError`.

For JSONL, the pointer is resolved independently within each logical JSONL
record. JSONL is never wrapped or interpreted as one aggregate JSON document.
Every non-empty line is one record, must be at most 1 MiB, and must decode to a
JSON object. Blank lines and non-object records are unsupported.

### CSV

CSV supports `SelectorKind.COLUMN` only. The selector expression must equal one
validated header exactly after the CSV parser decodes it; headers are not
case-folded, normalized, or renamed. Generated automatic mappings recognize a
run column only from one unique exact case-insensitive name in `run`, `run_id`,
or `trial`; and a seed column only from one unique exact case-insensitive name
`seed` or `random_seed`. A metric column is inferred only when exactly one
remaining column contains finite numeric values in every row.

If zero or multiple metric columns qualify, `probe()` returns a lower-confidence
match without a `metric_value` mapping. `extract()` does not choose a column; a
validated external `FieldMapping` is required.

### YAML and TOML

YAML/TOML configuration extraction supports `SelectorKind.DOTTED_PATH` only.
Each segment must match `[A-Za-z0-9_-]+`, segments are separated by one literal
dot, and empty, leading-dot, trailing-dot, escaped, bracket, wildcard, quoted,
or expression-like segments are unsupported. Resolution walks mapping keys
structurally and never indexes sequences.

A source key containing `.` cannot be silently split or escaped in v1. Such a
key is not representable through the dotted selector model and must be handled
by an explicit future mapping capability rather than guessed. Generic config
adapters omit unrepresentable leaves from automatic mappings and lower match
confidence. An externally supplied v1 dotted selector still cannot name them.

YAML uses the existing duplicate-rejecting `UniqueKeySafeLoader`; arbitrary
constructors, Python tags, recursive aliases, and unsafe loaders are forbidden.
TOML uses standard-library `tomllib` only.

## Generic adapter behavior

Generic JSON, JSONL, CSV, YAML, and TOML adapters always create observations
with `ExperimentRole.UNSPECIFIED`. They never infer baseline or candidate from
filenames, directories, keys, selectors, values, or neighboring artifacts.
Supplying a validated mapping may identify metric/run/seed/config selectors,
but does not give a generic adapter permission to assign an experiment role.

Generic JSON probes nested scalar leaves. A metric mapping is proposed only
when exactly one finite numeric leaf remains after uniquely recognized run/seed
leaves. Ambiguous numeric leaves produce a lower-confidence match with no
metric mapping. Generic JSONL applies the same rule across all records: one
candidate selector must exist and be finite in every record.

Generic YAML and TOML are configuration-first. They emit sorted scalar leaves
as `ConfigValue` entries and do not infer metrics. Sequences, objects as scalar
values, unrepresentable dotted keys, and unsupported values do not become
config observations.

## Native ClaimCI compatibility

### Native manifest

`claimci-native-manifest-v1` recognizes `research.yaml` or `research.yml` with
the existing ClaimCI claim/baseline/candidate structure. It uses safe,
duplicate-rejecting YAML parsing and validates every declared artifact value as
a portable relative path. The adapter resolves it lexically against the
manifest's repository-relative parent, rejects any traversal that escapes the
repository, and emits the resulting canonical `RepositoryPath` without reading
the referenced file. This preserves legacy manifest-relative paths while
keeping extraction passive and confined.

It emits one manifest metadata observation plus explicit baseline and candidate
observations. `ExperimentRole.BASELINE` and `ExperimentRole.CANDIDATE` are
preserved only because those roles are explicitly declared in the manifest.
Role and path provenance is `ProvenanceKind.MANIFEST_HINT`, includes the exact
dotted manifest key, and clearly states that no inference occurred.

This adapter does not create `RepoMapping`, call `RepoMapping.approve`, claim
`USER_APPROVED` trust, create `EphemeralAuditPlan`, invoke Audit, or construct
deterministic authority. Manifest roles are explicit evidence for the planner;
trust elevation remains a separate authenticated planner/mapping transition.

### Native results

`claimci-native-results-v1` recognizes the existing object with a `runs` array
of objects and optional `summary`. It recognizes a unique `seed` key and emits
one observation per run. A metric is mapped automatically only when the run
objects share exactly one finite numeric field other than seed. Multiple metric
keys are ambiguous and require an external mapping. The optional summary is
reported as configuration-like metadata; adapters do not compare or validate a
scientific claim.

### Native config

`claimci-native-config-v1` recognizes existing ClaimCI YAML configuration
objects. It emits sorted scalar `ConfigValue` leaves and finite numeric
`ComputeEvidence` only for explicit supported compute-proxy inputs already used
by ClaimCI, such as training steps and batch size. It reports the declared
values, not a fair-comparison conclusion. Its experiment role remains
`UNSPECIFIED`; only the native manifest may preserve explicit roles.

## Probe and extract behavior

`probe()` first checks artifact kind and a case-insensitive supported suffix.
An incompatible kind/suffix returns `None` without decoding. A recognized
artifact is fully parsed under all bounds. A confident, unambiguous shape
returns an `AdapterMatch` with validated mappings and high confidence. A valid
but ambiguous shape returns a lower-confidence match whose mapping tuple omits
the ambiguous target.

`extract()` accepts only a match whose adapter ID and path equal the receiving
adapter and artifact. It reparses after immediate integrity verification. Every
mapping target and selector kind is allowlisted per adapter. Missing required
metric mappings produce `AdapterSelectorError`; extraction never fills them by
guessing. Unused or duplicate target mappings are rejected. The output is one
`NormalizedEvidence` whose artifact and match are the validated input values.

## Explicit non-goals and unsupported cases

The v1 layer does not support:

- baseline/candidate inference by generic adapters;
- scientific claim evaluation, threshold checking, comparison, or verdicts;
- arbitrary JSONPath, jq, SQL, Python, regular-expression, or template syntax;
- JSONL cross-record/global selectors;
- CSV dialect guessing beyond deterministic standard comma-separated UTF-8;
- spreadsheet formats, Parquet, pickle, protobuf, HDF5, NumPy, or model files;
- YAML tags/constructors, YAML unsafe loading, or recursive/repeated aliases;
- dotted-path escaping for YAML/TOML keys containing dots;
- network access, artifact downloading, archive extraction, or dataset loading;
- parsing arbitrary training/evaluation datasets in this branch;
- plugins, dynamic imports, customer repository imports, or code execution;
- automatic approved mappings, audit plans, deterministic results, or Research
  Review calls.

## Test strategy

Adapter-specific tests use realistic passive fixtures for:

- nested JSON metrics and JSON Pointer escaping/indexing;
- CSV run/seed/metric rows and ambiguous numeric columns;
- repeated JSONL run observations and per-record pointers;
- nested YAML and TOML configuration values;
- the repository's existing `research.yaml`, results, and config files;
- malformed UTF-8 and syntax, duplicate keys/headers, inconsistent CSV rows;
- artifact, record, column, node, depth, and logical-record limits;
- traversal and nonportable manifest paths;
- NaN and infinity in every applicable format;
- mismatched path, adapter ID, size, and SHA during extraction;
- stable output ordering and complete source/selector/hash provenance;
- generic `UNSPECIFIED` roles and manifest-only explicit roles;
- ambiguous matches refusing extraction without external mapping;
- an import/AST guard against process, network, pickle, dynamic import, eval,
  unsafe YAML, customer-code loading, Audit, Review, workflow, and provider
  dependencies.

The full existing ClaimCI suite must remain green. No existing test is weakened,
and existing CLI, deterministic Audit, Research Review, workflows, and provider
behavior remain byte-for-byte outside this new package and its tests/docs.

## Follow-on integration contract

Repository discovery should iterate candidates and adapters independently,
catch `AdapterError` per artifact, and continue. The planner decides whether a
match is sufficient, constructs `MappingQuestion` for ambiguity, applies
authenticated `RepoMapping.approve` when appropriate, and assigns experiment
roles for generic evidence. Only later integration invokes the existing Audit
engine and converts its actual `AuditResult` into deterministic authority.
