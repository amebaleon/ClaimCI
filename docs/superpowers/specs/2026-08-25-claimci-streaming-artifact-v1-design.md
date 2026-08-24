# ClaimCI Streaming Artifact v1 Design

## Goal and frozen scope

Streaming Artifact v1 removes the whole-artifact memory bottleneck for large
structured scientific evidence without weakening ClaimCI's passive security,
exact-head identity, Metric Identity, planning, or Audit authority.

The work is based exactly on Metric Identity v1 commit
`381d2237e14e3f0ab997e11988fa747c777d955d`, whose certified parent is
`5068fb528b1af88393f110946ff13434c3800a15`. It changes ClaimCI Core only.
It does not modify ClaimCI-Web, either parent PR, the release conductor, main,
deployment, Cloudflare, staging, production, Replay, Stale Evidence, Verdict
Regression, Code/Markdown indexing, distributed processing, or vector storage.

## Existing whole-file boundary

The current limit is imposed before and after adapter selection:

1. Discovery's `inspect_artifact()` calls
   `capture_confined_regular_file()`, which collects 64 KiB chunks in a list
   and joins them before computing artifact metadata.
2. Exact-head materialization's `_capture_artifact()` independently collects
   and joins the complete file into a `PassiveArtifact`.
3. `_verify_integrity()` rejects more than 8 MiB and accepts only immutable
   bytes. JSONL then decodes the complete artifact, calls `splitlines()`, and
   materializes every decoded record. CSV/TSV decodes the complete artifact
   into `StringIO` and retains every row.
4. Materialization copies complete dataset bytes into its native scratch tree.
   Native Audit calls `Path.read_text()`, splits the whole dataset, and retains
   every record hash in a tuple.

Changing only adapter internals would leave discovery and materialization as
whole-file bottlenecks. V1 therefore introduces the source seam at confined
snapshot access and carries it through exact-head materialization.

## Authority pipeline

The approved authority pipeline is:

```text
exact confined source
  -> full incremental integrity verification
  -> bounded semantic scan over the same descriptor
  -> immutable ScanCompleteness
  -> only eligible exact evidence
  -> existing planning and exact-head materialization
  -> existing AuditResult verdict authority
```

Integrity completeness and semantic completeness are independent. A source
may be fully integrity-verified while its semantic scan is
`INCOMPLETE_BUDGET`. That state identifies the exact immutable artifact and
the exact inspected prefix but cannot establish anything about the uninspected
portion.

`AuditResult` remains the only verdict authority. Scan state filters evidence
eligibility and is itself evidence; it never computes or overrides a verdict.

## Trusted ArtifactSource

Core adds a factory-only `ArtifactSource` contract. A source is bound to:

- exact `RepositoryIdentity` and `GitCommitSha`;
- one trusted checkout root identity;
- one canonical repository-relative `RepositoryPath`;
- `ArtifactKind`;
- expected byte size;
- expected SHA-256;
- the complete issued `ArtifactCandidate`.

The source constructor is unavailable. A trusted Core factory accepts the
repository/head/runtime root and an issued candidate, repeats lexical path,
root, parent, symlink/reparse, regular-file, and descriptor identity checks,
and returns the source. Provider output cannot name a root, construct a source,
choose an opener, or supply a callback. No customer code, network access,
subprocess, formula, SQL, import, or repository execution is introduced.

Each scan opens the confined path through Core, compares pathname and handle
identity, and reads from that exact descriptor. Hashing and byte counting occur
incrementally. After the scan, Core rechecks descriptor, pathname, parent, and
root identities. Reopen support repeats the complete confinement procedure;
it never resolves an arbitrary path stored in evidence.

`PassiveArtifact` remains the existing small-file contract. Files at or below
the current materialization byte limit continue through immutable bytes.
Larger supported structured files use `ArtifactSource`. Both paths use the
same fixed adapter IDs, selectors, normalized meaning, evidence IDs, and Audit
rules. The streaming seam is additive rather than a global replacement.

## Integrity and semantic scanning

Core owns a bounded reader that updates:

- raw SHA-256;
- a domain-separated Evidence Trace byte commitment;
- exact bytes read;
- descriptor state.

The semantic parser consumes from this reader. When semantic work reaches a
budget or supported-unit limit, parsing stops and the reader discards remaining
bytes while continuing size/hash verification, subject to the independent
integrity-I/O budget. Finishing the integrity drain never upgrades semantic
state to `COMPLETE`.

No deterministic evidence from a source is eligible unless EOF was reached,
the exact expected byte count was observed, the expected SHA-256 matched, and
the descriptor/path identities remained stable. Truncation, substitution,
size mismatch, SHA mismatch, source mutation, or integrity-budget exhaustion
fails closed before planning or Audit.

Malformed UTF-8, malformed JSON/CSV, duplicate JSON keys, and other parse
failures produce `FAILED`. They do not preserve provisional evidence. A
logical-record or semantic budget limit can produce an integrity-verified
incomplete scan whose already-established monotonic dataset invalidator may
remain eligible under the rule below.

## ScanCompleteness

Every streaming scan returns an immutable `ScanCompleteness` with:

- purpose: `INTEGRITY_ONLY`, `SCHEMA`, `SELECTED_OBSERVATIONS`, or
  `DATASET_RECORDS`;
- state: `COMPLETE`, `INCOMPLETE_BUDGET`, `INCOMPLETE_LIMIT`, or `FAILED`;
- bounded reason code from a Core enum, never provider prose;
- records semantically scanned;
- semantic bytes inspected;
- integrity bytes read;
- expected bytes;
- `integrity_verified`;
- bounded peak logical-buffer bytes;
- bounded scratch bytes used where applicable.

`COMPLETE` requires verified integrity and full semantic processing through
EOF for the stated purpose. An incomplete/failed report cannot be silently
coerced to complete. Streaming extraction outcomes and streaming
`NormalizedEvidence` carry the report. Plan identity commits any streaming
completeness material it accepts. Exact-head revalidation creates fresh scan
reports, compares the relevant evidence, and passes dataset scan reports in a
factory-only Audit context. Evidence Trace records the scan purpose/state and
integrity result for streamed sources.

## Work and resource budgets

Defaults support scientifically ordinary artifacts while retaining hard
bounds:

- small bytes path: existing 16 MiB materialization limit;
- maximum streamed file/integrity bytes: 1 GiB per artifact;
- maximum semantic bytes: 512 MiB per scan;
- maximum semantic records: 250,000;
- maximum selected result observations retained: 100,000;
- maximum logical record: existing 1 MiB;
- maximum columns: existing 256;
- maximum JSON depth: existing 64 per record;
- maximum JSON nodes: existing 100,000 per record;
- maximum MetricCandidates: Metric Identity's existing 32 per artifact;
- maximum scratch allocation: 512 MiB per materialization;
- maximum reported example hashes: 128, with exact total counts retained.

Discovery and materialization also enforce aggregate integrity-I/O budgets so
many individually acceptable files cannot produce unbounded work. Limits are
typed and configurable downward for tests/callers, with Core-owned hard
ceilings. Wall-clock deadlines are not used because they would make identical
inputs nondeterministic; bytes, records, nodes, columns, observations, and
scratch pages bound the work instead.

## Streaming JSONL

The JSONL scanner reads bounded binary logical lines incrementally. It uses
strict UTF-8, preserves deterministic LF/CRLF handling, rejects blank result
records, protects duplicate keys, rejects non-finite numbers, and applies
depth/node bounds to each decoded record. No tuple of decoded records or whole
artifact text is created.

Schema discovery maintains only bounded common-leaf/type state across records.
For results such as:

```json
{"run":1,"acc":0.81,"loss":0.40}
{"run":2,"acc":0.82,"loss":0.30}
```

the complete schema scan yields `acc` and `loss`, not per-record candidates.
Candidate discovery is eligible only after a complete schema scan. Any
incomplete scan, candidate explosion, or unresolved schema inconsistency
produces no complete MetricCandidate set.

Selected result extraction retains only normalized selected metric/run/seed
observations, never raw records or irrelevant fields, and is hard-bounded by
the observation budget. Aggregate/result conclusions require a complete
selected-observation scan.

## Streaming CSV and TSV

CSV/TSV use Python's strict fixed dialects over a bounded incremental physical
line iterator. The iterator supports quoted multiline records while bounding
the cumulative bytes delivered for one logical record. UTF-8, NUL rejection,
header syntax, duplicate headers, field counts, 256 columns, finite numeric
conversion, and the existing selector validation remain unchanged.

Candidate discovery retains the header and one bounded numeric-eligibility bit
per column while scanning rows. It never retains all rows. Numeric candidates
are issued only after a complete schema scan.

An ordinary column selector may retain only bounded normalized observations.
A `TableSelector` retains at most its bounded selected rows but scans the
complete required table to prove exact cardinality. If the scan is incomplete,
no exact cardinality or selected table evidence is issued; returning the first
match is forbidden.

## Deterministic dataset spill

Large JSONL datasets are scanned record by record. Each valid record is
canonicalized with the existing deterministic JSON representation and hashed.
Hashes and multiplicities are inserted into a temporary SQLite database with
a fixed ClaimCI schema. The database is created with a randomized name under
the invocation's already-confined scratch root, uses parameterized statements
only, and accepts no customer SQL or identifiers.

The database file is created with restrictive permissions where supported.
Page/file growth is checked against the hard scratch budget after bounded
batches. Scratch exhaustion stops semantic insertion, drains the source for
integrity when permitted, and produces `INCOMPLETE_BUDGET`.

SQL computes exact complete multisets, intersection counts, bounded sorted
hash previews, and evaluation multiset commitments without loading all hashes
into RAM. The store exists only while native Audit consumes the factory-only
context. A `finally` path closes connections and removes database, journal,
and temporary sidecars after success, parse failure, integrity failure,
budget exhaustion, or Audit failure. It is never customer storage.

## Partial-scan rule authority

Partial scans may authorize only facts that are monotonic and existential:
once true for inspected exact records, additional records cannot make them
false. V1 recognizes one such existing rule: a confirmed exact train/eval
record match may support `DATASET.EXACT_LEAKAGE` when both involved sources
have fully verified integrity, even if one or both semantic dataset scans are
incomplete.

The finding identifies the exact scan states and inspected counts. It relies
only on matching canonical hashes actually inserted from valid inspected
records. It makes no statement about the unscanned portion.

An incomplete scan cannot support:

- `DATASET.NO_LEAKAGE`;
- `DATASET.EVALUATION_ALIGNED` or complete multiset comparison;
- complete MetricCandidate discovery;
- exact table cardinality;
- schema consistency;
- absence of duplicate seeds/runs;
- result aggregates, variances, or comparisons over the selected population;
- any equivalent absence or universal claim.

When dataset scans are integrity-verified but incomplete and no overlap has
been found, Audit emits a bounded insufficient-evidence scan finding instead
of `DATASET.NO_LEAKAGE`. When exact leakage is found, Audit may emit both the
existing invalidator and the bounded incomplete-scan limitation; the existing
`determine_verdict()` resolves the ordinary findings. Evaluation alignment is
attempted only from complete evaluation scans in V1.

## Materialization and Audit integration

Materialization chooses bytes or source by candidate size, without changing
the selected repository mapping. Every streamed source is rebuilt from the
runtime repository/head/root and the exact issued candidate. Fixed adapters,
selectors, roles, splits, and evidence are freshly revalidated.

Small results/config/datasets retain the existing native tree path. Streamed
result artifacts are reduced only to their complete selected normalized
observations before the existing result checker runs. Streamed datasets are
not copied into a second full byte buffer: materialization builds the bounded
scratch index and passes a trusted dataset scan context to `audit_research()`.
The manifest remains the claim/config path authority; Audit substitutes only
the exact dataset facts from that context. Direct Audit without a context is
unchanged.

Measurement Drift receives complete evaluation multiset identities computed
incrementally from the SQLite counts. Incomplete evaluation scans provide no
complete evaluation identity. Evidence Trace commits the already-verified
stream digest/size and scan report rather than requesting complete source
bytes.

## Metric Identity v1 compatibility

Streaming uses the same JSONL/CSV/TSV adapter IDs and selector identities as
the bytes path. Given identical path, hash, selector, raw metric name, adapter,
and role, both paths create the same logical `MetricCandidate` ID.

MetricCandidate extraction requires a complete schema scan. The existing
bounded conservative lexical identity and atomic baseline/candidate approval
are unchanged. At materialization both bound sources are independently rebuilt
and fully revalidated for repository/head, path, kind, size, SHA, adapter,
selector, raw name, canonical identity, and role. Either incomplete or stale
endpoint invalidates the entire pair and requires re-approval. Streaming adds
no global alias or partial pair authority.

## Compatibility and Hosted boundary

All public bytes-backed adapter functions and direct Audit behavior remain
available. Small artifacts continue to produce their historical plan IDs,
evidence, reports, and verdicts when streaming completeness is absent. Large
streaming evidence adds completeness material; it cannot collide with a legacy
plan that lacked that authority fact.

Core currently has no production orchestration function that turns Discovery
metadata into adapter input; callers construct `PassiveArtifact` bytes. Core
will expose the complete trusted source factory and streaming registry APIs for
a confined local checkout. If Hosted currently reads repository blobs fully
before invoking Core, the final report will state
`HOSTED_COMPANION_REQUIRED` and specify the minimal change: pass the exact-head
confined checkout root plus issued repository/head/path/kind/size/SHA metadata
to the Core factory rather than transporting full artifact bytes. No Core
security rule will be weakened to hide that integration requirement.

## Verification

Tests generate large artifacts under temporary directories; no large fixture
is committed. Focused stages prove source confinement/integrity, JSONL and
CSV/TSV streaming beyond 8 MiB and 10,000 records, per-unit limits, schema-level
candidates, exact TableSelector cardinality, dataset complete/partial rules,
SQLite cleanup, Metric Binding pair revalidation, and unchanged small bytes
behavior.

A structural memory regression uses a guarded source/scan instrument to prove
there is no unbounded read and no retained raw record/row collection. It checks
the reported peak logical buffer rather than fragile process RSS or timing.

The final gate runs the new streaming tests and the specified focused adapter,
dataset, Metric Identity, mapping/materialization, benchmark/TableSelector,
and Audit slices, followed by one complete Core suite, `compileall`, configured
lint if present, and `git diff --check`. The branch then opens exactly one Draft
Core PR stacked on `codex/metric-identity-v1` and stops without merge or deploy.
