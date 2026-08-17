# ClaimCI v0.3 Passive Dataset Identity Design

## Goal and boundary

This change closes the remaining production gap between Auto Discovery and the
existing deterministic dataset checks. ClaimCI may select repository-owned
JSON Lines artifacts as baseline/candidate train/evaluation inputs without a
customer-authored `research.yaml`.

The implementation is identity-only. It does not create another dataset
analysis engine, normalize rows, download data, execute repository content, or
infer scientific meaning from dataset records. Verified source bytes are copied
unchanged into the ephemeral native Audit tree; `claimci.dataset_check` remains
the only code that parses records and determines exact overlap/alignment.

## Trust model

Train/evaluation is repository mapping metadata, not a selector into JSONL
content. The shared contract therefore gains:

```python
class DatasetSplit(str, Enum):
    TRAIN = "train"
    EVAL = "eval"

@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    ...
    dataset_split: DatasetSplit | None = None
```

Only dataset bindings may carry a split. A dataset binding without a split is a
non-executable proposal. Executable mappings require exactly one train and one
evaluation binding for each baseline/candidate role. Split participates in
mapping identity, approval scoping, plan identity, runtime comparison, and
serialization. Changing it after planning fails closed.

Deterministic path-token inference may propose a split only when exactly one of
the locked train tokens (`train`, `training`) or evaluation tokens (`eval`,
`evaluation`, `test`) occurs in the complete portable repository path. Provider
proposals remain untrusted even at confidence 1.0. A `RepoMapping` created by
the existing explicit approval transition outranks inference; stale or invalid
approval still blocks fallback.

`research.yaml` continues to emit a `MANIFEST_HINT`, now with its declared
train/evaluation fields represented as split-bearing bindings. It remains an
optional advanced hint, not a prerequisite or authority elevation.

## Passive dataset adapter

Add one fixed built-in adapter:

```text
claimci-jsonl-dataset-v1
```

It accepts only a confined, SHA-256-bound `ArtifactKind.DATASET` whose suffix is
`.jsonl`. Probe and extraction recheck immutable bytes, size, path, kind, and
digest using the existing adapter core. It returns one `DatasetReference` to
the exact repository path with `split=None` and an `UNSPECIFIED` experiment
role. Its `AdapterMatch.mappings` is empty because there is no field selector.

The adapter does not parse, canonicalize, truncate, or reinterpret JSONL rows.
Malformed JSONL is passed unchanged through the trusted mapping/materialization
path and is rejected by the existing deterministic Audit. The adapter is
registered in the fixed registry; there is no plugin, dynamic import, customer
code execution, filesystem traversal, or network surface.

A small fixed-registry helper may run `probe()` then `extract()` for one already
constructed `PassiveArtifact`. It selects only the first fixed registry match,
does not fall through after a recognized controlled failure, and exposes no
extension API.

## Discovery and clarification

Discovery continues to classify bounded repository-relative dataset paths.
Inferred and manifest bindings retain their train/evaluation identity instead
of discarding it. Clear zero-config paths such as
`accuracy_candidate_eval.jsonl` produce a confidence-0.95 inferred mapping;
the adapter supplies only passive identity while the mapping supplies role and
split.

Two near-tied files for the same role/split produce the existing smallest useful
bounded question, for example:

> Which file contains the candidate train dataset?

Each choice carries the same requested role and split. ClaimCI asks at most one
question in a discovery result and caps choices at eight. If repository names
do not provide enough bounded role/split information to form useful literal
choices, ClaimCI returns partial/missing evidence rather than inventing an
identity. Provider-generated questions or split proposals cannot auto-execute.

## Planning and materialization

Planning uses selected binding metadata—not adapter-invented row metadata—to
require exactly one train and one evaluation dataset per role. A resolvable
upstream mapping question is returned before a generic representation failure;
an entirely absent dataset remains typed `PARTIAL` missing evidence.

At execution, the materializer revalidates:

- repository identity and head SHA;
- confined regular-file path with no symlink component;
- artifact kind, exact byte length, and SHA-256;
- selected adapter ID and empty selector set;
- selected role and `DatasetSplit` against the plan;
- a fresh identity-only adapter extraction over the captured bytes.

It then copies those captured bytes exactly once to fixed ephemeral
`baseline|candidate/train|eval.jsonl` paths. It performs no dataset download,
row conversion, or customer code execution. The generated `research.yaml` is
ephemeral implementation detail and is never represented as customer-authored
or committed.

## Compatibility and non-goals

No workflow, CLI, Audit rule, `dataset_check.py` behavior, Research Review
orchestration, provider limit, or hosted-service API changes are included.
Existing manifests and all non-dataset adapters remain compatible. Non-JSONL
datasets remain `PARTIAL` because the current normalized contracts cannot copy
them losslessly into the native Audit representation.

## Acceptance

- The real fixed dataset adapter replaces the cross-package test stub.
- A no-manifest repository with clear result/config/dataset names reaches the
  existing Audit and advisory one-call Review path.
- Dataset bytes in the ephemeral Audit tree exactly equal repository bytes.
- Missing datasets are `PARTIAL`; same-slot ambiguity is `MAPPING_NEEDED` with
  one bounded literal question; no verdict is guessed.
- Split/path/kind/adapter/hash drift fails closed before Audit.
- Provider-originated mappings do not auto-execute; explicit approval remains
  a separate transition.
- Existing CLI, deterministic findings/verdict precedence, and Review authority
  remain byte/behavior compatible outside the new hosted dataset path.

