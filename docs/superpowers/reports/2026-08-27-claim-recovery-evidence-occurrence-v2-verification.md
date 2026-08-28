# Claim recovery and evidence occurrence v2 verification

Date: 2026-08-28 (Asia/Seoul)

## Immutable verification boundary

- Repository: `amebaleon/ClaimCI`
- Base: `1c7ae750bf9b01485d8061cb047e1ef463eed24e`
- Verified implementation head: `1af824eec57a6a28400f4f2b32d460ecca8f12cf`
- Controlled fixture source tree: `90266b8ad6b63cda4f766e31039c75e0967f9b1c`
- Branch: `codex/zero-config-claim-evidence-v2`
- Review packet: `.superpowers/sdd/2026-08-27-claim-recovery-evidence-occurrence-v2/review-1c7ae75..1af824e.diff`

The verification-report commit necessarily follows the verified implementation
head and changes documentation only. The companion ClaimCI-Web PR must pin the
final report-bearing Core commit, not the pre-report implementation head.

## Outcome

The final Core gate passed. The unchanged controlled production-smoke fixture
produces exactly one deterministic claim and then terminates honestly as
`PARTIAL / required_threshold_not_recovered`; it does not terminate as
`scientific_routing_uncertain`, and no threshold is synthesized from arithmetic
or manifest metadata.

The full Core suite was invoked exactly once at the final gate:

```text
1396 passed, 16 skipped in 60.86s (0:01:00)
```

No merge, deployment, production run, OAuth action, or application-gate change
was performed.

## Environment

Command:

```powershell
python --version
python -c "import platform,sys; print('platform='+platform.platform()); print('machine='+platform.machine()); print('executable='+sys.executable)"
```

Transcript:

```text
Python 3.14.2
platform=Windows-11-10.0.26200-SP0
machine=AMD64
executable=C:\Users\windo\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

`pyproject.toml` declares Python `>=3.11`. Current repository automation uses
`ubuntu-latest` with Python 3.11 and does not declare an architecture matrix.
Core has no configured Linux/amd64 Python 3.13 gate.

## Exact final-gate transcripts

### Full test suite

Command:

```powershell
python -m pytest -q
```

Exit code: `0`

Transcript:

```text
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 15%]
........................................................................ [ 20%]
.............................................s.......................... [ 25%]
s....................................................................... [ 30%]
........................................................................ [ 35%]
........................................................................ [ 40%]
........................................................................ [ 45%]
............................s.....s..................................... [ 50%]
........................................................................ [ 56%]
........................................................................ [ 61%]
........................................................................ [ 66%]
........................................................................ [ 71%]
........................................................................ [ 76%]
..........................s............................s...s..ssss...... [ 81%]
.......s................................................................ [ 86%]
...............................................................s........ [ 91%]
..........................................s.ss.......................... [ 96%]
............................................                             [100%]
1396 passed, 16 skipped in 60.86s (0:01:00)
```

### Compilation

Command:

```powershell
python -m compileall claimci tests
```

Exit code: `0`

Transcript:

```text
Listing 'claimci'...
Listing 'claimci\\analysis'...
Listing 'claimci\\analysis\\adapters'...
Listing 'claimci\\analysis\\discovery'...
Listing 'claimci\\review'...
Listing 'tests'...
Compiling 'tests\\conftest.py'...
Listing 'tests\\fixtures'...
Listing 'tests\\fixtures\\adapters'...
Listing 'tests\\fixtures\\production_smoke'...
Compiling 'tests\\test_adapters_security_v03.py'...
Compiling 'tests\\test_analysis_claims_v03.py'...
Compiling 'tests\\test_analysis_review_bridge_v03.py'...
Compiling 'tests\\test_artifact_root_day2.py'...
Compiling 'tests\\test_audit.py'...
Compiling 'tests\\test_claim_types_v0.py'...
Compiling 'tests\\test_cli.py'...
Compiling 'tests\\test_config_check.py'...
Compiling 'tests\\test_console_hardening_day2.py'...
Compiling 'tests\\test_dataset_check.py'...
Compiling 'tests\\test_day2_demo.py'...
Compiling 'tests\\test_direction_day2.py'...
Compiling 'tests\\test_evidence_obligations_v1.py'...
Compiling 'tests\\test_evidence_trace_v1.py'...
Compiling 'tests\\test_external_install_v0.py'...
Compiling 'tests\\test_fixtures.py'...
Compiling 'tests\\test_github_day2.py'...
Compiling 'tests\\test_hardening_dataset_day2.py'...
Compiling 'tests\\test_hardening_github_day2.py'...
Compiling 'tests\\test_hardening_inputs_day2.py'...
Compiling 'tests\\test_hostile_zero_config_v03.py'...
Compiling 'tests\\test_manifest.py'...
Compiling 'tests\\test_markdown_day2.py'...
Compiling 'tests\\test_measurement_drift_v1.py'...
Compiling 'tests\\test_models.py'...
Compiling 'tests\\test_parsing_hardening_day2.py'...
Compiling 'tests\\test_report.py'...
Compiling 'tests\\test_result_check.py'...
Compiling 'tests\\test_review_cli_day3.py'...
Compiling 'tests\\test_review_evidence_day3.py'...
Compiling 'tests\\test_review_github_day3.py'...
Compiling 'tests\\test_review_hardening_day3.py'...
Compiling 'tests\\test_review_models_day3.py'...
Compiling 'tests\\test_review_orchestrator_day3.py'...
Compiling 'tests\\test_review_provider_day3.py'...
Compiling 'tests\\test_review_report_day3.py'...
Compiling 'tests\\test_review_sources_day3.py'...
Compiling 'tests\\test_review_tools_day3.py'...
Compiling 'tests\\test_review_workflow_day3.py'...
Compiling 'tests\\test_review_workflow_isolation_day3.py'...
Compiling 'tests\\test_streaming_security_v1.py'...
Compiling 'tests\\test_views_consistency_day2.py'...
```

### Installed dependency integrity

Command:

```powershell
python -m pip check
```

Exit code: `0`

Transcript:

```text
No broken requirements found.
```

### Configured lint and typecheck audit

Command:

```powershell
$candidates=@('setup.cfg','tox.ini','noxfile.py','mypy.ini','pyrightconfig.json','.ruff.toml','ruff.toml','.flake8','.pre-commit-config.yaml','Makefile','Taskfile.yml','Taskfile.yaml')
$present=$candidates | Where-Object { Test-Path -LiteralPath $_ }
if($present){'checker-configs:'; $present}else{'checker-configs: none'}
$matches=rg -n -i "ruff|mypy|pyright|pylint|flake8|black|isort|typecheck|(^|[^a-z])lint([^a-z]|$)" pyproject.toml .github/workflows scripts 2>$null
if($LASTEXITCODE -eq 0){'checker-references:'; $matches}else{'checker-references: none'}
```

Exit code: `0`

Transcript:

```text
checker-configs: none
checker-references: none
```

The current repository configures no lint or typecheck command. No unconfigured
checker was invented for this gate.

### Diff integrity

Command:

```powershell
git diff --check origin/main...HEAD
```

Exit code: `0`

Transcript: no stdout.

Command:

```powershell
git status --short
```

Exit code: `0`

Transcript: no stdout; the worktree was clean.

Command:

```powershell
git diff --name-status origin/main...HEAD
```

Exit code: `0`

Transcript:

```text
M claimci/analysis/__init__.py
M claimci/analysis/adapters/config.py
M claimci/analysis/adapters/core.py
M claimci/analysis/adapters/dataset.py
M claimci/analysis/adapters/native.py
M claimci/analysis/adapters/streaming.py
M claimci/analysis/adapters/structured.py
M claimci/analysis/adapters/tabular.py
M claimci/analysis/artifact_source.py
M claimci/analysis/claim_types.py
M claimci/analysis/contracts.py
M claimci/analysis/discovery/claims.py
M claimci/analysis/evidence_identity.py
M claimci/analysis/materialize.py
A docs/superpowers/plans/2026-08-27-claim-recovery-evidence-occurrence-v2.md
A docs/superpowers/specs/2026-08-27-claim-recovery-evidence-occurrence-v2-design.md
A tests/analysis_occurrence_support.py
A tests/fixtures/production_smoke/README.md
A tests/fixtures/production_smoke/baseline-config.yaml
A tests/fixtures/production_smoke/baseline-eval.jsonl
A tests/fixtures/production_smoke/baseline-results.json
A tests/fixtures/production_smoke/baseline-train.jsonl
A tests/fixtures/production_smoke/candidate-config.yaml
A tests/fixtures/production_smoke/candidate-eval.jsonl
A tests/fixtures/production_smoke/candidate-results-base.json
A tests/fixtures/production_smoke/candidate-results.json
A tests/fixtures/production_smoke/candidate-train.jsonl
A tests/fixtures/production_smoke/research.yaml
M tests/test_adapters_config_v03.py
M tests/test_adapters_core_v03.py
M tests/test_adapters_csv_v03.py
M tests/test_adapters_native_v03.py
M tests/test_adapters_registry_v03.py
M tests/test_adapters_security_v03.py
M tests/test_adapters_structured_v03.py
M tests/test_analysis_contracts_v03.py
M tests/test_analysis_cross_package_v03.py
M tests/test_auto_discovery_v03.py
M tests/test_benchmark_evidence_profile_v0.py
M tests/test_benchmark_table_adapter_v0.py
M tests/test_claim_types_v0.py
M tests/test_dataset_identity_v03.py
M tests/test_ephemeral_materialize_v03.py
M tests/test_ephemeral_planner_v03.py
A tests/test_evidence_identity_v2.py
M tests/test_metric_identity_v1.py
M tests/test_metric_streaming_security_integration.py
A tests/test_production_smoke_remediation.py
M tests/test_streaming_dataset_v1.py
M tests/test_streaming_jsonl_v1.py
M tests/test_streaming_materialize_v1.py
M tests/test_streaming_source_v1.py
M tests/test_streaming_tabular_v1.py
M tests/test_unified_analysis_v03.py
```

Command:

```powershell
git diff --stat origin/main...HEAD
```

Exit code: `0`

Transcript:

```text
 claimci/analysis/__init__.py                       |   8 +
 claimci/analysis/adapters/config.py                |   9 +-
 claimci/analysis/adapters/core.py                  |  83 ++-
 claimci/analysis/adapters/dataset.py               |  19 +-
 claimci/analysis/adapters/native.py                |  24 +-
 claimci/analysis/adapters/streaming.py             |  41 +-
 claimci/analysis/adapters/structured.py            |  16 +-
 claimci/analysis/adapters/tabular.py               |  16 +-
 claimci/analysis/artifact_source.py                |  53 +-
 claimci/analysis/claim_types.py                    |  58 +-
 claimci/analysis/contracts.py                      | 138 +++-
 claimci/analysis/discovery/claims.py               | 239 +++----
 claimci/analysis/evidence_identity.py              |  16 +
 claimci/analysis/materialize.py                    |  18 +-
 ...-08-27-claim-recovery-evidence-occurrence-v2.md | 726 +++++++++++++++++++++
 ...claim-recovery-evidence-occurrence-v2-design.md | 233 +++++++
 tests/analysis_occurrence_support.py               |  31 +
 tests/fixtures/production_smoke/README.md          |   9 +
 .../fixtures/production_smoke/baseline-config.yaml |  12 +
 .../fixtures/production_smoke/baseline-eval.jsonl  |   2 +
 .../production_smoke/baseline-results.json         |   8 +
 .../fixtures/production_smoke/baseline-train.jsonl |   2 +
 .../production_smoke/candidate-config.yaml         |  12 +
 .../fixtures/production_smoke/candidate-eval.jsonl |   2 +
 .../production_smoke/candidate-results-base.json   |   8 +
 .../production_smoke/candidate-results.json        |   8 +
 .../production_smoke/candidate-train.jsonl         |   2 +
 tests/fixtures/production_smoke/research.yaml      |  15 +
 tests/test_adapters_config_v03.py                  |  29 +-
 tests/test_adapters_core_v03.py                    |   4 +-
 tests/test_adapters_csv_v03.py                     |  31 +-
 tests/test_adapters_native_v03.py                  |  27 +-
 tests/test_adapters_registry_v03.py                |   1 +
 tests/test_adapters_security_v03.py                |   4 +-
 tests/test_adapters_structured_v03.py              |  27 +-
 tests/test_analysis_contracts_v03.py               | 105 ++-
 tests/test_analysis_cross_package_v03.py           | 181 ++++-
 tests/test_auto_discovery_v03.py                   | 233 ++++++-
 tests/test_benchmark_evidence_profile_v0.py        |  29 +-
 tests/test_benchmark_table_adapter_v0.py           |   5 +-
 tests/test_claim_types_v0.py                       | 339 ++++++++++
 tests/test_dataset_identity_v03.py                 |  27 +-
 tests/test_ephemeral_materialize_v03.py            |  73 ++-
 tests/test_ephemeral_planner_v03.py                |  84 ++-
 tests/test_evidence_identity_v2.py                 | 488 ++++++++++++++
 tests/test_metric_identity_v1.py                   |  21 +-
 .../test_metric_streaming_security_integration.py  |   9 +
 tests/test_production_smoke_remediation.py         | 153 +++++
 tests/test_streaming_dataset_v1.py                 |  32 +
 tests/test_streaming_jsonl_v1.py                   |  65 +-
 tests/test_streaming_materialize_v1.py             |  13 +-
 tests/test_streaming_source_v1.py                  |  54 ++
 tests/test_streaming_tabular_v1.py                 |  79 ++-
 tests/test_unified_analysis_v03.py                 |  13 +-
 54 files changed, 3651 insertions(+), 283 deletions(-)
```

The reviewed implementation allowlist is exactly 54 text files: 14 production
files, two design/plan documents, and 38 test/fixture files. There are no
renames, binary changes, mode changes, or submodule changes. This report is the
only additional path in the final report-bearing commit.

## Immutable SHA-256 ledger

### Design, plan, and repository configuration

```text
docs/superpowers/specs/2026-08-27-claim-recovery-evidence-occurrence-v2-design.md  0c68444cc1c372ae09cbfe9de178304f9f09117266b259770fb7cb903c996f6a
docs/superpowers/plans/2026-08-27-claim-recovery-evidence-occurrence-v2.md         23a83370a771b1f817b2024152f16ea7960b1ddde5c843fd4761c67348377c7a
pyproject.toml                                                                    db2df34cbaa8e41ba68784ca03b9c26e0811ab1f1a1711d50341a0ddad9fb5ec
```

### Changed production source

```text
claimci/analysis/__init__.py                    7bdd1435fdc79869d928ee353ec6c121694f7bb233379b0e6c9dcecc739bb6c6
claimci/analysis/adapters/config.py             99ca2f7e6679e8d78c36e7c3c1f1b13cfeb9c200990be2abf78a913f26400843
claimci/analysis/adapters/core.py               dd0b2d692cc13166ef8096260bb920f56995142abb28a8e40c15e2e4b35079c5
claimci/analysis/adapters/dataset.py            869646da6ac95b5f2681d2cb7f6a20d3f7d4ad799f63cbbea428e8c90d01b33a
claimci/analysis/adapters/native.py             23b78821132b2be9eb3675082d54b98c6a42028ff4e3517474970d7a68137a3f
claimci/analysis/adapters/streaming.py          76804183e95b99073c2c3ba7ad588ab153171075a90ba4fd90906746a0f7ba82
claimci/analysis/adapters/structured.py         00778241e9776735290cab8c8a40ea717621e08010ebedf742aa954d41f2aad6
claimci/analysis/adapters/tabular.py            108fc2212ae76e4594913bb3dbbdd78c2ffc4af698bbda57b9b8cac2945550c3
claimci/analysis/artifact_source.py             87edec6bb021010561f649e68a4c4009d9bb4d78413e7578437a0b15b323b973
claimci/analysis/claim_types.py                 31b9c86080fb2ab1e5b50c603f7ea215465ad9bebc4883e09828f1025ba4cc3b
claimci/analysis/contracts.py                   c12a98300ffc7c9d2198c65e0205128b9ea1d14e21529cd0c6c7e2e71880cc07
claimci/analysis/discovery/claims.py            98ac60a0348efcb0690a2a3adb17a38c59a01829638e4e00273a68dd64dea738
claimci/analysis/evidence_identity.py           4b69fa5d19cc2038ac6d6c8a648a1890e14ffb4ccca5c871044f792a185cf319
claimci/analysis/materialize.py                 1b07a6ffbe4b3f81ae7d25086b36e6bd429c2e4101b7ec8523e66e8ecafd9e10
```

### Controlled fixture

```text
tests/fixtures/production_smoke/README.md                        245906ae491e68f661e4b4571122bb3a9372211708af56bb07f28db87981e1bb
tests/fixtures/production_smoke/research.yaml                    8c7121eeb7603b816de26699640cd28be1ca2192ecfc057a1a8a0daf4cdd3d16
tests/fixtures/production_smoke/baseline-config.yaml             e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d
tests/fixtures/production_smoke/candidate-config.yaml            e1a1df4232410d5c7f7914003854edb7beeecabcb266da0d98f0bb020ac1f35d
tests/fixtures/production_smoke/baseline-eval.jsonl              62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600
tests/fixtures/production_smoke/candidate-eval.jsonl             62e6c7f6b0f53243d464c954488f94c4d0e708b3ba8284bd0f541e1c334b4600
tests/fixtures/production_smoke/baseline-results.json            8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493
tests/fixtures/production_smoke/candidate-results-base.json      8ced4ea6448fe152cdf8828950b73f24653a95682081c2f01c479579c3263493
tests/fixtures/production_smoke/candidate-results.json           7ea7c722aaae5e2739ae13800daa5f5ea12bd1675fda6e04994c95e22ce54dc3
tests/fixtures/production_smoke/baseline-train.jsonl             7ace1dd7aa40731aaee3c6e40d1d5d67f053e654488361e849b3d2b065626417
tests/fixtures/production_smoke/candidate-train.jsonl            1381bb732892ff2d772a5d06a0b7da2f3f43336195102073c7029799f390dc44
```

The ten approved head-fixture hashes exactly match the pinned acceptance map.
`candidate-results-base.json` is the separately stored base candidate result
and intentionally matches `baseline-results.json`.

## Review and security evidence

### Whole-branch review

An independent read-only reviewer examined the exact base-to-head packet and all
54 changed files. The initial review found one Important test-fixture occurrence
mismatch; production fail-closed behavior was correct. Commit
`1af824eec57a6a28400f4f2b32d460ecca8f12cf` aligned only five trusted test
fixture call sites. Re-review returned `APPROVED` with no Critical, Important,
or Minor findings. Assertions and production revalidation were not weakened.

### Scoped security review

Security coverage is continuous across the verified range:

1. Durable scan `b3aaa1a0-bc67-485e-bdcc-3f626a99d810` covered
   `1c7ae750bf9b01485d8061cb047e1ef463eed24e..305858fc396e0937f1746df0252d61c82e0ab4af`.
   It reviewed all 14 changed production files, completed 3/3 coverage, and
   produced zero candidates and zero findings.
2. Independent read-only delta review covered
   `305858fc396e0937f1746df0252d61c82e0ab4af..041c11305d7f420267bcb35d371f17a384a18f53`.
   It returned `APPROVED`, with 112 focused tests, 21 additional trust-boundary
   tests, an adversarial threshold probe, and bounded large-input regex probes.
3. Independent security addendum covered the test-only delta
   `041c11305d7f420267bcb35d371f17a384a18f53..1af824eec57a6a28400f4f2b32d460ecca8f12cf`.
   It returned `APPROVED`; no production/provider/schema/contract/authority
   surface changed, and 21 affected materialization tests passed.

The review covered provider/mapping non-authority, factory-only repository and
snapshot issuance, exact path/commit binding, selector canonicalization,
collision behavior, legacy decoding compatibility, and fail-closed
materialization. Provider payload schemas remain occurrence-free. Physical
occurrence identity remains separate from scientific role.

Durable scan artifact hashes:

```text
report.md       d1af7bc6b39abcfc78f4e76c51bf68f6c42567717d923e9ef3c9f55275cd36ce
findings.json   bb6c9899d4190435e6d0b3e412b757b5d80e576c8861856f982ea2cc869fe118
coverage.json   8c8dd863c9c80f04286b7c044f257262ce23ea260eb24da6bc6be96065d8e5db
threat_model.md 17a2b7781d347d18b7ba7f19fa506a485e11affdc20e128239205a982eec58af
results.sarif   8a58ab4629a59aad62abc42aab44880041354129af650f6e13d6b270567ac7c3
```

The TAC connector and CodeRabbit CLI were unavailable. No plugin or CodeRabbit
CLI was installed. A failed helper-setup attempt downloaded and unpacked an
`unzip` Debian package; it was not installed, and the artifacts were moved
recoverably outside the worktree to
`C:\Users\windo\AppData\Local\Temp\claimci-review-artifacts-b3aaa1a0`. The
required independent manual reviews completed instead.

## Compatibility and companion obligations

No Core D1 or storage migration exists or is required. Legacy bounded
`evidence-*` values remain accepted; reserved built-in `evidence-v2-*` values
are re-extracted under trusted runtime occurrence scope.

The following completed-result compatibility tests are deliberately deferred to
the ClaimCI-Web companion PR because Core has no completed hosted-result JSON
decoder:

1. Hosted decoder obligation — add an immutable legacy completed-result fixture
   in `runner/tests/test_contracts.py` and prove
   `runner/claimci_hosted_runner/contracts.py` decodes it with bounded legacy
   `evidence-*` identifiers unchanged and without requiring a v2 prefix.
2. Cloud decoder/hydration obligation — add the same immutable completed-result
   fixture to `cloud/src/analysis/contracts.test.ts` and
   `cloud/src/analysis/repository.test.ts`; prove
   `cloud/src/analysis/contracts.ts` decodes it and the repository loader
   hydrates it unchanged. Existing pending mapping approval must fail closed if
   regenerated question/choice identity changes; it must never silently retarget
   v2 evidence.

The companion PR must pin the exact final Core commit, remove or align Hosted's
duplicated natural-language grammar, preserve deterministic results under
provider uncertainty, and settle Pilot credit only from a persisted
deterministic Audit authority signal.

## Addendum — detached minimum-improvement Task 1 verification (2026-08-28)

This task-level addendum verifies the detached-threshold implementation against
the existing sealed branch. It does not replace the controller's final
whole-branch gate.

The focused RED command was run before production edits:

```powershell
python -m pytest -q tests/test_detached_threshold_source_binding.py tests/test_production_smoke_remediation.py
```

It exited `1` with `5 failed, 7 passed`: the three approved declarations were
not recovered, no private source binding existed, and the source-certified
native-Audit smoke case was not ready. A second pre-production matrix-only RED
run exited `1` with `4 failed, 4 passed, 2 deselected`; it additionally proved
that incompatible inline and detached values did not fail closed.

After implementation, the same focused command passed with `12 passed`. The
broader Core regression command covering claim types, deterministic/provider
discovery, planning, obligations, contracts, and review bridge passed with
`436 passed, 1 skipped`. The scoped trust-boundary subset passed with
`67 passed`.

The exact original smoke claim line remains byte-for-byte unchanged. The
same Review-issued PR-description document adds the next full line
`Declared minimum improvement: 0.05`; real evidence extraction, planning, and
native Audit reached `READY` and `SUPPORTED`.

`python -m pytest -q` was launched as the required full Core run. The desktop
terminal transport detached before returning its final summary, while the
pytest child completed; `.pytest_cache/v/cache/lastfailed` was empty
afterwards. No exit code or pass count is claimed for that invocation. The
focused evidence above is the task's conclusive executable result; the
controller must retain its one final captured whole-branch gate.

Additional integrity commands succeeded: `python -m compileall claimci tests`,
`python -m pip check` (`No broken requirements found`), and `git diff --check`.
The repository exposes no configured lint or type-check command: the checked
configuration files and `pyproject.toml`, workflow, and scripts references
contained none.

Scoped manual security review found no issue. It confirmed that only
deterministic discovery can issue the private document certificate; provider
schemas remain certificate/document/span-free; the canonical parser verifies
source id, hash, exact primary span, one metric, one declaration, and compatible
inline values; compiler/obligation revalidation reuse the binding; and public
JSON emits only historical canonical-claim fields. The durable security scan
could not be started because its selected uncommitted-diff digest was reported
stale immediately, and TAC status was unavailable in this environment. No
durable-scan result is represented as having passed.
