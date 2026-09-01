# Research Review exact-hint routing and cardinality root cause

## Scope and frozen evidence

This change is based on the single passive Opik #8075 acceptance run made from
ClaimCI Core `841d08b6e8e678577822f49a63f1fee1c0d0aa7e`. The external calibration is
not rerun by this work.

## Root causes

`discover_evidence()` used `_matches()` for two different routing origins.
Autonomous discovery appropriately applies changed-file and path-name relevance
heuristics. An exact provider proposal, however, had already passed safe-relative
normalization and exact membership in the issued `repository_paths` inventory.
Reapplying `_matches()` rejected two exact unchanged source hints at the
changed-file gate and seven exact source/test hints at filename heuristics. The
provider could not invent a path, but the implementation treated an exact proposal
as heuristic discovery.

The seven-versus-eight failure was not truncation. The claim-owned evidence was the
complete 10,787-byte Java test file and contained eight standalone `@Test`
annotation lines. The provider nevertheless affirmed the PR's count of seven.
Synthesis validation checked shape, claim coverage, citation ownership, and bounds,
but did not have a deterministic whole-file cardinality fact.

The missed `-1` sentinel difference shared the exact-hint routing failure and also
exposed prefix-only excerpt locality. The issued `DatasetService.java` excerpt ended
at line 386, while the changed enrichment and sentinel predicate were around lines
708-765. Citation ownership correctly prevented the behavior claim from borrowing
the complete test evidence owned by other claims.

## Bounded design

1. `_matches()` remains the relevance policy for autonomous heuristic discovery.
2. An exact indexed provider hint may select only SOURCE/TEST evidence without
   path-name or changed-file heuristics. All non-source/test kinds retain existing
   routing validation. Path identity, confinement, regular-file validation, byte
   and character limits, provenance, and claim-scoped ownership remain
   deterministic.
3. Complete exact-hint source/test files are issued completely. Oversized changed
   files use one bounded contiguous base/head-derived changed-region excerpt with
   accurate head line coordinates. Oversized unchanged files are explicitly marked
   incomplete with no claimed material locality; their bytes remain in the local
   evidence record but are omitted from provider synthesis evidence, and no citation
   can establish an exact whole-file or exact-local implementation claim.
4. Evidence records expose deterministic completeness and locality metadata. These
   fields describe what bytes were issued; they do not upgrade evidence provenance.
5. A narrow Java-test fact counts only complete, claim-owned TEST references whose
   exact hinted file matches the claimed test subject. A bounded linear lexical-state
   scan first excludes comments, strings, and character literals, then counts only a
   code line containing optional horizontal whitespace, `@Test`, optional horizontal
   whitespace, and no other token. Ambiguous or malformed lexical state yields no
   count. It does not execute Java or parse general Java syntax.
6. A normalized exact test-count claim may be affirmed only when that complete
   deterministic count matches. A mismatch is constrained to a contradiction
   interpretation; incomplete or unavailable eligible evidence is constrained to a
   cannot-verify interpretation. Provider output that violates the constraint fails
   closed at the existing synthesis-validation boundary.

Research Review remains advisory and non-blocking. ClaimCI Audit remains the sole
verdict authority. Provider calls remain capped at two, and no additional provider
operation is introduced.

## Rejected alternatives

- Relaxing `_matches()` globally would broaden autonomous discovery.
- CamelCase or domain keyword matching would be fuzzy, incomplete, and
  repository-specific.
- Treating provider hints as trusted manifest priority would launder authority.
- Increasing `max_file_chars` would weaken boundedness without guaranteeing
  locality.
- A prefix citation for an oversized unchanged file would be semantically hollow.
- Prompt-only cardinality wording and completeness metadata alone cannot correct a
  miscount over complete evidence.
- A general Java parser or sentinel-specific rule is unnecessary unless the bounded
  routing/locality regressions prove otherwise.
