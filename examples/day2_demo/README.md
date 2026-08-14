# ClaimCI Day 2 demo: a better score that does not support the claim

This self-contained fixture looks like a compelling accuracy improvement: three
baseline seeds average **0.60**, while one candidate seed reports **0.90** (a
30-point / 50% relative improvement against the 0.05 threshold). ClaimCI still
rejects it because the candidate uses a 3.0x compute proxy and one of the two
evaluation records shared by both arms is also present in candidate training after canonical JSON
normalization. The one-seed candidate is separately flagged as insufficient
evidence.

Run it from the repository root after installing ClaimCI:

```powershell
python -m pip install -e ".[dev]"
claimci audit examples/day2_demo/research.yaml
```

For the views used in a 60-second walkthrough:

```powershell
claimci audit examples/day2_demo/research.yaml --json > day2-demo.json
claimci audit examples/day2_demo/research.yaml --markdown > day2-demo.md
```

The human output makes the verdict and evidence readable in a terminal. Open
`day2-demo.md` in a Markdown preview to show the same recomputed metrics and
stable finding IDs in a PR-style report; the JSON file is the machine-readable
view of the exact same audit.

For a 60-second recording, capture four views: the headline and 0.60 to 0.90
metrics; the `CONFIG.COMPUTE_MISMATCH` 3.0x evidence; the one-seed warning next
to `RESULT.CLAIM_SUPPORTED`; and `DATASET.EXACT_LEAKAGE` showing one of two
evaluation records. End on the `NOT_SUPPORTED` Check-style headline.

This is a deterministic fixture, not a model-training benchmark. ClaimCI only
checks the declared local artifacts and exact canonical train/evaluation
overlap; it does not establish statistical significance, detect semantic or
near-duplicate leakage, verify artifact provenance, or run the experiment.
