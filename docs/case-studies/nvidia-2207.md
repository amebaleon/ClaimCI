# NVIDIA Model Optimizer #2207 — benchmark provenance

ClaimCI's [initial review](https://github.com/NVIDIA/Model-Optimizer/pull/2207#issuecomment-5544562513)
reported that the headline benchmark was not tied to a reproducible artifact
for the reviewed head. This was a provenance gap, not an allegation that the
numbers were false. A [follow-up](https://github.com/NVIDIA/Model-Optimizer/pull/2207#issuecomment-5582549145)
again requested baseline/head revisions, launch configuration, raw timing,
and memory evidence.

An independent [Claude review](https://github.com/NVIDIA/Model-Optimizer/pull/2207#pullrequestreview-5138584065)
explicitly cited @amebaleon's missing-artifact finding. A subsequent
[re-review](https://github.com/NVIDIA/Model-Optimizer/pull/2207#pullrequestreview-5145962678)
again cited it and required the PR body and changelog to describe the current
implementation before approval. An [inline review request](https://github.com/NVIDIA/Model-Optimizer/pull/2207#discussion_r3961356082)
explicitly called for current-head remeasurement or removal of the specific
figures from the changelog, citing @amebaleon's two comments.

The [final PR description](https://github.com/NVIDIA/Model-Optimizer/pull/2207)
removed the old 19.9x comparison and used new export measurements (168.6 seconds
for Qwen3-235B-A22B and 1359.7 seconds for the larger configuration, with a
1411.7-second repeat). It explicitly declined a speedup ratio and identified a
same-day baseline of main as outstanding. These are upstream-reported
measurements; ClaimCI did not independently reproduce those GPU runs.

The final changelog describes the export implementation without the prior
unreproducible numeric comparison. The PR merged on 2026-09-10.

This chronology shows a documented finding, explicit citation by another
reviewer, and subsequent evidence/wording changes. **It does not establish that
NVIDIA made those changes because of ClaimCI.**
