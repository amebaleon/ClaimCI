# ClaimCI external-install v0

This pilot runs ClaimCI from a separate private consumer repository without
copying ClaimCI Python source into that repository. It is intentionally limited
to private repositories owned by the same GitHub user as the private ClaimCI
repository. GitHub does not make private reusable workflows available to
arbitrary external repositories; organization and enterprise repositories have
the corresponding organization/enterprise boundary.

GitHub Actions settings for the private ClaimCI repository must allow access
from private repositories owned by the same user. GitHub warns that sharing a
private action or reusable workflow lets collaborators in the consuming
repository indirectly access it through workflow logs. Treat the provider
workflow as trusted release code and review every immutable SHA update.

## Generate the clean demo

The materializer accepts the reviewed reusable-workflow commit explicitly. It
rejects short, uppercase, non-hex, and all-zero SHAs, and refuses non-empty or
unsafe output paths:

```powershell
python scripts/materialize_external_demo.py `
  --workflow-sha <reviewed-claimci-external-workflow-sha> `
  --output C:\path\to\ClaimCI-Demo
```

Add `--enable-review` only when you want the generated consumer to contain the
trusted, disabled-by-default `.claimci/review.yaml` file. The fixture is the
known rejected-improvement study: baseline accuracy 0.60, candidate accuracy
0.90, 3x compute, one candidate run, and one exact train/evaluation leakage.
The generated tree contains only the manifest, declared artifacts, and one
SHA-pinned caller workflow; it contains no `claimci/`, `pyproject.toml`, or
installer code.

## Manual same-owner pilot

1. In the ClaimCI repository, review and commit the carrier (`action.yml`),
   then review and commit the rendered reusable workflow. Record the full
   40-character SHA of the workflow commit; never use a branch, tag, or
   placeholder in a consumer workflow.
2. Create a private customer repository under the same GitHub user. In ClaimCI
   repository Settings → Actions → General → Access, allow private repositories
   owned by that user to use the action/workflow.
3. Materialize the demo into an empty directory, or copy its caller workflow
   and artifacts into the customer repository. Keep the caller on the default
   branch and replace the workflow SHA only through a reviewed change.
4. Enable Actions and require the `ClaimCI Audit` check in branch protection.
   The caller uses `pull_request_target`, so its workflow file must remain
   base-branch-controlled and must not check out or execute PR workflow code.
5. If advisory review is approved, add the repository secret
   `OPENAI_API_KEY` and enable the trusted base `.claimci/review.yaml`. The
   deterministic Audit does not require this secret. Review is always neutral
   and advisory; only Audit can block a PR.
6. Open a test pull request and verify the Audit Check is attached to the PR
   head, the known fixture is `NOT_SUPPORTED`, and the optional Review Check is
   neutral. For a real study, replace only the manifest and declared artifacts.

## Security and troubleshooting

The reusable workflow checks out the consumer base and PR head separately and
treats PR files as passive confined data. Deterministic Audit keeps its reviewed
full-SHA action installer. The research-review job separately checks out the
reusable workflow's own repository at its exact immutable workflow SHA,
validates that checkout identity, and installs ClaimCI into an isolated runtime
from that source. Review configuration is read from the trusted consumer base.
If review is disabled, missing, or lacks a key, no provider call is made and the
neutral fallback remains advisory. When review is enabled, selected private
research text may be sent to the configured provider; disable or omit the config
when that egress is unacceptable. In particular, with
`pull_request_target`, pull_request_target fork-authored head content may be sent to the configured provider
when review is enabled.

The review workflow binds the exact pull-request base and head SHAs, derives and
validates one merge base from the local Git object graph, and materializes a
separate comparison checkout when that merge base differs from the requested
base. It runs `claimci review --preflight-only` without provider credentials or
the optional provider SDK. Only a preflight whose three gates are ready can
reach the later paid review step, and `OPENAI_API_KEY` is scoped to that step.
Both invocations repeat the same requested-base, comparison-base, basis, and
head coordinates; neither uses the synthetic workflow merge SHA.

For trusted local integrations, the declared CLI path is additive and requires
all six coordinate options together: `--requested-base-root`,
`--comparison-base-root`, `--requested-base-sha`, `--comparison-base-sha`,
`--comparison-basis` (`direct_base` or `merge_base`), and `--head-sha`.
Omitting the whole group keeps the bounded legacy pairwise mode available for
existing callers. Review report schema version 2 records the exact coordinates,
inventory completeness, all three gate results, scope completeness, projection
metrics, and the final status ceiling. `scope_complete` is independent of
provider `routing_incomplete`. Allowlisted submission/evaluation shell runners
are reported only as passive supporting configuration: they are never executed
and never treated as source code or executed results.

An installation failure usually means the ClaimCI repository is private without
the same-owner Actions access setting, the caller is not private, or the caller
uses an unavailable workflow SHA. A missing Check generally indicates Actions
is disabled or the caller's `checks: write` permission is restricted by an
organization policy. Do not solve access failures by adding a broad PAT to the
consumer workflow.
