# External-install demo fixture

This directory is an allowlisted, ClaimCI-free consumer fixture. Run the
materializer from the ClaimCI repository with the full reviewed reusable-workflow
SHA:

```powershell
python scripts/materialize_external_demo.py `
  --workflow-sha <reviewed-workflow-commit-sha> `
  --output C:\path\to\ClaimCI-Demo
```

Use `--enable-review` to add the trusted disabled review configuration. The
fixture intentionally reports `NOT_SUPPORTED`: the candidate moves from 0.60
to 0.90 accuracy, uses 3x compute, has one run, and leaks one exact evaluation
record into training. It contains no ClaimCI package, `pyproject.toml`, or
installer code.

The pilot is private/same-owner only. Configure the ClaimCI repository's
Actions access to allow private repositories owned by the same user. Keep the
caller workflow on the consumer default branch, pin the reusable workflow by
full SHA, and require the `ClaimCI Audit` check in branch protection. Review is
optional, neutral, and may send selected private content to the configured
provider only when the trusted base config and `OPENAI_API_KEY` are present.

See [the full external-install guide](../../docs/external-install-v0.md) for
the manual pilot sequence, fork/data-egress boundary, and troubleshooting.
