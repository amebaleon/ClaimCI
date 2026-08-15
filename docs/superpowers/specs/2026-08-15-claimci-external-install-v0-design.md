# ClaimCI External-Repository Installation v0 Design

Date: 2026-08-15

## Objective

Allow a separate repository that contains no ClaimCI Python source to run the
authoritative deterministic ClaimCI Audit and the optional advisory ClaimCI
Research Review from a small GitHub Actions caller workflow.

The v0 pilot boundary is intentionally narrow:

- ClaimCI remains private.
- The consumer is a separate private repository owned by the same GitHub user.
- The ClaimCI repository's Actions access setting explicitly allows private
  repositories owned by that user to consume its actions and reusable
  workflows.
- Arbitrary external users, public repositories, cross-organization sharing,
  a public automation repository, a GitHub App, and a hosted service remain out
  of scope.

## Product invariants

- `ClaimCI Audit` is the only blocking scientific authority. Its deterministic
  verdict-to-Check mapping is unchanged.
- `ClaimCI Research Review` is always advisory and its Check conclusion is
  always `neutral`.
- Review remains explicit opt-in through the consumer default branch's trusted
  `.claimci/review.yaml`.
- A complete review makes at most two provider calls: extraction and synthesis.
  Provider retries remain disabled.
- Deterministic audit works without `OPENAI_API_KEY` and without the optional
  OpenAI dependency.
- Pull-request head content is passive, untrusted data. It is never installed,
  imported, built, sourced, or executed.
- Provider credentials are scoped only to the read-only review job and never
  coexist with Checks-write credentials.

## GitHub distribution constraint

GitHub does not make private actions or reusable workflows available to
arbitrary external repositories. A private user-owned automation repository
can be shared only with private repositories owned by the same user. An
organization- or enterprise-owned repository has corresponding organization or
enterprise sharing boundaries. A public caller can consume only a public
reusable workflow.

Therefore v0 supports only the approved same-owner private pilot. A future
public/customer-capable distribution may publish the same action and reusable
workflow interface without changing the ClaimCI audit or review core.

GitHub also downloads a shared private action with a short-lived scoped token,
and warns that collaborators in the consumer repository can indirectly observe
the shared automation through workflow logs. The pilot documentation must state
this boundary.

## Architecture

### 1. Immutable ClaimCI installer action

A root `action.yml` is a small composite action used only as a trusted package
carrier. GitHub downloads the ClaimCI repository revision named in the action's
full-SHA `uses:` reference. The action installs ClaimCI from
`github.action_path`, never from the consumer checkout or the current working
directory.

The action accepts one boolean-like input, `install-review-dependencies`, with
allowed values `true` and `false`. `false` installs the deterministic package;
`true` installs the pinned `llm` extra. The value is passed through an
environment variable and strictly validated before it affects the shell
command.

The composite action is a transport helper only. It does not own permissions,
secrets, Check publication, checkout policy, or scientific decisions.

### 2. SHA-pinned reusable workflow

`.github/workflows/claimci-external.yml` is called with `workflow_call`. Its
interface contains only:

- optional string input `manifest-path`, default `research.yaml`;
- optional string input `review-config-path`, default
  `.claimci/review.yaml`;
- optional named secret `OPENAI_API_KEY`.

The customer pins this workflow using a full 40-character commit SHA. The
workflow itself pins the installer action to a separate full commit SHA. This
two-pin release is necessary because GitHub does not allow expressions in
`uses:` and a reusable workflow cannot dynamically reference its own revision.
The customer sees only the reusable-workflow SHA; the internal carrier SHA is
frozen in that workflow revision.

The reusable workflow has three security domains:

1. **Deterministic audit and publication.** It installs ClaimCI from the
   immutable carrier, checks out the consumer PR head to `pull-request` with
   persisted credentials disabled, runs `claimci audit` with
   `--artifact-root pull-request`, and publishes `ClaimCI Audit` against the
   explicit PR head SHA. Only deterministic verdicts control its conclusion.
2. **Optional research review.** It has `contents: read` only, checks out the
   consumer base SHA to `consumer-base` and head SHA to `pull-request`, installs
   the immutable carrier, loads review configuration only from
   `consumer-base`, and receives only `OPENAI_API_KEY`. Missing or disabled
   config produces zero provider calls. One `claimci review` invocation emits
   both JSON and Markdown and can make at most two provider calls.
3. **Advisory publication.** It has `checks: write`, receives no provider
   secret, performs no repository checkout, revalidates the bounded payload and
   PR head SHA, and publishes only a completed `neutral` Research Review Check.

The called workflow uses the caller's `github` event context, so
`github.event.pull_request.base.sha`, `head.sha`, repository, event path, and
Checks API target all refer to the consumer repository.

### 3. Minimal customer caller

The customer stores one base-branch-controlled file:

```yaml
name: ClaimCI

on:
  pull_request_target:
    types: [opened, synchronize, reopened, ready_for_review]

permissions: {}

jobs:
  claimci:
    permissions:
      contents: read
      checks: write
    uses: amebaleon/ClaimCI/.github/workflows/claimci-external.yml@WORKFLOW_COMMIT_SHA
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

`WORKFLOW_COMMIT_SHA` is replaced with the reviewed immutable release commit.
The caller does not check out code, run shell commands, inherit all secrets, or
accept a PR-controlled version input.

### 4. Consumer configuration and artifacts

The PR head contains `research.yaml` plus its declared baseline/candidate
configs, result artifacts, and train/evaluation datasets. These are parsed as
confined data by the deterministic auditor.

The consumer default branch may contain `.claimci/review.yaml`. Absence or
`enabled: false` disables external-provider review. The PR cannot enable review,
change the configured provider/model/limits, or redirect the trusted config
root. The `OPENAI_API_KEY` repository secret is optional; deterministic Audit
does not receive it.

## Release bootstrap

The initial immutable pilot requires two commits:

1. **Carrier commit:** add and verify `action.yml` and the ClaimCI package.
2. **Workflow commit:** render `claimci-external.yml` with the carrier commit's
   full SHA, then commit the rendered reusable workflow and documentation.

The consumer pins the second commit. A deterministic renderer rejects missing,
short, non-hex, or all-zero carrier SHAs, and the checked-in reusable workflow
must contain no template token, branch, or tag reference.

This bootstrap affects release packaging only. It does not add runtime version
resolution, mutable refs, or a PR-controlled install input.

## Failure behavior

- Missing/malformed `research.yaml` produces the existing deterministic
  insufficient-evidence or integration behavior and an authoritative Check.
- Missing/disabled review config performs no provider call and yields a neutral
  disabled review.
- Missing API key, SDK/provider failure, malformed provider output, or review
  limits yield a bounded advisory status and never alter Audit.
- Malformed, oversized, wrong-head, or non-neutral review handoff is replaced
  with the fixed neutral publisher fallback.
- Failure to access the private ClaimCI workflow/action is an installation
  failure. The pilot instructions diagnose the repository Actions access
  setting; they do not request a broad PAT or execute consumer code.

## Fixtures and release tooling

- An inert `templates/claimci-external.yml.tmpl` contains the explicit
  installer-SHA token. It is deliberately outside `.github/workflows` and has
  a `.tmpl` suffix so GitHub cannot treat the unreleased marker file as a
  callable workflow. The renderer emits the active workflow only after the
  marker is replaced by a verified immutable SHA.
- A deterministic renderer validates a full immutable SHA and produces the
  active workflow without changing other content.
- A demo materializer creates a clean `ClaimCI-Demo` tree containing the tiny
  caller, a known NOT_SUPPORTED `research.yaml` study, and an optional disabled
  review config. It accepts the reusable-workflow SHA as an explicit argument.
- Tests parse and validate the generated files, run the deterministic fixture
  locally, and prove the consumer tree contains no ClaimCI package.

## Verification

Automated tests use no network and no provider calls. They cover:

- full-SHA pins and deterministic rendering;
- `workflow_call` inputs and explicitly named optional secret;
- `pull_request_target` caller contract;
- consumer base/head checkout separation and passive head handling;
- install paths confined to the downloaded ClaimCI action;
- trusted base review config and optional no-key operation;
- provider/publisher credential separation;
- unchanged authoritative Audit and neutral Review conclusions;
- one review CLI invocation, two-call maximum, and zero retries;
- malformed/tampered/oversized handoff fallback;
- clean-repository fixture generation and deterministic audit behavior;
- private same-owner boundary documentation.

The complete existing offline suite and an independent read-only security and
product review are required before completion. No live OpenAI call is part of
this work.
