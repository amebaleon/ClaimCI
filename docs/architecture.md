# Architecture

ClaimCI can run entirely on the caller's machine or in their GitHub Actions job.

1. The CLI reads a research manifest and resolves artifact paths within the
   caller's explicit artifact root.
2. Deterministic adapters inspect configurations, dataset identities, and raw
   results, recompute supported metrics, and check evidence obligations.
3. Evidence references carry source identity and provenance. Missing or
   incomplete evidence remains visible rather than becoming a verified claim.
4. The report emits supported, not-supported, or insufficient-evidence outcomes
   as text, JSON, or Markdown.
5. Optional research review extracts claims and maps bounded evidence through
   a model provider. Its interpretation remains advisory and opt in.

See `claimci/cli.py`, `claimci/analysis/`, and `claimci/review/` for the code.
The immutable composite `action.yml` installs the selected Core revision.
The caller supplies its own compute, artifacts, and optional API credentials.

## Historical hosted architecture

The former commercial service connected a static Pages frontend to an API
Worker, GitHub App, D1 database, Queue, and container-backed analysis runner.
That service is retired from the maintenance-mode product. The separate
ClaimCI-Web repository preserves its reusable source and infrastructure
configuration; those files are historical and do not imply a live service.
The current public website is plain static HTML and CSS with no backend,
authentication, payment, model API, or analytics dependency.
