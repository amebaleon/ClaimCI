"""Apply the pull-request-scoped evidence discovery patch once."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def main() -> None:
    tools = (ROOT / "claimci/review/tools.py").read_text(encoding="utf-8")
    if "def select_relevant_manifest_audit_plans(" in tools:
        print("review scope patch already applied")
        return

    replace_once(
        "claimci/review/sources.py",
        '''def _change_candidate(path: str) -> bool:
    """Return paths whose base/head status can affect review file selection."""

    relative = Path(path)
    lowered = relative.as_posix().casefold()
    name = relative.name.casefold()
    return (
        _eligible_document(path)
        or relative.suffix.casefold() in {".py", ".js", ".ts", ".rs", ".go"}
        or lowered.startswith(("src/", "lib/", "claimci/", "tests/"))
        or name.startswith("test_")
    )
''',
        '''_CHANGE_EVIDENCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".go",
    ".sql",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
}


def _change_candidate(path: str) -> bool:
    """Return paths whose base/head status can affect review evidence."""

    relative = Path(path)
    lowered = relative.as_posix().casefold()
    name = relative.name.casefold()
    return (
        _eligible_document(path)
        or relative.suffix.casefold() in _CHANGE_EVIDENCE_SUFFIXES
        or lowered.startswith(("src/", "lib/", "claimci/", "tests/"))
        or name.startswith("test_")
    )
''',
    )

    replace_once(
        "claimci/review/tools.py",
        '''@dataclass(frozen=True)
class ManifestAuditPlan:
    """A path-confined, bounded manifest bundle reserved for later auditing."""

    manifest_path: str
    paths: tuple[str, ...]


def _freeze(value: Any) -> Any:
''',
        '''@dataclass(frozen=True)
class ManifestAuditPlan:
    """A path-confined, bounded manifest bundle reserved for later auditing."""

    manifest_path: str
    paths: tuple[str, ...]


def select_relevant_manifest_audit_plans(
    plans: Sequence[ManifestAuditPlan],
    *,
    changed_paths: Sequence[str],
) -> tuple[ManifestAuditPlan, ...]:
    """Keep audits whose manifest or declared artifact changed in this PR.

    An unrelated study elsewhere in a large repository must not reserve the
    review file budget, appear as deterministic authority, or enter synthesis.
    A no-base local review still marks every eligible file as changed, so this
    filter does not weaken standalone inspection.
    """

    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        raise ReviewError("manifest audit plans must be a sequence")
    if not all(isinstance(plan, ManifestAuditPlan) for plan in plans):
        raise ReviewError("manifest audit plans contain an invalid value")
    if not isinstance(changed_paths, Sequence) or isinstance(
        changed_paths, (str, bytes)
    ):
        raise ReviewError("changed manifest paths must be a sequence")

    changed: set[str] = set()
    for raw in changed_paths:
        relative = _relative(raw)
        if relative is None:
            raise ReviewError("changed manifest path is unsafe")
        changed.add(relative)

    return tuple(
        plan
        for plan in plans
        if any(path in changed for path in plan.paths)
    )


def _freeze(value: Any) -> Any:
''',
    )

    replace_once(
        "claimci/review/evidence.py",
        '''    by_path: dict[str, set[str]] = defaultdict(set)
    priority_order: list[str] = []
    missing: list[MissingEvidence] = []
''',
        '''    # With a trusted base checkout, heuristic discovery is confined to
    # files changed by the pull request. Unchanged files remain reachable only
    # through trusted manifest priority or an exact, route-valid provider hint.
    # Without a base checkout, every indexed path is treated as changed.
    heuristic_index = (
        sorted_index
        if changed_paths is None
        else tuple(path for path in sorted_index if path in changed)
    )

    by_path: dict[str, set[str]] = defaultdict(set)
    priority_order: list[str] = []
    missing: list[MissingEvidence] = []
''',
    )
    replace_once(
        "claimci/review/evidence.py",
        '''    for claim in claims:
        for path in sorted_index:
            if _matches(claim, path, changed):
                by_path[path].add(claim.claim_id)
''',
        '''    for claim in claims:
        for path in heuristic_index:
            if _matches(claim, path, changed):
                by_path[path].add(claim.claim_id)
''',
    )

    replace_once(
        "claimci/review/orchestrator.py",
        '''    plan_manifest_audits,
    run_manifest_audits,
)
''',
        '''    plan_manifest_audits,
    run_manifest_audits,
    select_relevant_manifest_audit_plans,
)
''',
    )
    replace_once(
        "claimci/review/orchestrator.py",
        '''        audit_plans = plan_manifest_audits(
            inputs.repository_root,
            manifest_candidates,
            limits=config.limits,
        )
        sources = _sources_after_manifest_reservation(
''',
        '''        audit_plans = select_relevant_manifest_audit_plans(
            plan_manifest_audits(
                inputs.repository_root,
                manifest_candidates,
                limits=config.limits,
            ),
            changed_paths=sources.changed_paths,
        )
        manifest_candidates = tuple(
            plan.manifest_path for plan in audit_plans
        )
        sources = _sources_after_manifest_reservation(
''',
    )

    replace_once(
        "README.md",
        '''Selected private repository content may therefore be sent to the configured
external provider when an owner enables this configuration. This boundary is
especially important for `pull_request_target` and fork pull requests: the
head checkout is passive data, while configuration and code come from the
trusted base branch. Disable or omit `.claimci/review.yaml` when external data
egress is not acceptable.
''',
        '''Selected private repository content may therefore be sent to the configured
external provider when an owner enables this configuration. With a trusted
base checkout, heuristic evidence discovery is confined to files changed by
the pull request. Unchanged files enter review only through a relevant
`research.yaml` whose manifest or declared artifact changed, or through an
exact route-valid evidence path recovered from the issued sources. Structured
research artifacts such as SQL, JSON, YAML, CSV, and TOML participate in the
bounded base/head change index. This boundary is especially important for
`pull_request_target` and fork pull requests: the head checkout is passive
data, while configuration and code come from the trusted base branch. Disable
or omit `.claimci/review.yaml` when external data egress is not acceptable.
''',
    )

    print("review scope patch applied")


if __name__ == "__main__":
    main()
