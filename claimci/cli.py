"""Command-line interface for ClaimCI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .audit import audit_research
from .models import ClaimCIError, Verdict
from .report import render_human, render_json, render_markdown
from .parsing import unique_json_object, validate_json_graph
from .review.config import load_review_config
from .review.inventory import InventoryVerificationError, build_git_change_inventory
from .review.models import (
    ComparisonBasis,
    DeclaredReviewCoordinates,
    ReviewConfig,
    ReviewError,
    ReviewInventoryFailure,
    ReviewPreflight,
    ReviewStatus,
    SnapshotIdentity,
    SnapshotRole,
)
from .review.orchestrator import ResearchReview, ReviewInputs, run_review
from .review.preflight import preflight_review
from .review.report import render_review_json, render_review_markdown


EXIT_CODES = {
    Verdict.SUPPORTED: 0,
    Verdict.NOT_SUPPORTED: 1,
    Verdict.INSUFFICIENT_EVIDENCE: 2,
}

MAX_EVENT_METADATA_BYTES = 1024 * 1024


def _write_console(stream: object, text: str) -> None:
    """Write without crashing when artifact text exceeds console encoding.

    Windows runners can expose legacy encodings such as cp949. Preserve every
    representable character and render the remainder as deterministic Python
    escape sequences instead of raising ``UnicodeEncodeError`` mid-audit.
    """

    encoding = getattr(stream, "encoding", None)
    if isinstance(encoding, str):
        try:
            text = text.encode(encoding, errors="backslashreplace").decode(encoding)
        except (LookupError, UnicodeError):
            text = text.encode("ascii", errors="backslashreplace").decode("ascii")
    writer = getattr(stream, "write")
    writer(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claimci",
        description="Verify whether research claims are supported by experiment artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="audit a research manifest")
    audit_parser.add_argument("research_file", help="path to research.yaml")
    audit_parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "confine the manifest and every resolved artifact path to this directory "
            "(used by trusted CI integrations)"
        ),
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit a machine-readable JSON report",
    )
    audit_parser.add_argument(
        "--markdown",
        action="store_true",
        dest="markdown_output",
        help="emit GitHub-flavored Markdown for a pull request or job summary",
    )
    review_parser = subparsers.add_parser(
        "review",
        help="run the opt-in advisory LLM research review",
    )
    review_parser.add_argument(
        "repository_root",
        help="checked-out pull-request repository to inspect",
    )
    review_parser.add_argument(
        "--base-root",
        default=None,
        help=(
            "legacy optional base checkout used to identify changed research documents; "
            "the declared coordinate path uses the six explicit coordinate options"
        ),
    )
    review_parser.add_argument(
        "--requested-base-root",
        default=None,
        help="trusted checkout at the exact requested pull-request base",
    )
    review_parser.add_argument(
        "--comparison-base-root",
        default=None,
        help="trusted checkout at the explicit direct or merge comparison base",
    )
    review_parser.add_argument(
        "--requested-base-sha",
        default=None,
        help="full lowercase Git object ID for the requested base checkout",
    )
    review_parser.add_argument(
        "--comparison-base-sha",
        default=None,
        help="full lowercase Git object ID for the comparison base checkout",
    )
    review_parser.add_argument(
        "--comparison-basis",
        choices=tuple(basis.value for basis in ComparisonBasis),
        default=None,
        help="explicit comparison semantics: direct_base or merge_base",
    )
    review_parser.add_argument(
        "--head-sha",
        default=None,
        help="full lowercase Git object ID for the passive head checkout",
    )
    review_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run only the three provider-free gates and render their advisory report",
    )
    review_parser.add_argument(
        "--config-root",
        default=None,
        help="trusted checkout containing .claimci/review.yaml (defaults to repository)",
    )
    review_parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="trusted review configuration path (relative to --config-root or absolute within it)",
    )
    review_parser.add_argument(
        "--event-json",
        default=None,
        help="GitHub event JSON file containing pull_request title/body",
    )
    review_parser.add_argument(
        "--pr-title",
        default=None,
        help="pull-request title (overrides event metadata)",
    )
    review_parser.add_argument(
        "--pr-description-file",
        default=None,
        help="UTF-8 file containing the pull-request description",
    )
    review_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the advisory JSON report",
    )
    review_parser.add_argument(
        "--markdown",
        action="store_true",
        dest="markdown_output",
        help="emit advisory GitHub-flavored Markdown (the default)",
    )
    review_parser.add_argument(
        "--json-output",
        dest="json_path",
        default=None,
        help="write advisory JSON to this path",
    )
    review_parser.add_argument(
        "--markdown-output",
        dest="markdown_path",
        default=None,
        help="write advisory Markdown to this path",
    )
    return parser


def _review_root(raw: str | None, label: str) -> Path:
    if raw is None:
        raise ReviewError(f"{label} is required")
    try:
        resolved = Path(raw).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError(f"{label} is invalid: {exc}") from exc
    if not resolved.is_dir():
        raise ReviewError(f"{label} is not a directory: {resolved}")
    return resolved


def _read_review_text(raw: str | None, label: str, *, max_chars: int) -> str:
    if raw is None:
        return ""
    try:
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            raise ValueError("metadata path must be a regular non-symlink file")
        with path.open("r", encoding="utf-8", newline=None) as handle:
            return handle.read(max_chars + 1)[:max_chars]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReviewError(f"could not read {label}: {type(exc).__name__}") from exc


def _event_metadata(raw: str | None) -> tuple[str, str]:
    if raw is None:
        return "", ""
    try:
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            raise ValueError("event metadata must be a regular non-symlink file")
        with path.open("rb") as handle:
            encoded = handle.read(MAX_EVENT_METADATA_BYTES + 1)
        if len(encoded) > MAX_EVENT_METADATA_BYTES:
            raise ValueError("event metadata exceeds the bounded input size")
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
        validate_json_graph(payload, label="GitHub event JSON")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ReviewError(f"could not load event metadata: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ReviewError("GitHub event JSON root must be an object")
    pull_request = payload.get("pull_request", {})
    if pull_request is None:
        pull_request = {}
    if not isinstance(pull_request, dict):
        raise ReviewError("GitHub event pull_request must be an object")
    title = pull_request.get("title", "")
    body = pull_request.get("body", "")
    if title is None:
        title = ""
    if body is None:
        body = ""
    if not isinstance(title, str) or not isinstance(body, str):
        raise ReviewError("GitHub event title and body must be strings or null")
    return title, body


def _bounded_pr_metadata(title: str, description: str, limit: int) -> tuple[str, str]:
    """Normalize and bound PR metadata before constructing review inputs."""

    normalized_title = title.replace("\r\n", "\n").replace("\r", "\n")[:limit]
    remaining = max(0, limit - len(normalized_title))
    normalized_description = (
        description.replace("\r\n", "\n").replace("\r", "\n")[:remaining]
    )
    return normalized_title, normalized_description


def _write_review_file(raw: str | None, content: str, label: str) -> None:
    if raw is None:
        return
    try:
        target = Path(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReviewError(f"could not write {label}: {type(exc).__name__}") from exc


_COORDINATE_OPTIONS = (
    "requested_base_root",
    "comparison_base_root",
    "requested_base_sha",
    "comparison_base_sha",
    "comparison_basis",
    "head_sha",
)


def _declared_coordinate_mode(args: argparse.Namespace) -> bool:
    present = tuple(getattr(args, name) is not None for name in _COORDINATE_OPTIONS)
    if any(present) and not all(present):
        raise ReviewError("all six coordinate options are required together")
    return all(present)


def _resolved_candidate_root(raw: str) -> Path | None:
    try:
        candidate = Path(raw)
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
        return resolved if resolved.is_dir() else None
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _trusted_config_root(raw: str) -> Path:
    """Resolve one explicit trusted root without following its final symlink."""

    try:
        candidate = Path(raw)
        if candidate.is_symlink():
            raise ValueError
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError
        return resolved
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ReviewError("config root is invalid") from exc


def _snapshot_or_none(
    root: Path | None,
    sha: str,
    role: SnapshotRole,
) -> SnapshotIdentity | None:
    if root is None:
        return None
    try:
        return SnapshotIdentity(role=role, root=root, sha=sha)
    except ReviewError:
        return None


def _preflight_only_review(preflight: ReviewPreflight) -> ResearchReview:
    if not isinstance(preflight, ReviewPreflight):
        raise ReviewError("preflight result is invalid")
    error_code = None
    error_message = None
    if not preflight.ready_for_provider:
        failure = next(
            gate
            for gate in preflight.gates
            if gate.disposition.value == "fail"
        )
        error_code = failure.reasons[0].code
        error_message = "Research review preflight did not pass."
    return ResearchReview(
        status=preflight.review_status_ceiling,
        preflight=preflight,
        error_code=error_code,
        error_message=error_message,
    )


def _emit_review_result(args: argparse.Namespace, result: ResearchReview) -> int:
    json_text = render_review_json(result)
    markdown_text = render_review_markdown(result)
    _write_review_file(args.json_path, json_text, "review JSON")
    _write_review_file(args.markdown_path, markdown_text, "review Markdown")

    # Explicit output paths are intended for CI artifacts; avoid duplicating a
    # large report on stdout when both views were requested as files. A view
    # flag always selects stdout, and with no flag Markdown is the default.
    if args.json_output:
        _write_console(sys.stdout, json_text)
    elif args.markdown_output:
        _write_console(sys.stdout, markdown_text)
    elif args.json_path is None and args.markdown_path is None:
        _write_console(sys.stdout, markdown_text)
    return 0


def _review_command(args: argparse.Namespace) -> int:
    """Run one advisory review and render one or both views."""

    declared_coordinates = _declared_coordinate_mode(args)
    head_root = (
        _resolved_candidate_root(args.repository_root)
        if declared_coordinates
        else None
    )
    repository_root = (
        head_root
        if head_root is not None
        else (
            Path(args.repository_root)
            if declared_coordinates
            else _review_root(args.repository_root, "repository root")
        )
    )
    base_root = None if args.base_root is None else _review_root(args.base_root, "base root")
    requested_root = (
        _resolved_candidate_root(args.requested_base_root)
        if declared_coordinates
        else None
    )
    comparison_root = (
        _resolved_candidate_root(args.comparison_base_root)
        if declared_coordinates
        else None
    )
    config_path = None if args.config_path is None else Path(args.config_path)
    if args.config_root is not None:
        config = load_review_config(
            _trusted_config_root(args.config_root),
            config_path,
        )
    elif declared_coordinates and requested_root is None:
        # A rejected identity root cannot be followed to discover policy. An
        # enabled in-memory policy exists only to render the inevitable Gate 1
        # root failure; that failure makes provider work unreachable.
        config = ReviewConfig(enabled=True)
    else:
        config = load_review_config(
            requested_root if declared_coordinates else repository_root,
            config_path,
        )
    if args.preflight_only and not config.enabled:
        return _emit_review_result(args, ResearchReview(status=ReviewStatus.DISABLED))
    event_title, event_body = _event_metadata(args.event_json)
    pr_title = event_title if args.pr_title is None else args.pr_title
    pr_description = event_body
    if args.pr_description_file is not None:
        pr_description = _read_review_text(
            args.pr_description_file,
            "pull-request description",
            max_chars=config.limits.max_context_chars,
        )
    if not isinstance(pr_title, str) or not isinstance(pr_description, str):
        raise ReviewError("pull-request title and description must be strings")
    pr_title, pr_description = _bounded_pr_metadata(
        pr_title,
        pr_description,
        config.limits.max_context_chars,
    )

    requested = None
    comparison = None
    head = None
    inventory: object | None = None
    coordinates = None
    inventory_failure = None
    if declared_coordinates:
        basis = ComparisonBasis(args.comparison_basis)
        invalid_root_roles = tuple(
            role
            for root, role in (
                (requested_root, SnapshotRole.REQUESTED_BASE),
                (comparison_root, SnapshotRole.COMPARISON_BASE),
                (head_root, SnapshotRole.HEAD),
            )
            if root is None
        )
        coordinates = DeclaredReviewCoordinates(
            requested_base_sha=args.requested_base_sha,
            comparison_base_sha=args.comparison_base_sha,
            head_sha=args.head_sha,
            comparison_basis=basis,
            invalid_root_roles=invalid_root_roles,
        )
        requested = _snapshot_or_none(
            requested_root,
            args.requested_base_sha,
            SnapshotRole.REQUESTED_BASE,
        )
        comparison = _snapshot_or_none(
            comparison_root,
            args.comparison_base_sha,
            SnapshotRole.COMPARISON_BASE,
        )
        head = _snapshot_or_none(
            head_root,
            args.head_sha,
            SnapshotRole.HEAD,
        )
        if invalid_root_roles:
            inventory = None
        elif requested is None or comparison is None or head is None:
            inventory_failure = ReviewInventoryFailure(
                code="PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"
            )
        else:
            try:
                inventory = build_git_change_inventory(
                    requested,
                    comparison,
                    head,
                    basis,
                )
            except InventoryVerificationError as exc:
                inventory_failure = ReviewInventoryFailure(code=exc.code)
            except ReviewError:
                inventory_failure = ReviewInventoryFailure(
                    code="PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED"
                )

    inputs = ReviewInputs(
        repository_root=repository_root,
        base_root=base_root,
        pr_title=pr_title,
        pr_description=pr_description,
        requested_base=requested,
        comparison_base=comparison,
        head=head,
        inventory=inventory,
        coordinates=coordinates,
        inventory_failure=inventory_failure,
    )

    # Keep this call singular: rendering both output formats must never rerun
    # provider calls or deterministic evidence discovery.
    result = (
        run_review(inputs, config, preflight_only=True)
        if args.preflight_only
        else run_review(inputs, config)
    )
    return _emit_review_result(args, result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        if args.json_output and args.markdown_output:
            parser.error("--json and --markdown are mutually exclusive")
        try:
            return _review_command(args)
        except ReviewError as exc:
            _write_console(sys.stderr, f"ClaimCI error: {exc}\n")
            return 2
    if args.json_output and args.markdown_output:
        parser.error("--json and --markdown are mutually exclusive")
    try:
        result = audit_research(args.research_file, artifact_root=args.artifact_root)
    except ClaimCIError as exc:
        _write_console(sys.stderr, f"ClaimCI error: {exc}\n")
        return 2
    if args.json_output:
        output = render_json(result)
    elif args.markdown_output:
        output = render_markdown(result)
    else:
        output = render_human(result)
    _write_console(sys.stdout, output)
    return EXIT_CODES[result.verdict]


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
