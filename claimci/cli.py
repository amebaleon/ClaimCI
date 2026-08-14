"""Command-line interface for ClaimCI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .audit import audit_research
from .models import ClaimCIError, Verdict
from .report import render_human, render_json, render_markdown


EXIT_CODES = {
    Verdict.SUPPORTED: 0,
    Verdict.NOT_SUPPORTED: 1,
    Verdict.INSUFFICIENT_EVIDENCE: 2,
}


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
