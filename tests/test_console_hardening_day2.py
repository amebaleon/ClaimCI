"""Cross-platform output boundaries for artifact-controlled Unicode."""

from __future__ import annotations

import io
import json

from claimci.cli import _write_console
from claimci.report import render_json, render_markdown
from tests.test_markdown_day2 import _result


def test_machine_json_is_ascii_safe_and_round_trips_unicode_metric() -> None:
    report = render_json(_result(metric="🧪-score"))

    report.encode("ascii")
    assert json.loads(report)["claim"]["metric"] == "🧪-score"


def test_markdown_writes_safely_to_cp949_console() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp949", newline="")

    _write_console(stream, render_markdown(_result(metric="🧪-score")))
    stream.flush()
    encoded = raw.getvalue()
    stream.detach()

    decoded = encoded.decode("cp949")
    assert "ClaimCI Audit" in decoded
    assert "\\U0001f9ea-score" in decoded
