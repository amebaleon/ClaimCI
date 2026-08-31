"""Apply the claim-heavy review timeout patch once."""

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


def replace_all(relative: str, old: str, new: str, expected: int) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{relative}: expected {expected} matches, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def main() -> None:
    models = (ROOT / "claimci/review/models.py").read_text(encoding="utf-8")
    if "0 < timeout <= 120" in models:
        print("review timeout patch already applied")
        return

    replace_all(
        "claimci/review/models.py",
        "timeout_seconds must be a finite number from 0 through 30",
        "timeout_seconds must be a finite number from 0 through 120",
        3,
    )
    replace_once(
        "claimci/review/models.py",
        "if not math.isfinite(timeout) or not 0 < timeout <= 30:",
        "if not math.isfinite(timeout) or not 0 < timeout <= 120:",
    )

    replace_once(
        "claimci/review/openai_provider.py",
        "or not 0 < float(timeout_seconds) <= 30",
        "or not 0 < float(timeout_seconds) <= 120",
    )
    replace_once(
        "claimci/review/openai_provider.py",
        "OpenAI timeout must be finite and from 0 through 30",
        "OpenAI timeout must be finite and from 0 through 120",
    )

    replace_once(
        ".claimci/review.yaml",
        "timeout_seconds: 30",
        "timeout_seconds: 90",
    )
    replace_once(
        "README.md",
        "timeout_seconds: 30",
        "timeout_seconds: 90",
    )
    replace_once(
        "README.md",
        "recorded for observability only.\n",
        "recorded for observability only. The checked-in configuration allows 90 seconds per provider call; trusted configurations may choose any finite timeout up to 120 seconds.\n",
    )

    replace_once(
        "tests/test_review_provider_day3.py",
        'math.nan, math.inf, -math.inf, 31.0',
        'math.nan, math.inf, -math.inf, 121.0',
    )

    print("review timeout patch applied")


if __name__ == "__main__":
    main()
