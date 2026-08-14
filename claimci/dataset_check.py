"""Detect exact train/evaluation overlap using canonical JSON hashes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import (
    ClaimCIError,
    DatasetHashes,
    Finding,
    Impact,
    OverlapSummary,
    Severity,
)
from .parsing import unique_json_object, validate_json_graph


def canonicalize_sample(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ClaimCIError(f"sample cannot be represented as finite JSON: {exc}") from exc
    return serialized.encode("utf-8")


def hash_sample(value: Any) -> str:
    return hashlib.sha256(canonicalize_sample(value)).hexdigest()


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON value {token}")


def load_dataset(path: Path) -> DatasetHashes:
    try:
        path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ClaimCIError(f"invalid dataset path {path!r}: {exc}") from exc
    try:
        # JSON Lines records are separated by LF.  ``str.splitlines`` also
        # treats U+2028/U+2029 and several other Unicode controls as line
        # boundaries, which would split a valid JSON string containing one of
        # those characters.  CRLF remains valid because the trailing CR is
        # JSON whitespace on the record line.
        lines = path.read_text(encoding="utf-8").split("\n")
    except FileNotFoundError as exc:
        raise ClaimCIError(f"dataset file not found: {path}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise ClaimCIError(f"could not read dataset file {path}: {exc}") from exc

    hashes: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=unique_json_object,
            )
            validate_json_graph(sample, label=f"dataset {path}")
            hashes.append(hash_sample(sample))
        except (
            json.JSONDecodeError,
            ValueError,
            RecursionError,
            ClaimCIError,
        ) as exc:
            raise ClaimCIError(
                f"invalid JSON in dataset {path} at line {line_number}: {exc}"
            ) from exc
    if not hashes:
        raise ClaimCIError(f"dataset file {path} contains no JSON records")
    return DatasetHashes(path=path.resolve(), hashes=tuple(hashes))


def check_dataset_leakage(
    experiment: str,
    train: DatasetHashes,
    eval: DatasetHashes,
) -> tuple[OverlapSummary, list[Finding]]:
    intersection = Counter(train.hashes) & Counter(eval.hashes)
    overlap_count = sum(intersection.values())
    eval_count = len(eval.hashes)
    overlap_rate = overlap_count / eval_count if eval_count else 0.0
    summary = OverlapSummary(
        experiment=experiment,
        train_count=len(train.hashes),
        eval_count=eval_count,
        overlap_count=overlap_count,
        overlap_rate=overlap_rate,
        overlapping_hashes=tuple(sorted(intersection)),
    )
    if overlap_count:
        finding = Finding(
            rule_id="DATASET.EXACT_LEAKAGE",
            severity=Severity.CRITICAL,
            title=f"{experiment.title()} evaluation data appears in training data",
            explanation=(
                f"Found {overlap_count} exact canonical overlap(s), covering "
                f"{overlap_rate:.1%} of evaluation records."
            ),
            evidence={
                "experiment": experiment,
                "train_count": len(train.hashes),
                "eval_count": eval_count,
                "overlap_count": overlap_count,
                "overlap_rate": overlap_rate,
                "hashes": list(summary.overlapping_hashes),
            },
            impact=Impact.INVALIDATES,
        )
    else:
        finding = Finding(
            rule_id="DATASET.NO_LEAKAGE",
            severity=Severity.VERIFIED,
            title=f"{experiment.title()} has no exact train/eval leakage",
            explanation="No canonical evaluation record hash appears in the training split.",
            evidence={
                "experiment": experiment,
                "train_count": len(train.hashes),
                "eval_count": eval_count,
                "overlap_count": 0,
                "overlap_rate": 0.0,
            },
            impact=Impact.NONE,
        )
    return summary, [finding]


def _hash_count_evidence(counts: Counter[str]) -> list[dict[str, int | str]]:
    """Serialize a hash counter in deterministic digest order."""

    return [{"hash": digest, "count": counts[digest]} for digest in sorted(counts)]


def check_evaluation_alignment(
    baseline_eval: DatasetHashes,
    candidate_eval: DatasetHashes,
) -> list[Finding]:
    """Verify both experiments used the exact same evaluation-record multiset."""

    baseline_counts = Counter(baseline_eval.hashes)
    candidate_counts = Counter(candidate_eval.hashes)
    matching = baseline_counts & candidate_counts
    baseline_only = baseline_counts - candidate_counts
    candidate_only = candidate_counts - baseline_counts
    evidence = {
        "baseline_eval_count": len(baseline_eval.hashes),
        "candidate_eval_count": len(candidate_eval.hashes),
        "matching_count": sum(matching.values()),
        "baseline_only_count": sum(baseline_only.values()),
        "candidate_only_count": sum(candidate_only.values()),
        "baseline_only": _hash_count_evidence(baseline_only),
        "candidate_only": _hash_count_evidence(candidate_only),
    }

    if baseline_counts == candidate_counts:
        return [
            Finding(
                rule_id="DATASET.EVALUATION_ALIGNED",
                severity=Severity.VERIFIED,
                title="Baseline and candidate evaluation records are aligned",
                explanation=(
                    "Both evaluation datasets contain the same canonical record "
                    "hash multiset."
                ),
                evidence=evidence,
                impact=Impact.NONE,
            )
        ]

    return [
        Finding(
            rule_id="DATASET.EVALUATION_MISMATCH",
            severity=Severity.CRITICAL,
            title="Baseline and candidate evaluation records differ",
            explanation=(
                "A valid comparison requires both experiments to use the same "
                "canonical evaluation-record multiset."
            ),
            evidence=evidence,
            impact=Impact.INVALIDATES,
        )
    ]
