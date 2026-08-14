"""Contract tests for exact JSONL canonicalization and leakage checks.

The dataset checker is deliberately byte-oriented: parsed JSON values are
canonicalized before hashing, and train/evaluation overlap is counted with a
multiset intersection.  These tests keep those rules independent of the
filesystem order or the presentation layer.
"""

from pathlib import Path

import pytest

import claimci.dataset_check as dataset_check
from claimci.audit import ClaimCIError
from claimci.dataset_check import (
    canonicalize_sample,
    check_dataset_leakage,
    hash_sample,
    load_dataset,
)
from claimci.models import Impact, Severity


KNOWN_SAMPLE_HASH = (
    "e2e557498c1d71d2dc357c72fdc86fa0b8a04f4717fc3a1a152ffa00eaf3a90d"
)
SAMPLE_A_HASH = (
    "8489a5deb454a360345c7868bca8672de92b446caf3d3b014af6a56e3d549d30"
)
SAMPLE_B_HASH = (
    "84a91dee31459ddf46933a42288dfd0ff0fb2a3aae2cc0e6ef84135c8cdd5f71"
)
SAMPLE_C_HASH = (
    "cf9b3160c4a5a9fac6789866c5920602fc6bb2cae40553b3a1641a2f3dbd2bd6"
)


def _write_jsonl(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_pair(
    tmp_path: Path,
    train_lines: tuple[str, ...],
    evaluation_lines: tuple[str, ...],
):
    train_path = _write_jsonl(tmp_path / "train.jsonl", *train_lines)
    evaluation_path = _write_jsonl(tmp_path / "eval.jsonl", *evaluation_lines)
    return load_dataset(train_path), load_dataset(evaluation_path)


def _finding_signature(findings):
    """Return the stable policy fields exposed in every finding."""

    return [(finding.rule_id, finding.severity, finding.impact) for finding in findings]


def test_known_sample_uses_sorted_compact_utf8_json_before_sha256(
    tmp_path: Path,
) -> None:
    sample = {"id": "a", "value": 1}

    assert canonicalize_sample(sample) == b'{"id":"a","value":1}'
    assert hash_sample(sample) == KNOWN_SAMPLE_HASH

    dataset_path = _write_jsonl(
        tmp_path / "sample.jsonl", '{"value":1,"id":"a"}'
    )
    dataset = load_dataset(dataset_path)
    assert dataset.path == dataset_path.resolve()
    assert dataset.hashes == (KNOWN_SAMPLE_HASH,)


def test_object_key_order_and_jsonl_whitespace_do_not_change_hash(tmp_path: Path) -> None:
    # The two textual records parse to the same object but exercise both the
    # JSONL loader and canonical serializer.
    train, evaluation = _load_pair(
        tmp_path,
        (' { "value": 1, "id": "a" } ',),
        ('{"id":"a","value":1}',),
    )

    summary, findings = check_dataset_leakage("baseline", train, evaluation)

    assert summary.train_count == 1
    assert summary.eval_count == 1
    assert summary.overlap_count == 1
    assert summary.overlap_rate == pytest.approx(1.0)
    assert _finding_signature(findings) == [
        ("DATASET.EXACT_LEAKAGE", Severity.CRITICAL, Impact.INVALIDATES)
    ]


def test_array_order_is_preserved_during_canonicalization() -> None:
    first = ["left", "right"]
    reversed_order = ["right", "left"]

    assert canonicalize_sample(first) == b'["left","right"]'
    assert hash_sample(first) != hash_sample(reversed_order)


def test_duplicate_aware_overlap_uses_counter_intersection_and_eval_denominator(
    tmp_path: Path,
) -> None:
    sample_a = '{"id":"a"}'
    sample_b = '{"id":"b"}'
    sample_c = '{"id":"c"}'
    # A occurs twice in train and three times in evaluation.  The multiset
    # intersection therefore contributes two records, not three or one.
    train, evaluation = _load_pair(
        tmp_path,
        (sample_a, sample_a, sample_b),
        (sample_a, sample_a, sample_a, sample_c),
    )

    summary, findings = check_dataset_leakage("candidate", train, evaluation)

    assert summary.experiment == "candidate"
    assert summary.train_count == 3
    assert summary.eval_count == 4
    assert summary.overlap_count == 2
    assert summary.overlap_rate == pytest.approx(0.5)
    assert _finding_signature(findings) == [
        ("DATASET.EXACT_LEAKAGE", Severity.CRITICAL, Impact.INVALIDATES)
    ]


def test_no_overlap_emits_verified_finding_with_no_verdict_impact(tmp_path: Path) -> None:
    train, evaluation = _load_pair(
        tmp_path,
        ('{"id":"train"}',),
        ('{"id":"eval"}',),
    )

    summary, findings = check_dataset_leakage("baseline", train, evaluation)

    assert summary.overlap_count == 0
    assert summary.overlap_rate == pytest.approx(0.0)
    assert _finding_signature(findings) == [
        ("DATASET.NO_LEAKAGE", Severity.VERIFIED, Impact.NONE)
    ]


def test_evaluation_alignment_accepts_same_duplicate_multiset_in_any_order(
    tmp_path: Path,
) -> None:
    baseline = load_dataset(
        _write_jsonl(
            tmp_path / "baseline-eval.jsonl",
            '{"id":"a"}',
            '{"id":"a"}',
            '{"id":"b"}',
        )
    )
    candidate = load_dataset(
        _write_jsonl(
            tmp_path / "candidate-eval.jsonl",
            '{ "id": "b" }',
            '{"id":"a"}',
            '{ "id": "a" }',
        )
    )

    findings = dataset_check.check_evaluation_alignment(baseline, candidate)

    assert _finding_signature(findings) == [
        ("DATASET.EVALUATION_ALIGNED", Severity.VERIFIED, Impact.NONE)
    ]
    assert findings[0].evidence == {
        "baseline_eval_count": 3,
        "candidate_eval_count": 3,
        "matching_count": 3,
        "baseline_only_count": 0,
        "candidate_only_count": 0,
        "baseline_only": [],
        "candidate_only": [],
    }


def test_evaluation_alignment_rejects_duplicate_multiplicity_with_sorted_evidence(
    tmp_path: Path,
) -> None:
    baseline = load_dataset(
        _write_jsonl(
            tmp_path / "baseline-eval.jsonl",
            '{"id":"c"}',
            '{"id":"a"}',
            '{"id":"a"}',
        )
    )
    candidate = load_dataset(
        _write_jsonl(
            tmp_path / "candidate-eval.jsonl",
            '{"id":"b"}',
            '{"id":"a"}',
            '{"id":"b"}',
        )
    )

    findings = dataset_check.check_evaluation_alignment(baseline, candidate)

    assert _finding_signature(findings) == [
        ("DATASET.EVALUATION_MISMATCH", Severity.CRITICAL, Impact.INVALIDATES)
    ]
    assert findings[0].evidence == {
        "baseline_eval_count": 3,
        "candidate_eval_count": 3,
        "matching_count": 1,
        "baseline_only_count": 2,
        "candidate_only_count": 2,
        "baseline_only": [
            {"hash": SAMPLE_A_HASH, "count": 1},
            {"hash": SAMPLE_C_HASH, "count": 1},
        ],
        "candidate_only": [{"hash": SAMPLE_B_HASH, "count": 2}],
    }


def test_blank_jsonl_lines_are_ignored(tmp_path: Path) -> None:
    train, evaluation = _load_pair(
        tmp_path,
        ("", "   ", '{"id":"a"}', "\t"),
        ("", '{"id":"a"}', "  "),
    )

    summary, findings = check_dataset_leakage("baseline", train, evaluation)

    assert summary.train_count == 1
    assert summary.eval_count == 1
    assert summary.overlap_count == 1
    assert findings[0].rule_id == "DATASET.EXACT_LEAKAGE"


def test_dataset_with_only_blank_lines_is_not_verified(tmp_path: Path) -> None:
    dataset_path = _write_jsonl(tmp_path / "empty.jsonl", "", "   ", "\t")

    with pytest.raises(ClaimCIError, match=r"(?i)no JSON records|empty"):
        load_dataset(dataset_path)


def test_nul_dataset_path_is_a_controlled_input_error() -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)dataset|path|read"):
        load_dataset(Path("\0"))


def test_malformed_json_reports_dataset_path_and_one_based_line_number(
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "broken.jsonl",
        '{"id":"ok"}',
        '{"id":',
    )

    with pytest.raises(ClaimCIError) as exc_info:
        load_dataset(dataset_path)

    message = str(exc_info.value)
    assert str(dataset_path) in message
    assert "line 2" in message


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_rejected(tmp_path: Path, literal: str) -> None:
    dataset_path = _write_jsonl(tmp_path / "nonfinite.jsonl", f'{{"value":{literal}}}')

    with pytest.raises(ClaimCIError, match=r"(?i)finite|nan|infinity"):
        load_dataset(dataset_path)


def test_json_decoder_recursion_is_a_controlled_dataset_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_jsonl(tmp_path / "recursive.jsonl", '{"id":"a"}')

    def exceed_recursion(*args, **kwargs):
        raise RecursionError("decoder nesting limit")

    monkeypatch.setattr(dataset_check.json, "loads", exceed_recursion)

    with pytest.raises(ClaimCIError, match=r"(?i)invalid JSON|recursion|nesting"):
        load_dataset(dataset_path)


def test_json_canonicalization_recursion_is_a_controlled_dataset_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_jsonl(tmp_path / "recursive.jsonl", '{"id":"a"}')
    real_loads = dataset_check.json.loads

    def exceed_recursion(*args, **kwargs):
        raise RecursionError("serializer nesting limit")

    monkeypatch.setattr(dataset_check.json, "dumps", exceed_recursion)
    # Loading still reaches canonicalization: json.loads is a distinct object
    # captured before the module-level json.dumps replacement above.
    monkeypatch.setattr(dataset_check.json, "loads", real_loads)

    with pytest.raises(ClaimCIError, match=r"(?i)finite JSON|recursion|nesting"):
        load_dataset(dataset_path)
