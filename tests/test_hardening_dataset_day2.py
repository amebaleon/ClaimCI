"""Adversarial Day 2 regressions for exact JSONL leakage boundaries.

These tests intentionally stay at the dataset checker boundary.  ClaimCI's
contract is exact canonical JSON equality within each experiment's train/eval
pair; it is not a semantic similarity detector and it must not treat records
shared by two experiments as leakage by themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimci.audit import ClaimCIError
from claimci.dataset_check import check_dataset_leakage, load_dataset


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _write_jsonl(path: Path, *records: object, newline: str = "\n") -> Path:
    payload = newline.join(json.dumps(record, ensure_ascii=False) for record in records)
    return _write_text(path, payload + newline)


def test_crlf_and_jsonl_whitespace_do_not_change_canonical_hashes(tmp_path: Path) -> None:
    """Transport line endings/indentation are not part of a sample identity."""

    crlf = _write_text(
        tmp_path / "crlf.jsonl",
        '  { "text": "café", "items": [1, 2] }  \r\n\t{"id": "two"}\r\n',
    )
    lf = _write_text(
        tmp_path / "lf.jsonl",
        '{"items":[1,2],"text":"café"}\n{"id":"two"}\n',
    )

    assert load_dataset(crlf).hashes == load_dataset(lf).hashes


def test_json_number_spellings_follow_documented_canonicalization_contract(
    tmp_path: Path,
) -> None:
    """JSON ``1`` and ``1.0`` remain distinct; equivalent float spellings match."""

    integer = load_dataset(_write_jsonl(tmp_path / "integer.jsonl", {"value": 1}))
    decimal = load_dataset(_write_jsonl(tmp_path / "decimal.jsonl", {"value": 1.0}))
    exponent = _write_text(tmp_path / "exponent.jsonl", '{"value":1e0}\n')

    assert integer.hashes != decimal.hashes
    assert decimal.hashes == load_dataset(exponent).hashes


@pytest.mark.parametrize(
    ("train_records", "eval_records"),
    [
        (({"id": "duplicate"}, {"id": "duplicate"}), ({"id": "eval-only"},)),
        (({"id": "train-only"},), ({"id": "duplicate"}, {"id": "duplicate"})),
    ],
)
def test_duplicates_confined_to_one_split_are_not_leakage(
    tmp_path: Path,
    train_records: tuple[object, ...],
    eval_records: tuple[object, ...],
) -> None:
    """Repeated records count toward split sizes but cannot overlap the other split."""

    train = load_dataset(_write_jsonl(tmp_path / "train.jsonl", *train_records))
    evaluation = load_dataset(_write_jsonl(tmp_path / "eval.jsonl", *eval_records))

    summary, findings = check_dataset_leakage("candidate", train, evaluation)

    assert summary.train_count == len(train_records)
    assert summary.eval_count == len(eval_records)
    assert summary.overlap_count == 0
    assert summary.overlap_rate == 0.0
    assert summary.overlapping_hashes == ()
    assert [finding.rule_id for finding in findings] == ["DATASET.NO_LEAKAGE"]


def test_multiset_overlap_hashes_and_evidence_are_sorted_and_repeatable(
    tmp_path: Path,
) -> None:
    """Counter intersection and finding evidence remain deterministic despite order."""

    train_records = ({"id": "z"}, {"id": "a"}, {"id": "a"})
    eval_records = ({"id": "a"}, {"id": "z"}, {"id": "a"}, {"id": "z"})
    train = load_dataset(_write_jsonl(tmp_path / "train.jsonl", *train_records))
    evaluation = load_dataset(_write_jsonl(tmp_path / "eval.jsonl", *eval_records))

    first_summary, first_findings = check_dataset_leakage("candidate", train, evaluation)
    second_summary, second_findings = check_dataset_leakage("candidate", train, evaluation)

    assert first_summary == second_summary
    assert first_findings == second_findings
    assert first_summary.overlap_count == 3
    assert first_summary.overlap_rate == pytest.approx(3 / 4)
    assert first_summary.overlapping_hashes == tuple(sorted(first_summary.overlapping_hashes))
    assert first_findings[0].evidence["hashes"] == list(first_summary.overlapping_hashes)


def test_shared_training_data_between_experiments_is_not_cross_experiment_leakage(
    tmp_path: Path,
) -> None:
    """Only baseline-train↔baseline-eval and candidate-train↔candidate-eval are checked."""

    shared_train_path = _write_jsonl(
        tmp_path / "shared-train.jsonl",
        {"id": "shared-1"},
        {"id": "shared-2"},
    )
    baseline_eval_path = _write_jsonl(tmp_path / "baseline-eval.jsonl", {"id": "baseline"})
    candidate_eval_path = _write_jsonl(tmp_path / "candidate-eval.jsonl", {"id": "candidate"})
    shared_train = load_dataset(shared_train_path)

    baseline_summary, baseline_findings = check_dataset_leakage(
        "baseline", shared_train, load_dataset(baseline_eval_path)
    )
    candidate_summary, candidate_findings = check_dataset_leakage(
        "candidate", shared_train, load_dataset(candidate_eval_path)
    )

    assert baseline_summary.overlap_count == candidate_summary.overlap_count == 0
    assert [finding.rule_id for finding in baseline_findings] == ["DATASET.NO_LEAKAGE"]
    assert [finding.rule_id for finding in candidate_findings] == ["DATASET.NO_LEAKAGE"]


def test_invalid_utf8_dataset_is_a_controlled_input_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid-utf8.jsonl"
    dataset_path.write_bytes(b'{"id":"ok"}\n\xff\n')

    with pytest.raises(ClaimCIError, match=r"(?i)could not read|utf|dataset") as exc_info:
        load_dataset(dataset_path)

    assert str(dataset_path) in str(exc_info.value)


def test_directory_dataset_path_is_a_controlled_input_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset-directory"
    dataset_path.mkdir()

    with pytest.raises(ClaimCIError, match=r"(?i)could not read|dataset|directory") as exc_info:
        load_dataset(dataset_path)

    assert str(dataset_path) in str(exc_info.value)


def test_missing_dataset_path_is_a_controlled_input_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "does-not-exist.jsonl"

    with pytest.raises(ClaimCIError, match=r"(?i)not found|dataset") as exc_info:
        load_dataset(dataset_path)

    assert str(dataset_path) in str(exc_info.value)


def test_duplicate_json_object_keys_are_rejected_as_ambiguous_samples(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "duplicate-key.jsonl"
    dataset_path.write_text(
        '{"id":"first","id":"second","text":"sample"}\n', encoding="utf-8"
    )

    with pytest.raises(ClaimCIError, match=r"(?i)duplicate|json|dataset"):
        load_dataset(dataset_path)


def test_deeply_nested_jsonl_is_a_controlled_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "deep.jsonl"
    dataset_path.write_text("[" * 2_000 + "0" + "]" * 2_000 + "\n", encoding="utf-8")

    with pytest.raises(ClaimCIError, match=r"(?i)nested|recursion|json|dataset"):
        load_dataset(dataset_path)


@pytest.mark.parametrize("bad_path", [None, 1, object()])
def test_non_path_dataset_inputs_are_controlled_errors(bad_path: object) -> None:
    with pytest.raises(ClaimCIError, match=r"(?i)path|dataset"):
        load_dataset(bad_path)  # type: ignore[arg-type]


def test_unicode_line_separator_inside_json_string_is_not_a_jsonl_boundary(
    tmp_path: Path,
) -> None:
    literal = tmp_path / "literal-separator.jsonl"
    escaped = tmp_path / "escaped-separator.jsonl"
    literal.write_text('{"text":"left\u2028right"}\n', encoding="utf-8")
    escaped.write_text('{"text":"left\\u2028right"}\n', encoding="utf-8")

    assert load_dataset(literal).hashes == load_dataset(escaped).hashes
