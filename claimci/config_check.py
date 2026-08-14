"""Configuration loading and fairness checks.

The configuration vocabulary is intentionally explicit for the Day 1 audit.
Unknown fields are preserved by :func:`load_config`, but only the supported
paths listed in the design participate in fairness findings.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, DecimalException, localcontext
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from .models import ClaimCIError, Finding, Impact, Severity
from .parsing import load_unique_yaml


_COMPUTE_RATIO_THRESHOLD = Decimal("1.5")
_MISSING = object()


def load_config(path: Path) -> Mapping[str, object]:
    """Load a YAML config and require a mapping at its root.

    Config parsing errors are surfaced as the public :class:`ClaimCIError`
    used by the audit layer.  The parsed mapping is returned without reducing
    it to the supported vocabulary; comparison itself is deliberately
    explicit in :func:`check_configs`.
    """

    try:
        config_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ClaimCIError(f"Invalid config path {path!r}: {exc}") from exc
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            parsed = load_unique_yaml(stream)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise ClaimCIError(f"Unable to load config {config_path}: {exc}") from exc

    if not isinstance(parsed, Mapping):
        raise ClaimCIError(f"Config root must be a mapping: {config_path}")
    return parsed


def _value(config: Mapping[str, object], *path: str) -> object:
    """Read a supported nested path, preserving missing-vs-null distinction."""

    current: object = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _decimal_number(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    try:
        numeric = Decimal(str(value))
    except (DecimalException, TypeError, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def _positive_number(value: object) -> Decimal | None:
    """Return finite positive Decimal values, excluding booleans."""

    numeric = _decimal_number(value)
    return numeric if numeric is not None and numeric > 0 else None


def _present_scalar(value: object) -> bool:
    if value is _MISSING or value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Real, Decimal)):
        return _decimal_number(value) is not None
    return False


def _missing_or_invalid_fields(config: Mapping[str, object]) -> list[str]:
    issues: list[str] = []
    for field in ("training_steps", "epochs", "batch_size"):
        if _positive_number(_value(config, field)) is None:
            issues.append(field)
    for path in (
        ("dataset", "identifier"),
        ("dataset", "version"),
        ("evaluation", "dataset_identifier"),
        ("evaluation", "dataset_version"),
        ("evaluation", "split"),
        ("model",),
    ):
        if not _present_scalar(_value(config, *path)):
            issues.append(".".join(path))
    if _positive_number(_value(config, "learning_rate")) is None:
        issues.append("learning_rate")
    return issues


def _evaluation_mapping_complete(config: Mapping[str, object]) -> bool:
    """Return whether the supported evaluation mapping is fully declared."""

    evaluation = _value(config, "evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    return all(
        _present_scalar(evaluation.get(field))
        for field in ("dataset_identifier", "dataset_version", "split")
    )


def _compute_proxies(config: Mapping[str, object]) -> dict[str, Decimal]:
    """Return every available proxy so both arms can choose one common basis."""

    training_steps = _positive_number(_value(config, "training_steps"))
    epochs = _positive_number(_value(config, "epochs"))
    batch_size = _positive_number(_value(config, "batch_size"))

    proxies: dict[str, Decimal] = {}
    if training_steps is not None and batch_size is not None:
        proxies["training_steps*batch_size"] = _multiply_exact(
            training_steps, batch_size
        )
    if training_steps is not None:
        proxies["training_steps"] = training_steps
    if epochs is not None and batch_size is not None:
        proxies["epochs*batch_size"] = _multiply_exact(epochs, batch_size)
    if epochs is not None:
        proxies["epochs"] = epochs
    return proxies


def _multiply_exact(left: Decimal, right: Decimal) -> Decimal:
    """Multiply finite decimals without the process-global 28-digit rounding."""

    precision = max(1, len(left.as_tuple().digits) + len(right.as_tuple().digits))
    with localcontext() as context:
        context.prec = precision
        return left * right


def _ratio_for_evidence(candidate: Decimal, baseline: Decimal) -> Decimal:
    """Compute enough ratio digits to make near-boundary evidence honest."""

    precision = max(
        50,
        len(candidate.as_tuple().digits) + len(baseline.as_tuple().digits) + 10,
    )
    with localcontext() as context:
        context.prec = precision
        return candidate / baseline


def _evidence_number(value: Decimal | None) -> float | str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return str(value)
    if not isfinite(number) or Decimal(str(number)) != value:
        return str(value)
    return number


def _at_least_ratio(
    candidate: Decimal, baseline: Decimal, threshold: Decimal
) -> bool:
    """Compare a positive Decimal ratio without context rounding.

    Decimal division obeys the active precision and can round a value just
    below the policy boundary up to exactly 1.5. Integer cross-products from
    ``as_integer_ratio`` preserve the declared numbers exactly.
    """

    candidate_num, candidate_den = candidate.as_integer_ratio()
    baseline_num, baseline_den = baseline.as_integer_ratio()
    threshold_num, threshold_den = threshold.as_integer_ratio()
    return (
        candidate_num * baseline_den * threshold_den
        >= threshold_num * candidate_den * baseline_num
    )


def _finding(
    *,
    rule_id: str,
    severity: Severity,
    title: str,
    explanation: str,
    evidence: Mapping[str, object],
    impact: Impact,
) -> Finding:
    """Construct a finding while keeping every call site's policy explicit."""

    return Finding(
        rule_id=rule_id,
        severity=severity,
        title=title,
        explanation=explanation,
        evidence=dict(evidence),
        impact=impact,
    )


def check_configs(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> list[Finding]:
    """Compare supported fairness fields and return deterministic findings.

    The candidate's compute proxy is the only comparison that can invalidate
    solely because of compute.  Other explicit differences remain visible but
    have ``Impact.NONE`` according to the Day 1 policy.
    """

    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("baseline and candidate configs must be mappings")

    findings: list[Finding] = []

    baseline_field_issues = _missing_or_invalid_fields(baseline)
    candidate_field_issues = _missing_or_invalid_fields(candidate)
    if baseline_field_issues or candidate_field_issues:
        findings.append(
            _finding(
                rule_id="CONFIG.MISSING_FIELDS",
                severity=Severity.WARNING,
                title="Required fairness config fields are missing or invalid",
                explanation=(
                    "ClaimCI cannot establish config comparability without every "
                    "documented Day 1 field."
                ),
                evidence={
                    "baseline": baseline_field_issues,
                    "candidate": candidate_field_issues,
                },
                impact=Impact.INSUFFICIENT,
            )
        )

    baseline_proxies = _compute_proxies(baseline)
    candidate_proxies = _compute_proxies(candidate)
    basis = next(
        (
            name
            for name in (
                "training_steps*batch_size",
                "training_steps",
                "epochs*batch_size",
                "epochs",
            )
            if name in baseline_proxies and name in candidate_proxies
        ),
        None,
    )
    baseline_proxy = baseline_proxies.get(basis) if basis is not None else None
    candidate_proxy = candidate_proxies.get(basis) if basis is not None else None
    ratio_decimal = (
        _ratio_for_evidence(candidate_proxy, baseline_proxy)
        if baseline_proxy is not None and candidate_proxy is not None
        else None
    )
    ratio = _evidence_number(ratio_decimal)
    compute_mismatch = (
        baseline_proxy is not None
        and candidate_proxy is not None
        and _at_least_ratio(
            candidate_proxy, baseline_proxy, _COMPUTE_RATIO_THRESHOLD
        )
    )

    if compute_mismatch:
        findings.append(
            _finding(
                rule_id="CONFIG.COMPUTE_MISMATCH",
                severity=Severity.CRITICAL,
                title="Candidate compute proxy is too large",
                explanation=(
                    "The candidate compute proxy is at least 1.5x the baseline "
                    "proxy."
                ),
                evidence={
                    "baseline_proxy": _evidence_number(baseline_proxy),
                    "candidate_proxy": _evidence_number(candidate_proxy),
                    "ratio": ratio,
                    "basis": basis,
                    "baseline_basis": basis,
                    "candidate_basis": basis,
                    "threshold": float(_COMPUTE_RATIO_THRESHOLD),
                    "exact_threshold_comparison": True,
                },
                impact=Impact.INVALIDATES,
            )
        )

    evaluation_baseline = _value(baseline, "evaluation")
    evaluation_candidate = _value(candidate, "evaluation")
    # A missing/invalid mapping already produces CONFIG.MISSING_FIELDS with an
    # insufficient-evidence impact.  Do not infer a proven held-out mismatch
    # from the missing value itself; only compare mappings once both sides
    # declare every supported evaluation field.
    if (
        _evaluation_mapping_complete(baseline)
        and _evaluation_mapping_complete(candidate)
        and evaluation_baseline != evaluation_candidate
    ):
        findings.append(
            _finding(
                rule_id="CONFIG.EVALUATION_MISMATCH",
                severity=Severity.CRITICAL,
                title="Evaluation mapping differs",
                explanation="Baseline and candidate evaluation mappings must match.",
                evidence={
                    "baseline": None
                    if evaluation_baseline is _MISSING
                    else evaluation_baseline,
                    "candidate": None
                    if evaluation_candidate is _MISSING
                    else evaluation_candidate,
                },
                impact=Impact.INVALIDATES,
            )
        )

    baseline_dataset = {
        "identifier": _value(baseline, "dataset", "identifier"),
        "version": _value(baseline, "dataset", "version"),
    }
    candidate_dataset = {
        "identifier": _value(candidate, "dataset", "identifier"),
        "version": _value(candidate, "dataset", "version"),
    }
    if baseline_dataset != candidate_dataset:
        findings.append(
            _finding(
                rule_id="CONFIG.TRAINING_DATASET_MISMATCH",
                severity=Severity.WARNING,
                title="Training dataset differs",
                explanation=(
                    "Training dataset identifier or version differs between "
                    "baseline and candidate."
                ),
                evidence={
                    "baseline": {
                        key: None if value is _MISSING else value
                        for key, value in baseline_dataset.items()
                    },
                    "candidate": {
                        key: None if value is _MISSING else value
                        for key, value in candidate_dataset.items()
                    },
                },
                impact=Impact.NONE,
            )
        )

    parameter_fields = ("training_steps", "epochs", "batch_size")
    parameter_differences = {
        field: (
            _value(baseline, field),
            _value(candidate, field),
        )
        for field in parameter_fields
        if _value(baseline, field) != _value(candidate, field)
    }
    # A large candidate compute advantage is already represented by the
    # critical compute rule.  The training-parameter warning is specifically
    # for differences that remain below that gating threshold (or where a
    # comparable ratio cannot be computed).
    if parameter_differences and not compute_mismatch:
        findings.append(
            _finding(
                rule_id="CONFIG.TRAINING_PARAMETER_DIFFERENCE",
                severity=Severity.WARNING,
                title="Training parameters differ",
                explanation=(
                    "Training steps, epochs, or batch size differ without "
                    "reaching the compute invalidation threshold."
                ),
                evidence={
                    "differences": {
                        field: {
                            "baseline": None if values[0] is _MISSING else values[0],
                            "candidate": None if values[1] is _MISSING else values[1],
                        }
                        for field, values in parameter_differences.items()
                    },
                    "baseline_proxy": _evidence_number(baseline_proxy),
                    "candidate_proxy": _evidence_number(candidate_proxy),
                    "ratio": ratio,
                    "basis": basis,
                },
                impact=Impact.NONE,
            )
        )

    baseline_learning_rate = _value(baseline, "learning_rate")
    candidate_learning_rate = _value(candidate, "learning_rate")
    if baseline_learning_rate != candidate_learning_rate:
        findings.append(
            _finding(
                rule_id="CONFIG.LEARNING_RATE_DIFFERENCE",
                severity=Severity.WARNING,
                title="Learning rate differs",
                explanation="Baseline and candidate learning rates differ.",
                evidence={
                    "baseline": None
                    if baseline_learning_rate is _MISSING
                    else baseline_learning_rate,
                    "candidate": None
                    if candidate_learning_rate is _MISSING
                    else candidate_learning_rate,
                },
                impact=Impact.NONE,
            )
        )

    baseline_model = _value(baseline, "model")
    candidate_model = _value(candidate, "model")
    if baseline_model != candidate_model:
        findings.append(
            _finding(
                rule_id="CONFIG.MODEL_DIFFERENCE",
                severity=Severity.INFO,
                title="Model differs",
                explanation="Baseline and candidate model identifiers differ.",
                evidence={
                    "baseline": None if baseline_model is _MISSING else baseline_model,
                    "candidate": None if candidate_model is _MISSING else candidate_model,
                },
                impact=Impact.NONE,
            )
        )

    if not findings:
        findings.append(
            _finding(
                rule_id="CONFIG.FAIRNESS_VERIFIED",
                severity=Severity.VERIFIED,
                title="Supported config fields match",
                explanation="All supported fairness configuration fields match.",
                evidence={
                    "baseline_proxy": _evidence_number(baseline_proxy),
                    "candidate_proxy": _evidence_number(candidate_proxy),
                    "ratio": ratio,
                    "basis": basis,
                },
                impact=Impact.NONE,
            )
        )

    return findings


__all__ = ["load_config", "check_configs"]
