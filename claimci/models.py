"""Stable data models shared by ClaimCI checks and output adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .measurement import MeasurementDriftReport


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    VERIFIED = "VERIFIED"
    INFO = "INFO"


class Impact(str, Enum):
    NONE = "NONE"
    INVALIDATES = "INVALIDATES"
    INSUFFICIENT = "INSUFFICIENT"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Direction(str, Enum):
    """Declared favorable direction for a metric.

    The wire values are deliberately lowercase and strict.  Manifest parsing
    is responsible for trimming surrounding whitespace before constructing the
    enum; direct enum construction therefore remains a useful guard against
    silently accepting non-canonical values such as ``"Higher"``.
    """

    HIGHER = "higher"
    LOWER = "lower"


class ClaimCIError(ValueError):
    """An expected, user-actionable input or artifact error."""


def coerce_direction(value: object, *, label: str = "direction") -> Direction:
    """Normalize a public direction value or raise a concise input error.

    Manifest and API callers may provide a string with surrounding whitespace;
    the normalized wire value is still required to be exactly ``higher`` or
    ``lower``.  Other scalar types (including booleans) are rejected rather
    than coerced through their string representation.
    """

    if isinstance(value, Direction):
        return value
    if isinstance(value, str):
        try:
            return Direction(value.strip())
        except ValueError:
            pass
    raise ClaimCIError(f"{label} must be either 'higher' or 'lower'")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    title: str
    explanation: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    impact: Impact = Impact.NONE


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    standard_deviation: float | None


@dataclass(frozen=True)
class DatasetHashes:
    path: Path
    hashes: tuple[str, ...]


@dataclass(frozen=True)
class OverlapSummary:
    experiment: str
    train_count: int
    eval_count: int
    overlap_count: int
    overlap_rate: float
    overlapping_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentPaths:
    config: Path
    results: Path
    train_dataset: Path
    eval_dataset: Path


@dataclass(frozen=True)
class ResearchSpec:
    manifest_path: Path
    metric: str
    minimum_improvement: float
    baseline: ExperimentPaths
    candidate: ExperimentPaths
    # Appended with a default so positional construction from Day 1 remains
    # valid and omitted manifest fields continue to mean higher-is-better.
    direction: Direction = Direction.HIGHER


@dataclass(frozen=True)
class AuditResult:
    verdict: Verdict
    findings: tuple[Finding, ...]
    manifest_path: Path = Path(".")
    metric: str = ""
    minimum_improvement: float = 0.0
    baseline_metrics: MetricSummary | None = None
    candidate_metrics: MetricSummary | None = None
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    overlaps: tuple[OverlapSummary, ...] = ()
    # Keep this field at the end to preserve the Day 1 positional shape.
    direction: Direction = Direction.HIGHER
    # Additive protocol companion. Legacy/direct Audit omits it completely.
    measurement_drift: "MeasurementDriftReport | None" = None
