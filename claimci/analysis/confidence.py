"""Validated confidence values shared by ClaimCI analysis contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Confidence:
    """A finite confidence score in the closed interval from zero through one."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("confidence must be a number")
        normalized = float(self.value)
        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError("confidence must be finite from 0 through 1")
        object.__setattr__(self, "value", normalized)

    def __float__(self) -> float:
        return self.value


__all__ = ["Confidence"]
