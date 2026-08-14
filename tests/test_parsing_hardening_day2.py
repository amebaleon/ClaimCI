"""Complexity boundaries for strict shared artifact parsing."""

from __future__ import annotations

import pytest
import yaml

from claimci.parsing import _validate_yaml_graph


class _CountingList(list[object]):
    def __init__(self, values: list[object]) -> None:
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def test_acyclic_shared_yaml_alias_graph_is_rejected_before_later_expansion() -> None:
    shared = _CountingList([{"leaf": 1}])
    root: object = [shared, shared, {"again": shared}]

    with pytest.raises(yaml.YAMLError, match="alias"):
        _validate_yaml_graph(root)

    assert shared.iterations == 1
