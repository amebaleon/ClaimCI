"""Compatibility adapters for existing ClaimCI manifest/result/config shapes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactKind,
    Confidence,
    ComputeEvidence,
    ConfigValue,
    DatasetReference,
    ExperimentRole,
    FieldMapping,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
)

from .config import _parse_yaml
from .core import (
    MAX_LOGICAL_RECORD_BYTES,
    MAX_RECORDS,
    AdapterLimitError,
    AdapterParseError,
    AdapterSelectorError,
    _adapter_provenance,
    _config_target,
    _evidence_id,
    _finite_number,
    _mapping,
    _resolve_dotted_path,
    _scalar_leaves,
    _supports,
    _validate_match,
    _verify_integrity,
)
from .structured import (
    _common_leaves,
    _dotted_from_pointer,
    _observation,
    _pointer_leaves,
    _result_mappings,
)


_MANIFEST_FIELDS = (
    "claim.metric",
    "claim.minimum_improvement",
    "baseline.config",
    "baseline.results",
    "baseline.train_dataset",
    "baseline.eval_dataset",
    "candidate.config",
    "candidate.results",
    "candidate.train_dataset",
    "candidate.eval_dataset",
)
_RESULT_TARGETS = frozenset({"metric_name", "metric_value", "run_id", "seed"})
_COMPUTE_INPUTS = frozenset({"training_steps", "epochs", "batch_size"})


def _manifest_mapping(
    artifact: PassiveArtifact, target: str, selector: str
) -> FieldMapping:
    return _mapping(
        artifact,
        adapter_id="claimci-native-manifest-v1",
        target_field=target,
        selector_kind=SelectorKind.DOTTED_PATH,
        selector=selector,
        inferred=False,
        provenance_kind=ProvenanceKind.MANIFEST_HINT,
    )


def _manifest_value(root: Mapping[str, object], selector: str) -> object:
    try:
        return _resolve_dotted_path(root, selector)
    except AdapterSelectorError as exc:
        raise AdapterParseError(
            f"native research manifest is missing {selector!r}"
        ) from exc


def _resolve_manifest_path(manifest_path: RepositoryPath, declared: object) -> RepositoryPath:
    if not isinstance(declared, str) or not declared or declared != declared.strip():
        raise AdapterParseError("manifest artifact path must be non-empty portable text")
    if (
        "\\" in declared
        or "\x00" in declared
        or any(ord(character) < 32 or ord(character) == 127 for character in declared)
        or PurePosixPath(declared).is_absolute()
        or PureWindowsPath(declared).is_absolute()
        or bool(PureWindowsPath(declared).drive)
    ):
        raise AdapterParseError("manifest artifact path must be portable and relative")

    stack = list(PurePosixPath(str(manifest_path)).parent.parts)
    for segment in declared.split("/"):
        if segment in {"", "."}:
            if segment == "":
                raise AdapterParseError("manifest artifact path is not canonical")
            continue
        if segment == "..":
            if not stack:
                raise AdapterParseError("manifest artifact path would escape repository")
            stack.pop()
            continue
        try:
            RepositoryPath(segment)
        except (TypeError, ValueError) as exc:
            raise AdapterParseError("manifest artifact path is not portable") from exc
        stack.append(segment)
    if not stack:
        raise AdapterParseError("manifest artifact path must resolve to a repository file")
    try:
        return RepositoryPath("/".join(stack))
    except (TypeError, ValueError) as exc:
        raise AdapterParseError("manifest artifact path is not repository-confined") from exc


def _parse_manifest(content: bytes) -> Mapping[str, object]:
    root = _parse_yaml(content)
    metric = _manifest_value(root, "claim.metric")
    minimum = _manifest_value(root, "claim.minimum_improvement")
    if not isinstance(metric, str) or not metric.strip():
        raise AdapterParseError("manifest claim.metric must be non-empty text")
    minimum_value = _finite_number(
        minimum, label="manifest claim.minimum_improvement"
    )
    if minimum_value < 0:
        raise AdapterParseError(
            "manifest claim.minimum_improvement must be non-negative"
        )
    for selector in _MANIFEST_FIELDS[2:]:
        if not isinstance(_manifest_value(root, selector), str):
            raise AdapterParseError(f"manifest {selector} must be a path string")
    direction = root.get("claim")
    if isinstance(direction, Mapping) and "direction" in direction:
        if direction["direction"] not in {"higher", "lower"}:
            raise AdapterParseError("manifest claim.direction must be higher or lower")
    return root


def _manifest_provenance(
    artifact: PassiveArtifact, selector: str
):
    return _adapter_provenance(
        artifact,
        adapter_id="claimci-native-manifest-v1",
        selector=selector,
        inferred=False,
        kind=ProvenanceKind.MANIFEST_HINT,
    )


def _experiment_observation(
    artifact: PassiveArtifact,
    root: Mapping[str, object],
    *,
    name: str,
    role: ExperimentRole,
) -> NormalizedObservation:
    paths = {
        field: _resolve_manifest_path(
            artifact.candidate.path,
            _manifest_value(root, f"{name}.{field}"),
        )
        for field in ("config", "results", "train_dataset", "eval_dataset")
    }
    return NormalizedObservation(
        provenance=_manifest_provenance(artifact, name),
        experiment_role=role,
        config_values=(
            ConfigValue(
                "config",
                str(paths["config"]),
                _manifest_provenance(artifact, f"{name}.config"),
            ),
            ConfigValue(
                "results",
                str(paths["results"]),
                _manifest_provenance(artifact, f"{name}.results"),
            ),
        ),
        dataset_references=(
            DatasetReference(
                paths["train_dataset"],
                "train",
                _manifest_provenance(artifact, f"{name}.train_dataset"),
            ),
            DatasetReference(
                paths["eval_dataset"],
                "eval",
                _manifest_provenance(artifact, f"{name}.eval_dataset"),
            ),
        ),
    )


class NativeManifestAdapter:
    """Translate explicit research.yaml declarations into manifest hints."""

    __slots__ = ()
    adapter_id = "claimci-native-manifest-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=frozenset({ArtifactKind.MANIFEST}),
            suffixes=frozenset({".yaml", ".yml"}),
        ) or PurePosixPath(str(artifact.candidate.path)).name.lower() not in {
            "research.yaml",
            "research.yml",
        }:
            return None
        _verify_integrity(artifact)
        root = _parse_manifest(artifact.content)
        for selector in _MANIFEST_FIELDS[2:]:
            _resolve_manifest_path(
                artifact.candidate.path,
                _manifest_value(root, selector),
            )
        selectors = list(_MANIFEST_FIELDS)
        claim = root["claim"]
        if isinstance(claim, Mapping) and "direction" in claim:
            selectors.insert(2, "claim.direction")
        mappings = tuple(_manifest_mapping(artifact, item, item) for item in selectors)
        provenance = _manifest_provenance(artifact, "<manifest structure>")
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(1),
            mappings=mappings,
            match_evidence=(provenance,),
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        root = _parse_manifest(artifact.content)
        selectors = list(_MANIFEST_FIELDS)
        claim_mapping = root["claim"]
        if isinstance(claim_mapping, Mapping) and "direction" in claim_mapping:
            selectors.insert(2, "claim.direction")
        expected = {item: item for item in selectors}
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.DOTTED_PATH,
            allowed_targets=frozenset(expected),
        )
        if {
            target: mapping.selector.expression for target, mapping in mappings.items()
        } != expected:
            raise AdapterSelectorError(
                "native manifest extraction requires its exact declared field mappings"
            )

        claim_values = [
            ConfigValue(
                "claim.metric",
                _manifest_value(root, "claim.metric"),
                _manifest_provenance(artifact, "claim.metric"),
            ),
            ConfigValue(
                "claim.minimum_improvement",
                _finite_number(
                    _manifest_value(root, "claim.minimum_improvement"),
                    label="manifest claim.minimum_improvement",
                ),
                _manifest_provenance(artifact, "claim.minimum_improvement"),
            ),
        ]
        if "claim.direction" in expected:
            claim_values.append(
                ConfigValue(
                    "claim.direction",
                    _manifest_value(root, "claim.direction"),
                    _manifest_provenance(artifact, "claim.direction"),
                )
            )
        observations = (
            NormalizedObservation(
                provenance=_manifest_provenance(artifact, "claim"),
                experiment_role=ExperimentRole.UNSPECIFIED,
                config_values=tuple(claim_values),
            ),
            _experiment_observation(
                artifact,
                root,
                name="baseline",
                role=ExperimentRole.BASELINE,
            ),
            _experiment_observation(
                artifact,
                root,
                name="candidate",
                role=ExperimentRole.CANDIDATE,
            ),
        )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=observations,
        )


def _parse_native_results(content: bytes) -> tuple[
    tuple[Mapping[str, object], ...], Mapping[str, object] | None
] | None:
    from .core import _parse_json

    root = _parse_json(content)
    if not isinstance(root, Mapping) or "runs" not in root:
        return None
    runs_value = root["runs"]
    if not isinstance(runs_value, list) or not runs_value:
        raise AdapterParseError("native results runs must be a non-empty array")
    if len(runs_value) > MAX_RECORDS:
        raise AdapterLimitError(
            f"native results exceed the {MAX_RECORDS}-records limit"
        )
    runs: list[Mapping[str, object]] = []
    for index, item in enumerate(runs_value):
        if not isinstance(item, Mapping):
            raise AdapterParseError(f"native results run {index} must be an object")
        encoded = json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_LOGICAL_RECORD_BYTES:
            raise AdapterLimitError(
                f"native results run {index} exceeds the 1 MiB logical record limit"
            )
        runs.append(item)
    summary = root.get("summary")
    if summary is not None and not isinstance(summary, Mapping):
        raise AdapterParseError("native results summary must be an object")
    return tuple(runs), summary


class NativeResultsAdapter:
    """Translate ClaimCI runs/summary JSON into normalized observations."""

    __slots__ = ()
    adapter_id = "claimci-native-results-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=frozenset({ArtifactKind.RESULTS}),
            suffixes=frozenset({".json"}),
        ):
            return None
        _verify_integrity(artifact)
        parsed = _parse_native_results(artifact.content)
        if parsed is None:
            return None
        runs, _summary = parsed
        mappings = _result_mappings(
            artifact,
            adapter_id=self.adapter_id,
            leaves=_common_leaves(runs),
        )
        unambiguous = any(item.target_field == "metric_value" for item in mappings)
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="/runs/*",
            inferred=not unambiguous,
        )
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(0.99 if unambiguous else 0.5),
            mappings=mappings,
            match_evidence=(provenance,),
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        parsed = _parse_native_results(artifact.content)
        if parsed is None:
            raise AdapterParseError("artifact does not contain native ClaimCI results")
        runs, summary = parsed
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.JSON_POINTER,
            allowed_targets=_RESULT_TARGETS,
        )
        metric = mappings.get("metric_value")
        if metric is None:
            raise AdapterSelectorError(
                "ambiguous native results require an external metric_value mapping"
            )
        observations: list[NormalizedObservation] = []
        for index, run in enumerate(runs):
            observations.append(
                _observation(
                    run,
                    mappings,
                    provenance=_adapter_provenance(
                        artifact,
                        adapter_id=self.adapter_id,
                        selector=f"record[{index}]{metric.selector.expression}",
                        inferred="inferred=true" in metric.provenance.detail,
                    ),
                )
            )
        if summary:
            try:
                config_values = tuple(
                    ConfigValue(
                        f"summary.{_dotted_from_pointer(pointer)}",
                        value,
                        _adapter_provenance(
                            artifact,
                            adapter_id=self.adapter_id,
                            selector=f"/summary{pointer}",
                            inferred=False,
                        ),
                    )
                    for pointer, value in _pointer_leaves(summary)
                    if not isinstance(value, (Mapping, list))
                )
            except AnalysisContractError as exc:
                raise AdapterParseError(
                    f"native results summary cannot be normalized safely: {exc}"
                ) from exc
            if config_values:
                observations.append(
                    NormalizedObservation(
                        provenance=_adapter_provenance(
                            artifact,
                            adapter_id=self.adapter_id,
                            selector="/summary",
                            inferred=False,
                        ),
                        experiment_role=ExperimentRole.UNSPECIFIED,
                        config_values=config_values,
                    )
                )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=tuple(observations),
        )


class NativeConfigAdapter:
    """Translate ClaimCI config YAML and expose raw compute inputs only."""

    __slots__ = ()
    adapter_id = "claimci-native-config-v1"

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        if not _supports(
            artifact,
            kinds=frozenset({ArtifactKind.CONFIG}),
            suffixes=frozenset({".yaml", ".yml"}),
        ):
            return None
        _verify_integrity(artifact)
        root = _parse_yaml(artifact.content)
        if not _COMPUTE_INPUTS.issubset(root):
            return None
        leaves, unsupported = _scalar_leaves(root)
        mappings = tuple(
            _mapping(
                artifact,
                adapter_id=self.adapter_id,
                target_field=_config_target(selector),
                selector_kind=SelectorKind.DOTTED_PATH,
                selector=selector,
                inferred=True,
            )
            for selector, _value in leaves
        )
        if not mappings:
            raise AdapterParseError("native config has no representable scalar values")
        provenance = _adapter_provenance(
            artifact,
            adapter_id=self.adapter_id,
            selector="<native config structure>",
            inferred=bool(unsupported),
        )
        return AdapterMatch(
            adapter_id=self.adapter_id,
            path=artifact.candidate.path,
            confidence=Confidence(0.98 if not unsupported else 0.75),
            mappings=mappings,
            match_evidence=(provenance,),
        )

    def extract(
        self, artifact: PassiveArtifact, match: AdapterMatch
    ) -> NormalizedEvidence:
        _verify_integrity(artifact)
        root = _parse_yaml(artifact.content)
        if not _COMPUTE_INPUTS.issubset(root):
            raise AdapterParseError("artifact does not contain native ClaimCI config")
        leaves, _unsupported = _scalar_leaves(root)
        allowed = frozenset(_config_target(selector) for selector, _value in leaves)
        mappings = _validate_match(
            artifact,
            match,
            adapter_id=self.adapter_id,
            selector_kind=SelectorKind.DOTTED_PATH,
            allowed_targets=allowed,
        )
        config_values: list[ConfigValue] = []
        compute: list[ComputeEvidence] = []
        for mapping in sorted(
            mappings.values(), key=lambda item: item.selector.expression
        ):
            selector = mapping.selector.expression
            if mapping.target_field != _config_target(selector):
                raise AdapterSelectorError(
                    "native config mapping target does not match selector"
                )
            value = _resolve_dotted_path(root, selector)
            if isinstance(value, (Mapping, list)):
                raise AdapterSelectorError("native config selector must resolve to scalar")
            config_values.append(ConfigValue(selector, value, mapping.provenance))
            if (
                selector in _COMPUTE_INPUTS
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
            ):
                compute.append(
                    ComputeEvidence(
                        selector,
                        _finite_number(value, label=f"native config {selector}"),
                        None,
                        mapping.provenance,
                    )
                )
        if not config_values:
            raise AdapterSelectorError("native config extraction requires mappings")
        observation = NormalizedObservation(
            provenance=_adapter_provenance(
                artifact,
                adapter_id=self.adapter_id,
                selector="<selected native config mappings>",
                inferred=False,
            ),
            experiment_role=ExperimentRole.UNSPECIFIED,
            config_values=tuple(config_values),
            compute_evidence=tuple(compute),
        )
        return NormalizedEvidence(
            evidence_id=_evidence_id(self.adapter_id, artifact),
            artifact=artifact.candidate,
            adapter_match=match,
            observations=(observation,),
        )


__all__ = [
    "NativeManifestAdapter",
    "NativeResultsAdapter",
    "NativeConfigAdapter",
]
