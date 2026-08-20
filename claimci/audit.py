"""Load a research manifest and orchestrate all deterministic ClaimCI checks."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from .config_check import check_configs, load_config
from .dataset_check import (
    check_dataset_leakage,
    check_evaluation_alignment,
    load_dataset,
)
from .models import (
    AuditResult,
    ClaimCIError,
    Direction,
    ExperimentPaths,
    Finding,
    Impact,
    ResearchSpec,
    Severity,
    Verdict,
    coerce_direction,
)
from .measurement import (
    MeasurementAuditContext,
    compare_measurement_protocols,
    recover_measurement_protocol_pair,
)
from .result_check import check_results, load_results
from .parsing import load_unique_yaml


REQUIRED_ARTIFACTS = ("config", "results", "train_dataset", "eval_dataset")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _experiment_paths(
    label: str,
    value: object,
    manifest_dir: Path,
    artifact_root: Path | None = None,
) -> ExperimentPaths:
    if not isinstance(value, Mapping):
        raise ClaimCIError(f"{label} must be a mapping")
    paths: dict[str, Path] = {}
    for field in REQUIRED_ARTIFACTS:
        raw_path = value.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ClaimCIError(f"{label}.{field} must be a non-empty path string")
        try:
            if artifact_root is not None:
                portable_parts = raw_path.replace("\\", "/").split("/")
                if ".." in portable_parts:
                    raise ClaimCIError(
                        f"{label}.{field} contains parent traversal outside artifact root"
                    )
                if PurePosixPath(raw_path).is_absolute() or PureWindowsPath(
                    raw_path
                ).is_absolute():
                    raise ClaimCIError(
                        f"{label}.{field} must be relative to the artifact root"
                    )
            path = Path(raw_path)
            resolved = (
                path if path.is_absolute() else manifest_dir / path
            ).resolve()
            if artifact_root is not None:
                try:
                    resolved.relative_to(artifact_root)
                except ValueError as exc:
                    raise ClaimCIError(
                        f"{label}.{field} resolves outside artifact root {artifact_root}"
                    ) from exc
            paths[field] = resolved
        except ClaimCIError:
            raise
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise ClaimCIError(f"{label}.{field} is not a valid path: {exc}") from exc
    return ExperimentPaths(**paths)


def load_research_spec(
    path: Path,
    *,
    artifact_root: Path | None = None,
) -> ResearchSpec:
    try:
        manifest_path = Path(path).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ClaimCIError(f"invalid research manifest path: {exc}") from exc
    resolved_artifact_root: Path | None = None
    if artifact_root is not None:
        try:
            resolved_artifact_root = Path(artifact_root).resolve()
            manifest_path.relative_to(resolved_artifact_root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise ClaimCIError(
                f"research manifest resolves outside artifact root {artifact_root}"
            ) from exc
    try:
        payload = load_unique_yaml(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClaimCIError(f"research manifest not found: {manifest_path}") from exc
    except OSError as exc:
        raise ClaimCIError(f"could not read research manifest {manifest_path}: {exc}") from exc
    except (UnicodeError, ValueError) as exc:
        raise ClaimCIError(f"could not read research manifest {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ClaimCIError(f"invalid YAML in research manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ClaimCIError("research manifest root must be a mapping")
    claim = payload.get("claim")
    if not isinstance(claim, Mapping):
        raise ClaimCIError("claim must be a mapping")
    metric = claim.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ClaimCIError("claim.metric must be a non-empty string")
    threshold = _number(claim.get("minimum_improvement"))
    if threshold is None or threshold < 0:
        raise ClaimCIError("claim.minimum_improvement must be a finite non-negative number")
    try:
        direction = coerce_direction(claim.get("direction", Direction.HIGHER))
    except ClaimCIError as exc:
        raise ClaimCIError(f"claim.direction {exc}") from exc
    if "baseline" not in payload:
        raise ClaimCIError("research manifest is missing baseline")
    if "candidate" not in payload:
        raise ClaimCIError("research manifest is missing candidate")

    return ResearchSpec(
        manifest_path=manifest_path,
        metric=metric.strip(),
        minimum_improvement=threshold,
        baseline=_experiment_paths(
            "baseline",
            payload["baseline"],
            manifest_path.parent,
            resolved_artifact_root,
        ),
        candidate=_experiment_paths(
            "candidate",
            payload["candidate"],
            manifest_path.parent,
            resolved_artifact_root,
        ),
        direction=direction,
    )


def determine_verdict(findings: list[Finding] | tuple[Finding, ...]) -> Verdict:
    if any(finding.impact is Impact.INVALIDATES for finding in findings):
        return Verdict.NOT_SUPPORTED
    if any(finding.impact is Impact.INSUFFICIENT for finding in findings):
        return Verdict.INSUFFICIENT_EVIDENCE
    return Verdict.SUPPORTED


def _artifact_failure(
    category: str,
    experiment: str,
    path: Path,
    error: ClaimCIError,
) -> Finding:
    missing = not path.exists()
    return Finding(
        rule_id=f"{category}.{'MISSING' if missing else 'INVALID'}",
        severity=Severity.CRITICAL,
        title=f"{experiment.title()} {category.lower()} artifact is unavailable",
        explanation=str(error),
        evidence={"experiment": experiment, "path": str(path)},
        impact=Impact.INSUFFICIENT,
    )


def audit_research(
    path: Path,
    *,
    artifact_root: Path | None = None,
    measurement_context: MeasurementAuditContext | None = None,
) -> AuditResult:
    if measurement_context is not None and type(
        measurement_context
    ) is not MeasurementAuditContext:
        raise TypeError(
            "measurement_context must be a trusted MeasurementAuditContext or null"
        )
    spec = load_research_spec(path, artifact_root=artifact_root)
    findings: list[Finding] = []
    overlaps = []

    configs: dict[str, Mapping[str, Any]] = {}
    for experiment, paths in (("baseline", spec.baseline), ("candidate", spec.candidate)):
        try:
            configs[experiment] = load_config(paths.config)
        except ClaimCIError as exc:
            findings.append(_artifact_failure("CONFIG", experiment, paths.config, exc))
    if len(configs) == 2:
        findings.append(
            Finding(
                rule_id="CONFIG.LOADED",
                severity=Severity.VERIFIED,
                title="Config files loaded successfully",
                explanation="Both declared YAML config artifacts were parsed as mappings.",
                evidence={
                    "baseline": str(spec.baseline.config),
                    "candidate": str(spec.candidate.config),
                },
                impact=Impact.NONE,
            )
        )
        findings.extend(check_configs(configs["baseline"], configs["candidate"]))

    loaded_results = {}
    for experiment, paths in (("baseline", spec.baseline), ("candidate", spec.candidate)):
        try:
            loaded_results[experiment] = load_results(paths.results, spec.metric)
        except ClaimCIError as exc:
            findings.append(_artifact_failure("RESULT", experiment, paths.results, exc))

    baseline_metrics = None
    candidate_metrics = None
    absolute_improvement = None
    relative_improvement = None
    if len(loaded_results) == 2:
        try:
            checked = check_results(
                loaded_results["baseline"],
                loaded_results["candidate"],
                spec.metric,
                spec.minimum_improvement,
                spec.direction,
            )
        except ClaimCIError as exc:
            findings.append(
                Finding(
                    rule_id="RESULT.INVALID",
                    severity=Severity.CRITICAL,
                    title="Result comparison could not be computed",
                    explanation=str(exc),
                    evidence={
                        "experiment": "comparison",
                        "baseline_path": str(spec.baseline.results),
                        "candidate_path": str(spec.candidate.results),
                    },
                    impact=Impact.INSUFFICIENT,
                )
            )
        else:
            baseline_metrics = checked.baseline_summary
            candidate_metrics = checked.candidate_summary
            absolute_improvement = checked.absolute_improvement
            relative_improvement = checked.relative_improvement
            findings.extend(checked.findings)

    loaded_datasets = {}
    for experiment, paths in (("baseline", spec.baseline), ("candidate", spec.candidate)):
        datasets = {}
        for split, dataset_path in (
            ("train", paths.train_dataset),
            ("eval", paths.eval_dataset),
        ):
            try:
                datasets[split] = load_dataset(dataset_path)
            except ClaimCIError as exc:
                findings.append(_artifact_failure("DATASET", experiment, dataset_path, exc))
        if len(datasets) == 2:
            overlap, leakage_findings = check_dataset_leakage(
                experiment, datasets["train"], datasets["eval"]
            )
            overlaps.append(overlap)
            findings.extend(leakage_findings)
        loaded_datasets[experiment] = datasets

    if all(
        "eval" in loaded_datasets[experiment]
        for experiment in ("baseline", "candidate")
    ):
        findings.extend(
            check_evaluation_alignment(
                loaded_datasets["baseline"]["eval"],
                loaded_datasets["candidate"]["eval"],
            )
        )

    verdict = determine_verdict(findings)
    measurement_drift = None
    if measurement_context is not None:
        baseline_eval = loaded_datasets.get("baseline", {}).get("eval")
        candidate_eval = loaded_datasets.get("candidate", {}).get("eval")
        protocols = recover_measurement_protocol_pair(
            measurement_context,
            metric=spec.metric,
            baseline_config=configs.get("baseline"),
            candidate_config=configs.get("candidate"),
            baseline_evaluation_hashes=(
                None if baseline_eval is None else baseline_eval.hashes
            ),
            candidate_evaluation_hashes=(
                None if candidate_eval is None else candidate_eval.hashes
            ),
        )
        measurement_drift = compare_measurement_protocols(
            protocols,
            native_rule_ids=tuple(item.rule_id for item in findings),
        )
    return AuditResult(
        manifest_path=spec.manifest_path,
        metric=spec.metric,
        minimum_improvement=spec.minimum_improvement,
        verdict=verdict,
        findings=tuple(findings),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        overlaps=tuple(overlaps),
        direction=spec.direction,
        measurement_drift=measurement_drift,
    )
