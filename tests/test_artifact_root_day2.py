"""Trusted-workflow artifact-root confinement contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from claimci.audit import audit_research, load_research_spec
from claimci.cli import main
from claimci.models import ClaimCIError, Verdict


def _set_candidate_results(manifest: Path, value: str) -> None:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["candidate"]["results"] = value
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_audit_accepts_artifacts_confined_to_explicit_root(study_factory) -> None:
    manifest = study_factory()

    result = audit_research(manifest, artifact_root=manifest.parent)

    assert result.verdict is Verdict.SUPPORTED


@pytest.mark.parametrize("spelling", ["../outside.json", "..\\outside.json"])
def test_artifact_root_rejects_parent_traversal(
    study_factory,
    tmp_path: Path,
    spelling: str,
) -> None:
    manifest = study_factory()
    (tmp_path / "outside.json").write_text('{"runs":[]}', encoding="utf-8")
    _set_candidate_results(manifest, spelling)

    with pytest.raises(ClaimCIError, match=r"(?i)artifact root|outside|candidate.results"):
        load_research_spec(manifest, artifact_root=manifest.parent)


def test_artifact_root_rejects_absolute_outside_path(
    study_factory,
    tmp_path: Path,
) -> None:
    manifest = study_factory()
    outside = tmp_path / "outside-results.json"
    outside.write_text('{"runs":[]}', encoding="utf-8")
    _set_candidate_results(manifest, str(outside.resolve()))

    with pytest.raises(ClaimCIError, match=r"(?i)artifact root|outside|candidate.results"):
        load_research_spec(manifest, artifact_root=manifest.parent)


def test_artifact_root_rejects_manifest_outside_root(study_factory, tmp_path: Path) -> None:
    manifest = study_factory()
    other_root = tmp_path / "other-root"
    other_root.mkdir()

    with pytest.raises(ClaimCIError, match=r"(?i)manifest|artifact root|outside"):
        load_research_spec(manifest, artifact_root=other_root)


def test_artifact_root_rejects_symlink_escape(study_factory, tmp_path: Path) -> None:
    manifest = study_factory()
    outside = tmp_path / "outside-results.json"
    outside.write_text('{"runs":[]}', encoding="utf-8")
    link = manifest.parent / "candidate" / "linked-results.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available for this test user")
    _set_candidate_results(manifest, "candidate/linked-results.json")

    with pytest.raises(ClaimCIError, match=r"(?i)artifact root|outside|candidate.results"):
        load_research_spec(manifest, artifact_root=manifest.parent)


def test_cli_artifact_root_option_confines_pr_tree(study_factory, capsys) -> None:
    manifest = study_factory()

    code = main(
        [
            "audit",
            str(manifest),
            "--artifact-root",
            str(manifest.parent),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert '"verdict": "SUPPORTED"' in captured.out
    assert captured.err == ""


def test_default_cli_remains_backward_compatible_with_shared_fixture_paths() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    result = audit_research(repository_root / "examples" / "valid" / "research.yaml")

    assert result.verdict is Verdict.SUPPORTED


def test_manifest_symlink_loop_resolution_is_a_controlled_error(monkeypatch) -> None:
    def raise_loop(_self, *args, **kwargs):
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(Path, "resolve", raise_loop)

    with pytest.raises(ClaimCIError, match=r"(?i)manifest|path|symlink loop"):
        load_research_spec(Path("looped-research.yaml"))


def test_artifact_symlink_loop_resolution_is_a_controlled_error(
    study_factory, monkeypatch
) -> None:
    manifest = study_factory()
    original_resolve = Path.resolve

    def raise_for_json(self, *args, **kwargs):
        if self.suffix == ".json":
            raise RuntimeError("symlink loop")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", raise_for_json)

    with pytest.raises(ClaimCIError, match=r"(?i)results|path|symlink loop"):
        load_research_spec(manifest)
