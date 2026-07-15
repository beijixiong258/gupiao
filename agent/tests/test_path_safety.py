"""Tests for research-output path validation."""

from pathlib import Path

import pytest

from src.tools.path_utils import safe_run_dir


def test_configured_run_root_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPYJ_ALLOWED_RUN_ROOTS", str(tmp_path))
    run_dir = tmp_path / "run_123"
    run_dir.mkdir()

    assert safe_run_dir(str(run_dir)) == run_dir.resolve()


def test_path_outside_run_roots_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("GPYJ_ALLOWED_RUN_ROOTS", str(allowed))

    with pytest.raises(ValueError, match="outside allowed run roots"):
        safe_run_dir(str(outside))


def test_unc_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="UNC paths"):
        safe_run_dir("\\\\server\\share\\run")
