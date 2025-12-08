"""Tests for data version manifest utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_data_versions as dv  # noqa: E402


def test_register_raw_entry_adds_manifest_entry(tmp_path: Path) -> None:
    """Registering a raw entry should persist metadata with checksum."""
    manifest = tmp_path / "manifest.json"
    data_file = tmp_path / "spy.parquet"
    data_file.write_bytes(b"dummy data")

    entry = dv.register_raw_entry("SPY", data_file, manifest_path=manifest)

    saved = json.loads(manifest.read_text())
    assert saved["entries"][-1]["dataset"] == "SPY"
    assert saved["entries"][-1]["md5"] == entry.md5
    assert saved["entries"][-1]["path"] == str(data_file)


def test_verify_manifest_detects_missing_files(tmp_path: Path) -> None:
    """verify_manifest should return errors when files are absent."""
    manifest = tmp_path / "manifest.json"
    missing_file = tmp_path / "missing.parquet"
    manifest_data = {
        "schema_version": 1,
        "entries": [
            {
                "dataset": "SPY",
                "path": str(missing_file),
                "md5": "abc",
                "downloaded_at": "2025-01-01T00:00:00Z",
            }
        ],
    }
    manifest.write_text(json.dumps(manifest_data))

    errors = dv.verify_manifest(manifest)

    assert errors and "missing file" in errors[0]


def test_verify_all_manifests_passes_with_empty_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_all_manifests should succeed when manifests have no entries."""
    raw_manifest = tmp_path / "raw.json"
    processed_manifest = tmp_path / "processed.json"
    raw_manifest.write_text(json.dumps({"schema_version": 1, "entries": []}))
    processed_manifest.write_text(json.dumps({"schema_version": 1, "entries": []}))

    monkeypatch.setattr(dv, "RAW_MANIFEST", raw_manifest)
    monkeypatch.setattr(dv, "PROCESSED_MANIFEST", processed_manifest)

    assert dv.verify_all_manifests() == 0

