"""Data version manifest utilities.

Provides helper functions to register raw/processed datasets and verify
that the manifest entries correspond to on-disk files with matching checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MANIFEST = PROJECT_ROOT / "data" / "raw" / "manifest.json"
PROCESSED_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.json"
SCHEMA_VERSION = 1


@dataclass
class RawEntry:
    dataset: str
    path: str
    md5: str
    downloaded_at: str


@dataclass
class ProcessedEntry:
    dataset: str
    version: int
    tier: str
    path: str
    md5: str
    generated_at: str
    source_raw_md5s: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def _compute_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_raw_entry(dataset: str, file_path: Path, manifest_path: Path = RAW_MANIFEST) -> RawEntry:
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    md5 = _compute_md5(file_path)
    entry = RawEntry(dataset=dataset, path=str(file_path), md5=md5, downloaded_at=_utc_now())

    manifest = _load_manifest(manifest_path)
    manifest.setdefault("entries", []).append(asdict(entry))
    manifest["schema_version"] = SCHEMA_VERSION
    _write_manifest(manifest_path, manifest)
    return entry


def register_processed_entry(
    dataset: str,
    version: int,
    tier: str,
    file_path: Path,
    source_raw_md5s: Iterable[str],
    manifest_path: Path = PROCESSED_MANIFEST,
) -> ProcessedEntry:
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    md5 = _compute_md5(file_path)
    entry = ProcessedEntry(
        dataset=dataset,
        version=version,
        tier=tier,
        path=str(file_path),
        md5=md5,
        generated_at=_utc_now(),
        source_raw_md5s=list(source_raw_md5s),
    )
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("entries", []).append(asdict(entry))
    manifest["schema_version"] = SCHEMA_VERSION
    _write_manifest(manifest_path, manifest)
    return entry


def _get_entry_timestamp(entry: dict[str, Any]) -> str:
    """Extract timestamp from entry (downloaded_at for raw, generated_at for processed)."""
    return entry.get("downloaded_at") or entry.get("generated_at") or ""


def verify_manifest(manifest_path: Path) -> list[str]:
    """Verify manifest entries against on-disk files.

    Only verifies the latest entry per path (by timestamp). Historical entries
    are kept for provenance but not verified against current files.
    """
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []

    # Group entries by path, keep only the latest per path
    latest_by_path: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        path = entry["path"]
        if path not in latest_by_path:
            latest_by_path[path] = entry
        else:
            # Keep the entry with the more recent timestamp
            if _get_entry_timestamp(entry) > _get_entry_timestamp(latest_by_path[path]):
                latest_by_path[path] = entry

    # Verify only the latest entry per path
    for entry in latest_by_path.values():
        file_path = Path(entry["path"])
        if not file_path.exists():
            errors.append(f"{manifest_path}: missing file {file_path}")
            continue
        actual_md5 = _compute_md5(file_path)
        if actual_md5 != entry.get("md5"):
            errors.append(f"{manifest_path}: checksum mismatch for {file_path}")
    return errors


def verify_all_manifests() -> int:
    errors: list[str] = []
    for manifest in (RAW_MANIFEST, PROCESSED_MANIFEST):
        if manifest.exists():
            errors.extend(verify_manifest(manifest))
    if errors:
        for error in errors:
            print(f"❌ {error}")
        print("Data version verification FAILED")
        return 1
    print("✅ Data version manifests verified")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage data version manifests.")
    sub = parser.add_subparsers(dest="command", required=True)

    verify_cmd = sub.add_parser("verify", help="Verify all manifest entries")
    verify_cmd.set_defaults(func=lambda _: verify_all_manifests())

    raw_cmd = sub.add_parser("register-raw", help="Register a raw dataset file")
    raw_cmd.add_argument("--dataset", required=True)
    raw_cmd.add_argument("--file", required=True, type=Path)
    raw_cmd.add_argument("--manifest", default=RAW_MANIFEST, type=Path)
    raw_cmd.set_defaults(
        func=lambda args: register_raw_entry(args.dataset, args.file, args.manifest) or 0,
    )

    proc_cmd = sub.add_parser("register-processed", help="Register a processed dataset file")
    proc_cmd.add_argument("--dataset", required=True)
    proc_cmd.add_argument("--version", required=True, type=int)
    proc_cmd.add_argument("--tier", required=True)
    proc_cmd.add_argument("--file", required=True, type=Path)
    proc_cmd.add_argument("--sources", nargs="+", default=[])
    proc_cmd.add_argument("--manifest", default=PROCESSED_MANIFEST, type=Path)
    proc_cmd.set_defaults(
        func=lambda args: register_processed_entry(
            args.dataset,
            args.version,
            args.tier,
            args.file,
            args.sources,
            args.manifest,
        )
        or 0
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())

