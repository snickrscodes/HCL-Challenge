"""Strict local assets for frozen B/adaptive/1337 + D2/1337 inference.

Historical full research freezes still require their retained research assets.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = SOURCE_ROOT / "evidence/ta_system_closure/ASSET_MANIFEST.json"


class AssetIntegrityError(ValueError):
    """A missing, unexpected or changed local inference dependency."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    value = json.loads(MANIFEST_PATH.read_text())
    if (
        value.get("version") != "ta-assets-v1"
        or value.get("condition") != "D2"
        or value.get("seed") != 1337
        or value.get("namespace") != "crema6_audio_votes_v1"
    ):
        raise AssetIntegrityError("Unsupported T+A asset manifest")
    return value


def contained_file(root, relative):
    """Resolve only regular files inside root; reject every symlink component."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in str(relative):
        raise AssetIntegrityError(f"Unsafe asset path: {relative}")
    root = Path(root).resolve()
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise AssetIntegrityError(f"Symlink dependency is not allowed: {relative}")
    if not path.resolve().is_relative_to(root):
        raise AssetIntegrityError(f"Dependency escapes asset root: {relative}")
    return path


def verify_sources(value=None):
    value = manifest() if value is None else value
    errors = []
    for root, entries in (
        (SOURCE_ROOT, value["source_sha256"]),
        (SOURCE_ROOT.parent, value.get("root_source_sha256", {})),
    ):
        for relative, expected in entries.items():
            path = contained_file(root, relative)
            if not path.is_file() or sha256(path) != expected:
                errors.append(relative)
    if errors:
        raise AssetIntegrityError("Changed runtime source: " + ", ".join(errors))


def verify_asset_root(asset_root, *, value=None, check_sources=True):
    """Check the complete exact asset tree before constructing any model."""
    value = manifest() if value is None else value
    root = Path(asset_root).expanduser().resolve()
    if not root.is_dir():
        raise AssetIntegrityError(f"Missing asset root: {root}; run prepare_ta_assets.py first")
    if check_sources:
        verify_sources(value)
    errors = []
    for relative, expected in value["files"].items():
        path = contained_file(root, relative)
        if not path.is_file():
            errors.append(f"missing {relative}")
        elif path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            errors.append(f"changed {relative}")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlink {path.relative_to(root).as_posix()}")
        elif path.is_file():
            observed.add(path.relative_to(root).as_posix())
    errors.extend(f"unexpected {name}" for name in sorted(observed - set(value["files"])))
    if errors:
        raise AssetIntegrityError("Asset verification failed: " + "; ".join(errors))
    return {
        "status": "PASS",
        "asset_root": str(root),
        "release_id": value["release_id"],
        "files": len(value["files"]),
        "bytes": sum(record["bytes"] for record in value["files"].values()),
        "scope": "exact inference dependency tree; no training caches or corpus media required",
    }


def prepare_assets(asset_root, *, baseline_root, b_root, c1_root, value=None):
    """Copy only predeclared hash-verified local assets; never download weights."""
    value = manifest() if value is None else value
    roots = {
        "A": Path(baseline_root).expanduser().resolve(),
        "B": Path(b_root).expanduser().resolve(),
        "C1": Path(c1_root).expanduser().resolve(),
        "source": SOURCE_ROOT,
    }
    sources = {}
    for relative, record in value["files"].items():
        path = contained_file(roots[record["source_root"]], record["source_path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssetIntegrityError(
                f"Missing or changed source dependency: {record['source_root']}/{record['source_path']}"
            )
        sources[relative] = path
    destination = Path(asset_root).expanduser().resolve()
    if destination.exists():
        # Reuse a complete bundle; never overwrite an incomplete or changed one.
        return verify_asset_root(destination, value=value)
    verify_sources(value)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, source in sources.items():
        target = contained_file(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    return verify_asset_root(destination, value=value)
