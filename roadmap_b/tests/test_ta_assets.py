"""Small integrity fixtures; no pretrained model or corpus access."""

import json

import pytest

from src import ta_assets as assets


def bundle(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"frozen bytes")
    value = {
        "release_id": "fixture",
        "files": {
            "weights.bin": {
                "bytes": 12,
                "sha256": assets.sha256(root / "weights.bin"),
                "source_root": "A",
                "source_path": "weights.bin",
            }
        },
        "source_sha256": {},
    }
    return root, value


def test_exact_tree_rejects_missing_changed_extra_and_symlink(tmp_path):
    root, value = bundle(tmp_path)
    assert assets.verify_asset_root(root, value=value)["status"] == "PASS"
    (root / "weights.bin").write_bytes(b"changed data")
    with pytest.raises(assets.AssetIntegrityError, match="changed weights.bin"):
        assets.verify_asset_root(root, value=value)
    (root / "weights.bin").unlink()
    with pytest.raises(assets.AssetIntegrityError, match="missing weights.bin"):
        assets.verify_asset_root(root, value=value)
    (root / "weights.bin").write_bytes(b"frozen bytes")
    (root / "private.wav").write_bytes(b"not part of inference")
    with pytest.raises(assets.AssetIntegrityError, match="unexpected private.wav"):
        assets.verify_asset_root(root, value=value)
    (root / "private.wav").unlink()
    (root / "alias").symlink_to(root / "weights.bin")
    with pytest.raises(assets.AssetIntegrityError, match="symlink alias"):
        assets.verify_asset_root(root, value=value)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "nested/../../outside", "nested\\outside"])
def test_manifest_path_cannot_escape_root(tmp_path, path):
    with pytest.raises(assets.AssetIntegrityError, match="Unsafe"):
        assets.contained_file(tmp_path, path)


def test_symlink_directory_cannot_reach_external_weight(tmp_path):
    source = tmp_path / "outside"
    source.mkdir()
    root = tmp_path / "assets"
    root.mkdir()
    (root / "nested").symlink_to(source, target_is_directory=True)
    with pytest.raises(assets.AssetIntegrityError, match="Symlink"):
        assets.contained_file(root, "nested/model.bin")


def test_source_identity_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "SOURCE_ROOT", tmp_path)
    (tmp_path / "runtime.py").write_text("saved")
    value = {"source_sha256": {"runtime.py": assets.sha256(tmp_path / "runtime.py")}}
    assets.verify_sources(value)
    (tmp_path / "runtime.py").write_text("modified")
    with pytest.raises(assets.AssetIntegrityError, match="runtime.py"):
        assets.verify_sources(value)


def test_prepare_checks_all_sources_before_creating_destination(tmp_path):
    source, value = bundle(tmp_path)
    (source / "weights.bin").write_bytes(b"tampered")
    destination = tmp_path / "new"
    with pytest.raises(assets.AssetIntegrityError, match="source dependency"):
        assets.prepare_assets(destination, baseline_root=source, b_root=source, c1_root=source, value=value)
    assert not destination.exists()


def test_prepare_reuses_valid_bundle_but_never_overwrites_changed_bundle(tmp_path):
    source, value = bundle(tmp_path)
    destination = tmp_path / "new"
    kwargs = {"baseline_root": source, "b_root": source, "c1_root": source, "value": value}
    assert assets.prepare_assets(destination, **kwargs)["files"] == 1
    assert assets.prepare_assets(destination, **kwargs)["status"] == "PASS"
    (destination / "weights.bin").write_bytes(b"local edit")
    with pytest.raises(assets.AssetIntegrityError, match="changed"):
        assets.prepare_assets(destination, **kwargs)
    assert (destination / "weights.bin").read_bytes() == b"local edit"


def test_release_manifest_requires_selected_namespace(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": "ta-assets-v1", "condition": "D3", "seed": 1337}))
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    with pytest.raises(assets.AssetIntegrityError, match="Unsupported"):
        assets.manifest()


def test_committed_manifest_is_minimal_and_bound_to_exact_source():
    value = assets.manifest()
    paths = list(value["files"])
    assert all("/D3/" not in p and "/D0/" not in p and "normalization/" not in p for p in paths)
    assert all("features/" not in p and "data/" not in p for p in paths)
    assert value["learned_parameters"] == 218895839 < 6_000_000_000
    assets.verify_sources(value)


def test_root_cli_source_identity_is_checked(tmp_path, monkeypatch):
    source = tmp_path / "roadmap_b"
    source.mkdir()
    monkeypatch.setattr(assets, "SOURCE_ROOT", source)
    entry = tmp_path / "ta.py"
    entry.write_text("fixed CLI")
    value = {"source_sha256": {}, "root_source_sha256": {"ta.py": assets.sha256(entry)}}
    assets.verify_sources(value)
    entry.write_text("changed policy selection")
    with pytest.raises(assets.AssetIntegrityError, match="ta.py"):
        assets.verify_sources(value)
