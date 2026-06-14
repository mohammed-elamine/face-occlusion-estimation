"""Tests for the dedicated face-mask store and its builder."""

from __future__ import annotations

import numpy as np
from PIL import Image

from face_occlusion.data.background_augment import BackgroundAugment
from face_occlusion.data.face_mask_store import FaceMaskStore


def _write_mask(path, mask_bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(path)


class TestFaceMaskStore:
    def test_load_mirrored_mask(self, tmp_path):
        store = FaceMaskStore(tmp_path)
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        _write_mask(tmp_path / "database1" / "img001.png", mask)
        out = store.load_mask("database1/img001.webp")  # .webp id -> .png mask
        assert out is not None
        assert out.dtype == bool and out.shape == (8, 8)
        assert np.array_equal(out, mask)

    def test_missing_returns_none(self, tmp_path):
        assert FaceMaskStore(tmp_path).load_mask("database1/nope.webp") is None

    def test_nested_path_and_suffix_swap(self, tmp_path):
        store = FaceMaskStore(tmp_path)
        # dots in directory components must be preserved; only the final suffix changes.
        p = store.mask_path("db/m.01x/13-FaceId-0_align.webp")
        assert p == tmp_path / "db" / "m.01x" / "13-FaceId-0_align.png"

    def test_has_mask(self, tmp_path):
        store = FaceMaskStore(tmp_path)
        assert not store.has_mask("a/b.webp")
        _write_mask(tmp_path / "a" / "b.png", np.ones((4, 4), bool))
        assert store.has_mask("a/b.webp")

    def test_plugs_into_background_augment(self, tmp_path):
        store = FaceMaskStore(tmp_path)
        mask = np.zeros((16, 16), bool)
        mask[4:12, 4:12] = True
        _write_mask(tmp_path / "x.png", mask)
        ba = BackgroundAugment(mask_lookup=store.load_mask, p=1.0, modes=("replace",), seed=0)
        img = Image.fromarray(np.full((16, 16, 3), 100, np.uint8))
        out = np.asarray(ba(img, "x.webp", 0))
        # Face pixels exact; background changed.
        assert np.array_equal(out[mask], np.full((int(mask.sum()), 3), 100))
        assert not np.array_equal(out[~mask], np.full((int((~mask).sum()), 3), 100))

    def test_no_mask_means_no_op(self, tmp_path):
        # An id with no stored mask returns the image unchanged (background aug no-ops).
        ba = BackgroundAugment(mask_lookup=FaceMaskStore(tmp_path).load_mask, p=1.0, seed=0)
        img = Image.fromarray(np.full((8, 8, 3), 100, np.uint8))
        assert np.array_equal(np.asarray(ba(img, "absent.webp", 0)), np.asarray(img))


class _FakeResult:
    def __init__(self, valid, masks):
        self.valid = valid
        self.masks = masks


class _FakeProvider:
    """Stub MediaPipe provider so the builder can be tested without mediapipe."""

    def __init__(self, valid=True):
        self.valid = valid

    def extract(self, image):
        h, w = np.asarray(image).shape[:2]
        if not self.valid:
            return _FakeResult(False, {})
        m = np.zeros((h, w), bool)
        m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return _FakeResult(True, {"face": m})


class TestBuildFaceMasks:
    def _setup(self, tmp_path, valid=True):
        import scripts.data.build_face_masks as builder

        img_root = tmp_path / "imgs"
        out = tmp_path / "masks"
        (img_root / "database1").mkdir(parents=True)
        Image.fromarray(np.full((20, 20, 3), 128, np.uint8)).save(img_root / "database1" / "a.webp")
        builder._PROVIDER = _FakeProvider(valid)
        builder._IMAGE_ROOT = img_root
        builder._OUT_DIR = out
        builder._OVERWRITE = False
        return builder, out

    def test_process_writes_mirrored_mask(self, tmp_path):
        builder, out = self._setup(tmp_path)
        status = builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))
        assert status[2] == "masked"
        assert (out / "database1" / "a.png").exists()
        # The written mask round-trips through the store.
        assert FaceMaskStore(out).load_mask("database1/a.webp") is not None

    def test_process_skips_existing(self, tmp_path):
        builder, _ = self._setup(tmp_path)
        builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))
        again = builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))
        assert again[2] == "skipped"

    def test_process_overwrite(self, tmp_path):
        builder, _ = self._setup(tmp_path)
        builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))
        builder._OVERWRITE = True
        assert builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))[2] == "masked"

    def test_process_no_face_writes_nothing(self, tmp_path):
        builder, out = self._setup(tmp_path, valid=False)
        assert builder._process_one(("database1/a.webp", "0.00_0.05", 0.0))[2] == "no_face"
        assert not (out / "database1" / "a.png").exists()

    def test_process_load_error(self, tmp_path):
        builder, out = self._setup(tmp_path)
        assert builder._process_one(("database1/missing.webp", "0.00_0.05", 0.0))[2] == "load_error"
        assert not (out / "database1" / "missing.png").exists()


class TestMediaPipeModelResolution:
    def test_search_order_prefers_models_dir(self):
        from face_occlusion.data.synthetic_occlusion import DEFAULT_MEDIAPIPE_MODEL_PATHS

        assert str(DEFAULT_MEDIAPIPE_MODEL_PATHS[0]) == "models/mediapipe/face_landmarker.task"
        # the scratch tmp/ path must no longer be a default search location
        assert all("tmp/" not in str(p) for p in DEFAULT_MEDIAPIPE_MODEL_PATHS)

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        from face_occlusion.data import synthetic_occlusion as so

        f = tmp_path / "custom.task"
        f.write_bytes(b"x")
        monkeypatch.setenv("FACE_OCCLUSION_MEDIAPIPE_FACE_LANDMARKER", str(f))
        assert so._resolve_mediapipe_model_asset_path() == f

    def test_ensure_noop_when_solutions_present(self, monkeypatch):
        from face_occlusion.data import synthetic_occlusion as so

        monkeypatch.setattr(so, "mediapipe_needs_model_asset", lambda: False)
        assert so.ensure_mediapipe_model() is None  # legacy backend bundles the model

    def test_ensure_no_download_returns_none_when_missing(self, monkeypatch):
        from face_occlusion.data import synthetic_occlusion as so

        monkeypatch.setattr(so, "mediapipe_needs_model_asset", lambda: True)
        monkeypatch.setattr(so, "_resolve_mediapipe_model_asset_path", lambda p=None: None)
        assert so.ensure_mediapipe_model(allow_download=False) is None
