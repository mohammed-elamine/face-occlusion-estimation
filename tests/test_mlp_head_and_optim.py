"""Tests for the MLP head, discriminative-LR optimizer, hue jitter, and TTA.

Model tests use a tiny timm ViT with pretrained=False (no network) to exercise the
MLP-head + LoRA + param_groups paths without downloading DINOv2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn as nn

from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.models.regressor import OcclusionRegressor

_TINY = "vit_tiny_patch16_224"


class _C(dict):
    """dict with attribute access + .get (mimics the project's Config wrapper)."""

    def __getattr__(self, k):
        return self[k]


class TestMlpHeadModel:
    def test_mlp_head_lora_forward_and_params(self):
        m = OcclusionRegressor(
            backbone=_TINY,
            pretrained=False,
            output_activation="sigmoid",
            mean_target=0.08,
            head={"type": "mlp", "hidden_dim": 64, "dropout": 0.1},
            lora={
                "enabled": True,
                "rank": 4,
                "alpha": 8,
                "target_modules": ["attn.qkv", "attn.proj"],
            },
        ).eval()
        out = m(torch.randn(2, 3, 224, 224)).y_pred
        assert out.shape == (2,)
        assert bool((out >= 0).all() and (out <= 1).all())  # sigmoid bounds
        # backbone frozen except LoRA; head trainable
        assert all(p.requires_grad for p in m.head.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        assert 0 < trainable < total  # not full fine-tuning

    def test_param_groups_two_lrs_no_wd_on_1d(self):
        m = OcclusionRegressor(
            backbone=_TINY,
            pretrained=False,
            head={"type": "mlp", "hidden_dim": 64},
            lora={"enabled": True, "rank": 4, "target_modules": ["attn.qkv", "attn.proj"]},
        )
        groups = m.param_groups(head_lr=1e-3, backbone_lr=2e-4, weight_decay=1e-4)
        lrs = {g["lr"] for g in groups}
        assert lrs == {1e-3, 2e-4}
        # every no-weight-decay group holds only 1-D params (norms/biases)
        for g in groups:
            if g["weight_decay"] == 0.0:
                assert all(p.ndim <= 1 for p in g["params"])
            else:
                assert all(p.ndim > 1 for p in g["params"])

    def test_linear_head_backcompat(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False).eval()
        assert m.head is None  # linear path: head lives inside the timm backbone
        out = m(torch.randn(2, 3, 224, 224)).y_pred
        assert out.shape == (2,)


def _stub_module(training: dict):
    m = OcclusionRegressor(
        backbone=_TINY,
        pretrained=False,
        head={"type": "mlp", "hidden_dim": 64},
        lora={"enabled": True, "rank": 4, "target_modules": ["attn.qkv", "attn.proj"]},
    )
    from face_occlusion.training.lit_module import FaceOcclusionLitModule

    mod = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(mod)
    mod.model = m
    mod.cfg = _C(training=_C(**training))
    return mod


class TestConfigureOptimizers:
    def test_discriminative_lrs_when_set(self):
        mod = _stub_module(
            dict(
                head_lr=1e-3, backbone_lr=2e-4, weight_decay=1e-4, max_epochs=5, learning_rate=2e-4
            )
        )
        out = mod.configure_optimizers()
        opt = out["optimizer"]
        assert {g["lr"] for g in opt.param_groups} == {1e-3, 2e-4}

    def test_warmup_per_step_scheduler(self):
        mod = _stub_module(
            dict(
                head_lr=1e-3,
                backbone_lr=2e-4,
                weight_decay=1e-4,
                max_epochs=5,
                learning_rate=2e-4,
                warmup_frac=0.1,
            )
        )
        mod._trainer = SimpleNamespace(estimated_stepping_batches=100)  # type: ignore[attr-defined]
        out = mod.configure_optimizers()
        assert out["lr_scheduler"]["interval"] == "step"

    def test_single_lr_fallback_unchanged(self):
        mod = _stub_module(dict(learning_rate=5e-4, weight_decay=0.01, max_epochs=5))
        out = mod.configure_optimizers()
        opt = out["optimizer"]
        assert {g["lr"] for g in opt.param_groups} == {5e-4}
        # per-epoch cosine (a scheduler object, not the per-step dict)
        assert not isinstance(out["lr_scheduler"], dict)


class TestHueAugmentation:
    def _find_color_jitter(self, compose):
        from torchvision import transforms

        for t in compose.transforms:
            if isinstance(t, transforms.RandomApply):
                for inner in t.transforms:
                    if isinstance(inner, transforms.ColorJitter):
                        return inner
            if isinstance(t, transforms.ColorJitter):
                return t
        return None

    def test_hue_flows_into_color_jitter(self):
        from face_occlusion.data.transforms import build_train_transform

        cfg = _C(
            augmentation=_C(
                resize=224,
                horizontal_flip_p=0.5,
                color_jitter_p=1.0,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                rotation_degrees=0,
            )
        )
        cj = self._find_color_jitter(build_train_transform(cfg))
        assert cj is not None and cj.hue == (-0.05, 0.05)


class _FlipSensitiveModel(nn.Module):
    """y_pred = top-left pixel of channel 0 -> changes under horizontal flip."""

    def forward(self, x):
        return OcclusionModelOutput(y_pred=x[:, 0, 0, 0])


class TestTTA:
    def _batch(self):
        images = torch.zeros(2, 3, 4, 4)
        images[0, 0, 0, 0] = 1.0  # asymmetric: top-left differs from top-right
        n = 2
        return {
            "image": images,
            "image_id": ["a", "b"],
            "filename": ["a.webp", "b.webp"],
            "path": ["a", "b"],
            "database": ["d", "d"],
            "source_subfolder": ["s", "s"],
            "group_id": ["g", "g"],
            "face_id": torch.zeros(n, dtype=torch.long),
        }

    def test_tta_changes_prediction(self):
        from face_occlusion.inference.predict import predict_dataframe

        model = _FlipSensitiveModel()
        batch = self._batch()
        plain = predict_dataframe(model, [batch], device="cpu", tta=False)
        tta = predict_dataframe(model, [batch], device="cpu", tta=True)
        # image[0] top-left=1, top-right=0 -> plain raw=1.0, tta raw=0.5
        assert plain["pred_raw"].iloc[0] == 1.0
        assert tta["pred_raw"].iloc[0] == 0.5
