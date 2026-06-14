"""Tests for the evaluation lenses (importance reweighting of the challenge metric)."""

from __future__ import annotations

import numpy as np
import pytest

from face_occlusion.metrics.eval_lenses import (
    DEFAULT_LENS_EDGES,
    LENS_NAMES,
    balanced_proportions,
    importance_weights,
    lens_weights,
    load_test_distribution,
)


def _targets(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.beta(1.5, 8.0, size=n), 0, 1)  # right-skewed like train


def test_official_lens_is_none():
    assert lens_weights("official", _targets()) is None


def test_unknown_lens_raises():
    with pytest.raises(ValueError):
        lens_weights("nonsense", _targets())


def test_weights_are_mean_normalised():
    for name in ("balanced", "test_matched"):
        w = lens_weights(name, _targets())
        assert w is not None
        assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)


def test_clip_caps_extreme_weights():
    y = _targets()
    capped = importance_weights(
        y, balanced_proportions(len(DEFAULT_LENS_EDGES) - 1), DEFAULT_LENS_EDGES, clip_max=3.0
    )
    # After clip+renormalise the max stays bounded near the cap (renorm can nudge it up a
    # little, but never to the raw ~1/p_train explosion of the sparse tail).
    assert capped.max() <= 3.0 * 1.5


def test_matching_train_distribution_gives_uniform_weights():
    # If the target distribution equals the empirical train distribution, weights are ~1.
    y = _targets()
    n_bins = len(DEFAULT_LENS_EDGES) - 1
    from face_occlusion.data.normalize import assign_occlusion_bin

    bins = assign_occlusion_bin(y, DEFAULT_LENS_EDGES)
    emp = np.bincount(bins, minlength=n_bins).astype(float)
    emp = emp / emp.sum()
    w = importance_weights(y, emp, DEFAULT_LENS_EDGES, clip_max=100.0)
    assert np.allclose(w, 1.0, atol=1e-2)


def test_balanced_upweights_rare_high_bins():
    # Under the balanced lens, rare high-occlusion rows get the largest weights.
    y = _targets()
    w = lens_weights("balanced", y)
    assert w[y >= 0.4].mean() > w[y < 0.05].mean()


def test_load_test_distribution_normalised():
    edges, props = load_test_distribution()
    assert len(props) == len(edges) - 1
    assert float(props.sum()) == pytest.approx(1.0, abs=1e-9)
    assert tuple(edges) == DEFAULT_LENS_EDGES  # config edges match the lens edges


def test_lens_names_resolve():
    y = _targets()
    for name in LENS_NAMES:
        w = lens_weights(name, y)
        assert w is None or w.shape == y.shape


def test_rebin_proportions_aggregates_and_normalises():
    from face_occlusion.metrics.eval_lenses import rebin_proportions

    src_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    src_props = [0.4, 0.3, 0.2, 0.1]
    out = rebin_proportions(src_edges, src_props, [0.0, 0.5, 1.0])
    assert float(out.sum()) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.7)  # 0.4 + 0.3
    assert out[1] == pytest.approx(0.3)  # 0.2 + 0.1


def test_rebin_proportions_splits_partial_overlap():
    from face_occlusion.metrics.eval_lenses import rebin_proportions

    # A single source bin [0, 1) with all mass, split onto [0,0.25),[0.25,1.0).
    out = rebin_proportions([0.0, 1.0], [1.0], [0.0, 0.25, 1.0])
    assert out[0] == pytest.approx(0.25)
    assert out[1] == pytest.approx(0.75)
