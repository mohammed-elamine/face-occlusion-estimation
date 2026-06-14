"""Tests for the shared target-normalization and occlusion-bin helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_occlusion.data.normalize import (
    PERCENT_SCALE_CUTOFF,
    assign_occlusion_bin,
    normalize_target,
)

BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]


class TestNormalizeTarget:
    def test_unit_is_noop(self):
        s = pd.Series([0.0, 0.5, 1.0])
        out = normalize_target(s, "unit")
        assert np.allclose(out.to_numpy(), [0.0, 0.5, 1.0])

    def test_percent_divides_by_100(self):
        out = normalize_target(pd.Series([0.0, 50.0, 100.0]), "percent")
        assert np.allclose(out.to_numpy(), [0.0, 0.5, 1.0])

    def test_auto_detects_percent(self):
        # Max well above the cutoff -> treated as percent.
        out = normalize_target(pd.Series([0.0, 50.0, 100.0]), "auto")
        assert np.allclose(out.to_numpy(), [0.0, 0.5, 1.0])

    def test_auto_leaves_unit_data(self):
        out = normalize_target(pd.Series([0.0, 0.3, 1.0]), "auto")
        assert np.allclose(out.to_numpy(), [0.0, 0.3, 1.0])

    def test_auto_cutoff_is_strict(self):
        # A max exactly at the cutoff is NOT divided (strictly greater).
        out = normalize_target(np.array([PERCENT_SCALE_CUTOFF]), "auto")
        assert np.allclose(out, [PERCENT_SCALE_CUTOFF])

    def test_series_in_series_out_array_in_array_out(self):
        assert isinstance(normalize_target(pd.Series([0.1]), "unit"), pd.Series)
        assert isinstance(normalize_target([0.1, 0.2], "unit"), np.ndarray)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            normalize_target(pd.Series([0.1]), "nonsense")


class TestAssignOcclusionBin:
    def test_bin_edges(self):
        vals = np.array([0.0, 0.049, 0.05, 0.15, 0.39, 0.40, 0.61, 1.0])
        bins = assign_occlusion_bin(vals, BINS)
        # [0,0.05)->0, 0.05->1, 0.15->2, 0.39->3, 0.40->4, 0.61->5, 1.0->5 (clipped)
        assert bins.tolist() == [0, 0, 1, 2, 3, 4, 5, 5]

    def test_closing_edge_clipped_into_last_bin(self):
        bins = assign_occlusion_bin(np.array([1.0]), BINS)
        assert bins.tolist() == [5]
        assert int(bins.max()) == len(BINS) - 2

    def test_requires_two_edges(self):
        with pytest.raises(ValueError):
            assign_occlusion_bin(np.array([0.1]), [0.0])
