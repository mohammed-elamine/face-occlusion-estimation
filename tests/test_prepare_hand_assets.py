"""Tests for the hand-asset segmentation + anchor derivation."""

from __future__ import annotations

import numpy as np
from scripts.data.prepare_hand_assets import derive_hand_anchors, segment_white_background


def _hand_on_white(size=200):
    """A dark 'hand' (vertical bar) on white, with the wrist entering the bottom."""
    rgb = np.full((size, size, 3), 250, dtype=np.uint8)  # white background
    # palm/fingers block (upper) + wrist column reaching the bottom edge.
    rgb[40:140, 70:130] = (90, 70, 60)  # palm
    rgb[140:size, 90:110] = (90, 70, 60)  # wrist to bottom edge
    return rgb


def test_segmentation_finds_hand_not_background():
    rgb = _hand_on_white()
    alpha = segment_white_background(rgb)
    assert alpha[90, 100]  # inside the palm -> hand
    assert not alpha[10, 10]  # corner -> white background
    # The hand occupies a sensible fraction (not everything, not nothing).
    frac = alpha.mean()
    assert 0.05 < frac < 0.6


def test_segmentation_fills_interior_holes():
    rgb = _hand_on_white()
    rgb[80:100, 90:110] = 250  # a white "ring" hole inside the palm
    alpha = segment_white_background(rgb)
    assert alpha[90, 100]  # hole is filled back into the hand


def test_anchors_palm_above_wrist_for_bottom_entry():
    alpha = segment_white_background(_hand_on_white(200))
    anchors = derive_hand_anchors(alpha)
    palm, wrist = anchors["palm"], anchors["wrist"]
    # Wrist enters from the bottom => wrist is lower (larger y) than the palm.
    assert wrist[1] > palm[1]
    # Wrist is near the bottom edge.
    assert wrist[1] > 0.85 * alpha.shape[0]
