import pytest

from brickgpt.masking import bricks_to_mask
from brickgpt.training import syntax_reward, silhouette_iou_reward, stability_reward, total_reward


@pytest.mark.parametrize('text,expected', [
    ('2x2 (0,0,0)\n', 1.0),
    ('2x6 (0,0,0)\n2x6 (2,0,0)\n', 1.0),
    ('garbage', -1.0),
    ('', -1.0),
    ('3x3 (0,0,0)\n', -1.0),  # 3x3 is not in the brick library
])
def test_syntax_reward(text, expected):
    assert syntax_reward(text) == expected


def test_silhouette_iou_perfect_and_partial():
    target = bricks_to_mask('2x2 (0,0,0)\n2x2 (2,0,0)\n')  # 8-cell footprint
    assert silhouette_iou_reward('2x2 (0,0,0)\n2x2 (2,0,0)\n', target) == 1.0
    # Only the first brick is generated -> 4/8 overlap.
    assert silhouette_iou_reward('2x2 (0,0,0)\n', target) == pytest.approx(0.5)
    # Empty prediction against a non-empty target.
    assert silhouette_iou_reward('', target) == 0.0


def test_silhouette_iou_both_empty_is_one():
    import numpy as np
    assert silhouette_iou_reward('', np.zeros((20, 20), dtype=np.float32)) == 1.0


@pytest.mark.parametrize('text,expected', [
    ('2x6 (0,0,0)\n2x6 (2,0,0)\n', 1.0),   # connected to ground, stable
    ('2x6 (0,0,0)\n2x6 (2,0,1)\n', -1.0),  # floating brick
    ('garbage', -1.0),
])
def test_stability_reward(text, expected):
    assert stability_reward(text) == expected


def test_total_reward_routing_drops_iou_for_null_mask():
    text = '2x2 (0,0,0)\n'
    target = bricks_to_mask(text)
    weights = (1.0, 1.0, 1.0)

    with_mask = total_reward(text, target, has_mask=True, weights=weights)
    without_mask = total_reward(text, target, has_mask=False, weights=weights)

    # IoU is perfect (1.0); routing should make the difference exactly w1 * IoU = 1.0.
    assert with_mask - without_mask == pytest.approx(1.0)
