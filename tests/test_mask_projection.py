import numpy as np
import pytest

from brickgpt.data import BrickStructure
from brickgpt.masking import MaskConditioningConfig, bricks_to_mask, null_mask


def test_top_down_mask_collapses_height():
    # Two 2x2 bricks stacked along z share the same footprint; the top-down mask is 2x2 of 1s.
    bricks = BrickStructure.from_txt('2x2 (0,0,0)\n2x2 (0,0,1)\n')
    mask = bricks.top_down_mask(axis=2)
    assert mask.shape == (20, 20)
    assert mask.dtype == np.float32
    assert mask.sum() == 4
    assert np.array_equal(mask[0:2, 0:2], np.ones((2, 2), dtype=np.float32))
    assert mask[2:, :].sum() == 0 and mask[:, 2:].sum() == 0


def test_bricks_to_mask_matches_structure():
    bricks_txt = '2x6 (0,0,0)\n2x6 (2,0,0)\n'
    mask = bricks_to_mask(bricks_txt)
    expected = BrickStructure.from_txt(bricks_txt).top_down_mask(axis=2)
    assert np.array_equal(mask, expected)
    # Footprint occupies x in [0,4), y in [0,6) -> 24 cells.
    assert mask.sum() == 24
    assert np.array_equal(mask[0:4, 0:6], np.ones((4, 6), dtype=np.float32))


def test_bricks_to_mask_skips_malformed_lines():
    mask = bricks_to_mask('2x6 (0,0,0)\nGARBAGE\n\n')
    assert mask.sum() == 12  # Only the one valid 2x6 brick contributes.


def test_null_mask_is_zeros():
    cfg = MaskConditioningConfig()
    m = null_mask(cfg)
    assert m.shape == (cfg.world_dim, cfg.world_dim)
    assert m.dtype == np.float32
    assert not m.any()


@pytest.mark.parametrize('world_dim', [10, 20, 32])
def test_mask_respects_world_dim(world_dim):
    cfg = MaskConditioningConfig(world_dim=world_dim)
    assert bricks_to_mask('2x2 (0,0,0)\n', cfg).shape == (world_dim, world_dim)
    assert null_mask(cfg).shape == (world_dim, world_dim)
