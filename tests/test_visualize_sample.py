import numpy as np
from PIL import Image

from brickgpt.masking import bricks_to_mask, three_view_masks
from brickgpt.visualize_sample import compose_2x2, mask_to_image, visualize_sample


BRICKS = '2x4 (0,0,0)\n2x4 (0,0,1)\n'  # a 2-layer stack


def test_three_view_masks_shapes_and_top_matches_projection():
    views = three_view_masks(BRICKS)
    assert set(views) == {'top', 'front', 'side'}
    assert views['top'].shape == (20, 20)   # (x, y)
    assert views['front'].shape == (20, 20)  # (x, z)
    assert views['side'].shape == (20, 20)   # (y, z)
    # The 'top' view is exactly the conditioning silhouette.
    np.testing.assert_array_equal(views['top'], bricks_to_mask(BRICKS))
    # A 2-layer stack occupies two z-rows in each elevation view.
    assert views['front'][:, :2].any() and not views['front'][:, 2:].any()


def test_mask_to_image_is_rgb_tile():
    img = mask_to_image(np.eye(20, dtype=np.float32), size=64)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64) and img.mode == 'RGB'


def test_compose_2x2_grid_size():
    tiles = [Image.new('RGB', (10, 10)) for _ in range(4)]
    sheet = compose_2x2(tiles, ['a', 'b', 'c', 'd'], tile=32)
    assert sheet.size == (64, 64)


def test_visualize_sample_writes_file(tmp_path):
    out = tmp_path / 'sheet.png'
    sheet = visualize_sample(BRICKS, str(out), render=False)  # skip Blender
    assert out.exists()
    assert sheet.size == (512, 512)  # 2x2 of 256px tiles
