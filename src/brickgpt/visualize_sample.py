"""
Smoke-test visualization: turn one brick sample into a single 2x2 contact sheet showing the
rendered LEGO structure alongside its three orthographic silhouette masks (top / front / side).

This is a developer tool for eyeballing what the mask-conditioning pipeline "sees": the same
``bricks`` string drives both the photorealistic render and the masks, so the panels should agree.

Usage::

    uv run visualize_sample --bricks_file structure.txt --out sample.png
    uv run visualize_sample --dataset AvaLovelace/StableText2Brick --split test --index 0 --out sample.png

Rendering needs the LDraw/Blender stack (``bpy`` + ``LDRAW_LIBRARY_PATH`` + the ImportLDraw
submodule). If that is unavailable the render panel is replaced with a placeholder and the three
mask panels are still produced, so the masks can always be inspected offline.
"""
import argparse
import logging
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from brickgpt.masking import MaskConditioningConfig, three_view_masks

logger = logging.getLogger(__name__)

TILE = 256  # Side length of each panel in the 2x2 grid.


def mask_to_image(mask: np.ndarray, size: int = TILE, height_axis: bool = False) -> Image.Image:
    """
    Renders a 2D binary mask as an upscaled grayscale tile (occupied = white).

    :param mask: A 2D array; nonzero entries are treated as occupied.
    :param size: Output tile side length in pixels.
    :param height_axis: If ``True``, treat the mask as an elevation ``(horizontal, height)`` and
        orient it so the ground sits at the bottom (transpose + vertical flip).
    """
    arr = np.asarray(mask)
    if height_axis:
        arr = np.flipud(arr.T)  # rows become height (ground at the bottom)
    arr = (arr > 0).astype(np.uint8) * 255
    img = Image.fromarray(arr, mode='L').convert('RGB')
    return img.resize((size, size), Image.NEAREST)


def _placeholder(text: str, size: int = TILE) -> Image.Image:
    img = Image.new('RGB', (size, size), color=(32, 32, 32))
    draw = ImageDraw.Draw(img)
    draw.text((8, size // 2 - 4), text, fill=(200, 200, 200))
    return img


def _label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, len(text) * 7 + 8, 16), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return img


def compose_2x2(images: list[Image.Image], labels: list[str], tile: int = TILE) -> Image.Image:
    """Pastes four equally-sized tiles into a labelled 2x2 grid (order: TL, TR, BL, BR)."""
    assert len(images) == len(labels) == 4, 'compose_2x2 expects exactly four panels'
    sheet = Image.new('RGB', (tile * 2, tile * 2), color=(16, 16, 16))
    positions = [(0, 0), (tile, 0), (0, tile), (tile, tile)]
    for img, label, pos in zip(images, labels, positions):
        sheet.paste(_label(img.resize((tile, tile)), label), pos)
    return sheet


def _try_render(bricks_txt: str, tile: int = TILE) -> Image.Image:
    """Renders the structure to an image via the LDraw/Blender stack, or a placeholder on failure."""
    try:
        from brickgpt.data import BrickStructure
        from brickgpt.render_bricks import render_bricks  # imports bpy lazily

        structure = BrickStructure.from_txt(bricks_txt)
        with tempfile.TemporaryDirectory() as d:
            ldr_path = os.path.join(d, 'structure.ldr')
            png_path = os.path.join(d, 'structure.png')
            with open(ldr_path, 'w') as f:
                f.write(structure.to_ldr())
            render_bricks(ldr_path, png_path)
            return Image.open(png_path).convert('RGB').resize((tile, tile))
    except Exception as e:  # noqa: BLE001 - rendering is best-effort for a smoke test
        logger.warning('Could not render structure (%s: %s); using placeholder.', type(e).__name__, e)
        return _placeholder('render unavailable', tile)


def visualize_sample(
        bricks_txt: str,
        out_path: str,
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
        render: bool = True,
) -> Image.Image:
    """
    Builds and saves the 2x2 contact sheet for a single brick sample.

    :return: The composed PIL image (also written to ``out_path``).
    """
    views = three_view_masks(bricks_txt, cfg)
    render_panel = _try_render(bricks_txt) if render else _placeholder('render disabled')
    sheet = compose_2x2(
        images=[
            render_panel,
            mask_to_image(views['top']),
            mask_to_image(views['front'], height_axis=True),
            mask_to_image(views['side'], height_axis=True),
        ],
        labels=['render', 'mask: top (z)', 'mask: front (y)', 'mask: side (x)'],
    )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    sheet.save(out_path)
    return sheet


def _load_bricks(args: argparse.Namespace) -> str:
    if args.bricks_file:
        with open(args.bricks_file) as f:
            return f.read()
    try:
        from datasets import load_dataset  # lazy: only the --dataset path needs it
    except ImportError as e:
        raise SystemExit(
            "The --dataset path needs the HuggingFace `datasets` package, which is not in the core "
            "deps. Install it with `uv sync --extra finetuning` (or `uv pip install datasets`), or "
            "pass a brick file with --bricks_file instead."
        ) from e

    ds = load_dataset(args.dataset, split=args.split)
    return ds[args.index]['bricks']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--bricks_file', type=str, help='Path to a text file of `HxW (x,y,z)` bricks.')
    src.add_argument('--dataset', type=str, help='HF dataset path to pull one sample from.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split (with --dataset).')
    parser.add_argument('--index', type=int, default=0, help='Row index (with --dataset).')
    parser.add_argument('--out', type=str, default='sample.png', help='Output PNG path.')
    parser.add_argument('--no_render', action='store_true', help='Skip the Blender render panel.')
    args = parser.parse_args()

    bricks_txt = _load_bricks(args)
    visualize_sample(bricks_txt, args.out, render=not args.no_render)
    print(f'Saved contact sheet to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
