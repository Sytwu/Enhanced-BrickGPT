import logging

import numpy as np

from brickgpt.data import Brick, BrickStructure

from .config import MaskConditioningConfig

logger = logging.getLogger(__name__)


def _parse_structure(bricks_txt: str, cfg: MaskConditioningConfig) -> BrickStructure:
    """Builds a :class:`BrickStructure` from text, skipping ill-formatted lines (robust to noise)."""
    bricks = []
    for line in bricks_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            bricks.append(Brick.from_txt(line))
        except ValueError:
            logger.warning('Skipping ill-formatted brick while building mask: %r', line)
            continue
    return BrickStructure(bricks, world_dim=cfg.world_dim)


def bricks_to_mask(bricks_txt: str, cfg: MaskConditioningConfig = MaskConditioningConfig()) -> np.ndarray:
    """
    Projects a brick structure (in text format) into a 2D binary silhouette mask.

    Reuses the occupancy grid built by :class:`~brickgpt.data.BrickStructure`, so no external
    renderer (PyTorch3D / Trimesh) is needed: the mask is the top-down (Z-axis) projection of the
    ground-truth voxels. Malformed lines are skipped so the function is robust to noisy data.

    :param bricks_txt: The brick structure in text format (one ``HxW (x,y,z)`` brick per line).
    :param cfg: Mask conditioning config controlling ``world_dim`` and ``projection_axis``.
    :return: A binary float32 array of shape ``(world_dim, world_dim)``.
    """
    return _parse_structure(bricks_txt, cfg).top_down_mask(cfg.projection_axis)


def three_view_masks(
        bricks_txt: str,
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
) -> dict[str, np.ndarray]:
    """
    Projects a brick structure into its three orthographic silhouettes (top / front / side).

    The voxel grid is indexed ``[x, y, z]`` (``x`` is the ``h`` axis, ``y`` the ``w`` axis, ``z`` the
    layer). Each view is the binary projection along one axis:

    - ``top``   = project along Z (axis 2) -> ``(x, y)``  (the silhouette used for conditioning)
    - ``front`` = project along Y (axis 1) -> ``(x, z)``  (an elevation; height is the second axis)
    - ``side``  = project along X (axis 0) -> ``(y, z)``  (an elevation; height is the second axis)

    These are returned in their *native* (un-rotated) orientation so they can be reused by the IoU
    reward; the visualization script handles display rotation. See [[mask-conditioning-feature]].

    :return: A dict ``{'top': ndarray, 'front': ndarray, 'side': ndarray}`` of binary float32 arrays.
    """
    structure = _parse_structure(bricks_txt, cfg)
    return {
        'top': structure.top_down_mask(2),
        'front': structure.top_down_mask(1),
        'side': structure.top_down_mask(0),
    }


def null_mask(cfg: MaskConditioningConfig = MaskConditioningConfig()) -> np.ndarray:
    """
    Returns the ``[NULL_MASK]`` placeholder: an all-zeros mask of the same shape as a real mask.
    Used for condition dropout and for unconditional (text-only) generation.

    :param cfg: Mask conditioning config controlling ``world_dim``.
    :return: A zeros float32 array of shape ``(world_dim, world_dim)``.
    """
    return np.zeros((cfg.world_dim, cfg.world_dim), dtype=np.float32)
