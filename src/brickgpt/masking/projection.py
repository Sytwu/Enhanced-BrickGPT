import logging

import numpy as np
import torch

from brickgpt.data import Brick, BrickStructure

from .config import MaskConditioningConfig, VIEW_ORDER

logger = logging.getLogger(__name__)

# Which occupancy-grid axis each view is projected along. The grid is indexed [x, y, z]:
#   top   -> project along Z (axis 2) -> (x, y)
#   front -> project along Y (axis 1) -> (x, z)
#   side  -> project along X (axis 0) -> (y, z)
VIEW_AXES: dict[str, int] = {'top': 2, 'front': 1, 'side': 0}


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
    return {name: structure.top_down_mask(axis) for name, axis in VIEW_AXES.items()}


def null_mask(cfg: MaskConditioningConfig = MaskConditioningConfig()) -> np.ndarray:
    """
    Returns the ``[NULL_MASK]`` placeholder: an all-zeros mask of the same shape as a real mask.
    Used for condition dropout and for unconditional (text-only) generation.

    :param cfg: Mask conditioning config controlling ``world_dim``.
    :return: A zeros float32 array of shape ``(world_dim, world_dim)``.
    """
    return np.zeros((cfg.world_dim, cfg.world_dim), dtype=np.float32)


def stack_views(bricks_txt: str, cfg: MaskConditioningConfig = MaskConditioningConfig()) -> np.ndarray:
    """
    Like :func:`three_view_masks`, but returns the views stacked into one array ordered by
    ``cfg.views`` (canonical :data:`~brickgpt.masking.config.VIEW_ORDER`), ready for the dataset /
    encoder.

    :return: A float32 array of shape ``(len(cfg.views), world_dim, world_dim)``.
    """
    views = three_view_masks(bricks_txt, cfg)
    return np.stack([views[name] for name in cfg.views], axis=0)


def views_to_tensors(
        views: dict[str, np.ndarray | None],
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
) -> tuple['torch.Tensor', 'torch.Tensor']:
    """
    Builds the ``(mask, has_mask)`` inference inputs from a partial dict of user-supplied views.

    This is the single-view (or subset) convenience path: the user passes only the views they have
    (e.g. ``{'top': arr}``); the rest are filled with the null mask and flagged absent. Always
    returns the full fixed-slot stack in ``cfg.views`` order with a leading batch dim of 1.

    :param views: Maps a subset of ``cfg.views`` to a ``(world_dim, world_dim)`` array. Missing keys
                  (or ``None`` values) are treated as absent views.
    :return: ``(mask, has_mask)`` of shapes ``(1, V, world_dim, world_dim)`` float and ``(1, V)`` bool.
    """
    masks, presence = [], []
    for name in cfg.views:
        arr = views.get(name)
        if arr is None:
            masks.append(null_mask(cfg))
            presence.append(False)
        else:
            masks.append(np.asarray(arr, dtype=np.float32))
            presence.append(True)
    mask = torch.from_numpy(np.stack(masks, axis=0)).float().unsqueeze(0)   # [1, V, H, W]
    has_mask = torch.tensor(presence, dtype=torch.bool).unsqueeze(0)        # [1, V]
    return mask, has_mask
