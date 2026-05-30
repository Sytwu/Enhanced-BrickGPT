"""
Reward functions for the GRPO RL phase (Phase 4).

These are fast, Tensor-friendly baselines. **Never call the Gurobi solver inside the RL loop** — use
the connectivity check (or the TODO Union-Find rewrite) for stability.

TODO(handoff): tune the reward shaping/weights, and vectorize ``silhouette_iou_reward`` and
``stability_reward`` to operate on a whole group of samples at once on the GPU.
"""
import logging

import numpy as np
import torch

from brickgpt.data import Brick, BrickStructure

from brickgpt.masking import MaskConditioningConfig

logger = logging.getLogger(__name__)


def syntax_reward(bricks_txt: str) -> float:
    """
    +1 if every non-empty line is a syntactically valid, in-library brick; -1 otherwise.

    Reuses :meth:`brickgpt.data.Brick.from_txt` (the same regex used during inference) and the
    brick-library lookup behind :attr:`Brick.brick_id`.
    """
    lines = [line.strip() for line in bricks_txt.splitlines() if line.strip()]
    if not lines:
        return -1.0
    for line in lines:
        try:
            brick = Brick.from_txt(line)
            _ = brick.brick_id  # Raises ValueError if dimensions are not in the library.
        except ValueError:
            return -1.0
    return 1.0


def bricks_to_occupancy_tensor(
        bricks_txt: str,
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
        device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """
    "Paints" the generated bricks into a 2D top-down occupancy tensor by splatting each brick's
    footprint slice. Stays on ``device`` so the IoU can be computed without a CPU round-trip.

    :return: A boolean tensor of shape ``(world_dim, world_dim)``.
    """
    occ = torch.zeros((cfg.world_dim, cfg.world_dim), dtype=torch.bool, device=device)
    for line in bricks_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            brick = Brick.from_txt(line)
        except ValueError:
            continue
        x0, x1 = max(brick.x, 0), min(brick.x + brick.h, cfg.world_dim)
        y0, y1 = max(brick.y, 0), min(brick.y + brick.w, cfg.world_dim)
        if x0 < x1 and y0 < y1:
            occ[x0:x1, y0:y1] = True
    return occ


def silhouette_iou_reward(
        bricks_txt: str,
        target_mask: np.ndarray | torch.Tensor,
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
        device: str | torch.device = 'cpu',
) -> float:
    """
    Pixel-wise IoU between the generated structure's top-down silhouette and the target mask.

    :return: IoU in ``[0, 1]`` (1.0 if both silhouettes are empty).
    """
    pred = bricks_to_occupancy_tensor(bricks_txt, cfg, device)
    target = torch.as_tensor(target_mask, device=device) > 0.5
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return (intersection.float() / union.float()).item()


def stability_reward(
        bricks_txt: str,
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
) -> float:
    """
    +1 if the structure is collision-free, in-bounds, has no floating bricks, and is fully connected
    to the ground; -1 otherwise. Uses the graph-based connectivity check (not Gurobi).

    TODO(handoff): return a graded reward (e.g. fraction of connected bricks) and replace the
    networkx connectivity check with a vectorized Union-Find / adjacency-matrix pass.
    """
    structure = BrickStructure(
        [Brick.from_txt(line) for line in bricks_txt.splitlines() if line.strip()],
        world_dim=cfg.world_dim,
    ) if syntax_reward(bricks_txt) > 0 else None

    if structure is None:
        return -1.0
    if structure.has_collisions() or structure.has_out_of_bounds_bricks():
        return -1.0
    return 1.0 if structure.is_connected() else -1.0


def total_reward(
        bricks_txt: str,
        target_mask: np.ndarray | torch.Tensor,
        has_mask: bool,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cfg: MaskConditioningConfig = MaskConditioningConfig(),
        device: str | torch.device = 'cpu',
) -> float:
    """
    Dynamic reward routing (Phase 5): drop the IoU term for ``[NULL_MASK]`` batches.

        total = (w1 if has_mask else 0) * IoU + w2 * Stability + w3 * Syntax
    """
    w1, w2, w3 = weights
    iou = silhouette_iou_reward(bricks_txt, target_mask, cfg, device) if has_mask else 0.0
    return (w1 if has_mask else 0.0) * iou \
        + w2 * stability_reward(bricks_txt, cfg) \
        + w3 * syntax_reward(bricks_txt)
