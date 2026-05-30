"""
Drive the constrained mask-conditioned generation loop (Step B) from an *in-memory* training model.

Used by the SFT IoU probe (does the model actually use the mask?) and, later, by GRPO rollouts. The
bridge reuses :meth:`LLM.from_model` so no checkpoint is reloaded from disk per eval.
"""
import logging

import numpy as np
import torch

from brickgpt.masking import MaskConditioningConfig, stack_views
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig
from brickgpt.models.llm import LLM
from brickgpt.training.rewards import silhouette_iou_from_structure, _valid_bricks

logger = logging.getLogger(__name__)


def build_mask_generator(masked_model, tokenizer, brickgpt_cfg: BrickGPTConfig) -> BrickGPT:
    """Wraps a :class:`BrickGPTWithMask` into a :class:`BrickGPT` that generates from its LLM + encoder."""
    device = next(masked_model.parameters()).device
    llm = LLM.from_model(masked_model.base, tokenizer, str(device))
    bg = BrickGPT(brickgpt_cfg, llm=llm)
    bg.mask_prefix_encoder = masked_model.mask_prefix_encoder
    return bg


def _structure_iou(bricks_txt: str, target_views: np.ndarray, mask_cfg: MaskConditioningConfig) -> float:
    """IoU of a generated structure's three silhouettes vs. the target (all views provided)."""
    bricks = _valid_bricks(bricks_txt)
    if bricks is None:
        return 0.0
    from brickgpt.data import BrickStructure
    structure = BrickStructure(bricks, world_dim=mask_cfg.world_dim)
    iou = silhouette_iou_from_structure(structure, target_views, [True] * mask_cfg.num_views, mask_cfg)
    return iou if iou is not None else 0.0


@torch.no_grad()
def iou_probe(
        masked_model,
        tokenizer,
        eval_examples: list[dict],
        mask_cfg: MaskConditioningConfig,
        brickgpt_cfg: BrickGPTConfig,
) -> dict[str, float]:
    """
    The SFT verification probe: for each held-out ``(caption, bricks)``, generate once *conditioned
    on the GT silhouette* and once with an *absent* (null) mask, and compare silhouette IoU to the GT.

    A positive ``iou_lift`` (masked > null) is direct evidence the model conditions on the mask
    rather than reproducing the caption prior. If the lift stays ~0, escalate to LoRA-on-LLM (per the
    confirmed SFT strategy).

    :param eval_examples: rows with ``captions`` (list) and ``bricks`` (str).
    :return: ``{'iou_masked', 'iou_null', 'iou_lift', 'n'}`` (means over the examples).
    """
    was_training = masked_model.training
    masked_model.eval()
    bg = build_mask_generator(masked_model, tokenizer, brickgpt_cfg)
    device = next(masked_model.parameters()).device

    masked_ious, null_ious = [], []
    for ex in eval_examples:
        caption = ex['captions'][0] if ex.get('captions') else ex['caption']
        target_views = stack_views(ex['bricks'], mask_cfg)                      # [V, H, W]
        mask = torch.from_numpy(target_views).unsqueeze(0).float().to(device)   # [1, V, H, W]
        present = torch.ones(1, mask_cfg.num_views, dtype=torch.bool, device=device)
        absent = torch.zeros(1, mask_cfg.num_views, dtype=torch.bool, device=device)

        masked_out = bg(caption, mask=mask, has_mask=present)
        null_out = bg(caption, mask=torch.zeros_like(mask), has_mask=absent)
        masked_ious.append(_structure_iou(masked_out['bricks'].to_txt(), target_views, mask_cfg))
        null_ious.append(_structure_iou(null_out['bricks'].to_txt(), target_views, mask_cfg))

    if was_training:
        masked_model.train()

    iou_masked = float(np.mean(masked_ious)) if masked_ious else 0.0
    iou_null = float(np.mean(null_ious)) if null_ious else 0.0
    return {'iou_masked': iou_masked, 'iou_null': iou_null,
            'iou_lift': iou_masked - iou_null, 'n': float(len(eval_examples))}
