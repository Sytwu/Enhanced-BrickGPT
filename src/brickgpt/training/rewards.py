"""
Reward functions for the GRPO RL phase (Step C / D3-D4).

Design (D3): a **syntax gate** short-circuits to ``-1`` for any ill-formatted / out-of-library
output (skipping structure building, IoU, and rendering). Otherwise the total reward sums a graded
**overlap** penalty, a connectivity-based **stability** term, a 3-view silhouette **IoU** averaged
over the *provided* views, and an optional CLIP **semantic** term.

These are fast and Tensor-friendly. **Never call the Gurobi solver inside the RL loop** -- stability
uses the connectivity graph. Rendering/CLIP (semantic) is seconds-per-sample like Gurobi: it is
*off by default* and the score must be passed in precomputed, so this module pulls in no Blender/CLIP
dependency.

Overlap vs stability double-count (D3.3): resolved by penalizing overlap **once**. The overlap term
is the only place collisions are penalized; stability is computed in an overlap-independent way
(:func:`~brickgpt.stability_analysis.connectivity_score` builds its graph from the brick list and is
robust to overlapping bricks), so a collision does not also force stability to ``-1``.
"""
import logging
from dataclasses import dataclass

import numpy as np
import torch

from brickgpt.data import Brick, BrickStructure
from brickgpt.masking import MaskConditioningConfig, VIEW_AXES
from brickgpt.stability_analysis import connectivity_score

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Ablation knobs for the GRPO reward (D4). Syntax is a hard gate, not a weighted term."""
    use_syntax_gate: bool = True
    use_overlap: bool = True
    use_stability: bool = True
    use_iou: bool = True
    use_semantic: bool = False   # off by default: needs rendering (seconds/sample)
    w_overlap: float = 1.0
    w_stability: float = 1.0
    w_iou: float = 1.0
    w_semantic: float = 1.0
    clip_lo: float = 0.15
    clip_hi: float = 0.35
    use_multi_turn: bool = False

    def __post_init__(self):
        if self.use_multi_turn and (self.use_iou or self.use_semantic):
            raise ValueError('multi-turn (step-level) rewards are incompatible with use_iou / '
                             'use_semantic, which need a complete trajectory.')


@dataclass
class StepReward:
    """Per-brick (per-turn) reward for the multi-turn GRPO path (step-only: syntax/overlap/stability).

    ``None`` components mean the term was disabled, or the brick was gated out (invalid / out-of-bounds)
    before structure-level terms could be computed.
    """
    line: str
    total: float
    syntax_ok: bool
    overlap: float | None = None
    stability: float | None = None


@dataclass
class RewardBreakdown:
    """Per-component reward, for logging (D6). ``None`` means a term was gated/disabled/not applicable."""
    total: float
    syntax_ok: bool
    overlap: float | None = None
    stability: float | None = None
    iou: float | None = None
    semantic: float | None = None

    def components(self) -> dict[str, float]:
        """The non-None components (for averaging into wandb), excluding ``total``/``syntax_ok``."""
        return {k: v for k, v in (('overlap', self.overlap), ('stability', self.stability),
                                  ('iou', self.iou), ('semantic', self.semantic)) if v is not None}


# --- primitives ----------------------------------------------------------------------------------

def _parse_line(line: str) -> Brick | None:
    """Parses one brick line; returns the :class:`Brick`, or ``None`` if invalid / not in library."""
    try:
        brick = Brick.from_txt(line)
        _ = brick.brick_id  # raises ValueError if dimensions are not in the library
    except ValueError:
        return None
    return brick


def _valid_bricks(bricks_txt: str) -> list[Brick] | None:
    """Parses every non-empty line; returns the brick list, or ``None`` if any line is invalid."""
    bricks = []
    for line in bricks_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        brick = _parse_line(line)
        if brick is None:
            return None
        bricks.append(brick)
    return bricks if bricks else None


def syntax_reward(bricks_txt: str) -> float:
    """+1 if every non-empty line is a syntactically valid, in-library brick; -1 otherwise."""
    return 1.0 if _valid_bricks(bricks_txt) is not None else -1.0


def _safe_structure(bricks: list[Brick], cfg: MaskConditioningConfig) -> BrickStructure | None:
    """Builds a BrickStructure, returning ``None`` if a brick lands outside the voxel grid (bad z)."""
    try:
        return BrickStructure(bricks, world_dim=cfg.world_dim)
    except (IndexError, ValueError):
        return None


def overlap_penalty(structure: BrickStructure) -> float:
    """
    Graded overlap penalty (D3.2): ``-Σ_v max(0, occupancy_v - 1)`` over voxels. 0 when collision-free
    (the usual case for rejection-sampled outputs); increasingly negative as bricks overlap.
    """
    occ = structure.voxel_occupancy
    return -float(np.clip(occ - 1, 0, None).sum())


def stability_reward_from_structure(structure: BrickStructure) -> float:
    """
    +1 if the structure is fully connected to the ground with no floating / out-of-bounds bricks;
    -1 otherwise. Overlap-independent: uses :func:`connectivity_score` on the brick list directly so
    a collision is *not* double-penalized here (it is handled by :func:`overlap_penalty`).
    """
    if structure.has_out_of_bounds_bricks() or structure.has_floating_bricks():
        return -1.0
    return 1.0 if connectivity_score(structure).max() < 1 else -1.0


def silhouette_iou_from_structure(
        structure: BrickStructure,
        target_views: np.ndarray | torch.Tensor,
        has_mask,
        cfg: MaskConditioningConfig,
) -> float | None:
    """
    Mean pixel-wise IoU between the generated structure's silhouettes and ``target_views``, averaged
    over the **provided** views only (D3.4 per-view routing).

    :param target_views: Target silhouettes of shape ``(V, H, W)`` in ``cfg.views`` order.
    :param has_mask: Per-view presence, shape ``(V,)`` (bool/0-1).
    :return: Mean IoU in ``[0, 1]``, or ``None`` if no view is provided (routed out).
    """
    target_views = np.asarray(target_views, dtype=np.float32) > 0.5
    presence = np.asarray(has_mask).astype(bool).reshape(-1)
    ious = []
    for vi, name in enumerate(cfg.views):
        if not presence[vi]:
            continue
        pred = structure.top_down_mask(VIEW_AXES[name]) > 0.5
        tgt = target_views[vi]
        inter = np.logical_and(pred, tgt).sum()
        union = np.logical_or(pred, tgt).sum()
        ious.append(1.0 if union == 0 else inter / union)
    return float(np.mean(ious)) if ious else None


def normalize_clip_score(score: float, cfg: RewardConfig) -> float:
    """Maps a raw CLIP cosine to [0,1] via ``clip((s - clip_lo) / (clip_hi - clip_lo), 0, 1)`` (D3.5)."""
    return float(np.clip((score - cfg.clip_lo) / (cfg.clip_hi - cfg.clip_lo), 0.0, 1.0))


# --- total ---------------------------------------------------------------------------------------

def compute_reward(
        bricks_txt: str,
        target_views: np.ndarray | torch.Tensor | None = None,
        has_mask=None,
        cfg: RewardConfig = RewardConfig(),
        mask_cfg: MaskConditioningConfig = MaskConditioningConfig(),
        clip_score: float | None = None,
) -> RewardBreakdown:
    """
    Computes the GRPO reward for a single completion, returning a :class:`RewardBreakdown` (D6).

    :param bricks_txt: The generated brick list.
    :param target_views: ``(V, H, W)`` target silhouettes (``cfg.views`` order); needed for IoU.
    :param has_mask: ``(V,)`` per-view presence; IoU is averaged over provided views only.
    :param cfg: Reward ablation config.
    :param mask_cfg: Mask/geometry config (``world_dim``, view order).
    :param clip_score: Precomputed raw CLIP cosine for the semantic term (rendering is the caller's
                       job; pass ``None`` to skip even when ``use_semantic`` is set).
    """
    # 1. Syntax gate -- short-circuit, skip everything else.
    bricks = _valid_bricks(bricks_txt)
    if cfg.use_syntax_gate and bricks is None:
        return RewardBreakdown(total=-1.0, syntax_ok=False)
    if bricks is None:  # gate disabled but still unparseable: nothing to score on.
        return RewardBreakdown(total=0.0, syntax_ok=False)

    structure = _safe_structure(bricks, mask_cfg)
    if structure is None:
        return RewardBreakdown(total=-1.0, syntax_ok=False)

    total = 0.0
    overlap = stability = iou = semantic = None

    if cfg.use_overlap:
        overlap = overlap_penalty(structure)
        total += cfg.w_overlap * overlap
    if cfg.use_stability:
        stability = stability_reward_from_structure(structure)
        total += cfg.w_stability * stability
    if cfg.use_iou and target_views is not None and has_mask is not None:
        iou = silhouette_iou_from_structure(structure, target_views, has_mask, mask_cfg)
        if iou is not None:  # None == no provided views (routed out)
            total += cfg.w_iou * iou
    if cfg.use_semantic and clip_score is not None:
        semantic = normalize_clip_score(clip_score, cfg)
        total += cfg.w_semantic * semantic

    return RewardBreakdown(total=total, syntax_ok=True,
                           overlap=overlap, stability=stability, iou=iou, semantic=semantic)


# --- multi-turn (per-brick step rewards) ---------------------------------------------------------

def stepwise_rewards(
        lines: list[str],
        cfg: RewardConfig = RewardConfig(),
        mask_cfg: MaskConditioningConfig = MaskConditioningConfig(),
) -> list[StepReward]:
    """
    Per-brick step rewards for the multi-turn GRPO path (D5 / step-level), with stability scored
    **once at the trajectory end** rather than per brick.

    Step-only by contract -- only the syntax gate, ``overlap`` (per brick), and ``stability`` (terminal)
    are used; the IoU and semantic terms need a complete trajectory and are *not* computed here (mirrors
    :meth:`RewardConfig.__post_init__`, which forbids ``use_multi_turn`` with those terms).

    Per line:

    * unparseable / not-in-library -> ``total = -1`` (syntax gate); the brick is **not** added, so a
      later valid brick is scored against the structure so far (not poisoned by the bad line).
    * out-of-bounds (``BrickStructure`` rejects it) -> ``total = -1``; not added.
    * otherwise added to the running structure, with ``total = w_overlap * Δoverlap`` where
      ``Δoverlap`` is the *incremental* overlap penalty of this brick (so the per-trajectory sum
      telescopes to the final overlap penalty, matching :func:`compute_reward`'s overlap magnitude).

    **Terminal stability:** stability is a global property a causal per-step check can't judge mid-build
    (a brick's support may arrive in a later turn), so it is computed *once* on the final structure and
    added to the **last committed brick's** step. The discounted return-to-go in
    :func:`~brickgpt.training.grpo_masked.compute_stepwise_advantages` then propagates that signal back
    to the earlier bricks -- without over-penalizing a not-yet-grounded prefix.

    :param lines: The brick lines of one completion, in generation order (caller splits on newlines).
    :return: One :class:`StepReward` per input line, in order. Only the terminal step carries a
             non-``None`` ``stability``.
    """
    added: list[Brick] = []
    prev_overlap = 0.0
    out: list[StepReward] = []
    last_committed = -1
    for line in lines:
        line = line.strip()
        brick = _parse_line(line) if line else None
        if brick is None:
            out.append(StepReward(line=line, total=-1.0, syntax_ok=False))
            continue
        structure = _safe_structure(added + [brick], mask_cfg)
        if structure is None:  # out-of-bounds: cannot build this brick into the grid
            out.append(StepReward(line=line, total=-1.0, syntax_ok=False))
            continue

        total = 0.0
        overlap = None
        if cfg.use_overlap:
            cum_overlap = overlap_penalty(structure)
            overlap = cum_overlap - prev_overlap          # incremental (<= 0)
            prev_overlap = cum_overlap
            total += cfg.w_overlap * overlap

        out.append(StepReward(line=line, total=total, syntax_ok=True, overlap=overlap))
        added.append(brick)
        last_committed = len(out) - 1

    # Terminal stability: one global connectivity check on the final structure, attached to the last
    # committed brick (return-to-go spreads the credit back over the earlier bricks).
    if cfg.use_stability and last_committed >= 0:
        stability = stability_reward_from_structure(_safe_structure(added, mask_cfg))
        out[last_committed].stability = stability
        out[last_committed].total += cfg.w_stability * stability
    return out
