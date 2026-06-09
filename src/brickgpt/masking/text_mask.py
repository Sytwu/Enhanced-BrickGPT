"""
Text-token mask conditioning: serialize the three orthographic silhouettes into a run-length
coordinate block that is appended to the user prompt (an alternative to the CNN prefix encoder).

Single source of truth shared by dataset prep (`prepare_text_mask_dataset`) and evaluation
(`scripts/eval_text_mask_iou.py`) so the prompt the model is trained on is byte-for-byte the prompt
it is evaluated on. The Stage-0 probe (`scripts/measure_mask_tokens.py`) showed per-pixel coordinate
listing is far too long (~1455 tok mean); the run-length form here averages ~369 tok with a bounded
tail, while staying coordinate-like so output bricks can attend back to it.
"""
from collections.abc import Sequence

import numpy as np

from brickgpt.models import create_instruction

from .config import MaskConditioningConfig, VIEW_ORDER
from .projection import three_view_masks

# Per-view header label and the (row-axis, col-axis) names of the 2D silhouette, matching
# brickgpt.masking.projection.VIEW_AXES (top->along Z->(x,y), front->along Y->(x,z), side->along X->(y,z)).
VIEW_LABELS: dict[str, tuple[str, str, str]] = {
    'top': ('z-proj', 'x', 'y'),
    'front': ('y-proj', 'x', 'z'),
    'side': ('x-proj', 'y', 'z'),
}

_BLOCK_HEADER = 'Target silhouettes (occupied columns per row, run-length):'


def _row_runs(row: np.ndarray) -> str:
    """Encodes one silhouette row as run-length column ranges, e.g. '3-19' or '0-2,4-5'. '' if empty."""
    cols = np.flatnonzero(np.asarray(row) > 0.5)
    if len(cols) == 0:
        return ''
    runs, start, prev = [], int(cols[0]), int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return ','.join(f'{a}' if a == b else f'{a}-{b}' for a, b in runs)


def serialize_views_rle(views: dict[str, np.ndarray], view_names: Sequence[str]) -> str:
    """
    Serializes the named views into the run-length mask block. ``view_names`` selects which views to
    include (the dropout-survivors); an empty selection returns ``''`` (no block at all).

    Each included view is one line ``Name (proj) [rows=<axis>, cols=<axis>]: i:runs; j:runs; ...``
    listing only the non-empty rows. Rows are emitted in ascending index, so the encoding is a
    canonical function of the silhouette (no ordering ambiguity for the model to learn around).
    """
    names = [n for n in VIEW_ORDER if n in view_names]
    if not names:
        return ''
    lines = [_BLOCK_HEADER]
    for name in names:
        proj, row_axis, col_axis = VIEW_LABELS[name]
        parts = [f'{i}:{runs}' for i, row in enumerate(views[name]) if (runs := _row_runs(row))]
        lines.append(f'{name.capitalize()} ({proj}) [rows={row_axis}, cols={col_axis}]: ' + '; '.join(parts))
    return '\n'.join(lines)


def build_user_content(
        caption: str,
        views: dict[str, np.ndarray] | None = None,
        view_names: Sequence[str] = (),
) -> str:
    """
    Builds the user-message content for the text-mask route: the standard BrickGPT instruction for
    ``caption``, with the run-length mask block (for the surviving ``view_names``) appended after it.

    When ``views`` is ``None`` or ``view_names`` is empty the result is *identical* to the original
    text-only instruction, so the unconditional (all-views-dropped) case matches the base task exactly.
    """
    instruction = create_instruction(caption)
    if views is None or not view_names:
        return instruction
    block = serialize_views_rle(views, view_names)
    return f'{instruction}\n\n{block}' if block else instruction


def sample_kept_views(
        rng: np.random.Generator,
        view_keep_probs: dict[str, float],
        p_uncond: float,
) -> tuple[str, ...]:
    """
    Per-view condition dropout for training. With probability ``p_uncond`` drops *all* views (a fully
    unconditional sample, so the model keeps its text-only ability and an unconditional branch is
    available for classifier-free guidance). Otherwise keeps each view independently with its
    ``view_keep_probs`` probability.
    """
    if rng.random() < p_uncond:
        return ()
    return tuple(name for name in VIEW_ORDER if rng.random() < view_keep_probs.get(name, 1.0))


def views_for_bricks(bricks_txt: str, cfg: MaskConditioningConfig = MaskConditioningConfig()) -> dict[str, np.ndarray]:
    """Convenience: GT silhouettes for a structure (thin wrapper over :func:`three_view_masks`)."""
    return three_view_masks(bricks_txt, cfg)
