"""
Stage 0 feasibility probe for the *text-based* mask conditioning idea.

The idea: instead of feeding the three orthographic silhouettes through a CNN (the existing Path-B
prefix-encoder approach), serialize each silhouette as a sorted list of 2D coordinates and append it
to the user prompt as plain text. That makes the whole thing reuse the original `trl sft` text
pipeline with zero new model code -- BUT the only real risk is sequence-length blow-up, because the
coordinate list grows with the silhouette *area*.

This script measures that risk on the real dataset. For every structure it:
  1. projects to the three silhouettes via `three_view_masks` (the same projection the reward uses),
  2. serializes them with `serialize_mask_block` (a reference format we can reuse for real),
  3. tokenizes the mask block with the actual Llama tokenizer (digit-level tokenization included),
and then reports the distribution of per-view areas, per-view / total mask-block token counts, and
how the mask block compares in size to the base instruction prompt and to the assistant brick list.

Run (offline once the model + dataset are cached -- see CLAUDE.md):

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    uv run python scripts/measure_mask_tokens.py
    uv run python scripts/measure_mask_tokens.py --split train --max_rows 2000 --save_csv /tmp/mask_tokens.csv
"""

import csv
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from brickgpt.masking import MaskConditioningConfig
from brickgpt.masking.projection import three_view_masks
from brickgpt.models import create_instruction

# 2D coordinate axis labels for each view, matching brickgpt.masking.projection.VIEW_AXES:
#   top   -> project along Z -> (x, y)
#   front -> project along Y -> (x, z)
#   side  -> project along X -> (y, z)
VIEW_LABELS: dict[str, tuple[str, str]] = {
    'top': ('z-proj', 'x,y'),
    'front': ('y-proj', 'x,z'),
    'side': ('x-proj', 'y,z'),
}


@dataclass
class Args:
    input_path: str = field(default='AvaLovelace/StableText2Brick')
    tokenizer: str = field(
        default='AvaLovelace/BrickGPT',
        metadata={'help': 'Tokenizer to count tokens with (defaults to the BrickGPT base tokenizer).'},
    )
    split: str = field(default='train', metadata={'help': 'Dataset split to analyze.'})
    world_dim: int = field(default=20)
    max_rows: int = field(default=0, metadata={'help': '0 = all rows; otherwise cap for a quick run.'})
    save_csv: str = field(default='', metadata={'help': 'Optional path to dump per-row measurements.'})


def _row_runs(row: np.ndarray) -> str:
    """Encodes one silhouette row as run-length column ranges, e.g. '3-19' or '0-2,4-5'."""
    cols = np.flatnonzero(row > 0)
    if len(cols) == 0:
        return ''
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return ','.join(f'{a}' if a == b else f'{a}-{b}' for a, b in runs)


def serialize_mask_block_rle(views: dict[str, np.ndarray]) -> str:
    """
    Compressed variant of :func:`serialize_mask_block`: instead of listing every occupied pixel, each
    silhouette row is given as run-length column ranges. Dense / convex silhouettes (the common case
    here) collapse from hundreds of coordinates to a handful of ranges, while staying coordinate-like
    so the model can still align it with the (x,y,z) output.

        Top (z-proj) [rows=x, cols=y]: 0:3-19; 1:3-19; 8:3,5-19; ...
    """
    lines = ['Target silhouettes (run-length occupied columns):']
    for name in ('top', 'front', 'side'):
        proj, axes = VIEW_LABELS[name]
        rows, cols = axes.split(',')
        parts = [f'{i}:{runs}' for i, row in enumerate(views[name]) if (runs := _row_runs(row))]
        lines.append(f'{name.capitalize()} ({proj}) [rows={rows}, cols={cols}]: ' + '; '.join(parts))
    return '\n'.join(lines)


def serialize_mask_block(views: dict[str, np.ndarray]) -> str:
    """
    Reference serialization of the three silhouettes into a prompt-ready text block.

    Each view becomes one line: a human-readable header plus the sorted list of occupied 2D pixels
    (the projected axis is dropped, so 'top' lists (x,y) etc.). `np.argwhere` returns coordinates in
    row-major order, i.e. already lexicographically sorted -- important so the model always sees a
    canonical ordering of what is conceptually an unordered set.

    Returns a string like:

        Target silhouettes (occupied cells):
        Top (z-proj) [x,y]: (1,1),(1,2),(2,1),(2,2)
        Front (y-proj) [x,z]: (1,0),(2,0),(1,1)
        Side (x-proj) [y,z]: (1,0),(2,0)
    """
    lines = ['Target silhouettes (occupied cells):']
    for name in ('top', 'front', 'side'):
        proj, axes = VIEW_LABELS[name]
        coords = np.argwhere(views[name] > 0)
        coord_str = ','.join(f'({int(a)},{int(b)})' for a, b in coords)
        lines.append(f'{name.capitalize()} ({proj}) [{axes}]: {coord_str}')
    return '\n'.join(lines)


def describe(name: str, values: list[int]) -> str:
    a = np.asarray(values, dtype=np.float64)
    pcts = np.percentile(a, [50, 90, 95, 99]) if len(a) else [0, 0, 0, 0]
    return (f'  {name:<22} mean={a.mean():8.1f}  p50={pcts[0]:6.0f}  p90={pcts[1]:6.0f}  '
            f'p95={pcts[2]:6.0f}  p99={pcts[3]:6.0f}  max={a.max():6.0f}')


def main():
    (args,) = HfArgumentParser(Args).parse_args_into_dataclasses()
    cfg = MaskConditioningConfig(world_dim=args.world_dim)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(args.input_path)[args.split]
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    n_tok = lambda s: len(tok(s, add_special_tokens=False)['input_ids'])

    rows = []  # per-structure measurements
    for ex in ds:
        bricks_txt = ex['bricks']
        views = three_view_masks(bricks_txt, cfg)
        areas = {name: int((views[name] > 0).sum()) for name in ('top', 'front', 'side')}

        block = serialize_mask_block(views)
        block_tokens = n_tok(block)
        block_rle_tokens = n_tok(serialize_mask_block_rle(views))
        per_view_tokens = {
            name: n_tok(f'{name}: ' + ','.join(
                f'({int(a)},{int(b)})' for a, b in np.argwhere(views[name] > 0)))
            for name in ('top', 'front', 'side')
        }
        # Base prompt uses the first caption; the mask block is identical across a row's captions.
        caption = ex['captions'][0] if ex.get('captions') else ''
        base_tokens = n_tok(create_instruction(caption))
        brick_tokens = n_tok(bricks_txt)

        rows.append({
            'area_top': areas['top'], 'area_front': areas['front'], 'area_side': areas['side'],
            'area_total': sum(areas.values()),
            'tok_top': per_view_tokens['top'], 'tok_front': per_view_tokens['front'],
            'tok_side': per_view_tokens['side'], 'tok_mask_block': block_tokens,
            'tok_mask_block_rle': block_rle_tokens,
            'tok_base_prompt': base_tokens, 'tok_bricks': brick_tokens,
            'tok_full_prompt': base_tokens + block_tokens,
            'pct_overhead': 100.0 * block_tokens / max(base_tokens, 1),
        })

    col = lambda k: [r[k] for r in rows]
    print(f'\n=== Text-mask token-budget probe: {args.input_path} [{args.split}], '
          f'{len(rows)} structures, world_dim={cfg.world_dim} ===\n')

    print('Silhouette area (occupied pixels per view):')
    for k in ('area_top', 'area_front', 'area_side', 'area_total'):
        print(describe(k, col(k)))

    print('\nTokens (Llama tokenizer, digit-level):')
    for k in ('tok_top', 'tok_front', 'tok_side', 'tok_mask_block', 'tok_mask_block_rle',
              'tok_base_prompt', 'tok_bricks', 'tok_full_prompt'):
        print(describe(k, col(k)))

    print('\nMask-block overhead vs. base instruction prompt:')
    print(describe('pct_overhead (%)', col('pct_overhead')))

    raw_mean = np.mean(col('tok_mask_block'))
    rle_mean = np.mean(col('tok_mask_block_rle'))
    print(f'\nCompression: raw mean {raw_mean:.0f} tok -> run-length mean {rle_mean:.0f} tok '
          f'({raw_mean / max(rle_mean, 1):.1f}x smaller)')

    v0 = three_view_masks(ds[0]['bricks'], cfg)
    print('\nSample run-length mask block (first structure):')
    print('  ' + serialize_mask_block_rle(v0).replace('\n', '\n  '))
    print()

    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'Wrote per-structure measurements to {args.save_csv}\n')


if __name__ == '__main__':
    main()
