"""
Like prepare_mask_dataset.py, but writes the three orthographic silhouette masks
(top / front / side) in human-readable JSON or TXT format instead of .npy.

JSON output (one file per split, e.g. train_masks.json):
    A JSON array where each element corresponds to one row in the dataset:
    {
        "idx":    <int>,
        "bricks": "<brick text>",
        "captions": ["..."],
        "top":   "00010000...\n...",   // 20 rows of 20 chars (0/1), newline-separated
        "front": "00010000...\n...",
        "side":  "00010000...\n..."
    }

TXT output (one file per split, e.g. train_masks.txt):
    Entries separated by a blank line. Each entry:
        IDX <n>
        BRICKS
        <brick line>
        ...
        === top ===
        00010000...
        ...
        === front ===
        ...
        === side ===
        ...

Run:
    uv run prepare_mask_dataset_text
    uv run prepare_mask_dataset_text --output_format txt --output_path datasets/masks_text
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from transformers import HfArgumentParser

from brickgpt.masking import MaskConditioningConfig, three_view_masks


@dataclass
class Args:
    input_path: str = field(
        default='AvaLovelace/StableText2Brick',
        metadata={'help': 'HuggingFace dataset path or local directory. '
                          'Must have "bricks" and "captions" fields.'},
    )
    output_path: str = field(
        default='datasets/masks_text',
        metadata={'help': 'Directory in which to save the output files.'},
    )
    output_format: str = field(
        default='json',
        metadata={'help': 'Output format: "json" or "txt".'},
    )
    world_dim: int = field(
        default=20,
        metadata={'help': 'World dimension; masks are (world_dim, world_dim).'},
    )
    splits: str = field(
        default='',
        metadata={'help': 'Comma-separated list of splits to process (e.g. "train,test"). '
                          'Empty = all splits in the dataset.'},
    )


def _mask_to_rows(mask) -> list[list[int]]:
    """Convert a (H, W) float32 array to a list-of-lists of 0/1 ints."""
    return [[int(v) for v in row] for row in mask.astype(int)]


def _mask_to_str(mask) -> str:
    """Convert a (H, W) float32 array to compact newline-separated rows of 0/1 chars."""
    return '\n'.join(''.join(str(int(v)) for v in row) for row in mask.astype(int))


def process_split_json(split, split_name: str, cfg: MaskConditioningConfig, out_dir: Path):
    out_file = out_dir / f'{split_name}_masks.json'
    records = []
    for idx, row in enumerate(split):
        bricks_txt = row['bricks']
        views = three_view_masks(bricks_txt, cfg)
        records.append({
            'idx': idx,
            'bricks': bricks_txt,
            'captions': row.get('captions', []),
            'top':   _mask_to_str(views['top']),
            'front': _mask_to_str(views['front']),
            'side':  _mask_to_str(views['side']),
        })
        if (idx + 1) % 500 == 0:
            print(f'  ... {idx + 1}/{len(split)} done')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} entries to {out_file}")


def process_split_txt(split, split_name: str, cfg: MaskConditioningConfig, out_dir: Path):
    out_file = out_dir / f'{split_name}_masks.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(split):
            bricks_txt = row['bricks']
            views = three_view_masks(bricks_txt, cfg)
            f.write(f'IDX {idx}\n')
            captions = row.get('captions', [])
            for cap in captions:
                f.write(f'CAPTION {cap}\n')
            f.write('BRICKS\n')
            f.write(bricks_txt.strip())
            f.write('\n')
            for view_name in ('top', 'front', 'side'):
                f.write(f'=== {view_name} ===\n')
                f.write(_mask_to_str(views[view_name]))
                f.write('\n')
            f.write('\n')  # blank line between entries
            if (idx + 1) % 500 == 0:
                print(f'  ... {idx + 1}/{len(split)} done')
    print(f"Saved {len(split)} entries to {out_file}")


def main():
    (args,) = HfArgumentParser(Args).parse_args_into_dataclasses()

    if args.output_format not in ('json', 'txt'):
        raise ValueError(f'--output_format must be "json" or "txt", got {args.output_format!r}')

    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    out_dir = Path(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset(args.input_path)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()] or list(dataset.keys())

    for split_name in splits:
        if split_name not in dataset:
            print(f"Split '{split_name}' not found in dataset, skipping.")
            continue
        split = dataset[split_name]
        print(f"Processing split '{split_name}' ({len(split)} rows)...")
        if args.output_format == 'json':
            process_split_json(split, split_name, cfg, out_dir)
        else:
            process_split_txt(split, split_name, cfg, out_dir)

    print(f'\nDone. Output written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
