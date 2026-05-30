import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser

from brickgpt.masking import MaskConditioningConfig, stack_views


@dataclass
class PrepareMaskDatasetArguments:
    input_path: str = field(
        default='AvaLovelace/StableText2Brick',
        metadata={'help': 'Path to the brick structure dataset. Must contain a "bricks" (string) field.'},
    )
    output_path: str = field(
        default='datasets/masks',
        metadata={'help': 'Directory in which to save the precomputed masks, one "<split>_masks.npy" per split. '
                          'Each file is an array of shape (num_rows, num_views, world_dim, world_dim), '
                          'indexed by row, with views ordered as MaskConditioningConfig.views.'},
    )
    world_dim: int = field(
        default=20,
        metadata={'help': 'World dimension; the native mask is (world_dim, world_dim).'},
    )


def main():
    """
    Offline mask-rendering script. Projects each ground-truth brick structure to its three
    orthographic silhouettes (top / front / side) and caches the stacks so training can read
    precomputed masks instead of re-projecting them on the fly. Projection reuses the voxel grid in
    :class:`~brickgpt.data.BrickStructure`, so no external renderer is required.
    """
    parser = HfArgumentParser(PrepareMaskDatasetArguments)
    (args,) = parser.parse_args_into_dataclasses()

    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    input_dataset = load_dataset(args.input_path)

    os.makedirs(args.output_path, exist_ok=True)
    for split_name, split in input_dataset.items():
        masks = np.stack([stack_views(bricks, cfg) for bricks in split['bricks']])  # [N, V, H, W]
        out_file = Path(args.output_path) / f'{split_name}_masks.npy'
        np.save(out_file, masks)
        print(f'Saved {masks.shape[0]} mask stacks {masks.shape[1:]} for split "{split_name}" to {out_file}')

    print(f'Precomputed masks saved to {os.path.abspath(args.output_path)}')


if __name__ == '__main__':
    main()
