import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser

from brickgpt.masking import MaskConditioningConfig, bricks_to_mask


@dataclass
class PrepareMaskDatasetArguments:
    input_path: str = field(
        default='AvaLovelace/StableText2Brick',
        metadata={'help': 'Path to the brick structure dataset. Must contain a "bricks" (string) field.'},
    )
    output_path: str = field(
        default='datasets/masks',
        metadata={'help': 'Directory in which to save the precomputed masks, one "<split>_masks.npy" per split. '
                          'Each file is an array of shape (num_rows, world_dim, world_dim), indexed by row.'},
    )
    world_dim: int = field(
        default=20,
        metadata={'help': 'World dimension; the native mask is (world_dim, world_dim).'},
    )
    projection_axis: int = field(
        default=2,
        metadata={'help': 'Axis to project the 3D occupancy grid along (2 = Z-axis top-down view).'},
    )


def main():
    """
    Offline mask-rendering script. Projects each ground-truth brick structure to a 2D top-down
    silhouette mask and caches the results so training can read precomputed masks instead of
    re-projecting them on the fly. Projection reuses the voxel grid in
    :class:`~brickgpt.data.BrickStructure`, so no external renderer is required.
    """
    parser = HfArgumentParser(PrepareMaskDatasetArguments)
    (args,) = parser.parse_args_into_dataclasses()

    cfg = MaskConditioningConfig(world_dim=args.world_dim, projection_axis=args.projection_axis)
    input_dataset = load_dataset(args.input_path)

    os.makedirs(args.output_path, exist_ok=True)
    for split_name, split in input_dataset.items():
        masks = np.stack([bricks_to_mask(bricks, cfg) for bricks in split['bricks']])
        out_file = Path(args.output_path) / f'{split_name}_masks.npy'
        np.save(out_file, masks)
        print(f'Saved {masks.shape[0]} masks for split "{split_name}" to {out_file}')

    print(f'Precomputed masks saved to {os.path.abspath(args.output_path)}')


if __name__ == '__main__':
    main()
