# when preparing mask, use "uv run 'src\brickgpt\prepare_mask_dataset.py' --do_prefix False"
# when encoder trained, use "uv run 'src\brickgpt\prepare_mask_dataset.py' --do_prefix True --encoder_ckpt 'THE_PATH_OF_ENCODER_WEIGHTS'"

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import HfArgumentParser

from brickgpt.masking import MaskConditioningConfig, MultiViewMaskPrefixEncoder, stack_views


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
    output_prefix_path: str = field(
        default='datasets/mask_prefixes',
        metadata={'help': 'Directory in which to save the precomputed multiview prefix tokens, '
                          'one "<split>_prefixes.npy" per split. '
                          'Each file is an array of shape (num_rows, V*num_prefix_tokens, llm_hidden_size).'},
    )
    world_dim: int = field(
        default=20,
        metadata={'help': 'World dimension; the native mask is (world_dim, world_dim).'},
    )
    do_prefix: bool = field(
        default=False,
        metadata={'help': 'Whether to precompute multiview prefix tokens in addition to masks.'},
    )
    device: str = field(
        default='cuda' if torch.cuda.is_available() else 'cpu',
        metadata={'help': 'Device to use for prefix token computation (cpu or cuda).'},
    )
    encoder_ckpt: str = field(
        default='',
        metadata={'help': 'Path to trained Mask Encoder weights (e.g., output/sft_masked/mask_encoder_final.pt)'}
    )


def main():
    """
    Offline mask and prefix-token rendering script. Projects each ground-truth brick structure to
    its three orthographic silhouettes (top / front / side) and caches the stacks. Optionally also
    precomputes the multiview prefix-token embeddings so training can read cached tokens instead of
    computing them on the fly.
    """
    parser = HfArgumentParser(PrepareMaskDatasetArguments)
    (args,) = parser.parse_args_into_dataclasses()

    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    if not args.do_prefix:
        os.makedirs(args.output_path, exist_ok=True)
        input_dataset = load_dataset(args.input_path)
        
        for split_name, split in input_dataset.items():
            print(f"Computing masks for split '{split_name}'...")
            masks = np.stack([stack_views(bricks, cfg) for bricks in split['bricks']])  # [N, V, H, W]
            out_file = Path(args.output_path) / f'{split_name}_masks.npy'
            np.save(out_file, masks)
            print(f"Saved {masks.shape[0]} mask stacks for split '{split_name}' to {out_file}\n")
    else:
        os.makedirs(args.output_prefix_path, exist_ok=True)
        encoder = MultiViewMaskPrefixEncoder(cfg).eval().to(args.device)
        if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
            print(f"@@@weights loaded: {args.encoder_ckpt}")
            encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=args.device))
        else:
            #Raise FileNotFoundError(f"Encoder checkpoint not found at {args.encoder_ckpt}. Please provide a valid path to pre-trained Mask Encoder weights.")
            print("!!!!Warning: I didn't find the encoder_ckpt, outputting random shitty Prefix Tokens!")

        for split_name in ["train", "test"]:
            mask_file = Path(args.output_path) / f'{split_name}_masks.npy'
            
            if not mask_file.exists():
                print(f"Can't find {mask_file}, Skipped.")
                continue

            print(f"Loading masks for split '{split_name}' and converting to prefix tokens...")
            masks = np.load(mask_file, mmap_mode='r')
            prefixes = []

            with torch.no_grad():
                for i, mask_stack in enumerate(masks):
                    mask_tensor = torch.from_numpy(mask_stack.copy()).float().unsqueeze(0).to(args.device)
                    prefix = encoder(mask_tensor).cpu().numpy()
                    prefixes.append(prefix[0])
                    
                    if (i + 1) % 1000 == 0:
                        print(f'  ... {i + 1}/{len(masks)} done')

            prefixes = np.stack(prefixes)
            prefix_file = Path(args.output_prefix_path) / f'{split_name}_prefixes.npy'
            np.save(prefix_file, prefixes)
            print(f"Saved {prefixes.shape[0]} prefix tokens for split '{split_name}' to {prefix_file}\n")
    # Prepare mask output directory
    os.makedirs(args.output_path, exist_ok=True)

    """# Optionally prepare prefix output directory and initialize encoder
    if args.do_prefix:
        os.makedirs(args.output_prefix_path, exist_ok=True)
        encoder = MultiViewMaskPrefixEncoder(cfg).eval().to(args.device)

    for split_name, split in input_dataset.items():
        # Precompute mask stacks
        masks = np.stack([stack_views(bricks, cfg) for bricks in split['bricks']])  # [N, V, H, W]
        out_file = Path(args.output_path) / f'{split_name}_masks.npy'
        np.save(out_file, masks)
        print(f'Saved {masks.shape[0]} mask stacks {masks.shape[1:]} for split "{split_name}" to {out_file}')

        if args.do_prefix:
            prefixes = []
            with torch.no_grad():
                for i, mask_stack in enumerate(masks):
                    # mask_stack is [V, H, W]; add batch dim and convert to tensor
                    mask_tensor = torch.from_numpy(mask_stack.copy()).float().unsqueeze(0).to(args.device)  # [1, V, H, W]
                    # All views are provided (has_mask=None means all present)
                    prefix = encoder(mask_tensor).cpu().numpy()  # [1, V*num_prefix_tokens, llm_hidden_size]
                    prefixes.append(prefix[0])  # Remove batch dim
                    if (i + 1) % 100 == 0:
                        print(f'  Computed {i + 1}/{len(masks)} prefix tokens...')

            prefixes = np.stack(prefixes)  # [N, V*num_prefix_tokens, llm_hidden_size]
            prefix_file = Path(args.output_prefix_path) / f'{split_name}_prefixes.npy'
            np.save(prefix_file, prefixes)
            print(f'Saved {prefixes.shape[0]} prefix tokens {prefixes.shape[1:]} for split "{split_name}" to {prefix_file}')
"""
    
    if args.do_prefix:
        print(f'Precomputed prefix tokens saved to {os.path.abspath(args.output_prefix_path)}')


if __name__ == '__main__':
    main()
