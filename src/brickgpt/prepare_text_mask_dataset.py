"""
Stage 1 of the text-token mask conditioning route: build the SFT JSONL with the run-length mask
block appended to each user prompt, plus per-view condition dropout.

Output is the same ``{"messages": [...]}`` chat format as :mod:`prepare_finetuning_dataset`, so it
feeds the original ``trl sft`` pipeline (``scripts/finetune.zsh``) with **zero model code** -- the
conditioning lives entirely in the prompt text. Remember the CLAUDE.md requirement to overwrite the
pretrained model's ``config.json`` / ``special_tokens_map.json`` / ``tokenizer_config.json`` with the
``finetuning_config_files/`` versions (so ``pad_token != eos_token``).

Dropout (see :func:`~brickgpt.masking.sample_kept_views`) makes the mask *optional*: some samples keep
all three views, some keep a subset, and a ``--p_uncond`` fraction drop all views (identical to the
original text-only task). This trains every view-subset combination and yields an unconditional branch
for classifier-free guidance / robustness to missing views.

    uv run prepare_text_mask_dataset --output_path datasets/text_mask
    uv run prepare_text_mask_dataset --output_path datasets/text_mask \
        --view_keep_prob 0.7 --p_uncond 0.15 --seed 0
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser

from brickgpt.masking import (
    MaskConditioningConfig, VIEW_ORDER, build_user_content, sample_kept_views, three_view_masks,
)


@dataclass
class PrepareTextMaskArguments:
    input_path: str = field(
        default='AvaLovelace/StableText2Brick',
        metadata={'help': 'Brick dataset with "captions" (list[str]) and "bricks" (str) fields.'},
    )
    output_path: str = field(
        default='datasets/text_mask',
        metadata={'help': 'Directory for the "<split>.jsonl" SFT files (chat "messages" format).'},
    )
    world_dim: int = field(default=20)
    view_keep_prob: float = field(
        default=0.7,
        metadata={'help': 'Per-view keep probability for condition dropout (applied to every view).'},
    )
    p_uncond: float = field(
        default=0.15,
        metadata={'help': 'Probability a sample drops ALL views (fully unconditional, == base task).'},
    )
    seed: int = field(default=0, metadata={'help': 'RNG seed for reproducible dropout.'})


def main():
    parser = HfArgumentParser(PrepareTextMaskArguments)
    (args,) = parser.parse_args_into_dataclasses()
    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    keep_probs = {name: args.view_keep_prob for name in VIEW_ORDER}
    rng = np.random.default_rng(args.seed)

    input_dataset = load_dataset(args.input_path)

    def create_messages(caption: str, bricks: str, views: dict) -> dict:
        kept = sample_kept_views(rng, keep_probs, args.p_uncond)
        return {'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': build_user_content(caption, views, kept)},
            {'role': 'assistant', 'content': bricks},
        ]}

    def convert_sample(batch) -> dict:
        out = []
        for bricks, captions in zip(batch['bricks'], batch['captions']):
            views = three_view_masks(bricks, cfg)   # GT silhouettes once per structure
            out.extend(create_messages(caption, bricks, views) for caption in captions)
        # Re-key list[dict] -> dict[list] for HF datasets batched map.
        return {'messages': [r['messages'] for r in out]}

    os.makedirs(args.output_path, exist_ok=True)
    for split_name, split in input_dataset.items():
        output_split = split.map(
            convert_sample,
            batched=True,
            remove_columns=split.column_names,
            desc=f'Building text-mask split "{split_name}"',
        )
        output_split.to_json(Path(args.output_path) / f'{split_name}.jsonl')

    print(f'Text-mask SFT dataset saved to {os.path.abspath(args.output_path)} '
          f'(view_keep_prob={args.view_keep_prob}, p_uncond={args.p_uncond})')


if __name__ == '__main__':
    main()
