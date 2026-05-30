"""
Phase 3 — Supervised fine-tuning (SFT) for mask-conditioned BrickGPT.

This replaces the ``trl sft`` CLI path ([scripts/finetune.zsh]) for the masked model, because that
CLI cannot host a custom multimodal model. We build a :class:`BrickGPTWithMask` (frozen LLM +
trainable mask encoder), a :class:`MaskBrickDataset`, and a :class:`MaskDataCollator`, then drive
them with TRL's ``SFTTrainer`` (the class) so we keep bf16 / checkpointing / wandb integration.

SCAFFOLD: the wiring below is complete enough to follow, but dataset paths, hyperparameters, and the
``SFTConfig`` are left as TODOs for the implementer. The model already returns ``loss``, which is all
``Trainer`` needs.

Fallback (see plan): if ``SFTTrainer`` + ``inputs_embeds`` label alignment is fiddly, swap in a plain
PyTorch training loop over the same :class:`BrickGPTWithMask` and :class:`MaskDataCollator`.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from transformers import AutoTokenizer

from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset, MaskDataCollator
from brickgpt.models.masked_brickgpt import BrickGPTWithMask

logger = logging.getLogger(__name__)


@dataclass
class SFTMaskedArguments:
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    mask_dir: str | None = field(
        default=None,
        metadata={'help': 'Optional directory of precomputed "<split>_masks.npy" from prepare_mask_dataset.'},
    )
    output_dir: str = field(default='output/sft_masked')


def build_dataset(args: SFTMaskedArguments, tokenizer, cfg: MaskConditioningConfig, split: str = 'train'):
    """Loads the HF dataset split and (optionally) its precomputed masks into a MaskBrickDataset."""
    from datasets import load_dataset  # Imported lazily; only needed at training time.

    data = load_dataset(args.dataset_name, split=split)
    masks = None
    if args.mask_dir is not None:
        masks = np.load(f'{args.mask_dir}/{split}_masks.npy', mmap_mode='r')
    return MaskBrickDataset(data, tokenizer, cfg, masks=masks, train=(split == 'train'))


def main():
    # TODO(handoff): parse SFTMaskedArguments + trl.SFTConfig (reuse hyperparameters from
    #   scripts/finetune.zsh: lr=2e-3, cosine, warmup=100, bf16). No LoRA here — the LLM is frozen.
    from trl import SFTConfig, SFTTrainer  # Imported lazily; part of the optional [finetuning] extra.

    args = SFTMaskedArguments()
    cfg = MaskConditioningConfig()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BrickGPTWithMask.from_pretrained(args.model_name_or_path, cfg)
    model.freeze_llm()  # Phase 3: train only the mask encoder + projection.

    train_dataset = build_dataset(args, tokenizer, cfg, split='train')
    collator = MaskDataCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    sft_config = SFTConfig(output_dir=args.output_dir, bf16=True)  # TODO: fill remaining hyperparameters.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        # NOTE: disable TRL's internal tokenization/packing — examples are pre-tokenized by the dataset.
    )
    trainer.train()  # TODO(handoff): verify SFTTrainer passes `mask`/`has_mask` through to forward.


if __name__ == '__main__':
    main()
