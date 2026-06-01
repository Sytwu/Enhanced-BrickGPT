"""
Step D -- Phase 1 SFT for mask-conditioned BrickGPT (self-written loop).

Stage 1 (D7): start from BrickGPT's released text-only SFT weights, **freeze the LLM**, and train
only the mask encoder + projection + view/presence embeddings with assistant-only CE and per-view
condition dropout. The frozen LLM forces the prefix to land in a region of embedding space the LLM
already understands (prefix-tuning / LLaVA-stage-1 style).

A custom loop (not ``trl.SFTTrainer``) is used so the pre-tokenized + custom-tensor (``mask`` /
``has_mask``) batch is passed straight through, and so SFT and GRPO share one logging UX
(:class:`~brickgpt.training.logging_utils.RunLogger`: tqdm + console + optional wandb).

The built-in **IoU probe** (D6) periodically generates conditioned-on-GT-silhouette vs. null-mask and
reports the IoU lift -- direct evidence the model uses the mask. If the lift stays ~0, escalate to
LoRA-on-LLM (the confirmed fallback).

Run::

    uv run train_sft_masked --output_dir output/sft_masked --max_steps 4000 --mask_dir datasets/masks
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, get_cosine_schedule_with_warmup

from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset, MaskDataCollator
from brickgpt.models.brickgpt import BrickGPTConfig
from brickgpt.models.masked_brickgpt import BrickGPTWithMask
from brickgpt.training.generation import ce_delta_probe, iou_probe
from brickgpt.training.logging_utils import RunLogger

logger = logging.getLogger(__name__)


@dataclass
class SFTMaskedArguments:
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    mask_dir: str | None = field(
        default=None,
        metadata={'help': 'Optional dir of precomputed "<split>_masks.npy" (N, V, H, W) from prepare_mask_dataset.'},
    )
    train_split: str = field(default='train')
    probe_split: str = field(default='test')
    output_dir: str = field(default='output/sft_masked')

    caption_dropout_p: float = field(
        default=0.0,
        metadata={'help': 'CFG-style text dropout (see MaskConditioningConfig.caption_dropout_p). '
                          'Makes the mask non-redundant so the encoder gets gradient even with a frozen '
                          'LLM that already fits caption->bricks. Try ~0.5 if the IoU lift / CE delta is ~0.'},
    )

    # Optimization (defaults follow scripts/finetune.zsh: lr=2e-3 cosine, warmup=100, bf16).
    lr: float = field(default=2e-3)
    weight_decay: float = field(default=0.0)
    warmup_steps: int = field(default=100)
    max_steps: int = field(default=4000)
    batch_size: int = field(default=2)
    grad_accum_steps: int = field(default=8)
    max_grad_norm: float = field(default=1.0)
    num_workers: int = field(default=2)
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Backprop reaches the input-level prefix through the frozen LLM, so '
                          'activation memory is large; checkpointing trades compute for memory.'},
    )

    # Logging / eval / checkpoints.
    report_to: str = field(default='auto', metadata={'help': "'auto' | 'wandb' | 'none'."})
    wandb_project: str = field(default='brickgpt-mask-sft')
    run_name: str | None = field(default=None)
    log_every: int = field(default=50)
    probe_every: int = field(default=500, metadata={'help': 'Run the IoU probe every N optimizer steps (0=off).'})
    ce_probe_every: int = field(
        default=100,
        metadata={'help': 'Run the (cheap, teacher-forced) CE-delta probe every N optimizer steps (0=off). '
                          'Logs ce_masked/ce_null/ce_delta; a positive, rising ce_delta is the earliest '
                          'evidence the frozen-LLM encoder is learning to use the mask.'},
    )
    probe_n: int = field(default=8)
    save_every: int = field(default=1000)


def build_dataset(args, tokenizer, cfg, split, train):
    from datasets import load_dataset
    data = load_dataset(args.dataset_name, split=split)
    masks = None
    if args.mask_dir is not None:
        masks = np.load(f'{args.mask_dir}/{split}_masks.npy', mmap_mode='r')
    return MaskBrickDataset(data, tokenizer, cfg, masks=masks, train=train), data


def _save(model, cfg, args, tag):
    import json
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    # The LLM is frozen, so only the mask encoder is trained: save it alone (lean; GRPO reloads the
    # base from model_name_or_path and loads this on top).
    torch.save(model.mask_prefix_encoder.state_dict(), f'{args.output_dir}/mask_encoder_{tag}.pt')
    with open(f'{args.output_dir}/sft_meta.json', 'w') as f:
        json.dump({'model_name_or_path': args.model_name_or_path, 'mask_config': cfg.__dict__}, f, indent=2)
    logger.info('Saved checkpoint (%s) to %s', tag, args.output_dir)


def main():
    logging.basicConfig(level=logging.INFO)
    (args,) = HfArgumentParser(SFTMaskedArguments).parse_args_into_dataclasses()
    cfg = MaskConditioningConfig(caption_dropout_p=args.caption_dropout_p)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BrickGPTWithMask.from_pretrained(args.model_name_or_path, cfg, torch_dtype=torch.bfloat16).to(device)
    model.freeze_llm()
    if args.gradient_checkpointing:
        model.base.gradient_checkpointing_enable()
        model.base.enable_input_require_grads()

    train_ds, _ = build_dataset(args, tokenizer, cfg, args.train_split, train=True)
    collator = MaskDataCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=args.num_workers, drop_last=True)

    # Held-out examples shared by both probes (IoU + CE-delta).
    probe_examples = []
    if args.probe_every or args.ce_probe_every:
        try:
            _, probe_data = build_dataset(args, tokenizer, cfg, args.probe_split, train=False)
        except (ValueError, FileNotFoundError, KeyError):
            logger.warning('probe_split %r unavailable; probing on the train head instead.', args.probe_split)
            _, probe_data = build_dataset(args, tokenizer, cfg, args.train_split, train=False)
        probe_examples = [probe_data[i] for i in range(min(args.probe_n, len(probe_data)))]
    probe_cfg = BrickGPTConfig(model_name_or_path=args.model_name_or_path, use_gurobi=False,
                               max_regenerations=0, max_brick_rejections=10, max_bricks=40)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)
    run = RunLogger(total=args.max_steps, phase='sft', report_to=args.report_to,
                    console_every=args.log_every, project=args.wandb_project,
                    name=args.run_name, config={**vars(args), **{'mask_' + k: v for k, v in cfg.__dict__.items()}})

    model.train()
    step, micro, running = 0, 0, 0.0
    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch, use_cache=False)
        loss = out.loss / args.grad_accum_steps
        loss.backward()
        running += out.loss.item()
        micro += 1

        if micro % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run.log({'loss': running / args.grad_accum_steps, 'lr': sched.get_last_lr()[0]}, step=step)
            running = 0.0

            if args.ce_probe_every and step % args.ce_probe_every == 0 and probe_examples:
                ce = ce_delta_probe(model, tokenizer, probe_examples, cfg)
                run.write(f'[sft] step {step} CE probe: masked={ce["ce_masked"]:.4f} '
                          f'null={ce["ce_null"]:.4f} delta={ce["ce_delta"]:+.4f}')
                run.log(ce, step=step, advance=0)
            if args.probe_every and step % args.probe_every == 0 and probe_examples:
                metrics = iou_probe(model, tokenizer, probe_examples, cfg, probe_cfg)
                run.write(f'[sft] step {step} IoU probe: masked={metrics["iou_masked"]:.3f} '
                          f'null={metrics["iou_null"]:.3f} lift={metrics["iou_lift"]:+.3f}')
                run.log(metrics, step=step, advance=0)
            if args.save_every and step % args.save_every == 0:
                _save(model, cfg, args, tag=f'step{step}')

    _save(model, cfg, args, tag='final')
    run.close()


if __name__ == '__main__':
    main()
