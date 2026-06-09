"""
Self-written SFT loop for the **text-token** mask conditioning route (Stage 1).

Unlike the CNN Path-B (:mod:`brickgpt.training.sft_masked`, which freezes the LLM and trains a mask
*encoder*), here the mask is plain text in the prompt -- the run-length silhouette block built by
:func:`~brickgpt.masking.build_user_content`. There is no encoder; we teach the LLM itself to read
that block and follow it, by adapting it with **LoRA** (the same merge-BrickGPT-then-add-LoRA setup as
:func:`brickgpt.training.grpo_text.load_policy`). Loss is the standard assistant-only causal-LM CE --
no auxiliary loss; mask-following is learned implicitly because the GT bricks are consistent with the
GT silhouettes (see TODO §text-mask). Per-view condition dropout (:func:`sample_kept_views`) makes the
mask optional and yields an unconditional branch.

The reason for a self-written loop (vs. the original ``trl sft`` CLI) is the in-loop **per-view
delta-IoU probe**: every ``--probe_every`` steps it generates with the GT mask and without it, and logs
``iou_{top,front,side}`` for each plus the lift (mask - nomask). That is the training-curve evidence
the model is actually using the mask. The probe uses fast *unconstrained* generation (one pass, no
logit masking); the authoritative headline number comes from ``scripts/eval_text_mask_iou.py`` (which
uses the full constrained decoder) on the saved checkpoint.

    uv run train_sft_text_mask --output_dir output/text_mask_sft \
        --epochs 1 --probe_every 200 --probe_n 48
"""
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from brickgpt.data import BrickStructure
from brickgpt.masking import (
    MaskConditioningConfig, VIEW_ORDER, build_user_content, sample_kept_views, three_view_masks,
)
from brickgpt.training.grpo_text import load_policy
from brickgpt.training.logging_utils import RunLogger
from brickgpt.training.rewards import _valid_bricks, silhouette_iou_per_view

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = 'You are a helpful assistant.'


@dataclass
class SFTTextMaskArguments:
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    train_split: str = field(default='train')
    probe_split: str = field(default='test')
    output_dir: str = field(default='output/text_mask_sft')
    max_train_rows: int = field(default=0, metadata={'help': '0 = full split; else cap rows (quick runs/smoke).'})

    world_dim: int = field(default=20)
    view_keep_prob: float = field(default=0.7, metadata={'help': 'Per-view keep prob (condition dropout).'})
    p_uncond: float = field(default=0.15, metadata={'help': 'Prob of dropping ALL views (unconditional).'})

    epochs: int = field(default=1)
    max_steps: int = field(default=0, metadata={'help': '0 = run full epochs; else cap optimizer steps.'})
    batch_size: int = field(default=4)
    grad_accum: int = field(default=4)
    lr: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.03)
    max_grad_norm: float = field(default=1.0)
    max_seq_len: int = field(default=2048, metadata={'help': 'Drop examples whose tokenized length exceeds this. '
                                                             'Typical RLE prompt (~500) + bricks (~1084) ~= 1600.'})
    gradient_checkpointing: bool = field(default=True)
    seed: int = field(default=0)

    # Extra adapter(s) to merge into the frozen backbone *after* BrickGPT and *before* the fresh
    # trainable SFT LoRA -- e.g. a GRPO checkpoint (`output/grpo_text_2k/adapter_final`). Comma list.
    # Each was trained on the merged-BrickGPT backbone, so its delta must be applied on top of the
    # BrickGPT merge (its own adapter_config records raw Llama as base, which is only informational).
    init_adapters: str | None = field(default=None)

    # LoRA (same knobs as grpo_text.load_policy, which we reuse).
    resume_from: str | None = field(default=None)
    start_step: int = field(default=0)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='q_proj,k_proj,v_proj,o_proj')

    # Per-view delta-IoU probe.
    probe_every: int = field(default=200, metadata={'help': 'Run the IoU probe every N optimizer steps (0=off).'})
    probe_n: int = field(default=48)
    probe_max_new_tokens: int = field(default=400)

    save_every: int = field(default=0, metadata={'help': 'Save the LoRA adapter every N steps (0=only final).'})
    save_merged: bool = field(default=True, metadata={'help': 'Also write a merged full model at the end for eval.'})
    report_to: str = field(default='auto')
    wandb_project: str = field(default='brickgpt-text-mask-sft')
    run_name: str | None = field(default=None)
    log_every: int = field(default=20)


class TextMaskBrickDataset(Dataset):
    """
    One example per caption, tokenized as a chat with the run-length mask block in the user turn and
    assistant-only labels. Views are projected once per structure; per-view dropout is resampled every
    ``__getitem__`` so each epoch sees fresh view subsets (incl. fully-unconditional samples).
    """

    def __init__(self, data, tokenizer, cfg, view_keep_prob, p_uncond, train=True, rng_seed=0):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.keep_probs = {name: view_keep_prob for name in VIEW_ORDER}
        self.p_uncond = p_uncond
        self.train = train
        self.rng = np.random.default_rng(rng_seed)
        self.bricks, self.views, self.index = [], [], []
        for row_idx, row in enumerate(data):
            self.bricks.append(row['bricks'])
            self.views.append(three_view_masks(row['bricks'], cfg))
            for caption in row['captions']:
                self.index.append((row_idx, caption))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row_idx, caption = self.index[i]
        views = self.views[row_idx]
        kept = sample_kept_views(self.rng, self.keep_probs, self.p_uncond) if self.train else VIEW_ORDER
        content = build_user_content(caption, views, kept)

        prompt_ids = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}],
            add_generation_prompt=True, tokenize=True)
        full_ids = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content},
             {'role': 'assistant', 'content': self.bricks[row_idx]}], tokenize=True)
        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100
        return {'input_ids': torch.tensor(full_ids), 'labels': torch.tensor(labels)}


class Collator:
    """
    Right-pads input_ids / labels to the batch max (labels padded with -100). Drops examples longer
    than ``max_seq_len`` *here* (lazily) rather than pre-tokenizing the whole dataset up front -- the
    length varies per call anyway because of view dropout. Returns ``None`` if the whole batch is
    dropped (the training loop skips it).
    """

    def __init__(self, pad_token_id, max_seq_len=None):
        self.pad_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, feats):
        if self.max_seq_len:
            feats = [f for f in feats if f['input_ids'].size(0) <= self.max_seq_len]
        if not feats:
            return None
        max_len = max(f['input_ids'].size(0) for f in feats)
        ids, attn, labels = [], [], []
        for f in feats:
            pad = max_len - f['input_ids'].size(0)
            ids.append(F.pad(f['input_ids'], (0, pad), value=self.pad_id))
            attn.append(F.pad(torch.ones_like(f['input_ids']), (0, pad), value=0))
            labels.append(F.pad(f['labels'], (0, pad), value=-100))
        return {'input_ids': torch.stack(ids), 'attention_mask': torch.stack(attn),
                'labels': torch.stack(labels)}


@torch.no_grad()
def _generate(model, tokenizer, content, device, max_new_tokens):
    """Unconstrained one-pass generation of a brick list for a prompt (probe only)."""
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}],
        add_generation_prompt=True, return_tensors='pt').to(device)
    out = model.generate(
        input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
        do_sample=True, temperature=1.0, top_k=None, top_p=None, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def iou_probe(model, tokenizer, examples, cfg, device, max_new_tokens, seed=0):
    """
    Per-view delta-IoU probe: for each held-out structure, generate with the GT mask block and without
    it, build each structure, and compare its silhouettes to the GT. Returns means + the mask-vs-nomask
    lift per view. Invalid completions -> empty structure -> IoU vs GT (typically ~0), so a model that
    ignores the mask scores no lift.
    """
    was_training = model.training
    model.eval()
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    acc = {c: {v: [] for v in VIEW_ORDER} for c in ('mask', 'nomask')}
    for ex in examples:
        caption = ex['captions'][0] if ex.get('captions') else ex['caption']
        gt_views = three_view_masks(ex['bricks'], cfg)
        for cond, kept in (('mask', VIEW_ORDER), ('nomask', ())):
            text = _generate(model, tokenizer, build_user_content(caption, gt_views, kept),
                             device, max_new_tokens)
            bricks = _valid_bricks(text)
            structure = BrickStructure(bricks, world_dim=cfg.world_dim) if bricks else BrickStructure([])
            ious = silhouette_iou_per_view(structure, gt_views)
            for v in VIEW_ORDER:
                acc[cond][v].append(ious[v])
    if was_training:
        model.train()
    metrics = {}
    for v in VIEW_ORDER:
        m, n = float(np.mean(acc['mask'][v])), float(np.mean(acc['nomask'][v]))
        metrics[f'iou_mask_{v}'] = m
        metrics[f'iou_nomask_{v}'] = n
        metrics[f'iou_lift_{v}'] = m - n
    metrics['iou_lift_mean'] = float(np.mean([metrics[f'iou_lift_{v}'] for v in VIEW_ORDER]))
    return metrics


def load_stacked_policy(args, device):
    """
    Like :func:`grpo_text.load_policy`, but merges a *chain* of adapters into the frozen backbone
    before adding the fresh trainable SFT LoRA: ``model_name_or_path`` (BrickGPT) first, then each of
    ``--init_adapters`` in order (e.g. a GRPO checkpoint). With no ``init_adapters`` this is exactly
    equivalent to ``load_policy``. Starting SFT from GRPO-2k needs the backbone to be
    raw Llama -> merge BrickGPT -> merge GRPO -> fresh LoRA, which a single-merge ``load_policy`` can't do.
    """
    extra = [a.strip() for a in (args.init_adapters or '').split(',') if a.strip()]
    if not extra:
        return load_policy(args, device)

    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    peft_cfg = PeftConfig.from_pretrained(args.model_name_or_path)
    backbone = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    for adapter in [args.model_name_or_path, *extra]:
        backbone = PeftModel.from_pretrained(backbone, adapter).merge_and_unload()
        logger.info('Merged adapter into backbone: %s', adapter)
    backbone = backbone.to(device)

    if args.resume_from:
        model = PeftModel.from_pretrained(backbone, args.resume_from, is_trainable=True)
        logger.info('Resumed LoRA adapter from %s (start_step=%d)', args.resume_from, args.start_step)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(',') if m.strip()],
            task_type='CAUSAL_LM', bias='none',
        )
        model = get_peft_model(backbone, lora_cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def _save_adapter(model, output_dir, tag):
    out = os.path.join(output_dir, f'adapter_{tag}')
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    logger.info('Saved LoRA adapter (%s) to %s', tag, out)


def main():
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer, HfArgumentParser, get_cosine_schedule_with_warmup
    from datasets import load_dataset

    (args,) = HfArgumentParser(SFTTextMaskArguments).parse_args_into_dataclasses()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = load_stacked_policy(args, device)  # merge BrickGPT (+ optional init_adapters) + fresh LoRA
    model.train()

    train_data = load_dataset(args.dataset_name, split=args.train_split)
    if args.max_train_rows:
        train_data = train_data.select(range(min(args.max_train_rows, len(train_data))))
    ds = TextMaskBrickDataset(train_data, tokenizer, cfg, args.view_keep_prob, args.p_uncond,
                              train=True, rng_seed=args.seed)
    # Over-long examples (mask block + bricks) are dropped lazily in the collator (length varies with
    # view dropout), so startup does no full-dataset tokenization pass.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=Collator(tokenizer.pad_token_id or tokenizer.eos_token_id, args.max_seq_len))

    probe_data = load_dataset(args.dataset_name, split=args.probe_split)
    probe_examples = [probe_data[i] for i in range(min(args.probe_n, len(probe_data)))]

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    run = RunLogger(total=total_steps, phase='sft-text-mask', report_to=args.report_to,
                    console_every=args.log_every, project=args.wandb_project, name=args.run_name,
                    config=vars(args))

    step = args.start_step
    micro = 0
    opt.zero_grad(set_to_none=True)
    done = False
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None:   # whole micro-batch dropped for length
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, use_cache=False).loss / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run.log({'loss': loss.item() * args.grad_accum, 'lr': sched.get_last_lr()[0]}, step=step)

            if args.probe_every and step % args.probe_every == 0:
                m = iou_probe(model, tokenizer, probe_examples, cfg, device, args.probe_max_new_tokens, args.seed)
                run.log(m, step=step)
                logger.info('[probe step %d] lift mean=%+.4f | top %+.4f front %+.4f side %+.4f',
                            step, m['iou_lift_mean'], m['iou_lift_top'], m['iou_lift_front'], m['iou_lift_side'])
            if args.save_every and step % args.save_every == 0:
                _save_adapter(model, args.output_dir, f'step{step}')
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    _save_adapter(model, args.output_dir, 'final')
    if args.save_merged:
        merged_dir = os.path.join(args.output_dir, 'merged_final')
        model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info('Saved merged full model to %s (point eval_text_mask_iou.py --model_name_or_path here)', merged_dir)
    run.close()


if __name__ == '__main__':
    main()
