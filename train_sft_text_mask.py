"""
SFT fine-tuning with top/front/side binary strings in the prompt (text-mask approach).

Loads the base BrickGPT model, applies LoRA to the attention layers, and fine-tunes on
the local JSON dataset where each row already has ``top``/``front``/``side`` fields.
The LLM learns to read the binary silhouette strings and generate matching brick structures.

Run:
    uv run train_sft_text_mask.py \
        --train_data dataset01/mask01/train_masks.json \
        --eval_data  dataset01/mask01/test_masks.json  \
        --output_dir output/sft_text_mask

After training the script saves both the LoRA adapter and a merged full model at
``<output_dir>/merged``, which can be passed directly to eval_per_view_iou.py.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    train_data: str = field(default='dataset01/mask01/train_masks.json')
    eval_data: str = field(default='dataset01/mask01/test_masks.json')
    output_dir: str = field(default='output/sft_text_mask')

    # LoRA
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

    # Optimisation
    lr: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=50)
    max_steps: int = field(default=500)
    batch_size: int = field(default=1)
    grad_accum_steps: int = field(default=8)
    max_grad_norm: float = field(default=1.0)
    max_length: int = field(default=1024, metadata={'help': 'Truncate sequences longer than this.'})

    # Logging / saving
    log_every: int = field(default=20)
    save_every: int = field(default=200)
    eval_every: int = field(default=200)
    eval_n: int = field(default=32, metadata={'help': 'Number of eval examples for loss probe.'})

    num_workers: int = field(default=0)
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# collator
# ---------------------------------------------------------------------------

def collate_text(features: list[dict], pad_id: int) -> dict:
    max_len = max(f['input_ids'].size(0) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad = max_len - f['input_ids'].size(0)
        input_ids.append(F.pad(f['input_ids'], (0, pad), value=pad_id))
        attention_mask.append(F.pad(f['attention_mask'], (0, pad), value=0))
        labels.append(F.pad(f['labels'], (0, pad), value=-100))
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }


# ---------------------------------------------------------------------------
# dataset wrapper (truncation on top of MaskBrickDataset text-mask path)
# ---------------------------------------------------------------------------

class TruncatedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, max_length: int):
        self.base = base_ds
        self.max_length = max_length

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        for key in ('input_ids', 'attention_mask', 'labels'):
            if item[key].size(0) > self.max_length:
                item[key] = item[key][:self.max_length]
        return item


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    (args,) = HfArgumentParser(TrainArgs).parse_args_into_dataclasses()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device: %s', device)

    # ---- data ----------------------------------------------------------------
    with open(args.train_data) as f:
        train_rows = json.load(f)
    with open(args.eval_data) as f:
        eval_rows = json.load(f)

    # ---- tokenizer & base model ----------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    except ImportError:
        raise ImportError('peft is required: uv pip install peft')

    # Load BrickGPT. If it is a PEFT model, merge the existing adapter before adding ours.
    raw = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    if isinstance(raw, PeftModel):
        logger.info('Base model is a PeftModel — merging existing adapter before applying new LoRA.')
        model = raw.merge_and_unload()
    else:
        logger.info('Base model loaded as %s — applying new LoRA directly.', type(raw).__name__)
        model = raw

    # ---- LoRA ----------------------------------------------------------------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing: trade compute for memory (essential on 8 GB GPU).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ---- datasets ------------------------------------------------------------
    from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset

    cfg = MaskConditioningConfig()
    train_ds = TruncatedDataset(
        MaskBrickDataset(train_rows, tokenizer, cfg, train=True, use_text_mask=True),
        args.max_length,
    )
    eval_ds = TruncatedDataset(
        MaskBrickDataset(eval_rows[:args.eval_n], tokenizer, cfg, train=False, use_text_mask=True),
        args.max_length,
    )
    logger.info('Train examples: %d  |  Eval examples: %d', len(train_ds), len(eval_ds))

    pad_id = tokenizer.pad_token_id
    collate = lambda feats: collate_text(feats, pad_id)

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # ---- optimiser -----------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)

    # ---- training loop -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step, micro, running_loss = 0, 0, 0.0
    data_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / args.grad_accum_steps
        loss.backward()
        running_loss += out.loss.item()
        micro += 1

        if micro % args.grad_accum_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            avg = running_loss / args.grad_accum_steps
            logger.info('step %4d/%d  loss=%.4f  lr=%.2e', step, args.max_steps, avg, sched.get_last_lr()[0])
            running_loss = 0.0

        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for eb in eval_loader:
                    eb = {k: v.to(device) for k, v in eb.items()}
                    eval_losses.append(model(**eb).loss.item())
            model.train()
            logger.info('step %4d  eval_loss=%.4f', step, sum(eval_losses) / len(eval_losses))

        if args.save_every and step % args.save_every == 0:
            ckpt = Path(args.output_dir) / f'step{step}'
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            logger.info('Saved adapter checkpoint to %s', ckpt)

    # ---- final save ----------------------------------------------------------
    adapter_dir = Path(args.output_dir) / 'adapter_final'
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info('Saved final adapter to %s', adapter_dir)

    # merge LoRA weights into the base model and save a standalone checkpoint
    logger.info('Merging LoRA weights...')
    merged = model.merge_and_unload()
    # Strip leftover peft_config from the original BrickGPT adapter so
    # save_pretrained treats this as a plain model.
    for attr in ('peft_config', '_hf_peft_config_loaded'):
        if hasattr(merged, attr):
            try:
                delattr(merged, attr)
            except Exception:
                setattr(merged, attr, None)
    merged_dir = Path(args.output_dir) / 'merged'
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info('Saved merged model to %s  (use --model %s for eval)', merged_dir, merged_dir)


if __name__ == '__main__':
    main()
