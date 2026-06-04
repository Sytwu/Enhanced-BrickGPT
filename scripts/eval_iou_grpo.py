"""
Exp 2.0 eval: did IoU-GRPO improve the mask-conditioning IoU lift over the SFT seed?

Loads ONE base BrickGPTWithMask, then swaps in each mask-encoder checkpoint (SFT seed vs GRPO final)
and runs the same `iou_probe` (mask-vs-null IoU lift) on the same held-out test examples with a fixed
seed per probe, for a fair comparison. Prints a small table; does not write EXP.md.

Usage:
    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        uv run python scripts/eval_iou_grpo.py --probe_n 64
"""
import argparse

import numpy as np
import torch

from brickgpt.masking import MaskConditioningConfig
from brickgpt.models.brickgpt import BrickGPTConfig
from brickgpt.models.masked_brickgpt import BrickGPTWithMask
from brickgpt.training.generation import iou_probe
from transformers import AutoTokenizer
from datasets import load_dataset


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_encoder(model, ckpt_path, tokenizer, examples, cfg, probe_cfg, device, seed):
    sd = torch.load(ckpt_path, map_location=device)
    model.mask_prefix_encoder.load_state_dict(sd)
    seed_all(seed)
    return iou_probe(model, tokenizer, examples, cfg, probe_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', default='AvaLovelace/BrickGPT')
    ap.add_argument('--dataset_name', default='AvaLovelace/StableText2Brick')
    ap.add_argument('--probe_split', default='test')
    ap.add_argument('--sft_ckpt', default='output/sft_masked_2k71j4zp4/mask_encoder_final.pt')
    ap.add_argument('--grpo_ckpt', default='output/grpo_masked_2k71j4zp4/mask_encoder_final.pt')
    ap.add_argument('--probe_n', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = MaskConditioningConfig()          # same default both training scripts use
    probe_cfg = BrickGPTConfig(model_name_or_path=args.model_name_or_path, use_gurobi=False,
                               max_regenerations=0, max_brick_rejections=10, max_bricks=40)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BrickGPTWithMask.from_pretrained(args.model_name_or_path, cfg, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    data = load_dataset(args.dataset_name, split=args.probe_split)
    examples = [data[i] for i in range(min(args.probe_n, len(data)))]
    print(f'Probing on {len(examples)} held-out "{args.probe_split}" examples (seed={args.seed}).')

    rows = []
    for tag, ckpt in [('SFT seed', args.sft_ckpt), ('GRPO final', args.grpo_ckpt)]:
        m = probe_encoder(model, ckpt, tokenizer, examples, cfg, probe_cfg, device, args.seed)
        rows.append((tag, m))
        print(f'[{tag:10s}] iou_masked={m["iou_masked"]:.4f}  iou_null={m["iou_null"]:.4f}  '
              f'iou_lift={m["iou_lift"]:+.4f}  (n={int(m["n"])})')

    sft, grpo = rows[0][1], rows[1][1]
    print('\n=== Exp 2.0 verdict ===')
    print(f'  SFT  iou_lift = {sft["iou_lift"]:+.4f}')
    print(f'  GRPO iou_lift = {grpo["iou_lift"]:+.4f}')
    print(f'  Δlift (GRPO - SFT) = {grpo["iou_lift"] - sft["iou_lift"]:+.4f}')
    print(f'  Δ iou_masked       = {grpo["iou_masked"] - sft["iou_masked"]:+.4f}')


if __name__ == '__main__':
    main()
