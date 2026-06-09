"""
Per-view IoU eval for the text-token mask conditioning route (top / front / side reported separately).

For each held-out structure it generates the brick list under one or more conditions and measures the
IoU of each generated silhouette against the ground-truth silhouette:

  * ``nomask`` -- prompt is the plain instruction (no mask block). On the *un-fine-tuned* base model
    this is the BASELINE FLOOR the plan calls for: how much a generated shape coincidentally matches
    the GT shape from text alone.
  * ``mask``   -- prompt carries the run-length mask block for all three views (the real conditioning).

When both conditions are run, the per-view delta ``mask - nomask`` is the mask-usage lift. The script
formats the prompt with the exact same `build_user_content` used for training, so train/eval match.

    # Baseline line first (base model, no mask):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        uv run python scripts/eval_text_mask_iou.py --conditions nomask --probe_n 64

    # After SFT, lift on the fine-tuned checkpoint:
    ... uv run python scripts/eval_text_mask_iou.py \
        --model_name_or_path output/text_mask_sft --conditions nomask,mask --probe_n 64
"""
import argparse
import csv

import numpy as np
import torch
from datasets import load_dataset

from brickgpt.masking import MaskConditioningConfig, VIEW_ORDER, build_user_content, three_view_masks
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig
from brickgpt.training.rewards import silhouette_iou_per_view


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', default='AvaLovelace/BrickGPT',
                    help='Base model for the baseline; a fine-tuned dir for the lift.')
    ap.add_argument('--dataset_name', default='AvaLovelace/StableText2Brick')
    ap.add_argument('--split', default='test')
    ap.add_argument('--conditions', default='nomask',
                    help='Comma list of {nomask,mask}. Baseline = "nomask" on the base model.')
    ap.add_argument('--probe_n', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_bricks', type=int, default=40)
    ap.add_argument('--save_csv', default='')
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    assert all(c in ('nomask', 'mask') for c in conditions), 'conditions must be nomask/mask'

    mask_cfg = MaskConditioningConfig()
    bg_cfg = BrickGPTConfig(model_name_or_path=args.model_name_or_path, use_gurobi=False,
                            max_regenerations=0, max_brick_rejections=10, max_bricks=args.max_bricks)
    bg = BrickGPT(bg_cfg)
    # Feed our pre-built (instruction + optional mask block) verbatim instead of re-wrapping the caption.
    bg.instruction_fn = lambda content: content

    data = load_dataset(args.dataset_name, split=args.split)
    examples = [data[i] for i in range(min(args.probe_n, len(data)))]
    print(f'Eval on {len(examples)} held-out "{args.split}" examples | model={args.model_name_or_path} '
          f'| conditions={conditions} | seed={args.seed}\n')

    # cond -> view -> list of per-example IoUs
    scores = {c: {v: [] for v in VIEW_ORDER} for c in conditions}
    rows = []
    for idx, ex in enumerate(examples):
        caption = ex['captions'][0] if ex.get('captions') else ex['caption']
        gt_views = three_view_masks(ex['bricks'], mask_cfg)
        row = {'idx': idx, 'caption': caption[:60]}
        for cond in conditions:
            view_names = VIEW_ORDER if cond == 'mask' else ()
            content = build_user_content(caption, gt_views, view_names)
            seed_all(args.seed + idx)   # same seed per example across conditions -> paired comparison
            structure = bg(content)['bricks']
            ious = silhouette_iou_per_view(structure, gt_views)
            for v in VIEW_ORDER:
                scores[cond][v].append(ious[v])
                row[f'{cond}_{v}'] = round(ious[v], 4)
        rows.append(row)

    print(f'{"condition":<10} | ' + ' | '.join(f'{v:>7}' for v in VIEW_ORDER) + ' |    mean')
    print('-' * 56)
    means = {}
    for cond in conditions:
        m = {v: float(np.mean(scores[cond][v])) for v in VIEW_ORDER}
        means[cond] = m
        overall = float(np.mean([m[v] for v in VIEW_ORDER]))
        print(f'{cond:<10} | ' + ' | '.join(f'{m[v]:7.4f}' for v in VIEW_ORDER) + f' | {overall:7.4f}')

    if 'mask' in means and 'nomask' in means:
        d = {v: means['mask'][v] - means['nomask'][v] for v in VIEW_ORDER}
        overall = float(np.mean(list(d.values())))
        print('-' * 56)
        print(f'{"Δ lift":<10} | ' + ' | '.join(f'{d[v]:+7.4f}' for v in VIEW_ORDER) + f' | {overall:+7.4f}')

    if args.save_csv and rows:
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nPer-example IoUs written to {args.save_csv}')


if __name__ == '__main__':
    main()
