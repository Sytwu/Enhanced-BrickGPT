"""
Mask-condition eval with **render metrics + top-K example dump** (the heavier sibling of
``scripts/eval_text_mask_iou.py``, which only does per-view IoU).

For one **track** (a text-mask SFT model + the start point it was fine-tuned from) we generate each
held-out structure under three conditions with the **constrained** decoder (the deployed inference
path, same as the IoU eval):

  * ``baseline`` -- the SFT model's *start* point, **no mask** (BrickGPT for the ``base`` track;
    GRPO-2k for the ``grpo2k`` track). The floor: shape match from text alone.
  * ``nomask``   -- the SFT model, **no mask** (what the extra SFT did to the no-mask prior).
  * ``mask``     -- the SFT model, **+ the three-view RLE mask block** (the real conditioning).

Per generation we record per-view silhouette **IoU** (vs GT), **CLIP** (render vs caption) and
**DINOv2** (render vs the GT structure's render). Then we report the aggregate table and **dump the
top-K examples** by two criteria, with everything needed to render / animate them later:

  * **top-K by IoU lift**  -- largest ``mean_3view(mask_IoU - nomask_IoU)`` (where the mask helped most).
  * **top-K by CLIP lift** -- largest ``mask_CLIP - nomask_CLIP``.

Each dumped example carries the GT bricks + conditioning silhouette and, per condition, the generated
brick list, per-view IoU, CLIP and DINOv2 -- so a downstream script can render the three side by side.

Run one track per process (so two tracks can run on two GPUs via ``.venv/bin/python`` -- NOT concurrent
``uv run``, which swaps bpy/numpy and breaks rendering)::

    env -u LD_LIBRARY_PATH LDRAW_LIBRARY_PATH=$PWD/ldraw CUDA_VISIBLE_DEVICES=0 \\
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      .venv/bin/python scripts/eval_text_mask_render.py --track base   --probe_n 64
    ... CUDA_VISIBLE_DEVICES=1 ... --track grpo2k --probe_n 64
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
from datasets import load_dataset

from brickgpt.masking import MaskConditioningConfig, VIEW_ORDER, build_user_content, three_view_masks
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig
from brickgpt.training.render_score import RenderScorer
from brickgpt.training.rewards import silhouette_iou_per_view
from brickgpt.training.semantic import _structure_from_txt

logger = logging.getLogger(__name__)

# Each track = (start point fine-tuned from, the SFT model). 'base' starts from released BrickGPT;
# 'grpo2k' starts from the GRPO-2k policy (BrickGPT backbone + grpo_text_2k adapter, merged).
TRACKS = {
    'base': {
        'start': {'kind': 'pretrained', 'name': 'AvaLovelace/BrickGPT'},
        'sft': 'output/text_mask_sft_base/merged_final',
    },
    'grpo2k': {
        'start': {'kind': 'grpo_merged', 'base_model': 'AvaLovelace/BrickGPT',
                  'adapter': 'output/grpo_text_2k/adapter_final'},
        'sft': 'output/text_mask_sft_grpo2k/merged_final',
    },
}
CONDITIONS = ('baseline', 'nomask', 'mask')


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bg_config(model_name_or_path, max_bricks):
    """The deployed constrained-decoder config used by the IoU eval (connectivity, no rollback)."""
    return BrickGPTConfig(model_name_or_path=model_name_or_path, use_gurobi=False,
                          max_regenerations=0, max_brick_rejections=10, max_bricks=max_bricks)


def _build_bg(spec_or_path, device, max_bricks):
    """A BrickGPT whose ``instruction_fn`` is identity (we feed pre-built instruction+mask content)."""
    if isinstance(spec_or_path, str):  # a local/full model dir or HF id (PEFT auto-loaded)
        bg = BrickGPT(_bg_config(spec_or_path, max_bricks))
    elif spec_or_path['kind'] == 'pretrained':
        bg = BrickGPT(_bg_config(spec_or_path['name'], max_bricks))
    elif spec_or_path['kind'] == 'grpo_merged':
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from brickgpt.models.llm import LLM
        peft_cfg = PeftConfig.from_pretrained(spec_or_path['base_model'])
        base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path,
                                                    torch_dtype=torch.bfloat16)
        backbone = PeftModel.from_pretrained(base, spec_or_path['base_model']).merge_and_unload()
        merged = PeftModel.from_pretrained(backbone, spec_or_path['adapter']).merge_and_unload().to(device)
        tokenizer = AutoTokenizer.from_pretrained(spec_or_path['base_model'])
        bg = BrickGPT(_bg_config('unused', max_bricks), llm=LLM.from_model(merged, tokenizer, device))
    else:
        raise ValueError(f'unknown start spec {spec_or_path}')
    bg.instruction_fn = lambda content: content  # feed (instruction [+ mask block]) verbatim
    return bg


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', choices=list(TRACKS), default='base')
    ap.add_argument('--dataset_name', default='AvaLovelace/StableText2Brick')
    ap.add_argument('--split', default='test')
    ap.add_argument('--probe_n', type=int, default=64)
    ap.add_argument('--top_k', type=int, default=20, help='How many examples to dump per criterion.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_bricks', type=int, default=40)
    ap.add_argument('--world_dim', type=int, default=20)
    ap.add_argument('--render_samples', type=int, default=32)
    ap.add_argument('--render_resolution', type=int, default=224)
    ap.add_argument('--dino_model', default='facebook/dinov2-base')
    ap.add_argument('--output_dir', default='output')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    track = TRACKS[args.track]
    mask_cfg = MaskConditioningConfig()

    logger.info('Building models for track=%s ...', args.track)
    bg_start = _build_bg(track['start'], device, args.max_bricks)   # baseline condition
    bg_sft = _build_bg(track['sft'], device, args.max_bricks)       # nomask + mask conditions
    bg_for = {'baseline': bg_start, 'nomask': bg_sft, 'mask': bg_sft}
    scorer = RenderScorer(device, args.dino_model, args.render_samples, args.render_resolution)

    data = load_dataset(args.dataset_name, split=args.split)
    examples = [data[i] for i in range(min(args.probe_n, len(data)))]
    logger.info('Eval on %d held-out "%s" examples | track=%s | conditions=%s',
                len(examples), args.split, args.track, CONDITIONS)

    # cond -> view -> per-example IoU; cond -> per-example CLIP / DINO
    iou_scores = {c: {v: [] for v in VIEW_ORDER} for c in CONDITIONS}
    clip_scores = {c: [] for c in CONDITIONS}
    dino_scores = {c: [] for c in CONDITIONS}
    per_example = []  # full record for the dump

    for idx, ex in enumerate(examples):
        caption = ex['captions'][0] if ex.get('captions') else ex['caption']
        gt_views = three_view_masks(ex['bricks'], mask_cfg)  # dict view -> binary 2D array

        # GT render -> DINOv2 reference feature (rendered once, reused across conditions)
        gt_feat = None
        gt_struct = _structure_from_txt(ex['bricks'], args.world_dim)
        if gt_struct is not None:
            gt_img = scorer.render(gt_struct, f'{args.track}_gt_{idx}')
            if gt_img is not None:
                gt_feat = scorer.dino_feat(gt_img)

        rec = {'idx': idx, 'caption': caption, 'gt_bricks': ex['bricks'],
               'gt_mask': {v: gt_views[v].astype(int).tolist() for v in VIEW_ORDER},
               'conditions': {}}
        for cond in CONDITIONS:
            view_names = VIEW_ORDER if cond == 'mask' else ()
            content = build_user_content(caption, gt_views, view_names)
            seed_all(args.seed + idx)  # identical RNG start per example across conditions -> paired
            structure = bg_for[cond](content)['bricks']
            ious = silhouette_iou_per_view(structure, gt_views)
            for v in VIEW_ORDER:
                iou_scores[cond][v].append(ious[v])
            img = scorer.render(structure, f'{args.track}_{cond}_{idx}')
            clip = scorer.clip_cosine(img, caption) if img is not None else None
            dino = (scorer.dino_feat(img) @ gt_feat.T).item() if (img is not None and gt_feat is not None) else None
            if clip is not None:
                clip_scores[cond].append(clip)
            if dino is not None:
                dino_scores[cond].append(dino)
            rec['conditions'][cond] = {
                'bricks': structure.to_txt(),
                'iou': {v: round(float(ious[v]), 4) for v in VIEW_ORDER},
                'iou_mean': round(float(np.mean([ious[v] for v in VIEW_ORDER])), 4),
                'clip': round(clip, 4) if clip is not None else None,
                'dino': round(dino, 4) if dino is not None else None,
            }
        # paired lifts (mask - nomask)
        iou_lift = float(np.mean([rec['conditions']['mask']['iou'][v] - rec['conditions']['nomask']['iou'][v]
                                  for v in VIEW_ORDER]))
        m_clip, n_clip = rec['conditions']['mask']['clip'], rec['conditions']['nomask']['clip']
        clip_lift = (m_clip - n_clip) if (m_clip is not None and n_clip is not None) else None
        rec['iou_lift_mean'] = round(iou_lift, 4)
        rec['clip_lift'] = round(clip_lift, 4) if clip_lift is not None else None
        per_example.append(rec)
        logger.info('[%d/%d] iou_lift=%+.3f clip_lift=%s', idx + 1, len(examples), iou_lift,
                    f'{clip_lift:+.3f}' if clip_lift is not None else 'n/a')

    # ---- aggregate table
    def agg(cond):
        row = {'condition': cond}
        for v in VIEW_ORDER:
            row[v] = float(np.mean(iou_scores[cond][v])) if iou_scores[cond][v] else 0.0
        row['iou_mean'] = float(np.mean([row[v] for v in VIEW_ORDER]))
        row['clip'] = float(np.mean(clip_scores[cond])) if clip_scores[cond] else None
        row['dino'] = float(np.mean(dino_scores[cond])) if dino_scores[cond] else None
        return row
    table = [agg(c) for c in CONDITIONS]

    print(f'\n=== track={args.track} | n={len(examples)} ===')
    hdr = f'{"condition":<10} | ' + ' | '.join(f'{v:>7}' for v in VIEW_ORDER) + ' |  iou_mn |    CLIP |    DINO'
    print(hdr + '\n' + '-' * len(hdr))
    for row in table:
        line = (f'{row["condition"]:<10} | ' + ' | '.join(f'{row[v]:7.4f}' for v in VIEW_ORDER) +
                f' | {row["iou_mean"]:7.4f} | ' +
                (f'{row["clip"]:7.4f}' if row['clip'] is not None else '    n/a') + ' | ' +
                (f'{row["dino"]:7.4f}' if row['dino'] is not None else '    n/a'))
        print(line)
    # mask - nomask lift line
    nm, mk = table[CONDITIONS.index('nomask')], table[CONDITIONS.index('mask')]
    print('-' * len(hdr))
    dv = {v: mk[v] - nm[v] for v in VIEW_ORDER}
    dclip = (mk['clip'] - nm['clip']) if (mk['clip'] is not None and nm['clip'] is not None) else None
    print(f'{"Δ mask":<10} | ' + ' | '.join(f'{dv[v]:+7.4f}' for v in VIEW_ORDER) +
          f' | {np.mean(list(dv.values())):+7.4f} | ' +
          (f'{dclip:+7.4f}' if dclip is not None else '    n/a') + ' |     n/a')

    # ---- top-K dumps
    by_iou = sorted(per_example, key=lambda r: r['iou_lift_mean'], reverse=True)[:args.top_k]
    by_clip = sorted([r for r in per_example if r['clip_lift'] is not None],
                     key=lambda r: r['clip_lift'], reverse=True)[:args.top_k]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'mask_render_{args.track}.json')
    payload = {
        'meta': {
            'script': 'scripts/eval_text_mask_render.py', 'track': args.track,
            'start': track['start'], 'sft': track['sft'],
            'dataset': args.dataset_name, 'split': args.split, 'n': len(examples),
            'seed': args.seed, 'top_k': args.top_k, 'decoder': 'constrained (deployed inference path)',
            'clip': 'OpenCLIP ViT-B-32/openai (render vs caption)',
            'dino': f'{args.dino_model} (gen render vs GT render)',
            'conditions': 'baseline=start-point nomask; nomask=SFT nomask; mask=SFT +3-view RLE mask',
        },
        'table': table,
        'top_by_iou_lift': by_iou,
        'top_by_clip_lift': by_clip,
        'all_examples': per_example,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nDumped table + top-{args.top_k} (IoU lift & CLIP lift) + all {len(examples)} examples '
          f'-> {out_path}')


if __name__ == '__main__':
    main()
