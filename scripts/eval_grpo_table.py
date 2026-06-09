"""
Build the **GRPO results table** for EXP.md: baseline / 1k / 2k / 2k+mask, with render+physics metrics.

This is the heavier sibling of ``brickgpt.training.eval_grpo_text`` (which only does the fast
valid/stable/overlap/CLIP/diversity numbers and the base-vs-adapter toggle). Here we add the three
expensive columns the EXP.md table wants and that the training-time eval deliberately skips:

* **continuous Gurobi stability** -- one MIQP solve per structure (WLS license; ``GRB_LICENSE_FILE``),
  reduced to a per-structure margin in ``[0, 1]`` (``1 - red`` over occupied voxels; higher = more
  stable). We report the binary ``stable_rate`` (Gurobi ``is_stable``) *and* the continuous mean/min
  from the **same** solve, so the two definitions can't drift. ``--gurobi_time_limit`` caps each solve.
* **DINOv2 score** -- image-image cosine between the render of the generated structure and the render
  of its **ground-truth** structure (the GT brick list shipped with each caption). GT is rendered once
  per caption and its DINOv2 feature cached across the caption's K samples.
* **mean_runtime** -- wall-clock generation time per sample (unconstrained decode only; no scaffolding).
* **scaffold_runtime / mean_regenerations** -- a *second* pass that runs each model through the FULL
  deployed inference scaffolding (``BrickGPT.__call__``: logit masking + rejection sampling + physics
  rollback, ``use_gurobi=False`` connectivity backend) and reports its per-structure wall-clock and the
  mean number of physics-rollback regenerations. This is the cost the unconstrained ``mean_runtime``
  excludes; a well-trained GRPO policy needs fewer regenerations -> cheaper scaffolding. ``--scaffold
  False`` skips it; ``--scaffold_max_regen`` caps the rollback count.

Each structure is rendered **once** and that single image feeds both CLIP (image-text vs caption) and
DINOv2 (image-image vs GT), so adding DINO does not double render cost.

Rows (``--rows``):
  baseline  base BrickGPT (``AvaLovelace/BrickGPT``, adapter disabled)
  1k        ``output/grpo_text/adapter_final``      (adapter on, over the same merged backbone)
  2k        ``output/grpo_text_2k/adapter_final``
  2kmask    ``output/text_mask_sft_grpo2k/merged_final`` -- a *full merged model*, evaluated with the
            SAME no-mask prompts as the others (the agreed apples-to-apples 口徑).

Network is needed once (DINOv2 download); Gurobi WLS also validates over the network at solve time.
Run (start with baseline only to validate the pipeline + get a real per-model timing)::

    env -u LD_LIBRARY_PATH \\
      GRB_LICENSE_FILE=$PWD/gurobi.lic CUDA_VISIBLE_DEVICES=1 \\
      uv run python scripts/eval_grpo_table.py --rows baseline

then, once happy, ``--rows baseline 1k 2k 2kmask``.
"""
import argparse
import json
import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from brickgpt.training.render_score import RenderScorer
from brickgpt.training.rewards import RewardConfig, _valid_bricks, compute_reward
from brickgpt.training.semantic import _structure_from_txt

logger = logging.getLogger(__name__)

ROW_SPECS = {
    'baseline': {'kind': 'adapter', 'adapter_dir': None},  # adapter disabled
    '1k':       {'kind': 'adapter', 'adapter_dir': 'output/grpo_text/adapter_final'},
    '2k':       {'kind': 'adapter', 'adapter_dir': 'output/grpo_text_2k/adapter_final'},
    '2kmask':   {'kind': 'merged',  'merged_dir':  'output/text_mask_sft_grpo2k/merged_final'},
}


@dataclass
class Args:
    rows: list
    base_model: str
    dataset_name: str
    split: str
    num_prompts: int
    samples_per_prompt: int
    max_gen_tokens: int
    temperature: float
    seed: int
    render_samples: int
    render_resolution: int
    world_dim: int
    gurobi_time_limit: float
    dino_model: str
    scaffold: bool
    scaffold_only: bool
    scaffold_max_regen: int
    scaffold_samples: int
    output_md: str | None


# --------------------------------------------------------------------------- models / generation
def _load_adapter_backbone(base_model, device):
    """Merged BrickGPT backbone (adapter disabled = baseline; adapter on = a trained GRPO policy)."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    peft_cfg = PeftConfig.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(base, base_model).merge_and_unload().to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return backbone, tokenizer


def _load_merged(merged_dir, device):
    """A standalone full model (the 2k+mask SFT->GRPO merge), evaluated with plain no-mask prompts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    return model, tokenizer


@torch.no_grad()
def _generate(model, tokenizer, caption, args, device):
    """K unconstrained completions; returns (texts, mean_gen_seconds_per_sample)."""
    from brickgpt.models import create_instruction
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
         {'role': 'user', 'content': create_instruction(caption)}],
        add_generation_prompt=True, return_tensors='pt').to(device)
    attn = torch.ones_like(prompt_ids)
    texts, dur = [], 0.0
    for _ in range(args.samples_per_prompt):
        t0 = time.time()
        out = model.generate(
            input_ids=prompt_ids, attention_mask=attn, do_sample=True, num_return_sequences=1,
            max_new_tokens=args.max_gen_tokens, temperature=args.temperature, top_k=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True)
        dur += time.time() - t0
        gen = out.sequences[0, prompt_ids.shape[1]:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    return texts, dur / max(args.samples_per_prompt, 1)


# --------------------------------------------------------------------------- continuous Gurobi
def gurobi_stability(structure, time_limit):
    """
    One MIQP solve -> (binary_stable, mean_margin, min_margin), margins in [0, 1] over occupied voxels
    (``1 - red``; higher = more stable). Floating/colliding/oob structures are physically unstable and
    can't be solved -> (False, 0.0, 0.0). Consistent with ``BrickStructure.is_stable`` (``red.max()<1``).
    """
    from brickgpt.stability_analysis import StabilityConfig, stability_score
    from brickgpt.data.brick_library import brick_library
    if structure.has_floating_bricks() or structure.has_collisions() or structure.has_out_of_bounds_bricks():
        return False, 0.0, 0.0
    cfg = StabilityConfig(world_dimension=(structure.world_dim,) * 3, time_limit=time_limit)
    red, *_ = stability_score(structure.to_json(), brick_library, cfg)  # per-voxel red channel
    occ = structure.voxel_occupancy > 0
    if not np.any(occ):
        return False, 0.0, 0.0
    margin = 1.0 - red[occ]
    binary_stable = bool(red[occ].max() < 1)
    return binary_stable, float(margin.mean()), float(margin.min())


# --------------------------------------------------------------------------- scaffold (rollback) timing
@torch.no_grad()
def scaffold_stats(model, tokenizer, captions, args, device):
    """
    Run each caption through the FULL inference scaffolding (``BrickGPT.__call__``: per-token logit
    masking + per-brick rejection sampling + physics-informed rollback) and report what that
    deployed path costs -- which the unconstrained ``mean_runtime`` deliberately excludes:

    * **scaffold_runtime**     -- mean wall-clock seconds per generated structure (the whole scaffolded
      call, rollback included).
    * **mean_regenerations**   -- mean number of physics-rollback regenerations
      (``__call__``'s ``n_regenerations``); a well-trained GRPO policy needs fewer.

    Stability inside the rollback loop uses the **connectivity** backend (``use_gurobi=False``) -- the
    deployed ``--use_gurobi False`` path; Gurobi is far too slow to call inside this loop (CLAUDE.md).
    ``--scaffold_max_regen`` caps the rollback count so a hopeless baseline can't hang the batch.
    """
    from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig
    from brickgpt.models.llm import LLM
    cfg = BrickGPTConfig(use_gurobi=False, max_regenerations=args.scaffold_max_regen,
                         world_dim=args.world_dim)
    bg = BrickGPT(cfg, llm=LLM.from_model(model, tokenizer, device))
    durs, regens = [], []
    for ci, cap in enumerate(captions):
        for _ in range(args.scaffold_samples):
            t0 = time.time()
            out = bg(cap)
            durs.append(time.time() - t0)
            regens.append(out['n_regenerations'])
    return (float(np.mean(durs)) if durs else 0.0,
            float(np.mean(regens)) if regens else 0.0)


# --------------------------------------------------------------------------- per-model eval
def eval_model(name, gen_fn, captions, gt_texts, scorer, reward_cfg, args):
    valid = total = 0
    g_stable = 0  # Gurobi binary stable count (over valid)
    overlaps, n_bricks, clip_raws, dinos = [], [], [], []
    g_means, g_mins, runtimes = [], [], []
    for ci, cap in enumerate(captions):
        texts, mean_gen_s = gen_fn(cap)
        runtimes.append(mean_gen_s)
        # GT DINO feature: render the ground-truth structure once per caption, cache its feature.
        gt_feat = None
        gt_struct = _structure_from_txt(gt_texts[ci], args.world_dim)
        if gt_struct is not None:
            gt_img = scorer.render(gt_struct, f'gt_{ci}')
            if gt_img is not None:
                gt_feat = scorer.dino_feat(gt_img)
        for si, text in enumerate(texts):
            total += 1
            bd = compute_reward(text, cfg=reward_cfg, clip_score=None)
            if not bd.syntax_ok:
                continue
            valid += 1
            if bd.overlap is not None:
                overlaps.append(bd.overlap)
            n_bricks.append(len(_valid_bricks(text)))
            structure = _structure_from_txt(text, args.world_dim)
            if structure is None:  # syntactically valid but unbuildable (e.g. oob) -> unstable
                g_means.append(0.0); g_mins.append(0.0)
                continue
            # one Gurobi solve -> binary + continuous
            b_stable, g_mean, g_min = gurobi_stability(structure, args.gurobi_time_limit)
            if b_stable:
                g_stable += 1
            g_means.append(g_mean); g_mins.append(g_min)
            # one render -> CLIP + DINO
            img = scorer.render(structure, f'{name}_{ci}_{si}')
            if img is not None:
                clip_raws.append(scorer.clip_cosine(img, cap))
                if gt_feat is not None:
                    dinos.append((scorer.dino_feat(img) @ gt_feat.T).item())
    return {
        'model': name, 'n': total,
        'valid_rate': valid / max(total, 1),
        'stable_rate': g_stable / max(valid, 1),
        'gurobi_mean': float(np.mean(g_means)) if g_means else 0.0,
        'gurobi_min': float(np.mean(g_mins)) if g_mins else 0.0,
        'clip': float(np.mean(clip_raws)) if clip_raws else None,
        'dino': float(np.mean(dinos)) if dinos else None,
        'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'mean_bricks': float(np.mean(n_bricks)) if n_bricks else 0.0,
        'mean_runtime': float(np.mean(runtimes)) if runtimes else 0.0,
    }


def print_table(rows):
    cols = [('model', 10, 's'), ('valid_rate', 11, '.3f'), ('stable_rate', 12, '.3f'),
            ('gurobi_mean', 12, '.3f'), ('gurobi_min', 11, '.3f'), ('clip', 9, '.4f'),
            ('dino', 9, '.4f'), ('mean_overlap', 13, '.3f'), ('mean_bricks', 12, '.2f'),
            ('mean_runtime', 13, '.2f'), ('scaffold_runtime', 17, '.2f'),
            ('mean_regenerations', 19, '.2f')]
    head = ''.join(f'{c:>{w}}' for c, w, _ in cols)
    print('\n' + head + '\n' + '-' * len(head))
    for r in rows:
        cells = []
        for c, w, fmt in cols:
            v = r[c]
            cells.append(f'{("n/a" if v is None else v):>{w}}' if (fmt == 's' or v is None)
                         else f'{v:>{w}{fmt}}')
        print(''.join(cells))
    print()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', nargs='+', default=['baseline'], choices=list(ROW_SPECS))
    p.add_argument('--base_model', default='AvaLovelace/BrickGPT')
    p.add_argument('--dataset_name', default='AvaLovelace/StableText2Brick')
    p.add_argument('--split', default='test', help='HELD-OUT split for reporting; GRPO trained on train.')
    p.add_argument('--num_prompts', type=int, default=20)
    p.add_argument('--samples_per_prompt', type=int, default=4)
    p.add_argument('--max_gen_tokens', type=int, default=400)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--render_samples', type=int, default=32)
    p.add_argument('--render_resolution', type=int, default=224)
    p.add_argument('--world_dim', type=int, default=20)
    p.add_argument('--gurobi_time_limit', type=float, default=30.0)
    p.add_argument('--dino_model', default='facebook/dinov2-base')
    p.add_argument('--scaffold', type=lambda s: s.lower() not in ('false', '0', 'no'), default=True,
                   help='Also time the full inference scaffolding (logit-masking + rejection + rollback) '
                        'per model -> scaffold_runtime + mean_regenerations columns. --scaffold False to skip.')
    p.add_argument('--scaffold_only', action='store_true',
                   help='Skip the unconstrained metrics + Gurobi + render entirely and ONLY run the '
                        'scaffold pass (scaffold_runtime + mean_regenerations). Use to splice these two '
                        'columns into an existing results JSON cheaply.')
    p.add_argument('--scaffold_max_regen', type=int, default=20,
                   help='Cap on physics-rollback regenerations during the scaffold pass (deployed default '
                        'is 100; capped here so a hopeless baseline cannot hang the batch).')
    p.add_argument('--scaffold_samples', type=int, default=1,
                   help='Scaffolded generations per caption for the runtime/regen aggregate (1 is enough; '
                        'constrained decoding is slow).')
    p.add_argument('--output_md', default=None, help='Append a markdown table here (e.g. EXP.md scratch).')
    a = p.parse_args()
    return Args(**vars(a))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from datasets import load_dataset
    data = load_dataset(args.dataset_name, split=args.split)
    # Sample (caption, gt_bricks) pairs; fixed seed so every row sees identical prompts + GT.
    pairs = [(c, r['bricks']) for r in data for c in r['captions']]
    chosen = random.sample(pairs, min(args.num_prompts, len(pairs)))
    captions = [c for c, _ in chosen]
    gt_texts = [g for _, g in chosen]
    logger.info('Evaluating rows=%s on %d captions x %d samples (Gurobi TimeLimit=%.0fs).',
                args.rows, len(captions), args.samples_per_prompt, args.gurobi_time_limit)

    if args.scaffold_only:
        args.scaffold = True  # the whole point of this mode
    reward_cfg = RewardConfig(use_overlap=True, use_stability=True, use_iou=False, use_semantic=False)
    # In --scaffold_only mode we skip CLIP/DINO/Gurobi/render entirely, so don't load the scorer.
    scorer = None if args.scaffold_only else \
        RenderScorer(device, args.dino_model, args.render_samples, args.render_resolution)

    def attach_scaffold(row, model, tokenizer):
        """Time the full deployed scaffolding for `row`'s active model; -> scaffold_runtime + regens."""
        if args.scaffold:
            sr, mr = scaffold_stats(model, tokenizer, captions, args, device)
            logger.info('%s scaffold: %.2fs/struct, %.2f regenerations/struct', row['model'], sr, mr)
        else:
            sr, mr = None, None
        row['scaffold_runtime'] = sr
        row['mean_regenerations'] = mr
        return row

    _NUMERIC_COLS = ('valid_rate', 'stable_rate', 'gurobi_mean', 'gurobi_min', 'clip', 'dino',
                     'mean_overlap', 'mean_bricks', 'mean_runtime')

    def make_row(name, gen, model, tokenizer):
        """eval_model (full metrics) + scaffold pass, or scaffold-only stub when --scaffold_only."""
        if args.scaffold_only:
            row = {'model': name, 'n': 0, **{c: None for c in _NUMERIC_COLS}}
        else:
            row = eval_model(name, gen, captions, gt_texts, scorer, reward_cfg, args)
        return attach_scaffold(row, model, tokenizer)

    results = []
    # 'baseline'/'1k'/'2k' share one merged backbone (load once, then swap/disable adapters in place);
    # '2kmask' is a separate full model. Adapter swaps use load_adapter/set_adapter; baseline is the
    # disabled-adapter (or bare-backbone) policy -- the exact base BrickGPT, same trick as the GRPO KL ref.
    adapter_rows = [r for r in args.rows if ROW_SPECS[r]['kind'] == 'adapter']
    if adapter_rows:
        backbone, tokenizer = _load_adapter_backbone(args.base_model, device)
        from peft import PeftModel
        peft_model = None  # lazily created when the first real adapter is loaded
        for r in adapter_rows:
            adir = ROW_SPECS[r]['adapter_dir']
            t0 = time.time()
            if adir is None:  # baseline = base BrickGPT (no adapter influence)
                logger.info('--- %s (base BrickGPT, adapter disabled) ---', r)
                if peft_model is None:
                    model = backbone.eval()  # bare merged base
                    def gen(cap, _m=model):
                        return _generate(_m, tokenizer, cap, args, device)
                    row = make_row(r, gen, model, tokenizer)
                else:
                    with peft_model.disable_adapter():
                        def gen(cap, _m=peft_model):
                            return _generate(_m, tokenizer, cap, args, device)
                        row = make_row(r, gen, peft_model, tokenizer)
            else:
                logger.info('--- %s (adapter %s) ---', r, adir)
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(backbone, adir, adapter_name=r).to(device)
                else:
                    peft_model.load_adapter(adir, adapter_name=r)
                peft_model.set_adapter(r)
                peft_model.eval()
                def gen(cap, _m=peft_model):
                    return _generate(_m, tokenizer, cap, args, device)
                row = make_row(r, gen, peft_model, tokenizer)
            logger.info('%s done in %.1f min', r, (time.time() - t0) / 60)
            results.append(row)
            print_table([row])

    for r in args.rows:
        if ROW_SPECS[r]['kind'] != 'merged':
            continue
        mdir = ROW_SPECS[r]['merged_dir']
        logger.info('--- %s (merged model %s, no-mask prompts) ---', r, mdir)
        t0 = time.time()
        model, tokenizer = _load_merged(mdir, device)
        def gen(cap, _m=model, _t=tokenizer):
            return _generate(_m, _t, cap, args, device)
        row = make_row(r, gen, model, tokenizer)
        logger.info('%s done in %.1f min', r, (time.time() - t0) / 60)
        results.append(row)
        print_table([row])

    # order results to match the requested --rows
    results.sort(key=lambda x: args.rows.index(x['model']))
    print('\n===== FINAL =====')
    print_table(results)
    print(json.dumps(results, indent=2))
    if args.output_md:
        with open(args.output_md, 'a') as f:
            f.write('\n' + json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
