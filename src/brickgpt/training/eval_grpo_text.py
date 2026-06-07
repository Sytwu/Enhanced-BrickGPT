"""
Evaluate a text-only GRPO checkpoint (Path A) against the **base** BrickGPT it was trained from.

Both models share one merged backbone: the trained LoRA adapter ON is the GRPO policy, and
``disable_adapter()`` recovers the released BrickGPT exactly (same trick the GRPO KL reference uses).
So the comparison is strictly base-vs-base+LoRA on identical prompts -- no confound from a separate load.

For each of ``num_prompts`` captions we draw ``samples_per_prompt`` **unconstrained** completions
(``do_sample``, ``temperature`` -- the training-time decoder, *not* the inference-time logit-masking /
rejection-sampling scaffolding) and report, per model:

* **valid_rate**   -- fraction whose every line is a syntactically valid, in-library brick.
* **stable_rate**  -- fraction (of *valid* completions) that are ground-connected with no float/oob.
* **mean_overlap** -- mean overlap penalty over valid completions (0 = collision-free; the goal).
* **mean_bricks**  -- mean brick count over valid completions (a crude size/collapse signal).
* **clip_raw**     -- mean *raw* CLIP cosine (render vs caption) over valid completions, if
                      ``--use_semantic``. This is the unmapped image-text similarity.
* **semantic**     -- mean of ``clip_raw`` after the reward's [0,1] normalization
                      (``clip((s - clip_lo) / (clip_hi - clip_lo), 0, 1)``); this is the value the GRPO
                      reward actually optimizes. Reported alongside ``clip_raw`` so the headline number
                      is comparable to the published CLIP-score scale.
* **uniq_intra**   -- mean fraction of *distinct* structures within a prompt's K samples (1.0 = all
                      different; low = the model ignores sampling noise -> mode-collapse within a prompt).
* **uniq_global**  -- distinct structures across *all* generations / total valid (low = the model emits
                      the same few structures regardless of caption -> caption-agnostic collapse).

The point: GRPO is "good" only if it lifts valid/stable **without** crushing diversity (uniq_*). Run::

    uv run python -m brickgpt.training.eval_grpo_text --adapter_dir output/grpo_text/adapter_final
"""
import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from brickgpt.training.rewards import RewardConfig, _valid_bricks, compute_reward

logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    base_model: str = field(default='AvaLovelace/BrickGPT')
    adapter_dir: str = field(default='output/grpo_text/adapter_final')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    split: str = field(default='train')

    num_prompts: int = field(default=20, metadata={'help': 'Distinct captions to evaluate on.'})
    samples_per_prompt: int = field(default=4, metadata={'help': 'K unconstrained completions per caption.'})
    max_gen_tokens: int = field(default=400)
    temperature: float = field(default=1.0)
    seed: int = field(default=0)

    use_semantic: bool = field(default=True, metadata={'help': 'Render + CLIP-score each valid completion '
                                                               'vs its caption (heavy; needs LDraw/render).'})
    render_samples: int = field(default=32)
    render_resolution: int = field(default=224)


def _canonical(bricks_txt: str) -> str | None:
    """A canonical, order-independent signature of a valid structure (for dedup); None if invalid."""
    bricks = _valid_bricks(bricks_txt)
    if bricks is None:
        return None
    return '\n'.join(sorted(b.to_txt() for b in bricks))


def _load(args, device):
    """Merged BrickGPT backbone + trained adapter as one model (adapter on = GRPO, disabled = base)."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_cfg = PeftConfig.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(base, args.base_model).merge_and_unload().to(device)
    model = PeftModel.from_pretrained(backbone, args.adapter_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    return model, tokenizer


@torch.no_grad()
def _generate(model, tokenizer, caption, args, device):
    from brickgpt.models import create_instruction
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
         {'role': 'user', 'content': create_instruction(caption)}],
        add_generation_prompt=True, return_tensors='pt').to(device)
    attn = torch.ones_like(prompt_ids)
    texts = []
    for _ in range(args.samples_per_prompt):
        out = model.generate(
            input_ids=prompt_ids, attention_mask=attn, do_sample=True, num_return_sequences=1,
            max_new_tokens=args.max_gen_tokens, temperature=args.temperature, top_k=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True)
        gen = out.sequences[0, prompt_ids.shape[1]:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    return texts


def _eval_model(name, gen_fn, captions, scorer, reward_cfg, args):
    """Generates + scores all prompts under one model (the caller toggles adapter on/off via gen_fn)."""
    valid = stable = total = 0
    overlaps, n_bricks, semantics, clip_raws = [], [], [], []
    uniq_intra, all_sigs = [], []
    for cap in captions:
        texts = gen_fn(cap)
        sigs = []
        for text in texts:
            total += 1
            clip = scorer.score(text, cap) if scorer is not None else None
            bd = compute_reward(text, cfg=reward_cfg, clip_score=clip)
            if not bd.syntax_ok:
                continue
            valid += 1
            if bd.stability == 1.0:
                stable += 1
            if bd.overlap is not None:
                overlaps.append(bd.overlap)
            if bd.semantic is not None:
                semantics.append(bd.semantic)
            if clip is not None:  # raw cosine, before the reward's [0,1] normalization
                clip_raws.append(clip)
            bricks = _valid_bricks(text)
            n_bricks.append(len(bricks))
            sig = _canonical(text)
            sigs.append(sig)
            all_sigs.append(sig)
        if sigs:
            uniq_intra.append(len(set(sigs)) / len(sigs))
    return {
        'model': name, 'n': total, 'valid_rate': valid / max(total, 1),
        'stable_rate': stable / max(valid, 1), 'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'mean_bricks': float(np.mean(n_bricks)) if n_bricks else 0.0,
        'clip_raw': float(np.mean(clip_raws)) if clip_raws else None,
        'semantic': float(np.mean(semantics)) if semantics else None,
        'uniq_intra': float(np.mean(uniq_intra)) if uniq_intra else 0.0,
        'uniq_global': len(set(all_sigs)) / max(valid, 1),
    }


def _print_table(rows):
    cols = [('model', 12, 's'), ('valid_rate', 11, '.3f'), ('stable_rate', 12, '.3f'),
            ('mean_overlap', 13, '.3f'), ('mean_bricks', 12, '.2f'), ('clip_raw', 10, '.4f'),
            ('semantic', 10, '.4f'), ('uniq_intra', 11, '.3f'), ('uniq_global', 12, '.3f')]
    head = ''.join(f'{c:>{w}}' for c, w, _ in cols)
    print('\n' + head)
    print('-' * len(head))
    for r in rows:
        cells = []
        for c, w, fmt in cols:
            v = r[c]
            if fmt == 's' or v is None:
                cells.append(f'{("n/a" if v is None else v):>{w}}')
            else:
                cells.append(f'{v:>{w}{fmt}}')
        print(''.join(cells))
    print()


def main():
    logging.basicConfig(level=logging.INFO)
    from transformers import HfArgumentParser
    (args,) = HfArgumentParser(EvalArguments).parse_args_into_dataclasses()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from datasets import load_dataset
    data = load_dataset(args.dataset_name, split=args.split)
    all_caps = [c for r in data for c in r['captions']]
    captions = random.sample(all_caps, min(args.num_prompts, len(all_caps)))
    logger.info('Evaluating on %d captions x %d samples each.', len(captions), args.samples_per_prompt)

    model, tokenizer = _load(args, device)
    reward_cfg = RewardConfig(use_overlap=True, use_stability=True, use_iou=False,
                              use_semantic=args.use_semantic)
    scorer = None
    if args.use_semantic:
        from brickgpt.training.semantic import SemanticScorer
        scorer = SemanticScorer(device, render_samples=args.render_samples,
                                img_resolution=args.render_resolution)
        logger.info('Semantic ON: rendering each valid completion (this dominates runtime).')

    def gen(cap):
        return _generate(model, tokenizer, cap, args, device)

    # base = adapter disabled (released BrickGPT); grpo = adapter on (trained policy). Same prompts.
    logger.info('--- base BrickGPT (adapter disabled) ---')
    with model.disable_adapter():
        base_row = _eval_model('base', gen, captions, scorer, reward_cfg, args)
    logger.info('--- GRPO (adapter on) ---')
    grpo_row = _eval_model('grpo', gen, captions, scorer, reward_cfg, args)

    _print_table([base_row, grpo_row])


if __name__ == '__main__':
    main()
