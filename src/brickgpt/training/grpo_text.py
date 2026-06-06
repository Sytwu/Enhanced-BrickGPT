"""
Text-only GRPO post-training for the **original** (mask-free) BrickGPT -- Path A (LoRA-on-LLM).

This is the no-mask counterpart of :mod:`brickgpt.training.grpo_masked`. Where the masked loop freezes
the LLM and pushes the policy gradient through a trainable mask-prefix encoder (Path B), here there is
no mask and no prefix: the policy is the LLM itself, adapted with **LoRA**. The KL penalty anchors the
LoRA-adapted policy to the **frozen base BrickGPT** -- realized for free by PEFT's
``disable_adapter()`` context (policy == base when the adapter is off), so no separate reference copy is
held.

Rollouts generate the **full brick list in one pass** with plain ``generate(input_ids=...)`` -- no
per-brick rejection sampling and no logit masking -- so (a) syntax / overlap / stability give real
reward signal (validity is *shaped* by the reward, not enforced by the decoder) and (b) the
teacher-forced log-probs match the sampling policy exactly (keep ``temperature=1.0``).

Reward (:class:`~brickgpt.training.rewards.RewardConfig`): a syntax **gate** (-1 short-circuit) plus a
graded **overlap** penalty, a connectivity-based **stability** term, and an optional CLIP **semantic**
term. The mask-only **IoU** term is unavailable here (no target silhouettes) and is forced off. Note:
without IoU or semantic, *no reward term references the caption*, so the loop only sharpens
validity/stability; turn on ``--use_semantic`` (with a render+CLIP hook) for caption grounding.

Two advantage granularities (shared with the masked loop):

* **Single-turn (default):** one completion = the whole brick list -> one group-relative advantage
  broadcast over all of its tokens.
* **Multi-turn (``--use_multi_turn``):** split on newline tokens into per-brick turns; each brick gets a
  per-step syntax + overlap reward with **stability scored once on the final structure**, then a
  discounted return-to-go gives a per-step advantage on each brick's token span.

Run::

    uv run train_grpo_text --output_dir output/grpo_text
"""
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from brickgpt.training.grpo_masked import (_brick_token_spans, compute_group_advantages,
                                           compute_stepwise_advantages)
from brickgpt.training.logging_utils import RunLogger
from brickgpt.training.rewards import RewardConfig, compute_reward, stepwise_rewards

logger = logging.getLogger(__name__)


@dataclass
class GRPOTextArguments:
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    train_split: str = field(default='train')
    output_dir: str = field(default='output/grpo_text')

    group_size: int = field(default=8, metadata={'help': 'G: completions sampled per prompt.'})
    max_gen_tokens: int = field(default=400)
    temperature: float = field(default=1.0)
    max_steps: int = field(default=2000)
    lr: float = field(default=1e-5)
    beta_kl: float = field(default=0.04, metadata={'help': 'KL penalty anchoring the LoRA policy to the '
                                                           'frozen base BrickGPT (adapter disabled). 0=off.'})
    max_grad_norm: float = field(default=1.0)
    gradient_checkpointing: bool = field(default=True)

    # LoRA (Path A) hyper-parameters.
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default='q_proj,k_proj,v_proj,o_proj',
        metadata={'help': 'Comma-separated attention/MLP projection names to adapt with LoRA.'},
    )

    # Reward toggles + weights. IoU is mask-only and always off here; semantic needs a render+CLIP hook.
    use_overlap: bool = field(default=True)
    use_stability: bool = field(default=True)
    w_overlap: float = field(default=1.0, metadata={'help': 'Overlap penalty weight. Overlap is unbounded '
                                                            '(-Σ extra occupancy), so <1 keeps it from '
                                                            'swamping the ±1 stability / [0,1] semantic terms.'})
    w_stability: float = field(default=1.0)
    w_semantic: float = field(default=1.0)
    use_semantic: bool = field(default=False, metadata={'help': 'Caption-grounded CLIP term: render each '
                                                                'completion + score vs caption. Heavy '
                                                                '(seconds/completion); needs LDraw + render '
                                                                'setup (see DEVELOP.md). Single-turn only.'})
    render_samples: int = field(default=64, metadata={'help': 'CYCLES samples per semantic render (low is '
                                                              'fine for CLIP; cuts render time vs the 512 '
                                                              'inference default).'})
    render_resolution: int = field(default=256, metadata={'help': 'Render resolution for the semantic term.'})

    # Multi-turn: per-brick step rewards + per-step advantages (step-only: syntax/overlap/stability).
    use_multi_turn: bool = field(default=False)
    gamma: float = field(default=1.0, metadata={'help': 'Discount for per-brick return-to-go (multi-turn).'})

    report_to: str = field(default='auto')
    wandb_project: str = field(default='brickgpt-text-grpo')
    run_name: str | None = field(default=None)
    log_every: int = field(default=10)
    save_every: int = field(default=500)


class TextPromptSet:
    """Yields a random caption from the dataset (the only conditioning the original BrickGPT has)."""

    def __init__(self, data):
        self.captions = [c for r in data for c in r['captions']]

    def sample(self) -> str:
        return random.choice(self.captions)


def _sequence_logprobs(model, prompt_ids, gen_ids) -> torch.Tensor:
    """
    Per-token log-probs of ``gen_ids`` under ``model``, teacher-forced over ``[prompt, gen]``.

    Grad flows iff the model's params (the LoRA adapter) require it; call under
    ``model.disable_adapter()`` + ``no_grad`` to get the frozen-base reference log-probs.

    :return: ``(num_gen_tokens,)`` log-probs.
    """
    seq = torch.cat([prompt_ids, gen_ids], dim=1)
    logits = model(input_ids=seq, use_cache=False).logits                 # [1, P+G, V]
    start = prompt_ids.shape[1]
    gen_logits = logits[:, start - 1: start - 1 + gen_ids.shape[1], :]    # predicts each gen token
    logp = torch.log_softmax(gen_logits.float(), dim=-1)
    return logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


def load_policy(args, device):
    """
    Builds the GRPO policy for Path A.

    ``AvaLovelace/BrickGPT`` is itself a LoRA adapter on ``meta-llama/Llama-3.2-1B-Instruct`` (not a full
    model), so we load the base + BrickGPT adapter and **merge BrickGPT into the weights** -- making its
    fine-tuned text->bricks behavior the frozen backbone -- *before* adding a fresh trainable LoRA for the
    RL policy. (Calling ``get_peft_model`` directly on the auto-adapted model would instead stack a second
    adapter that deactivates BrickGPT, reverting generation to the raw base Llama.) The frozen merged
    backbone is then the KL reference: ``disable_adapter()`` recovers exactly the released BrickGPT.
    """
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    peft_cfg = PeftConfig.from_pretrained(args.model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(base, args.model_name_or_path).merge_and_unload()
    backbone = backbone.to(device)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(',') if m.strip()],
        task_type='CAUSAL_LM', bias='none',
    )
    model = get_peft_model(backbone, lora_cfg)
    if args.gradient_checkpointing:
        # Backprop reaches the LoRA params through the (frozen) base; checkpointing trades compute for
        # memory. enable_input_require_grads lets checkpointing track through the frozen embeddings.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def rollout(model, tokenizer, prompt_ids, args):
    """
    Samples G full completions from the LoRA policy via ``generate(input_ids=prompt_ids, ...)``.

    Generation runs the model in **eval** mode: gradient checkpointing (kept on for the log-prob
    backward) forces ``use_cache=False`` in *train* mode, which kills the KV cache; Llama has no dropout
    so eval-mode sampling and train-mode log-probs share the same distribution (exact at temperature 1).
    Each completion is generated separately so ``gen_ids`` are unpadded and line up with
    :func:`_sequence_logprobs`.

    :return: ``group_size`` tuples ``(gen_ids[1D, new tokens only], decoded_text)``.
    """
    attn = torch.ones_like(prompt_ids)
    was_training = model.training
    model.eval()
    completions = []
    try:
        for _ in range(args.group_size):
            out = model.generate(
                input_ids=prompt_ids, attention_mask=attn,
                do_sample=True, num_return_sequences=1, max_new_tokens=args.max_gen_tokens,
                temperature=args.temperature, top_k=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True)
            gen_ids = out.sequences[0, prompt_ids.shape[1]:]  # strip the prompt; keep new tokens only
            completions.append((gen_ids.detach(), tokenizer.decode(gen_ids, skip_special_tokens=True)))
    finally:
        model.train(was_training)
    return completions


def _instruction(caption):
    from brickgpt.models import create_instruction
    return create_instruction(caption)


def main():
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer, HfArgumentParser
    (args,) = HfArgumentParser(GRPOTextArguments).parse_args_into_dataclasses()
    if args.use_multi_turn:
        if args.use_semantic:
            logger.warning('Multi-turn GRPO is step-only; forcing use_semantic off (it needs a complete '
                           'trajectory render).')
        reward_cfg = RewardConfig(use_multi_turn=True, use_overlap=args.use_overlap,
                                  use_stability=args.use_stability, use_iou=False, use_semantic=False,
                                  w_overlap=args.w_overlap, w_stability=args.w_stability)
    else:
        reward_cfg = RewardConfig(use_overlap=args.use_overlap, use_stability=args.use_stability,
                                  use_iou=False, use_semantic=args.use_semantic,
                                  w_overlap=args.w_overlap, w_stability=args.w_stability,
                                  w_semantic=args.w_semantic)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = load_policy(args, device)
    model.train()

    # Semantic (CLIP) scorer: the only caption-grounded reward term. Heavy (render + CLIP per
    # completion), so built only when actually used and only for the single-turn path.
    scorer = None
    if reward_cfg.use_semantic:
        from brickgpt.training.semantic import SemanticScorer
        scorer = SemanticScorer(device, render_samples=args.render_samples,
                                img_resolution=args.render_resolution)
        logger.info('Semantic reward ON: rendering %d completions/step (samples=%d, res=%d). This is the '
                    'loop bottleneck.', args.group_size, args.render_samples, args.render_resolution)

    from datasets import load_dataset
    prompts = TextPromptSet(load_dataset(args.dataset_name, split=args.train_split))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    run = RunLogger(total=args.max_steps, phase='grpo', report_to=args.report_to,
                    console_every=args.log_every, project=args.wandb_project, name=args.run_name,
                    config=vars(args))

    for step in range(1, args.max_steps + 1):
        caption = prompts.sample()
        prompt_ids = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user', 'content': _instruction(caption)}],
            add_generation_prompt=True, return_tensors='pt').to(device)

        completions = rollout(model, tokenizer, prompt_ids, args)

        # Reward + advantage. Single-turn: one scalar reward / advantage per completion. Multi-turn:
        # per-brick step rewards split on newline tokens, with a per-step advantage per token span.
        if args.use_multi_turn:
            spans_per = [_brick_token_spans(gen_ids, tokenizer) for gen_ids, _ in completions]
            steps_per = [stepwise_rewards([t for _, _, t in spans], reward_cfg) for spans in spans_per]
            step_r = [[s.total for s in steps] for steps in steps_per]
            adv_per = compute_stepwise_advantages(step_r, args.gamma)
            returns = torch.tensor([float(sum(r)) for r in step_r], dtype=torch.float32, device=device)
        else:
            breakdowns = []
            for _, text in completions:
                clip_score = scorer.score(text, caption) if scorer is not None else None
                breakdowns.append(compute_reward(text, cfg=reward_cfg, clip_score=clip_score))
            returns = torch.tensor([b.total for b in breakdowns], dtype=torch.float32, device=device)
            scalar_adv = compute_group_advantages(returns)

        # GRPO loss = -A * logp + beta * KL(policy || base), token-mean. Policy = LoRA adapter on;
        # reference = same model with the adapter disabled (frozen base BrickGPT).
        opt.zero_grad(set_to_none=True)
        pg_terms, kl_terms = [], []
        for g, (gen_ids, _) in enumerate(completions):
            if gen_ids.numel() == 0:
                continue
            gen_ids_b = gen_ids.unsqueeze(0).to(device)
            logp = _sequence_logprobs(model, prompt_ids, gen_ids_b)
            with torch.no_grad(), model.disable_adapter():  # reference = frozen base (adapter off)
                ref_logp = _sequence_logprobs(model, prompt_ids, gen_ids_b)
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0  # k3 estimator, per token
            if args.use_multi_turn:  # per-token advantage: brick t's span carries A_t, else 0
                adv = torch.zeros(logp.shape[0], dtype=logp.dtype, device=device)
                for (s, e, _t), a in zip(spans_per[g], adv_per[g]):
                    adv[s:e] = a
            else:
                adv = scalar_adv[g]
            pg_terms.append(-(adv * logp).mean())
            kl_terms.append(kl.mean())

        if not pg_terms:
            run.log({'reward': returns.mean().item(), 'skipped': 1.0}, step=step)
            continue
        pg_loss = torch.stack(pg_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        loss = pg_loss + args.beta_kl * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        opt.step()

        metrics = {'reward': returns.mean().item(), 'reward_std': returns.std().item(),
                   'loss': loss.item(), 'pg_loss': pg_loss.item(), 'kl': kl_loss.item()}
        if args.use_multi_turn:
            all_steps = [s for steps in steps_per for s in steps]
            metrics['bricks_per_traj'] = float(np.mean([len(steps) for steps in steps_per]))
            metrics['step_reward'] = float(np.mean([s.total for s in all_steps])) if all_steps else 0.0
            metrics['syntax_ok'] = float(np.mean([s.syntax_ok for s in all_steps])) if all_steps else 0.0
            for key in ('overlap', 'stability'):
                vals = [getattr(s, key) for s in all_steps if getattr(s, key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        else:
            metrics['adv_abs'] = scalar_adv.abs().mean().item()
            metrics['syntax_ok'] = float(np.mean([b.syntax_ok for b in breakdowns]))
            for key in ('overlap', 'stability', 'semantic'):
                vals = [getattr(b, key) for b in breakdowns if getattr(b, key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        run.log(metrics, step=step)

        if args.save_every and step % args.save_every == 0:
            _save(model, args, f'step{step}')

    _save(model, args, 'final')
    run.close()


def _save(model, args, tag):
    """Saves the LoRA adapter (the only trained params) to ``<output_dir>/adapter_<tag>``."""
    out = f'{args.output_dir}/adapter_{tag}'
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    logger.info('Saved LoRA adapter (%s) to %s', tag, out)


if __name__ == '__main__':
    main()
