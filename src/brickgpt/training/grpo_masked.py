"""
Step E -- Phase 2 GRPO post-training for mask-conditioned BrickGPT (self-written loop).

Stage 2 (D7): load the SFT mask encoder, **freeze it**, attach **LoRA to the LLM**, and optimize the
reward (:class:`~brickgpt.training.rewards.RewardConfig`) with group-relative advantages and a KL
penalty to the frozen reference policy (the same model with the LoRA adapter disabled -- no separate
copy needed).

Rollouts (both modes) generate the **full completion in one pass** from the mask-prefilled cache (no
per-brick rejection / no logit masking), so (a) syntax/overlap/stability give real reward signal and
(b) teacher-forced log-probs match the sampling policy exactly. Validity is shaped by the reward
(syntax gate), not enforced by the decoder.

Two advantage granularities:

* **Single-turn (D5, default):** one completion = the whole brick list -> one scalar reward ->
  one group-relative advantage broadcast over all of its tokens.
* **Multi-turn (``--use_multi_turn``):** the completion is split on newline tokens into per-brick
  turns; each brick gets a per-step syntax + overlap reward, with **stability scored once on the final
  structure** and attached to the last brick (a causal per-step check can't see a support that arrives
  in a later turn). A discounted return-to-go then gives each brick a **per-step** advantage on just
  its token span -- finer credit assignment for which brick collided / broke the build. IoU/semantic
  are unavailable here (they need a complete trajectory) and are forced off.

A custom loop (not ``trl.GRPOTrainer``) is required: GRPOTrainer assumes text prompts and a fixed
reward signature, which fights both the prefix-embed injection and the per-view NULL-mask routing.

Run::

    uv run train_grpo_masked --sft_checkpoint output/sft_masked --output_dir output/grpo_masked
"""
import json
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from brickgpt.masking import MaskConditioningConfig, stack_views
from brickgpt.models.masked_brickgpt import BrickGPTWithMask
from brickgpt.training.logging_utils import RunLogger
from brickgpt.training.rewards import RewardConfig, compute_reward, stepwise_rewards

logger = logging.getLogger(__name__)


@dataclass
class GRPOMaskedArguments:
    sft_checkpoint: str | None = field(
        default=None,
        metadata={'help': 'Dir with mask_encoder_final.pt + sft_meta.json from SFT. If None, starts '
                          'from a fresh encoder (smoke only).'},
    )
    model_name_or_path: str = field(default='AvaLovelace/BrickGPT')
    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    train_split: str = field(default='train')
    output_dir: str = field(default='output/grpo_masked')

    group_size: int = field(default=8, metadata={'help': 'G: completions sampled per prompt.'})
    max_gen_tokens: int = field(default=400)
    temperature: float = field(default=1.0)
    max_steps: int = field(default=2000)
    lr: float = field(default=1e-5)
    beta_kl: float = field(default=0.04, metadata={'help': 'KL penalty coefficient to the frozen reference.'})
    max_grad_norm: float = field(default=1.0)
    gradient_checkpointing: bool = field(default=True)

    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: tuple[str, ...] = field(default=('q_proj', 'v_proj'))

    # Reward toggles (D4). Semantic stays off by default (needs rendering).
    use_overlap: bool = field(default=True)
    use_stability: bool = field(default=True)
    use_iou: bool = field(default=True)
    use_semantic: bool = field(default=False)

    # Multi-turn (D5): per-brick step rewards + per-step (per-turn) advantages. Step-only by contract
    # -- IoU/semantic need a full trajectory and are forced off when this is set.
    use_multi_turn: bool = field(default=False)
    gamma: float = field(default=1.0, metadata={'help': 'Discount for per-brick return-to-go (multi-turn).'})

    report_to: str = field(default='auto')
    wandb_project: str = field(default='brickgpt-mask-grpo')
    run_name: str | None = field(default=None)
    log_every: int = field(default=10)
    save_every: int = field(default=500)


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO advantage: standardize rewards within the sampled group ``(r - mean) / (std + eps)``."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def compute_stepwise_advantages(step_rewards: list[list[float]], gamma: float) -> list[list[float]]:
    """
    Per-brick (per-turn) GRPO advantages for the multi-turn path.

    For each trajectory, compute the discounted **return-to-go** ``G_t = r_t + gamma * G_{t+1}``; then
    standardize **per step index** across the group -- at index ``t`` only over the trajectories that
    actually have a ``t``-th brick. This per-index baseline removes the positional bias whereby earlier
    bricks have systematically larger returns-to-go. A step reached by ``< 2`` trajectories gets
    advantage ``0`` (no group signal there).

    :param step_rewards: One list of per-brick step rewards per trajectory (ragged across trajectories).
    :return: Same ragged shape; ``advantages[g][t]`` is the advantage for brick ``t`` of trajectory ``g``.
    """
    returns: list[list[float]] = []
    for rs in step_rewards:
        g = [0.0] * len(rs)
        acc = 0.0
        for t in range(len(rs) - 1, -1, -1):
            acc = rs[t] + gamma * acc
            g[t] = acc
        returns.append(g)

    advantages = [[0.0] * len(g) for g in returns]
    max_t = max((len(g) for g in returns), default=0)
    for t in range(max_t):
        idx = [i for i in range(len(returns)) if len(returns[i]) > t]
        if len(idx) < 2:
            continue
        vals = np.array([returns[i][t] for i in idx], dtype=np.float64)
        mean, std = vals.mean(), vals.std()
        for i, v in zip(idx, vals):
            advantages[i][t] = float((v - mean) / (std + 1e-8))
    return advantages


def _brick_token_spans(gen_ids: torch.Tensor, tokenizer) -> list[tuple[int, int, str]]:
    """
    Split a 1-D ``gen_ids`` into per-brick ``(start, end, line_text)`` spans on newline tokens.

    A turn boundary is a generated token whose decoded text contains ``\\n`` (the brick format ends each
    line with a single ``)\\n`` token). Generation stops at ``eos_token_id``; a trailing non-empty run
    with no closing newline is emitted as a final span too (typically gated to ``-1`` by the syntax
    check). Empty segments are dropped, so the spans align 1:1 with the lines fed to
    :func:`~brickgpt.training.rewards.stepwise_rewards`. Indices are into ``gen_ids`` so they line up
    with the per-token log-probs from :func:`_sequence_logprobs`.
    """
    eos = tokenizer.eos_token_id
    ids = gen_ids.tolist()
    spans: list[tuple[int, int, str]] = []
    start, buf = 0, []
    for i, tid in enumerate(ids):
        if tid == eos:
            break
        buf.append(tid)
        if '\n' in tokenizer.decode([tid]):
            text = tokenizer.decode(buf, skip_special_tokens=True).strip()
            if text:
                spans.append((start, i + 1, text))
            start, buf = i + 1, []
    if buf:  # trailing tokens with no closing newline (or stopped by EOS / max tokens)
        text = tokenizer.decode(buf, skip_special_tokens=True).strip()
        if text:
            spans.append((start, start + len(buf), text))
    return spans


def _sequence_logprobs(base, prefix_embeds, prompt_ids, gen_ids) -> torch.Tensor:
    """
    Per-token log-probs of ``gen_ids`` under ``base``, teacher-forced over [prefix, prompt, gen].

    :return: ``(num_gen_tokens,)`` log-probs. Grad flows iff ``base`` params (LoRA) require it.
    """
    embed = base.get_input_embeddings()
    seq = torch.cat([prefix_embeds, embed(prompt_ids), embed(gen_ids)], dim=1)
    # We pass inputs_embeds directly (bypassing the embedding hook), and the embeddings are frozen, so
    # seq carries no grad. Mark it grad-requiring so gradient checkpointing tracks through to LoRA.
    if torch.is_grad_enabled():
        seq.requires_grad_(True)
    logits = base(inputs_embeds=seq, use_cache=False).logits          # [1, T+P+G, V]
    start = prefix_embeds.shape[1] + prompt_ids.shape[1]
    gen_logits = logits[:, start - 1: start - 1 + gen_ids.shape[1], :]  # predicts each gen token
    logp = torch.log_softmax(gen_logits.float(), dim=-1)
    return logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


class GRPOPromptSet:
    """Yields ``(caption, mask[V,H,W], has_mask[V], target_views[V,H,W])`` with per-view dropout (D2)."""

    def __init__(self, data, cfg: MaskConditioningConfig):
        self.cfg = cfg
        self.rows = [(c, r['bricks']) for r in data for c in r['captions']]

    def sample(self):
        caption, bricks_txt = random.choice(self.rows)
        target = stack_views(bricks_txt, self.cfg)                       # [V, H, W] (full, for reward)
        v = self.cfg.num_views
        k = int(np.random.choice(v + 1, p=np.asarray(self.cfg.view_keep_probs)))
        presence = np.zeros(v, dtype=bool)
        presence[random.sample(range(v), k)] = True
        mask = target * presence[:, None, None]                          # dropped views zeroed
        return caption, mask, presence, target


def load_policy(args, cfg, device):
    """Builds BrickGPTWithMask, loads the SFT encoder, freezes encoder + base, and adds LoRA to the LLM."""
    from peft import LoraConfig, get_peft_model

    name = args.model_name_or_path
    if args.sft_checkpoint and os.path.exists(f'{args.sft_checkpoint}/sft_meta.json'):
        with open(f'{args.sft_checkpoint}/sft_meta.json') as f:
            name = json.load(f).get('model_name_or_path', name)

    model = BrickGPTWithMask.from_pretrained(name, cfg, torch_dtype=torch.bfloat16).to(device)
    if args.sft_checkpoint and os.path.exists(f'{args.sft_checkpoint}/mask_encoder_final.pt'):
        sd = torch.load(f'{args.sft_checkpoint}/mask_encoder_final.pt', map_location=device)
        model.mask_prefix_encoder.load_state_dict(sd)
        logger.info('Loaded SFT mask encoder from %s', args.sft_checkpoint)
    else:
        logger.warning('No SFT encoder found; starting GRPO from a fresh encoder (smoke only).')

    model.mask_prefix_encoder.requires_grad_(False)  # encoder frozen in Stage 2
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=list(args.lora_target_modules), task_type='CAUSAL_LM')
    model.base = get_peft_model(model.base, lora)    # only LoRA params on the base are trainable
    if args.gradient_checkpointing:
        model.base.gradient_checkpointing_enable()
        model.base.enable_input_require_grads()
    return model, name


@torch.no_grad()
def rollout(model, tokenizer, prefix_embeds, prompt_ids, args, device):
    """Samples G full completions from the mask-prefilled cache. Returns list of gen_id tensors + texts."""
    from brickgpt.models.llm import LLM
    llm = LLM.from_model(model.base, tokenizer, str(device))
    completions = []
    llm.prefill_with_embeds(prefix_embeds, prompt_ids)
    llm.save_state()
    for _ in range(args.group_size):
        gen_ids = llm(None, return_as_ids=True, max_new_tokens=args.max_gen_tokens,
                      temperature=args.temperature, top_k=None, top_p=None)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append((gen_ids.detach(), text))
        llm.rollback_to_saved_state()
    return completions


def main():
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer, HfArgumentParser
    (args,) = HfArgumentParser(GRPOMaskedArguments).parse_args_into_dataclasses()
    cfg = MaskConditioningConfig()
    if args.use_multi_turn:
        if args.use_iou or args.use_semantic:
            logger.warning('Multi-turn GRPO is step-only; forcing use_iou/use_semantic off '
                           '(they need a complete trajectory).')
        reward_cfg = RewardConfig(use_multi_turn=True, use_overlap=args.use_overlap,
                                  use_stability=args.use_stability, use_iou=False, use_semantic=False)
    else:
        reward_cfg = RewardConfig(use_overlap=args.use_overlap, use_stability=args.use_stability,
                                  use_iou=args.use_iou, use_semantic=args.use_semantic)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model, _ = load_policy(args, cfg, device)
    model.train()

    from datasets import load_dataset
    prompts = GRPOPromptSet(load_dataset(args.dataset_name, split=args.train_split), cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    run = RunLogger(total=args.max_steps, phase='grpo', report_to=args.report_to,
                    console_every=args.log_every, project=args.wandb_project, name=args.run_name,
                    config=vars(args))
    embed_base = model.base  # PeftModel; get_input_embeddings works through it

    for step in range(1, args.max_steps + 1):
        caption, mask, presence, target = prompts.sample()
        prompt_ids = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user', 'content': _instruction(caption)}],
            add_generation_prompt=True, return_tensors='pt').to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
        has_mask_t = torch.from_numpy(presence).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix_embeds = model.mask_prefix_encoder(mask_t, has_mask_t).to(next(model.base.parameters()).dtype)

        completions = rollout(model, tokenizer, prefix_embeds, prompt_ids, args, device)

        # Reward + advantage. Single-turn: one scalar reward / advantage per completion. Multi-turn:
        # per-brick step rewards split on newline tokens, with a per-step advantage per token span.
        if args.use_multi_turn:
            spans_per = [_brick_token_spans(gen_ids, tokenizer) for gen_ids, _ in completions]
            steps_per = [stepwise_rewards([t for _, _, t in spans], reward_cfg, cfg) for spans in spans_per]
            step_r = [[s.total for s in steps] for steps in steps_per]
            adv_per = compute_stepwise_advantages(step_r, args.gamma)
            returns = torch.tensor([float(sum(r)) for r in step_r], dtype=torch.float32, device=device)
        else:
            breakdowns = [compute_reward(text, target, presence, cfg=reward_cfg, mask_cfg=cfg)
                          for _, text in completions]
            returns = torch.tensor([b.total for b in breakdowns], dtype=torch.float32, device=device)
            scalar_adv = compute_group_advantages(returns)

        # Policy + reference log-probs; GRPO loss = -A * logp + beta * KL(policy || ref), token-mean.
        opt.zero_grad(set_to_none=True)
        pg_terms, kl_terms = [], []
        for g, (gen_ids, _) in enumerate(completions):
            if gen_ids.numel() == 0:
                continue
            gen_ids_b = gen_ids.unsqueeze(0).to(device)
            logp = _sequence_logprobs(embed_base, prefix_embeds, prompt_ids, gen_ids_b)
            with torch.no_grad():
                with model.base.disable_adapter():
                    ref_logp = _sequence_logprobs(embed_base, prefix_embeds, prompt_ids, gen_ids_b)
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
            for key in ('overlap', 'stability', 'iou', 'semantic'):
                vals = [getattr(b, key) for b in breakdowns if getattr(b, key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        run.log(metrics, step=step)

        if args.save_every and step % args.save_every == 0:
            _save(model, args, f'step{step}')

    _save(model, args, 'final')
    run.close()


def _instruction(caption):
    from brickgpt.models import create_instruction
    return create_instruction(caption)


def _save(model, args, tag):
    os.makedirs(args.output_dir, exist_ok=True)
    model.base.save_pretrained(f'{args.output_dir}/lora_{tag}')  # LoRA adapter only
    logger.info('Saved LoRA adapter (%s) to %s', tag, args.output_dir)


if __name__ == '__main__':
    main()
