"""
Phase 5 — GRPO reinforcement-learning fine-tuning for mask-conditioned BrickGPT.

STUB. Loads the SFT checkpoint, **freezes the mask encoder**, and applies **LoRA to the LLM**
(matching ``q_proj,v_proj`` in [scripts/finetune.zsh]). For each prompt it samples ``G`` complete
brick sequences (single-turn), scores them with :mod:`brickgpt.training.rewards` using dynamic
reward routing, computes group-normalized GRPO advantages, and updates the LoRA parameters with a KL
penalty to a frozen reference policy.

A custom loop (rather than ``trl.GRPOTrainer``) is used because GRPOTrainer assumes text prompts and
a fixed reward-fn signature, which fights both the prefix-embed injection and the NULL-mask reward
routing.

TODO(handoff):
  - Batched prefix-embed generation (pre-fill the KV-cache from the mask prefix; reuse the
    token-by-token logit-masking loop in brickgpt.models.brickgpt).
  - KL/reference-policy handling and the GRPO objective.
  - Reward weights (w1, w2, w3) config and logging.
"""
import logging
from dataclasses import dataclass, field

import torch

from brickgpt.masking import MaskConditioningConfig
from brickgpt.training.rewards import total_reward

logger = logging.getLogger(__name__)


@dataclass
class GRPOMaskedArguments:
    sft_checkpoint: str = field(default='output/sft_masked')
    group_size: int = field(default=8, metadata={'help': 'G: completions sampled per prompt.'})
    reward_weights: tuple[float, float, float] = field(default=(1.0, 1.0, 1.0))  # (IoU, stability, syntax)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_target_modules: tuple[str, ...] = field(default=('q_proj', 'v_proj'))
    output_dir: str = field(default='output/grpo_masked')


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO advantage: standardize rewards within each sampled group ``(r - mean) / (std + eps)``."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def score_completions(completions, target_mask, has_mask, args: GRPOMaskedArguments,
                      cfg: MaskConditioningConfig) -> torch.Tensor:
    """Scores a group of completions with dynamic reward routing (drops IoU for NULL_MASK)."""
    return torch.tensor([
        total_reward(c, target_mask, has_mask, weights=args.reward_weights, cfg=cfg)
        for c in completions
    ], dtype=torch.float32)


def main():
    # TODO(handoff): implement the full GRPO loop:
    #   1. Load SFT BrickGPTWithMask; freeze mask encoder; attach LoRA to the LLM via peft.
    #   2. For each (caption, mask): sample G completions via the prefix-embed generation path.
    #   3. rewards = score_completions(...); advantages = compute_group_advantages(rewards).
    #   4. Policy-gradient update on LoRA params with KL penalty to a frozen reference policy.
    raise NotImplementedError('GRPO training loop is a Phase-5 stub; see module docstring TODOs.')


if __name__ == '__main__':
    main()
