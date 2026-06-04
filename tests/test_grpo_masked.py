"""
Offline tests for the Phase-2 IoU-GRPO loop (Step E), **Path B / encoder-trainable** variant.

No download: a real-but-tiny ``LlamaForCausalLM`` built from config (à la ``test_prefill.py``) exercises
the actual GRPO gradient path -- a frozen LLM with a trainable mask prefix, the policy/reference
log-prob plumbing (:func:`_sequence_logprobs`), and the group-advantage standardisation. These pin down
the Path-B invariants: the **prefix carries the gradient** (encoder moves, LLM stays frozen) and the KL
reference (a frozen copy of the encoder) is a **no-op at init** (policy == reference).
"""
import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from brickgpt.masking import MaskConditioningConfig
from brickgpt.models.masked_brickgpt import BrickGPTWithMask
from brickgpt.training.grpo_masked import (GRPOMaskedArguments, _sequence_logprobs,
                                           compute_group_advantages, rollout)


def _tiny_llama(hidden: int = 64, vocab: int = 128, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )
    return LlamaForCausalLM(config)


def _path_b_model(hidden: int = 64):
    """Path-B setup mirroring ``load_policy``: train the encoder, freeze the LLM, snapshot a ref encoder."""
    cfg = MaskConditioningConfig(pretrained_backbone=False, llm_hidden_size=hidden)
    model = BrickGPTWithMask(_tiny_llama(hidden=hidden), cfg)
    model.base.requires_grad_(False)
    model.mask_prefix_encoder.requires_grad_(True)
    ref_encoder = copy.deepcopy(model.mask_prefix_encoder).requires_grad_(False)
    ref_encoder.eval()
    return model, ref_encoder, cfg


def _inputs(cfg, prompt_len: int = 5, gen_len: int = 7, vocab: int = 128):
    torch.manual_seed(1)
    mask = torch.rand(1, cfg.num_views, cfg.world_dim, cfg.world_dim)
    has_mask = torch.ones(1, cfg.num_views, dtype=torch.bool)
    prompt_ids = torch.randint(2, vocab, (1, prompt_len))
    gen_ids = torch.randint(2, vocab, (1, gen_len))
    return mask, has_mask, prompt_ids, gen_ids


def test_path_b_only_encoder_is_trainable():
    """Freeze the LLM, train the prefix: the optimizer set is exactly the mask encoder."""
    model, _, _ = _path_b_model()
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    enc = {id(p) for p in model.mask_prefix_encoder.parameters()}
    assert trainable == enc                                         # nothing but the encoder trains
    assert all(not p.requires_grad for p in model.base.parameters())


def test_sequence_logprobs_shape_and_grad_reaches_encoder():
    """A policy-gradient step flows grad through the frozen LLM into the encoder, never into the LLM."""
    model, _, cfg = _path_b_model()
    mask, has_mask, prompt_ids, gen_ids = _inputs(cfg)

    # Prefix computed WITH grad (Path B) and reused as the policy log-prob input.
    prefix = model.mask_prefix_encoder(mask, has_mask)
    assert prefix.requires_grad
    logp = _sequence_logprobs(model.base, prefix, prompt_ids, gen_ids)
    assert logp.shape == (gen_ids.shape[1],)                        # one log-prob per generated token

    adv = torch.tensor(0.7)                                         # a nonzero group advantage
    loss = -(adv * logp).mean()
    loss.backward()

    enc_grads = [p.grad for p in model.mask_prefix_encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)  # prefix carries the gradient
    assert all(p.grad is None for p in model.base.parameters())     # LLM untouched (frozen)


def test_kl_reference_is_noop_at_init():
    """At init the policy encoder == the frozen ref encoder, so the k3 KL term is ~0.

    Both are evaluated in ``eval`` mode here to isolate the weight-identity property -- in the real loop
    the policy encoder runs in ``train`` mode (BatchNorm/dropout), so the seed KL is small but nonzero.
    """
    model, ref_encoder, cfg = _path_b_model()
    model.mask_prefix_encoder.eval()
    mask, has_mask, prompt_ids, gen_ids = _inputs(cfg)

    with torch.no_grad():
        prefix = model.mask_prefix_encoder(mask, has_mask)
        ref_prefix = ref_encoder(mask, has_mask)
        logp = _sequence_logprobs(model.base, prefix, prompt_ids, gen_ids)
        ref_logp = _sequence_logprobs(model.base, ref_prefix, prompt_ids, gen_ids)
    kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0       # k3 estimator, per token
    assert torch.allclose(prefix, ref_prefix, atol=1e-5)            # identical encoders
    assert kl.abs().max() < 1e-4                                    # ⇒ no KL pressure at the SFT seed


class _StubTokenizer:
    """Minimal tokenizer for the rollout plumbing test (generate needs pad/eos ids + decode)."""
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(str(int(i)) for i in ids)


def test_rollout_returns_unpadded_completions_and_restores_train_mode():
    """Rollout generates G completions via ``generate(inputs_embeds=...)`` in eval mode, then restores
    the caller's train mode (the bug that broke conditioning: train-mode grad-checkpointing forced
    ``use_cache=False``)."""
    model, _, cfg = _path_b_model()
    mask, has_mask, prompt_ids, _ = _inputs(cfg)
    prefix = model.mask_prefix_encoder(mask, has_mask)
    args = GRPOMaskedArguments(group_size=3, max_gen_tokens=6, temperature=1.0)

    model.train()                                                  # caller's mode before the rollout
    comps = rollout(model, _StubTokenizer(), prefix, prompt_ids, args, 'cpu')

    assert model.base.training is True                             # eval flipped back to train after
    assert len(comps) == args.group_size
    for gen_ids, text in comps:
        assert gen_ids.dim() == 1 and gen_ids.numel() >= 1         # new tokens only, unpadded, non-empty
        assert gen_ids.numel() <= args.max_gen_tokens
        assert isinstance(text, str)


def test_compute_group_advantages_standardizes():
    """GRPO advantage standardizes within the group: zero mean, unit-ish std, monotone in reward."""
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    adv = compute_group_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-3                       # standardized by the (unbiased) std
    assert torch.all(adv[1:] > adv[:-1])                           # preserves reward ordering
