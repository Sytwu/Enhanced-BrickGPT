"""
Offline tests for the Phase-1 SFT diagnostics + training step (Step D).

No download: a chat-template stub tokenizer + a tiny stand-in LM exercise the CE-delta probe, and a
real-but-tiny ``LlamaForCausalLM`` built from config (à la ``test_prefill.py``) exercises the
freeze-LLM training step -- using a real attention stack so the mask prefix actually conditions the
assistant tokens (a non-attention stand-in would only couple the prefix to the first one).
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from brickgpt.masking import MaskConditioningConfig
from brickgpt.models.masked_brickgpt import BrickGPTWithMask
from brickgpt.training.generation import ce_delta_probe


class FakeTokenizer:
    """Chat-template stub: one token per content word, an assistant header on add_generation_prompt."""
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [100]                                   # BOS
        for m in messages:
            ids.append(200)                           # role header
            ids += [300] * max(1, len(m['content'].split()))
        if add_generation_prompt:
            ids.append(201)                           # assistant generation header
        return ids


class TinyCausalLM(nn.Module):
    """Minimal causal LM accepting inputs_embeds and returning an HF-style loss (offline stand-in)."""

    def __init__(self, vocab: int = 512, hidden: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


# Two single-caption rows; valid bricks so projection.stack_views builds real silhouettes.
DATA = [
    {'captions': ['red car'], 'bricks': '2x2 (0,0,0)\n2x2 (0,0,1)\n'},
    {'captions': ['a small house'], 'bricks': '2x6 (0,0,0)\n'},
]


def _probe_model():
    cfg = MaskConditioningConfig(pretrained_backbone=False)
    model = BrickGPTWithMask(TinyCausalLM(hidden=cfg.llm_hidden_size), cfg)
    model.freeze_llm()
    return model, cfg


def test_ce_delta_probe_returns_keys_and_arithmetic():
    model, cfg = _probe_model()
    out = ce_delta_probe(model, FakeTokenizer(), DATA, cfg)

    assert set(out) == {'ce_masked', 'ce_null', 'ce_delta', 'n'}
    assert out['n'] == 2                                          # two single-caption rows
    assert all(math.isfinite(v) for v in out.values())           # no NaN / inf
    assert abs(out['ce_delta'] - (out['ce_null'] - out['ce_masked'])) < 1e-5


def test_ce_delta_probe_restores_training_mode():
    model, cfg = _probe_model()

    model.train()
    ce_delta_probe(model, FakeTokenizer(), DATA, cfg)
    assert model.training is True

    model.eval()
    ce_delta_probe(model, FakeTokenizer(), DATA, cfg)
    assert model.training is False


def _tiny_llama(hidden: int = 64, vocab: int = 128, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )
    return LlamaForCausalLM(config)


def _sft_batch(cfg, seq: int = 12, prompt_len: int = 5, vocab: int = 128) -> dict:
    """One batch with a contiguous assistant-only label region (prompt masked to -100)."""
    torch.manual_seed(1)
    input_ids = torch.randint(2, vocab, (1, seq))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones(1, seq, dtype=torch.long),
        'labels': labels,
        'mask': torch.rand(1, cfg.num_views, cfg.world_dim, cfg.world_dim),
        'has_mask': torch.ones(1, cfg.num_views, dtype=torch.bool),
    }


def test_sft_step_updates_only_encoder_and_reduces_loss():
    """Freeze the LLM, train the mask prefix on one batch: encoder moves, LLM is untouched, loss drops."""
    cfg = MaskConditioningConfig(pretrained_backbone=False, llm_hidden_size=64)
    model = BrickGPTWithMask(_tiny_llama(hidden=64), cfg)
    model.freeze_llm()
    batch = _sft_batch(cfg)

    base_before = [p.clone() for p in model.base.parameters()]
    enc_before = [p.clone() for p in model.mask_prefix_encoder.parameters()]

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(not p.requires_grad for p in model.base.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-2)

    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]                                  # prefix-tuning overfits the batch
    assert all(torch.equal(p, b) for p, b in zip(model.base.parameters(), base_before))   # LLM frozen
    assert any(not torch.equal(p, b)                                                       # encoder moved
               for p, b in zip(model.mask_prefix_encoder.parameters(), enc_before))
