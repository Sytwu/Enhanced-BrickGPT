from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from brickgpt.masking import MaskConditioningConfig
from brickgpt.models.masked_brickgpt import BrickGPTWithMask


class TinyCausalLM(nn.Module):
    """A minimal causal LM that accepts ``inputs_embeds`` and returns a HF-style loss, for offline tests."""

    def __init__(self, vocab: int = 64, hidden: int = 2048):
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


def _batch(bsz=2, seq=5, vocab=64):
    return {
        'input_ids': torch.randint(0, vocab, (bsz, seq)),
        'attention_mask': torch.ones(bsz, seq, dtype=torch.long),
        'labels': torch.randint(0, vocab, (bsz, seq)),
        'mask': torch.rand(bsz, 1, 20, 20),
    }


def test_forward_prepends_prefix_and_returns_loss():
    cfg = MaskConditioningConfig(pretrained_backbone=False)
    model = BrickGPTWithMask(TinyCausalLM(hidden=cfg.llm_hidden_size), cfg)
    batch = _batch(bsz=2, seq=5)
    out = model(**batch)

    assert out.loss is not None and out.loss.dim() == 0
    # Sequence is extended by num_prefix_tokens mask tokens.
    assert out.logits.shape[1] == 5 + cfg.num_prefix_tokens


def test_freeze_llm_trains_only_mask_encoder():
    cfg = MaskConditioningConfig(pretrained_backbone=False)
    model = BrickGPTWithMask(TinyCausalLM(hidden=cfg.llm_hidden_size), cfg)
    model.freeze_llm()

    model(**_batch()).loss.backward()

    assert all(p.grad is None for p in model.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mask_prefix_encoder.parameters())


def test_forward_requires_mask():
    cfg = MaskConditioningConfig(pretrained_backbone=False)
    model = BrickGPTWithMask(TinyCausalLM(hidden=cfg.llm_hidden_size), cfg)
    batch = _batch()
    del batch['mask']
    try:
        model(**batch)
        assert False, 'expected ValueError when mask is missing'
    except ValueError:
        pass
