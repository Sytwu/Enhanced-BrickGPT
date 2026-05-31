"""
Test suite for mask dataset with precomputed prefix tokens.
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset, MaskDataCollator
from brickgpt.models.masked_brickgpt import BrickGPTWithMask


def _cfg(**kwargs):
    return MaskConditioningConfig(pretrained_backbone=False, **kwargs)


@pytest.fixture
def dummy_data():
    """Create a simple dummy dataset with 3 samples."""
    return [
        {
            'bricks': '1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat',
            'captions': ['A small brick structure', 'Another description'],
        },
        {
            'bricks': '1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat',
            'captions': ['Brick number 2'],
        },
        {
            'bricks': '1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat',
            'captions': ['Final brick'],
        },
    ]


@pytest.fixture
def tokenizer():
    """Load a tokenizer for testing."""
    from transformers import AutoTokenizer
    # Use Llama tokenizer which has built-in chat template
    try:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    except Exception:
        # Fallback: set chat template manually for GPT2
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_mask_dataset_with_precomputed_prefixes(dummy_data, tokenizer):
    """Test that MaskBrickDataset can load and return precomputed prefix tokens."""
    cfg = _cfg()
    
    # Create dummy precomputed prefix tokens: [num_rows, V*num_prefix_tokens, llm_hidden_size]
    num_rows = 3
    num_tokens = cfg.num_views * cfg.num_prefix_tokens
    hidden_size = cfg.llm_hidden_size
    
    prefixes = np.random.randn(num_rows, num_tokens, hidden_size).astype(np.float32)
    
    dataset = MaskBrickDataset(
        data=dummy_data,
        tokenizer=tokenizer,
        cfg=cfg,
        prefixes=prefixes,
        train=False,
    )
    
    # Test __getitem__ returns prefix_embeds when available
    example = dataset[0]
    assert 'prefix_embeds' in example
    assert example['prefix_embeds'].shape == (num_tokens, hidden_size)
    assert isinstance(example['prefix_embeds'], torch.Tensor)


def test_mask_data_collator_with_prefixes(dummy_data, tokenizer):
    """Test that MaskDataCollator properly batches precomputed prefix tokens."""
    cfg = _cfg()
    num_rows = 3
    num_tokens = cfg.num_views * cfg.num_prefix_tokens
    hidden_size = cfg.llm_hidden_size
    
    prefixes = np.random.randn(num_rows, num_tokens, hidden_size).astype(np.float32)
    
    dataset = MaskBrickDataset(
        data=dummy_data,
        tokenizer=tokenizer,
        cfg=cfg,
        prefixes=prefixes,
        train=False,
    )
    
    collator = MaskDataCollator(pad_token_id=tokenizer.pad_token_id or 0)
    
    # Collate a batch
    batch_indices = [0, 1]
    batch = collator([dataset[i] for i in batch_indices])
    
    # Check batch contains prefix_embeds with correct shape
    assert 'prefix_embeds' in batch
    assert batch['prefix_embeds'].shape[0] == 2  # batch size
    assert batch['prefix_embeds'].shape[1] == num_tokens
    assert batch['prefix_embeds'].shape[2] == hidden_size


def test_dataset_without_prefixes_still_works(dummy_data, tokenizer):
    """Test that dataset still works without precomputed prefixes (backward compatibility)."""
    cfg = _cfg()
    
    dataset = MaskBrickDataset(
        data=dummy_data,
        tokenizer=tokenizer,
        cfg=cfg,
        prefixes=None,  # No precomputed prefixes
        train=False,
    )
    
    # Should still work, just without prefix_embeds in output
    example = dataset[0]
    assert 'prefix_embeds' not in example
    assert 'mask' in example
    assert 'has_mask' in example


def test_model_with_precomputed_prefixes():
    """Test that BrickGPTWithMask can accept and use precomputed prefix embeddings."""
    from transformers import AutoModelForCausalLM
    
    cfg = _cfg()
    
    # Create a small model for testing
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            'gpt2', torch_dtype=torch.float32
        )
    except Exception:
        pytest.skip('Cannot download gpt2 model')
    
    model = BrickGPTWithMask(base_model, cfg)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    num_tokens = cfg.num_views * cfg.num_prefix_tokens
    hidden_size = cfg.llm_hidden_size
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Create precomputed prefix embeddings
    prefix_embeds = torch.randn(batch_size, num_tokens, hidden_size, dtype=torch.float32)
    
    # Forward pass using precomputed prefixes
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=prefix_embeds,
        )
    
    assert output.logits.shape == (batch_size, seq_len + num_tokens, 50257)


def test_model_backward_compat_with_masks():
    """Test that BrickGPTWithMask still works with masks (backward compatibility)."""
    from transformers import AutoModelForCausalLM
    
    cfg = _cfg()
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            'gpt2', torch_dtype=torch.float32
        )
    except Exception:
        pytest.skip('Cannot download gpt2 model')
    
    model = BrickGPTWithMask(base_model, cfg)
    model.eval()
    
    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Create dummy mask stack
    mask = torch.randn(batch_size, cfg.num_views, 20, 20, dtype=torch.float32)
    
    # Forward pass using masks (traditional way)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask=mask,
        )
    
    num_tokens = cfg.num_views * cfg.num_prefix_tokens
    assert output.logits.shape == (batch_size, seq_len + num_tokens, 50257)
