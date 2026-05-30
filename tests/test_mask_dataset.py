import random

import torch

from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset, MaskDataCollator


class FakeTokenizer:
    """
    Minimal stand-in for a chat tokenizer so the dataset can be tested offline.

    Mirrors the structure of a real chat template: ``add_generation_prompt=True`` appends an
    assistant-header token so that ``len(prompt_ids)`` lands exactly where the assistant *content*
    begins in the full (no-generation-prompt) sequence.
    """
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [100]  # BOS
        for m in messages:
            ids.append(200)                       # role header
            ids += [300] * len(m['content'].split())  # one token per word of content
        if add_generation_prompt:
            ids.append(201)                       # assistant generation header
        return ids


DATA = [
    {'captions': ['red car', 'a small house with windows'], 'bricks': '2x2 (0,0,0)\n2x2 (0,0,1)\n'},
    {'captions': ['a chair'], 'bricks': '2x6 (0,0,0)\n'},
]


def test_dataset_flattening_and_item_shapes():
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    assert len(ds) == 3  # 2 captions + 1 caption

    item = ds[0]
    assert set(item) == {'input_ids', 'attention_mask', 'labels', 'mask', 'has_mask'}
    assert item['input_ids'].dim() == 1
    assert item['input_ids'].dtype == torch.long
    assert item['labels'].shape == item['input_ids'].shape
    assert item['attention_mask'].shape == item['input_ids'].shape
    assert torch.all(item['attention_mask'] == 1)
    assert item['mask'].shape == (1, 20, 20)


def test_labels_are_assistant_only():
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    labels = ds[0]['labels']
    assert (labels == -100).any()              # prompt tokens are masked
    assert (labels != -100).any()              # assistant tokens are supervised
    # The masked region is a contiguous prefix.
    first_supervised = int((labels != -100).nonzero()[0])
    assert torch.all(labels[:first_supervised] == -100)


def test_condition_dropout_always_and_never():
    cfg_always = MaskConditioningConfig(condition_dropout_p=1.0)
    ds = MaskBrickDataset(DATA, FakeTokenizer(), cfg_always, train=True)
    item = ds[0]
    assert item['has_mask'] is False
    assert not item['mask'].any()              # null mask is all zeros

    cfg_never = MaskConditioningConfig(condition_dropout_p=0.0)
    ds2 = MaskBrickDataset(DATA, FakeTokenizer(), cfg_never, train=True)
    item2 = ds2[0]
    assert item2['has_mask'] is True
    assert item2['mask'].any()

    # Dropout is disabled outside training even with p=1.0.
    ds3 = MaskBrickDataset(DATA, FakeTokenizer(), cfg_always, train=False)
    assert ds3[0]['has_mask'] is True


def test_condition_dropout_rate_is_about_30_percent():
    random.seed(0)
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(condition_dropout_p=0.3), train=True)
    n = 3000
    dropped = sum(0 if ds[0]['has_mask'] else 1 for _ in range(n))
    assert 0.26 < dropped / n < 0.34


def test_collator_pads_and_stacks():
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    collator = MaskDataCollator(pad_token_id=FakeTokenizer.pad_token_id)
    batch = collator([ds[0], ds[1], ds[2]])

    b, max_len = 3, max(ds[i]['input_ids'].size(0) for i in range(3))
    assert batch['input_ids'].shape == (b, max_len)
    assert batch['attention_mask'].shape == (b, max_len)
    assert batch['labels'].shape == (b, max_len)
    assert batch['mask'].shape == (b, 1, 20, 20)
    assert batch['has_mask'].shape == (b,)
    assert batch['has_mask'].dtype == torch.bool
    # Padding positions: pad_token_id in input_ids, -100 in labels, 0 in attention_mask.
    shortest = min(range(3), key=lambda i: ds[i]['input_ids'].size(0))
    pad_len = max_len - ds[shortest]['input_ids'].size(0)
    if pad_len:
        assert torch.all(batch['attention_mask'][shortest, -pad_len:] == 0)
        assert torch.all(batch['labels'][shortest, -pad_len:] == -100)
        assert torch.all(batch['input_ids'][shortest, -pad_len:] == FakeTokenizer.pad_token_id)
