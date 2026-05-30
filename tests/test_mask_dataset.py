import numpy as np
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

V = MaskConditioningConfig().num_views  # 3


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
    assert item['mask'].shape == (V, 20, 20)
    assert item['has_mask'].shape == (V,)
    assert item['has_mask'].dtype == torch.bool


def test_three_views_differ():
    # A single flat 2x6 brick (ds[2]) projects differently along each axis:
    #   top=(x,y) 2x6,  front=(x,z) 2x1,  side=(y,z) 6x1.  (A symmetric cube would NOT differ.)
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    mask = ds[2]['mask']
    assert not torch.equal(mask[0], mask[1])
    assert not torch.equal(mask[0], mask[2])


def test_labels_are_assistant_only():
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    labels = ds[0]['labels']
    assert (labels == -100).any()              # prompt tokens are masked
    assert (labels != -100).any()              # assistant tokens are supervised
    # The masked region is a contiguous prefix.
    first_supervised = int((labels != -100).nonzero()[0])
    assert torch.all(labels[:first_supervised] == -100)


def test_per_view_dropout_keep_all_and_none():
    # view_keep_probs over the number of kept views {0,1,2,3}.
    keep_all = MaskConditioningConfig(view_keep_probs=(0.0, 0.0, 0.0, 1.0))
    item = MaskBrickDataset(DATA, FakeTokenizer(), keep_all, train=True)[0]
    assert bool(item['has_mask'].all())
    assert item['mask'].any()                  # at least one provided view is non-empty

    keep_none = MaskConditioningConfig(view_keep_probs=(1.0, 0.0, 0.0, 0.0))
    item = MaskBrickDataset(DATA, FakeTokenizer(), keep_none, train=True)[0]
    assert not item['has_mask'].any()
    assert not item['mask'].any()              # all views dropped -> all-zeros

    # Dropout is disabled outside training: all views provided regardless of probs.
    item = MaskBrickDataset(DATA, FakeTokenizer(), keep_none, train=False)[0]
    assert bool(item['has_mask'].all())


def test_per_view_dropout_keeps_exactly_one():
    cfg = MaskConditioningConfig(view_keep_probs=(0.0, 1.0, 0.0, 0.0))
    ds = MaskBrickDataset(DATA, FakeTokenizer(), cfg, train=True)
    for _ in range(20):
        item = ds[0]
        assert int(item['has_mask'].sum()) == 1
        # Dropped views are zeroed; the kept (top, non-empty) view is not.
        assert item['mask'][~item['has_mask']].sum() == 0


def test_per_view_dropout_distribution():
    np.random.seed(0)
    cfg = MaskConditioningConfig(view_keep_probs=(0.10, 0.60, 0.15, 0.15))
    ds = MaskBrickDataset(DATA, FakeTokenizer(), cfg, train=True)
    n = 4000
    counts = np.zeros(V + 1)
    for _ in range(n):
        counts[int(ds[0]['has_mask'].sum())] += 1
    freq = counts / n
    assert np.allclose(freq, cfg.view_keep_probs, atol=0.03)


def test_collator_pads_and_stacks():
    ds = MaskBrickDataset(DATA, FakeTokenizer(), MaskConditioningConfig(), train=False)
    collator = MaskDataCollator(pad_token_id=FakeTokenizer.pad_token_id)
    batch = collator([ds[0], ds[1], ds[2]])

    b, max_len = 3, max(ds[i]['input_ids'].size(0) for i in range(3))
    assert batch['input_ids'].shape == (b, max_len)
    assert batch['attention_mask'].shape == (b, max_len)
    assert batch['labels'].shape == (b, max_len)
    assert batch['mask'].shape == (b, V, 20, 20)
    assert batch['has_mask'].shape == (b, V)
    assert batch['has_mask'].dtype == torch.bool
    # Padding positions: pad_token_id in input_ids, -100 in labels, 0 in attention_mask.
    shortest = min(range(3), key=lambda i: ds[i]['input_ids'].size(0))
    pad_len = max_len - ds[shortest]['input_ids'].size(0)
    if pad_len:
        assert torch.all(batch['attention_mask'][shortest, -pad_len:] == 0)
        assert torch.all(batch['labels'][shortest, -pad_len:] == -100)
        assert torch.all(batch['input_ids'][shortest, -pad_len:] == FakeTokenizer.pad_token_id)
