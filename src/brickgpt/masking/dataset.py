import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from brickgpt.models import create_instruction

from .config import MaskConditioningConfig
from .projection import bricks_to_mask, null_mask


class MaskBrickDataset(Dataset):
    """
    PyTorch dataset that pairs each ``(caption, bricks)`` example with a 2D top-down silhouette mask.

    Each underlying row is expected to expose a ``captions`` list and a ``bricks`` string (the schema
    of ``AvaLovelace/StableText2Brick``). Rows are flattened to one example per caption, mirroring
    :func:`brickgpt.prepare_finetuning_dataset.convert_sample`.

    ``__getitem__`` returns a single tokenized example together with its mask. During training the
    mask is replaced with the all-zeros ``[NULL_MASK]`` with probability ``cfg.condition_dropout_p``
    (condition dropout), and a ``has_mask`` flag records whether a real mask was kept (used later for
    RL reward routing).
    """

    def __init__(
            self,
            data: Sequence[dict],
            tokenizer: Any,
            cfg: MaskConditioningConfig = MaskConditioningConfig(),
            masks: np.ndarray | None = None,
            train: bool = True,
            system_prompt: str = 'You are a helpful assistant.',
    ):
        """
        :param data: A sequence of rows, each with ``captions`` (list[str]) and ``bricks`` (str).
        :param tokenizer: A HuggingFace tokenizer exposing ``apply_chat_template``.
        :param cfg: Mask conditioning configuration.
        :param masks: Optional precomputed masks of shape ``(num_rows, world_dim, world_dim)``,
                      indexed by row. If ``None``, masks are computed on the fly from ``bricks``.
        :param train: If ``True``, applies condition dropout in ``__getitem__``.
        :param system_prompt: System message used to build the conversation.
        """
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.masks = masks
        self.train = train
        self.system_prompt = system_prompt

        # Flatten rows to one example per caption, keeping a pointer to the row's bricks/mask.
        self.bricks: list[str] = []
        self.index: list[tuple[int, str]] = []
        for row_idx, row in enumerate(data):
            self.bricks.append(row['bricks'])
            for caption in row['captions']:
                self.index.append((row_idx, caption))

    def __len__(self) -> int:
        return len(self.index)

    def _get_mask(self, row_idx: int) -> np.ndarray:
        if self.masks is not None:
            return np.asarray(self.masks[row_idx], dtype=np.float32)
        return bricks_to_mask(self.bricks[row_idx], self.cfg)

    def __getitem__(self, i: int) -> dict:
        row_idx, caption = self.index[i]
        bricks_txt = self.bricks[row_idx]

        # Resolve the mask, applying condition dropout during training.
        mask = self._get_mask(row_idx)
        has_mask = True
        if self.train and random.random() < self.cfg.condition_dropout_p:
            mask = null_mask(self.cfg)
            has_mask = False

        # Build the conversation. The prompt is identical to the text-only model so the LLM's
        # learned behavior transfers; the mask is injected later as prefix embeddings, not text.
        prompt_messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': create_instruction(caption)},
        ]
        full_messages = prompt_messages + [{'role': 'assistant', 'content': bricks_txt}]

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True,
        )
        full_ids = self.tokenizer.apply_chat_template(full_messages, tokenize=True)

        # Assistant-only labels: mask out the prompt tokens, supervise only the brick tokens.
        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100

        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mask': torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),  # [1, H, W]
            'has_mask': has_mask,
        }


class MaskDataCollator:
    """
    Collates :class:`MaskBrickDataset` examples into a padded batch.

    Right-pads ``input_ids`` / ``attention_mask`` / ``labels`` to the batch's longest sequence and
    stacks the per-example masks. The ``num_prefix_tokens`` mask tokens are prepended (with ``-100``
    labels) inside the model's forward pass, not here, so the text labels stay aligned with the text.
    """

    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f['input_ids'].size(0) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - f['input_ids'].size(0)
            input_ids.append(F.pad(f['input_ids'], (0, pad), value=self.pad_token_id))
            attention_mask.append(F.pad(f['attention_mask'], (0, pad), value=0))
            labels.append(F.pad(f['labels'], (0, pad), value=self.label_pad_id))

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
            'mask': torch.stack([f['mask'] for f in features]),
            'has_mask': torch.tensor([f['has_mask'] for f in features], dtype=torch.bool),
        }
