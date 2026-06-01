import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from brickgpt.models import create_instruction

from .config import MaskConditioningConfig
from .projection import stack_views


class MaskBrickDataset(Dataset):
    """
    PyTorch dataset that pairs each ``(caption, bricks)`` example with the three orthographic
    silhouette masks (top / front / side) of its structure.

    Each underlying row is expected to expose a ``captions`` list and a ``bricks`` string (the schema
    of ``AvaLovelace/StableText2Brick``). Rows are flattened to one example per caption, mirroring
    :func:`brickgpt.prepare_finetuning_dataset.convert_sample`.

    ``__getitem__`` returns one tokenized example together with a ``[V, H, W]`` mask stack and a
    ``[V]`` boolean ``has_mask`` flag. During training **per-view condition dropout** (D2) drops a
    random subset of views: dropped views are replaced with the all-zeros ``[NULL_MASK]`` and flagged
    absent. The number of kept views is sampled from ``cfg.view_keep_probs`` (biased toward a single
    provided view to match single-view inference); which views are kept is uniform. The presence
    flags are used later for per-view IoU reward routing and for the encoder's presence embedding.
    
    If precomputed multiview prefix tokens are provided, they will be included in the output dict
    and can be used directly by the model instead of computing them on-the-fly.
    """

    def __init__(
            self,
            data: Sequence[dict],
            tokenizer: Any,
            cfg: MaskConditioningConfig = MaskConditioningConfig(),
            masks: np.ndarray | None = None,
            prefixes: np.ndarray | None = None,
            train: bool = True,
            system_prompt: str = 'You are a helpful assistant.',
    ):
        """
        :param data: A sequence of rows, each with ``captions`` (list[str]) and ``bricks`` (str).
        :param tokenizer: A HuggingFace tokenizer exposing ``apply_chat_template``.
        :param cfg: Mask conditioning configuration.
        :param masks: Optional precomputed masks of shape ``(num_rows, V, world_dim, world_dim)``
                      (``V == len(cfg.views)``), indexed by row, as written by
                      :mod:`brickgpt.prepare_mask_dataset`. If ``None``, masks are projected on the
                      fly from ``bricks``.
        :param prefixes: Optional precomputed multiview prefix tokens of shape 
                         ``(num_rows, V*num_prefix_tokens, llm_hidden_size)``, indexed by row.
                         If provided, these are used instead of computing prefix tokens from masks.
        :param train: If ``True``, applies per-view condition dropout in ``__getitem__``.
        :param system_prompt: System message used to build the conversation.
        """
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.masks = masks
        self.prefixes = prefixes
        self.train = train
        self.system_prompt = system_prompt

        if len(cfg.view_keep_probs) != cfg.num_views + 1:
            raise ValueError(
                f'cfg.view_keep_probs must have len(views)+1={cfg.num_views + 1} entries '
                f'(P over the number of kept views 0..{cfg.num_views}), got {len(cfg.view_keep_probs)}.'
            )

        # Flatten rows to one example per caption, keeping a pointer to the row's bricks/mask.
        self.bricks: list[str] = []
        self.index: list[tuple[int, str]] = []
        for row_idx, row in enumerate(data):
            self.bricks.append(row['bricks'])
            for caption in row['captions']:
                self.index.append((row_idx, caption))

    def __len__(self) -> int:
        return len(self.index)

    def _get_views(self, row_idx: int) -> np.ndarray:
        """Returns the ``[V, world_dim, world_dim]`` mask stack for a row (precomputed or on the fly)."""
        if self.masks is not None:
            return np.asarray(self.masks[row_idx], dtype=np.float32)
        return stack_views(self.bricks[row_idx], self.cfg)

    def _get_prefix(self, row_idx: int) -> np.ndarray | None:
        """Returns precomputed ``[V*num_prefix_tokens, llm_hidden_size]`` prefix tokens, if available."""
        if self.prefixes is not None:
            return np.asarray(self.prefixes[row_idx], dtype=np.float32)
        return None

    def _sample_presence(self) -> np.ndarray:
        """Per-view condition dropout (D2): which views to keep. Returns a ``[V]`` bool array."""
        v = self.cfg.num_views
        if not self.train:
            return np.ones(v, dtype=bool)
        k = int(np.random.choice(v + 1, p=np.asarray(self.cfg.view_keep_probs, dtype=np.float64)))
        kept = random.sample(range(v), k)
        presence = np.zeros(v, dtype=bool)
        presence[kept] = True
        return presence

    def __getitem__(self, i: int) -> dict:
        row_idx, caption = self.index[i]
        bricks_txt = self.bricks[row_idx]

        # Resolve the views and per-view condition dropout: drop -> null mask + absent flag.
        views = self._get_views(row_idx)                  # [V, H, W]
        presence = self._sample_presence()                # [V] bool

        # Caption dropout (CFG-style): with prob caption_dropout_p, blank the text caption so the mask
        # is non-redundant and the encoder gets a real gradient. Force >=1 view present in that case so
        # the example is not fully unconditioned (which would carry no conditioning signal at all).
        if self.train and self.cfg.caption_dropout_p and random.random() < self.cfg.caption_dropout_p:
            caption = ''
            if not presence.any():
                presence[random.randrange(self.cfg.num_views)] = True

        views = views * presence[:, None, None].astype(np.float32)

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

        out = {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mask': torch.from_numpy(np.ascontiguousarray(views)).float(),  # [V, H, W]
            'has_mask': torch.from_numpy(presence),                         # [V] bool
        }
        
        # Optionally include precomputed prefix tokens
        prefix = self._get_prefix(row_idx)
        if prefix is not None:
            out['prefix_embeds'] = torch.from_numpy(prefix).float()  # [V*num_prefix_tokens, D]

        return out


class MaskDataCollator:
    """
    Collates :class:`MaskBrickDataset` examples into a padded batch.

    Right-pads ``input_ids`` / ``attention_mask`` / ``labels`` to the batch's longest sequence and
    stacks the per-example mask stacks (``[B, V, H, W]``) and presence flags (``[B, V]``). The mask
    prefix tokens are prepended (with ``-100`` labels) inside the model's forward pass, not here, so
    the text labels stay aligned with the text.
    
    If precomputed prefix tokens are available in the examples, they are stacked and included in the
    output batch as ``prefix_embeds`` (``[B, V*num_prefix_tokens, llm_hidden_size]``).
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

        out = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
            'mask': torch.stack([f['mask'] for f in features]),          # [B, V, H, W]
            'has_mask': torch.stack([f['has_mask'] for f in features]),  # [B, V] bool
        }
        
        # Optionally include precomputed prefix tokens if they're in all examples
        if all('prefix_embeds' in f for f in features):
            out['prefix_embeds'] = torch.stack([f['prefix_embeds'] for f in features])  # [B, V*T, D]
        
        return out
