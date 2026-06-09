import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from brickgpt.models import create_instruction

from .config import MaskConditioningConfig
from .projection import stack_views


def _encode_rle_row(row: str) -> str:
    """Run-length encode one binary row string.

    Prefix: 'A' = starts with 0, 'B' = starts with 1.
    Run lengths 1-9  → digit character ('1'..'9').
    Run lengths 10-20 → lowercase letter ('a'=10 .. 'k'=20).
    """
    chars = [c for c in row if c in '01']
    if not chars:
        return ''
    runs: list[int] = []
    current = chars[0]
    count = 1
    for c in chars[1:]:
        if c == current:
            count += 1
        else:
            runs.append(count)
            current = c
            count = 1
    runs.append(count)

    def _enc(n: int) -> str:
        return str(n) if n <= 9 else chr(ord('a') + n - 10)

    return ('A' if chars[0] == '0' else 'B') + ''.join(_enc(r) for r in runs)


def _rle_mask(mask_str: str) -> str:
    """Apply RLE encoding to every row of a multi-line mask string."""
    return '\n'.join(_encode_rle_row(row) for row in mask_str.strip().splitlines())


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
            use_text_mask: bool = False,
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
        :param use_text_mask: If ``True``, prepend the top/front/side binary strings to the user
                              message as plain text instead of using the mask-encoder prefix path.
                              Requires each data row to have ``top``, ``front``, and ``side`` fields.
                              Items returned by ``__getitem__`` will only contain ``input_ids``,
                              ``attention_mask``, and ``labels`` (no mask tensors).
        """
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.masks = masks
        self.prefixes = prefixes
        self.train = train
        self.system_prompt = system_prompt
        self.use_text_mask = use_text_mask

        if len(cfg.view_keep_probs) != cfg.num_views + 1:
            raise ValueError(
                f'cfg.view_keep_probs must have len(views)+1={cfg.num_views + 1} entries '
                f'(P over the number of kept views 0..{cfg.num_views}), got {len(cfg.view_keep_probs)}.'
            )

        # Flatten rows to one example per caption, keeping a pointer to the row's bricks/mask.
        self.bricks: list[str] = []
        self.mask_strings: list[dict] = []
        self.index: list[tuple[int, str]] = []
        for row_idx, row in enumerate(data):
            self.bricks.append(row['bricks'])
            self.mask_strings.append({
                'top': _rle_mask(row.get('top', '')) if use_text_mask else row.get('top', ''),
                'front': _rle_mask(row.get('front', '')) if use_text_mask else row.get('front', ''),
                'side': _rle_mask(row.get('side', '')) if use_text_mask else row.get('side', ''),
            })
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

        if self.use_text_mask:
            return self._getitem_text_mask(row_idx, caption, bricks_txt)

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

    def _getitem_text_mask(self, row_idx: int, caption: str, bricks_txt: str) -> dict:
        """Text-mask path: inject top/front/side binary strings into the user message."""
        ms = self.mask_strings[row_idx]

        if self.train and self.cfg.caption_dropout_p and random.random() < self.cfg.caption_dropout_p:
            caption = ''

        mask_block = (
            "The following shows the silhouette of the target structure from three sides "
            "(A=starts with 0, B=starts with 1; 1-9=run length, a-k=run length 10-20):\n\n"
            f"Top view:\n{ms['top']}\n\n"
            f"Front view:\n{ms['front']}\n\n"
            f"Side view:\n{ms['side']}\n\n"
        )
        prompt_messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': mask_block + create_instruction(caption)},
        ]
        full_messages = prompt_messages + [{'role': 'assistant', 'content': bricks_txt}]

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True, return_tensors='pt',
        )['input_ids'][0].tolist()
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, return_tensors='pt',
        )['input_ids'][0].tolist()

        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100

        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


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
