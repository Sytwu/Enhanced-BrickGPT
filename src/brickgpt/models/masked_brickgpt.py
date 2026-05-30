import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from brickgpt.masking import MaskConditioningConfig, MaskPrefixEncoder


class BrickGPTWithMask(nn.Module):
    """
    Wraps a causal LLM with a :class:`~brickgpt.masking.MaskPrefixEncoder`.

    The 2D mask is encoded into ``num_prefix_tokens`` prefix-token embeddings that are prepended to
    the text token embeddings before the LLM, so the model conditions generation on the silhouette.
    The module returns a standard HuggingFace causal-LM output (with ``loss`` when ``labels`` are
    given), so it can be driven directly by a HF/TRL ``Trainer``.

    Phase 3 (SFT): call :meth:`freeze_llm` to train only the mask encoder + projection.

    TODO(Phase 5): add an inference path that pre-fills the KV-cache from the prefix embeddings so
    the token-by-token logit-masking generation loop in :mod:`brickgpt.models.brickgpt` can run with
    mask conditioning. See the plan's "Open risks" section.
    """

    def __init__(self, base_model: nn.Module, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        self.base = base_model
        self.cfg = cfg
        self.mask_prefix_encoder = MaskPrefixEncoder(cfg)

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path: str,
            cfg: MaskConditioningConfig = MaskConditioningConfig(),
            **kwargs,
    ) -> 'BrickGPTWithMask':
        base = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base, cfg)

    def freeze_llm(self) -> None:
        """Freeze the LLM body; keep the mask encoder + projection trainable (Phase-3 SFT)."""
        self.base.requires_grad_(False)
        self.mask_prefix_encoder.requires_grad_(True)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            has_mask: torch.Tensor | None = None,  # carried for RL reward routing; unused here
            **kwargs,
    ):
        if mask is None:
            raise ValueError('BrickGPTWithMask.forward requires a `mask` tensor of shape (B, 1, H, W).')

        text_embeds = self.base.get_input_embeddings()(input_ids)            # [B, S, D]
        prefix_embeds = self.mask_prefix_encoder(mask).to(text_embeds.dtype)  # [B, T, D]
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        bsz, num_prefix = prefix_embeds.shape[:2]
        prefix_attn = torch.ones(bsz, num_prefix, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full((bsz, num_prefix), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
