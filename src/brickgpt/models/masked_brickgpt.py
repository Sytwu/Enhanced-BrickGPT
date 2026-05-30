import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from brickgpt.masking import MaskConditioningConfig, MultiViewMaskPrefixEncoder


class BrickGPTWithMask(nn.Module):
    """
    Wraps a causal LLM with a :class:`~brickgpt.masking.MultiViewMaskPrefixEncoder`.

    The three orthographic silhouettes (top / front / side) are encoded into
    ``len(views) * num_prefix_tokens`` prefix-token embeddings that are prepended to the text token
    embeddings before the LLM, so the model conditions generation on the silhouette. The module
    returns a standard HuggingFace causal-LM output (with ``loss`` when ``labels`` are given), so it
    can be driven directly by a HF/TRL ``Trainer``.

    Phase 3 (SFT): call :meth:`freeze_llm` to train only the mask encoder + projection + the
    view/presence embeddings.

    The mask-conditioned *generation* path (KV-cache prefill) lives in
    :mod:`brickgpt.models.brickgpt`; :meth:`prefix_embeds` exposes the prefix used there.
    """

    def __init__(self, base_model: nn.Module, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        self.base = base_model
        self.cfg = cfg
        self.mask_prefix_encoder = MultiViewMaskPrefixEncoder(cfg)

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
        """Freeze the LLM body; keep the mask encoder + projection + embeddings trainable (Phase-3 SFT)."""
        self.base.requires_grad_(False)
        self.mask_prefix_encoder.requires_grad_(True)

    def prefix_embeds(self, mask: torch.Tensor, has_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes the mask stack into prefix-token embeddings.

        :param mask: Mask stack of shape ``(B, V, H, W)``.
        :param has_mask: Optional ``(B, V)`` presence flags.
        :return: Prefix embeddings of shape ``(B, V * num_prefix_tokens, D)``.
        """
        return self.mask_prefix_encoder(mask, has_mask)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            has_mask: torch.Tensor | None = None,
            **kwargs,
    ):
        if mask is None:
            raise ValueError('BrickGPTWithMask.forward requires a `mask` tensor of shape (B, V, H, W).')

        text_embeds = self.base.get_input_embeddings()(input_ids)                       # [B, S, D]
        prefix_embeds = self.prefix_embeds(mask, has_mask).to(text_embeds.dtype)         # [B, T, D]
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
