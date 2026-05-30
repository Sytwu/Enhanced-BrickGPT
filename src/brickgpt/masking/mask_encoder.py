import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import MaskConditioningConfig

# Output feature dimension of the supported torchvision backbones (after replacing the classifier).
_BACKBONE_FEATURE_DIM = {
    'resnet18': 512,
}


class MaskEncoder(nn.Module):
    """
    Encodes a single-channel 2D binary mask into a feature vector with a (optionally pretrained)
    torchvision CNN backbone. The first convolution is rewired to accept 1-channel input; when
    pretrained, its weights are initialized by summing the pretrained RGB filters so the ImageNet
    signal is preserved.

    TODO(handoff): tune the backbone choice, input normalization, and resize interpolation mode.
    """

    def __init__(self, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        if cfg.backbone not in _BACKBONE_FEATURE_DIM:
            raise ValueError(f'Unsupported mask-encoder backbone: {cfg.backbone!r}. '
                             f'Supported: {sorted(_BACKBONE_FEATURE_DIM)}')
        self.feature_dim = _BACKBONE_FEATURE_DIM[cfg.backbone]
        self.input_size = cfg.encoder_input_size

        weights = 'DEFAULT' if cfg.pretrained_backbone else None
        net = getattr(torchvision.models, cfg.backbone)(weights=weights)

        # Rewire the first conv to accept a single channel.
        old_conv = net.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if cfg.pretrained_backbone:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        net.conv1 = new_conv

        net.fc = nn.Identity()  # Expose the pooled feature vector instead of class logits.
        self.net = net

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        :param mask: A mask tensor of shape ``(B, 1, H, W)``.
        :return: A feature tensor of shape ``(B, feature_dim)``.
        """
        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError(f'Expected mask of shape (B, 1, H, W), got {tuple(mask.shape)}')
        mask = F.interpolate(mask, size=(self.input_size, self.input_size), mode='nearest')
        return self.net(mask)


class MaskProjection(nn.Module):
    """
    A 2-layer MLP mapping the encoder feature vector to ``num_prefix_tokens`` prefix-token embeddings
    of the LLM hidden size.
    """

    def __init__(self, feature_dim: int, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        self.num_prefix_tokens = cfg.num_prefix_tokens
        self.llm_hidden_size = cfg.llm_hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, cfg.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.projection_hidden_dim, cfg.num_prefix_tokens * cfg.llm_hidden_size),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        :param feat: A feature tensor of shape ``(B, feature_dim)``.
        :return: Prefix-token embeddings of shape ``(B, num_prefix_tokens, llm_hidden_size)``.
        """
        out = self.mlp(feat)
        return out.view(-1, self.num_prefix_tokens, self.llm_hidden_size)


class MaskPrefixEncoder(nn.Module):
    """
    End-to-end module turning a *single* 2D mask into prefix-token embeddings for the LLM:
    ``mask -> MaskEncoder -> MaskProjection -> prefix embeddings``.

    Kept for the single-view path / tests. The 3-view model uses
    :class:`MultiViewMaskPrefixEncoder`.
    """

    def __init__(self, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = MaskEncoder(cfg)
        self.projection = MaskProjection(self.encoder.feature_dim, cfg)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        :param mask: A mask tensor of shape ``(B, 1, H, W)``.
        :return: Prefix-token embeddings of shape ``(B, num_prefix_tokens, llm_hidden_size)``.
        """
        return self.projection(self.encoder(mask))


class MultiViewMaskPrefixEncoder(nn.Module):
    """
    Turns the three orthographic silhouettes (top / front / side) into a single block of prefix-token
    embeddings for the LLM.

    A **shared** :class:`MaskEncoder` + :class:`MaskProjection` encode every view (fewer params, and
    the pretrained init is reused for all three). Two learned embeddings, added to each view's prefix
    tokens, let the frozen-init shared encoder be disambiguated downstream:

    - **view-type embedding** (``top`` / ``front`` / ``side``): "which silhouette is this".
    - **presence embedding** (``provided`` / ``absent``): distinguishes a *dropped* view (null mask,
      D2 condition dropout) from a *provided* view whose silhouette merely happens to be empty.

    The output always has ``len(views) * num_prefix_tokens`` tokens (fixed slots), so batching is
    trivial; absent views still occupy their slots (null mask + *absent* presence embedding).
    """

    def __init__(self, cfg: MaskConditioningConfig = MaskConditioningConfig()):
        super().__init__()
        self.cfg = cfg
        self.num_views = cfg.num_views
        self.num_prefix_tokens = cfg.num_prefix_tokens
        self.llm_hidden_size = cfg.llm_hidden_size

        self.encoder = MaskEncoder(cfg)
        self.projection = MaskProjection(self.encoder.feature_dim, cfg)

        # One vector per view-type / presence-state, broadcast across that view's prefix tokens.
        self.view_embedding = (
            nn.Embedding(self.num_views, cfg.llm_hidden_size) if cfg.use_view_embedding else None
        )
        self.presence_embedding = (
            nn.Embedding(2, cfg.llm_hidden_size) if cfg.use_presence_embedding else None
        )

    def forward(self, mask: torch.Tensor, has_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param mask: Mask stack of shape ``(B, V, H, W)`` (``V == len(cfg.views)``).
        :param has_mask: Optional bool/0-1 tensor of shape ``(B, V)`` marking provided views. If
                         ``None``, every view is treated as provided.
        :return: Prefix-token embeddings of shape ``(B, V * num_prefix_tokens, llm_hidden_size)``.
        """
        if mask.dim() != 4 or mask.size(1) != self.num_views:
            raise ValueError(f'Expected mask of shape (B, {self.num_views}, H, W), got {tuple(mask.shape)}')
        b, v, h, w = mask.shape

        # Encode every view with the shared backbone in one batched pass.
        feat = self.encoder(mask.reshape(b * v, 1, h, w))               # [B*V, feat_dim]
        prefix = self.projection(feat).view(b, v, self.num_prefix_tokens, self.llm_hidden_size)

        if self.view_embedding is not None:
            view_ids = torch.arange(v, device=mask.device)
            prefix = prefix + self.view_embedding(view_ids)[None, :, None, :]

        if self.presence_embedding is not None:
            if has_mask is None:
                has_mask = torch.ones(b, v, dtype=torch.bool, device=mask.device)
            prefix = prefix + self.presence_embedding(has_mask.long())[:, :, None, :]

        return prefix.reshape(b, v * self.num_prefix_tokens, self.llm_hidden_size)
