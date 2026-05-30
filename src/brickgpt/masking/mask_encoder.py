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
    End-to-end module turning a 2D mask into prefix-token embeddings for the LLM:
    ``mask -> MaskEncoder -> MaskProjection -> prefix embeddings``.
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
