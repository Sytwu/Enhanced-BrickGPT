from dataclasses import dataclass, field

# Canonical view ordering. The mask stack, presence flags, view-type embeddings, and the IoU reward
# all index views in this order, so it is defined once and imported everywhere.
VIEW_ORDER: tuple[str, ...] = ('top', 'front', 'side')


@dataclass
class MaskConditioningConfig:
    """
    Configuration for the 2D-mask visual conditioning pipeline (data, encoder, and projection).
    Shared by the dataset, the mask encoder, and the training scripts so that all components
    agree on the prefix-token count, hidden size, and mask geometry.
    """
    backbone: str = field(
        default='resnet18',
        metadata={'help': 'Name of the torchvision CNN backbone used by the mask encoder.'},
    )
    views: tuple[str, ...] = field(
        default=VIEW_ORDER,
        metadata={'help': 'Orthographic silhouette views used as the visual condition, in order. '
                          'Each view contributes num_prefix_tokens prefix tokens (fixed slots).'},
    )
    use_view_embedding: bool = field(
        default=True,
        metadata={'help': 'Add a learned per-view-type embedding to each view\'s prefix tokens, so the '
                          'shared encoder\'s outputs can be disambiguated (top vs front vs side).'},
    )
    use_presence_embedding: bool = field(
        default=True,
        metadata={'help': 'Add a learned presence embedding (provided vs absent) to each view\'s prefix '
                          'tokens, distinguishing "view not given" from "view given but empty silhouette".'},
    )
    num_prefix_tokens: int = field(
        default=8,
        metadata={'help': 'Number of prefix tokens PER VIEW. The total prefix length is '
                          'len(views) * num_prefix_tokens.'},
    )
    llm_hidden_size: int = field(
        default=2048,
        metadata={'help': 'Hidden size of the LLM (2048 for Llama-3.2-1B). Prefix tokens are projected to this size.'},
    )
    world_dim: int = field(
        default=20,
        metadata={'help': 'World dimension; the native mask is (world_dim, world_dim).'},
    )
    projection_axis: int = field(
        default=2,
        metadata={'help': 'Axis to project the 3D occupancy grid along (2 = Z-axis top-down view).'},
    )
    encoder_input_size: int = field(
        default=64,
        metadata={'help': 'Spatial size the mask is resized to before being fed to the CNN backbone.'},
    )
    projection_hidden_dim: int = field(
        default=1024,
        metadata={'help': 'Hidden dimension of the 2-layer projection MLP.'},
    )
    encoder_resize_mode: str = field(
        default='nearest',
        metadata={'help': 'Interpolation mode used to resize the input mask before the CNN encoder.'},
    )
    normalize_mask: bool = field(
        default=True,
        metadata={'help': 'Whether to normalize the single-channel mask before the CNN encoder.'},
    )
    mask_mean: float = field(
        default=0.5,
        metadata={'help': 'Mean value used for mask normalization.'},
    )
    mask_std: float = field(
        default=0.5,
        metadata={'help': 'Std value used for mask normalization.'},
    )
    condition_dropout_p: float = field(
        default=0.3,
        metadata={'help': 'DEPRECATED (single-view). Kept for backward compatibility; the 3-view dataset '
                          'uses view_keep_probs instead.'},
    )
    view_keep_probs: tuple[float, ...] = field(
        default=(0.10, 0.60, 0.15, 0.15),
        metadata={'help': 'Per-view condition dropout (D2): categorical distribution over the NUMBER of '
                          'views kept, indexed 0..len(views). Default keeps 0/1/2/3 views with prob '
                          '0.10/0.60/0.15/0.15 -- biased toward a single provided view to match the '
                          'single-view inference distribution, while still seeing all-3 and text-only. '
                          'Which views are kept (given the count) is chosen uniformly.'},
    )
    pretrained_backbone: bool = field(
        default=True,
        metadata={'help': 'Whether to initialize the CNN backbone with pretrained ImageNet weights.'},
    )

    @property
    def num_views(self) -> int:
        return len(self.views)

    @property
    def total_prefix_tokens(self) -> int:
        """Total prefix length prepended to the text sequence: len(views) * num_prefix_tokens."""
        return self.num_views * self.num_prefix_tokens
