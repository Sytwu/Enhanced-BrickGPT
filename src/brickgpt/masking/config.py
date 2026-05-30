from dataclasses import dataclass, field


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
    num_prefix_tokens: int = field(
        default=8,
        metadata={'help': 'Number of prefix tokens the mask is projected into and prepended to the text sequence.'},
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
    condition_dropout_p: float = field(
        default=0.3,
        metadata={'help': 'Probability of replacing the mask with the null mask during training (condition dropout).'},
    )
    pretrained_backbone: bool = field(
        default=True,
        metadata={'help': 'Whether to initialize the CNN backbone with pretrained ImageNet weights.'},
    )
