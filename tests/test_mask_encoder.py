import pytest
import torch

from brickgpt.masking import MaskConditioningConfig, MaskEncoder, MaskPrefixEncoder


def _cfg(**kwargs):
    # pretrained_backbone=False avoids any network download during tests.
    return MaskConditioningConfig(pretrained_backbone=False, **kwargs)


def test_encoder_first_conv_is_single_channel():
    enc = MaskEncoder(_cfg())
    assert enc.net.conv1.in_channels == 1
    assert enc.feature_dim == 512


def test_encoder_forward_shape():
    enc = MaskEncoder(_cfg())
    feat = enc(torch.rand(2, 1, 20, 20))  # native 20x20 mask
    assert feat.shape == (2, 512)


def test_prefix_encoder_output_shape_and_gradients():
    cfg = _cfg()
    enc = MaskPrefixEncoder(cfg)
    out = enc(torch.rand(2, 1, 20, 20))
    assert out.shape == (2, cfg.num_prefix_tokens, cfg.llm_hidden_size)

    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_num_prefix_tokens_is_configurable():
    cfg = _cfg(num_prefix_tokens=4)
    out = MaskPrefixEncoder(cfg)(torch.rand(2, 1, 20, 20))
    assert out.shape == (2, 4, cfg.llm_hidden_size)


def test_invalid_backbone_raises():
    with pytest.raises(ValueError):
        MaskEncoder(_cfg(backbone='not_a_real_backbone'))


def test_forward_rejects_wrong_channel_count():
    enc = MaskEncoder(_cfg())
    with pytest.raises(ValueError):
        enc(torch.rand(2, 3, 20, 20))
