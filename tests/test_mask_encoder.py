import pytest
import torch

from brickgpt.masking import (
    MaskConditioningConfig, MaskEncoder, MaskPrefixEncoder, MultiViewMaskPrefixEncoder,
)


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


# --- MultiViewMaskPrefixEncoder (3-view) ---------------------------------------------------------

def test_multiview_output_is_fixed_prefix_length():
    cfg = _cfg()
    enc = MultiViewMaskPrefixEncoder(cfg)
    mask = torch.rand(2, cfg.num_views, 20, 20)
    out = enc(mask, has_mask=torch.ones(2, cfg.num_views, dtype=torch.bool))
    assert out.shape == (2, cfg.num_views * cfg.num_prefix_tokens, cfg.llm_hidden_size)


def test_multiview_rejects_wrong_view_count():
    cfg = _cfg()
    enc = MultiViewMaskPrefixEncoder(cfg)
    with pytest.raises(ValueError):
        enc(torch.rand(2, cfg.num_views + 1, 20, 20))


def test_presence_embedding_changes_output_on_identical_mask():
    # An all-zeros mask is ambiguous on its own; the presence embedding must disambiguate
    # "view absent" from "view provided but empty".
    cfg = _cfg()
    enc = MultiViewMaskPrefixEncoder(cfg).eval()
    zeros = torch.zeros(1, cfg.num_views, 20, 20)
    provided = enc(zeros, has_mask=torch.ones(1, cfg.num_views, dtype=torch.bool))
    absent = enc(zeros, has_mask=torch.zeros(1, cfg.num_views, dtype=torch.bool))
    assert not torch.allclose(provided, absent)


def test_view_embedding_distinguishes_views():
    # The same silhouette fed as different views should land on different prefix tokens.
    cfg = _cfg()
    enc = MultiViewMaskPrefixEncoder(cfg).eval()
    same = torch.rand(1, 1, 20, 20).repeat(1, cfg.num_views, 1, 1)
    out = enc(same, has_mask=torch.ones(1, cfg.num_views, dtype=torch.bool))
    out = out.view(1, cfg.num_views, cfg.num_prefix_tokens, cfg.llm_hidden_size)
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_disabling_embeddings_makes_identical_views_collapse():
    cfg = _cfg(use_view_embedding=False, use_presence_embedding=False)
    enc = MultiViewMaskPrefixEncoder(cfg).eval()
    same = torch.rand(1, 1, 20, 20).repeat(1, cfg.num_views, 1, 1)
    out = enc(same).view(1, cfg.num_views, cfg.num_prefix_tokens, cfg.llm_hidden_size)
    # With no view/presence offsets and a shared encoder, identical views map identically.
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)


def test_multiview_gradients_flow():
    cfg = _cfg()
    enc = MultiViewMaskPrefixEncoder(cfg)
    out = enc(torch.rand(2, cfg.num_views, 20, 20))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
