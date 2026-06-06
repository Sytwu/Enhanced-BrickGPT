"""
Semantic (CLIP) reward hook for the text GRPO loop.

The semantic term is the **only** reward that references the caption, so it is what makes GRPO optimize
"the structure looks like what was asked for" rather than just "valid + stable". It is expensive
(Blender render + CLIP encode, *seconds per completion* -- the same order as the Gurobi solver the
reward module warns against), so it is **off by default** and isolated here: ``bpy`` / ``open_clip`` are
imported lazily, so non-semantic runs and the offline test suite never touch them.

Pipeline per completion: brick text -> :class:`BrickStructure` -> ``.ldr`` -> Blender render (CYCLES) ->
OpenCLIP image-text cosine. The raw cosine is returned; :func:`~brickgpt.training.rewards.compute_reward`
maps it to ``[0, 1]`` via ``clip_lo/clip_hi``. Invalid / unrenderable completions return ``None`` (the
syntax gate already drives those to ``-1``), so no render is wasted on garbage.

Requires the render prerequisites from ``DEVELOP.md``: the LDraw parts library (``LDRAW_LIBRARY_PATH``),
``ImportLDraw/loadldraw/background.exr``, and -- because ``import bpy`` crashes under a populated
``LD_LIBRARY_PATH`` -- launching with ``env -u LD_LIBRARY_PATH``.
"""
import logging
import os
import tempfile

import torch

from brickgpt.data import Brick, BrickStructure

logger = logging.getLogger(__name__)


def _structure_from_txt(bricks_txt: str, world_dim: int) -> BrickStructure | None:
    """Parses a completion into a buildable :class:`BrickStructure`, or ``None`` if invalid/out-of-bounds."""
    bricks = []
    for line in bricks_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            brick = Brick.from_txt(line)
            _ = brick.brick_id  # raises if dimensions are not in the library
        except ValueError:
            return None
        bricks.append(brick)
    if not bricks:
        return None
    try:
        return BrickStructure(bricks, world_dim=world_dim)
    except (IndexError, ValueError):
        return None


class SemanticScorer:
    """
    Renders a completion and scores it against the caption with OpenCLIP (raw cosine).

    :param device: Torch device for the CLIP model.
    :param clip_arch / clip_pretrained: OpenCLIP model spec (default ``ViT-B-32`` / ``openai``, matching
        :mod:`brickgpt.infer`).
    :param render_samples: CYCLES samples per render. CLIP is robust to noise, so a low value (e.g. 64)
        cuts render time several-fold vs. the inference default of 512.
    :param img_resolution: Render resolution; 256 is plenty for ViT-B-32's 224px input.
    :param world_dim: Voxel grid size for building the structure (must match the reward's ``mask_cfg``).
    """

    def __init__(self, device: str, clip_arch: str = 'ViT-B-32', clip_pretrained: str = 'openai',
                 render_samples: int = 64, img_resolution: int = 256, world_dim: int = 20):
        import open_clip  # lazy: only pulled in when semantic reward is actually used
        self.device = device
        self.render_samples = render_samples
        self.img_resolution = img_resolution
        self.world_dim = world_dim
        model, _, preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(clip_arch)
        self._tmp = tempfile.mkdtemp(prefix='brickgpt_grpo_render_')

    @torch.no_grad()
    def score(self, bricks_txt: str, caption: str) -> float | None:
        """
        Returns the raw CLIP cosine of the rendered completion vs. ``caption``, or ``None`` if the
        completion is invalid / unrenderable (so the caller leaves ``clip_score=None`` and the syntax
        gate handles it). Render/CLIP failures are caught and logged, also returning ``None``.
        """
        structure = _structure_from_txt(bricks_txt, self.world_dim)
        if structure is None:
            return None
        from PIL import Image
        from brickgpt.render_bricks import render_bricks
        ldr_path = os.path.join(self._tmp, 'structure.ldr')
        img_path = os.path.join(self._tmp, 'structure.png')
        try:
            with open(ldr_path, 'w') as f:
                f.write(structure.to_ldr())
            render_bricks(ldr_path, img_path, square_image=True, instructions_look=False,
                          img_resolution=self.img_resolution, samples=self.render_samples)
            image = self.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(self.device)
            text = self.tokenizer([caption]).to(self.device)
            img_feat = self.model.encode_image(image)
            txt_feat = self.model.encode_text(text)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).item()
        except Exception as e:  # render is a long external pipeline; never let it kill the RL step
            logger.warning('Semantic render/CLIP failed (%s); skipping semantic term for this sample.', e)
            return None
