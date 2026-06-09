"""
Shared render + CLIP + DINOv2 scorer for the EXP.md evaluation scripts.

A structure is rendered to a single image (Blender/LDraw) and that one image feeds both:

* **CLIP**   -- OpenCLIP ViT-B-32 image-**text** cosine (render vs caption).
* **DINOv2** -- ``facebook/dinov2-base`` image-**image** cosine (render vs a cached *ground-truth*
  render feature); shape fidelity, caption-independent.

Used by ``scripts/eval_grpo_table.py`` (GRPO table) and ``scripts/eval_text_mask_render.py`` (mask
condition table + top-K example dump) so the two reports use byte-identical scoring.
"""
import logging
import os
import tempfile

import torch

logger = logging.getLogger(__name__)


class RenderScorer:
    """Renders a structure once, then scores CLIP (image-text) and DINOv2 (image-image vs a GT feature)."""

    def __init__(self, device, dino_model='facebook/dinov2-base', render_samples=32, img_resolution=224):
        import open_clip
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.render_samples = render_samples
        self.img_resolution = img_resolution
        clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip = clip.to(device).eval()
        self.clip_preprocess = preprocess
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.dino = AutoModel.from_pretrained(dino_model).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)
        self._tmp = tempfile.mkdtemp(prefix='brickgpt_render_')

    def render(self, structure, tag):
        """Structure -> PIL image (RGB), or None on render failure."""
        from PIL import Image
        from brickgpt.render_bricks import render_bricks
        ldr = os.path.join(self._tmp, f'{tag}.ldr')
        png = os.path.join(self._tmp, f'{tag}.png')
        try:
            with open(ldr, 'w') as f:
                f.write(structure.to_ldr())
            render_bricks(ldr, png, square_image=True, instructions_look=False,
                          img_resolution=self.img_resolution, samples=self.render_samples)
            return Image.open(png).convert('RGB')
        except Exception as e:
            logger.warning('render failed (%s); skipping image metrics for this structure.', e)
            return None

    @torch.no_grad()
    def clip_cosine(self, image, caption):
        img = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        txt = self.clip_tokenizer([caption]).to(self.device)
        f_i = self.clip.encode_image(img); f_t = self.clip.encode_text(txt)
        f_i /= f_i.norm(dim=-1, keepdim=True); f_t /= f_t.norm(dim=-1, keepdim=True)
        return (f_i @ f_t.T).item()

    @torch.no_grad()
    def dino_feat(self, image):
        inputs = self.dino_processor(images=image, return_tensors='pt').to(self.device)
        out = self.dino(**inputs)
        feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(1)
        feat = feat.float()
        return feat / feat.norm(dim=-1, keepdim=True)
