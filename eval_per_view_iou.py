"""
Per-view silhouette IoU comparison: text-only vs. mask-strings-in-prompt.

Usage:
    uv run eval_per_view_iou.py --data dataset01/mask01/test_masks.json --n_samples 50
    # With CLIP score (slow, requires LDraw library):
    uv run eval_per_view_iou.py --data dataset01/mask01/test_masks.json --n_samples 50 --compute_clip

For each sample this script generates the brick structure twice:
  - Original : plain text prompt, no mask information
  - Masked   : top/front/side RLE strings prepended to the prompt (from the JSON)

Computes per-view pixel-wise IoU (top / front / side), valid rate, avg brick count,
and optionally CLIP score (text-image cosine similarity via rendered PNG).
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers

from brickgpt.data import BrickStructure
from brickgpt.masking import MaskConditioningConfig, VIEW_AXES
from brickgpt.masking.dataset import _rle_mask
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig, create_instruction
from brickgpt.training.rewards import _valid_bricks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_mask_string(mask_str: str) -> np.ndarray:
    """'01110...\\n...' -> (H, W) float32 array."""
    rows = mask_str.strip().splitlines()
    return np.array([[float(c) for c in row] for row in rows], dtype=np.float32)


def per_view_iou(
        bricks_txt: str,
        target_views: dict[str, np.ndarray],
        cfg: MaskConditioningConfig,
) -> tuple[dict[str, float], bool, int]:
    """Pixel-wise IoU for each view independently.

    Returns (iou_dict, is_valid, num_bricks).
    iou_dict has 0.0 per view when the structure is unparseable.
    """
    bricks = _valid_bricks(bricks_txt)
    if not bricks:
        return {name: 0.0 for name in cfg.views}, False, 0
    try:
        structure = BrickStructure(bricks, world_dim=cfg.world_dim)
    except (IndexError, ValueError):
        return {name: 0.0 for name in cfg.views}, False, 0

    result = {}
    for name in cfg.views:
        pred = structure.top_down_mask(VIEW_AXES[name]) > 0.5
        tgt = target_views[name] > 0.5
        inter = float(np.logical_and(pred, tgt).sum())
        union = float(np.logical_or(pred, tgt).sum())
        result[name] = 1.0 if union == 0 else inter / union
    return result, True, len(bricks)


def make_masked_instruction_fn(top: str, front: str, side: str):
    """Returns an instruction_fn that prepends RLE-encoded views to the prompt."""
    top_rle = _rle_mask(top)
    front_rle = _rle_mask(front)
    side_rle = _rle_mask(side)

    def fn(caption: str) -> str:
        mask_block = (
            "The following shows the silhouette of the target structure from three sides "
            "(A=starts with 0, B=starts with 1; 1-9=run length, a-k=run length 10-20):\n\n"
            f"Top view:\n{top_rle}\n\n"
            f"Front view:\n{front_rle}\n\n"
            f"Side view:\n{side_rle}\n\n"
        )
        return mask_block + create_instruction(caption)
    return fn


# ---------------------------------------------------------------------------
# rendering + CLIP
# ---------------------------------------------------------------------------

_bpy_initialized = False


def _render_to_png(structure: BrickStructure, png_path: str,
                   ldraw_path: str, samples: int) -> bool:
    """Render a BrickStructure to a PNG using bpy + ImportLDraw.

    Returns True on success, False on any error.
    """
    global _bpy_initialized

    ldr_content = structure.to_ldr()
    if not ldr_content.strip():
        return False

    # Make ImportLDraw importable (it lives inside the project root, not installed).
    proj_dir = str(Path(__file__).resolve().parent)
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)

    os.environ['LDRAW_LIBRARY_PATH'] = ldraw_path

    tmp_ldr = os.path.join(tempfile.gettempdir(), f'_brick_{os.getpid()}.ldr')
    try:
        with open(tmp_ldr, 'w') as f:
            f.write(ldr_content)

        import bpy
        try:
            import ImportLDraw
            from ImportLDraw.loadldraw.loadldraw import Options, Configure, loadFromFile
        except ImportError as e:
            print(f'  [render] ImportLDraw not importable: {e}')
            return False

        # Reset scene before each render to avoid object accumulation.
        try:
            bpy.ops.wm.read_homefile(use_empty=True)
        except Exception:
            pass

        # Render engine and GPU setup.
        bpy.context.scene.render.engine = 'CYCLES'
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
            prefs.get_devices()
            for d in prefs.devices:
                d['use'] = int(d['name'].startswith('NVIDIA'))
        except Exception:
            pass
        bpy.context.scene.cycles.samples = samples

        # ImportLDraw options.
        plugin_path = Path(ImportLDraw.__file__).parent
        Options.ldrawDirectory = ldraw_path
        Options.instructionsLook = False
        Options.useLogoStuds = True
        Options.useUnofficialParts = True
        Options.gaps = True
        Options.studLogoDirectory = str(plugin_path / 'studs')
        Options.LSynthDirectory = str(plugin_path / 'lsynth')
        Options.verbose = 0
        Options.overwriteExistingMaterials = True
        Options.overwriteExistingMeshes = True
        Options.scale = 0.01
        Options.createInstances = True
        Options.removeDoubles = True
        Options.positionObjectOnGroundAtOrigin = True
        Options.flattenHierarchy = False
        Options.edgeSplit = True
        Options.addBevelModifier = True
        Options.bevelWidth = 0.5
        Options.addEnvironmentTexture = True
        Options.scriptDirectory = str(plugin_path / 'loadldraw')
        Options.addWorldEnvironmentTexture = True
        Options.addGroundPlane = True
        Options.setRenderSettings = True
        Options.removeDefaultObjects = True
        Options.positionCamera = True
        Options.cameraBorderPercent = 0.05

        Configure()
        loadFromFile(None, tmp_ldr)

        bpy.context.scene.render.resolution_x = 256
        bpy.context.scene.render.resolution_y = 256
        bpy.context.scene.camera.data.angle = math.radians(45)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = str(Path(png_path).resolve())
        bpy.ops.render.render(write_still=True)

        return Path(png_path).exists()
    except Exception as e:
        print(f'  [render error] {e}')
        return False
    finally:
        if os.path.exists(tmp_ldr):
            try:
                os.unlink(tmp_ldr)
            except OSError:
                pass


def _make_clip_scorer(clip_model_name: str, clip_pretrained: str, device: str):
    """Returns a callable (structure, caption) -> float."""
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    model = model.to(device).eval()

    def score(structure: BrickStructure, caption: str, ldraw_path: str, samples: int) -> float:
        tmp_png = os.path.join(tempfile.gettempdir(), f'_brick_{os.getpid()}.png')
        try:
            if not _render_to_png(structure, tmp_png, ldraw_path, samples):
                return float('nan')
            img = preprocess(Image.open(tmp_png)).unsqueeze(0).to(device)
            text = tokenizer([caption]).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
                txt_feat = model.encode_text(text)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                return float((img_feat @ txt_feat.T).item())
        except Exception as e:
            print(f'  [clip error] {e}')
            return float('nan')
        finally:
            if os.path.exists(tmp_png):
                try:
                    os.unlink(tmp_png)
                except OSError:
                    pass

    return score


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Per-view IoU: original vs. mask-string-in-prompt')
    parser.add_argument('--model', default='AvaLovelace/BrickGPT',
                        help='HuggingFace repo or local path of the BrickGPT model')
    parser.add_argument('--data', default='dataset01/mask01/test_masks.json',
                        help='Path to the JSON file with top/front/side strings')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--world_dim', type=int, default=20)
    parser.add_argument('--max_bricks', type=int, default=150)
    parser.add_argument('--max_brick_rejections', type=int, default=50)
    parser.add_argument('--max_regenerations', type=int, default=0)
    # CLIP score options
    parser.add_argument('--compute_clip', action='store_true',
                        help='Render each structure and compute CLIP text-image similarity')
    parser.add_argument('--ldraw_path',
                        default=str(Path(__file__).resolve().parent / 'ldraw'),
                        help='Path to the LDraw parts library')
    parser.add_argument('--render_samples', type=int, default=32,
                        help='Blender Cycles sample count for rendering (lower = faster)')
    parser.add_argument('--clip_model', default='ViT-B-32')
    parser.add_argument('--clip_pretrained', default='openai')
    args = parser.parse_args()

    transformers.set_seed(args.seed)

    with open(args.data) as f:
        data = json.load(f)
    if args.n_samples > 0:
        data = data[:args.n_samples]

    cfg = MaskConditioningConfig(world_dim=args.world_dim)
    brickgpt_cfg = BrickGPTConfig(
        model_name_or_path=args.model,
        world_dim=args.world_dim,
        max_bricks=args.max_bricks,
        max_brick_rejections=args.max_brick_rejections,
        max_regenerations=args.max_regenerations,
    )
    bg = BrickGPT(brickgpt_cfg)
    orig_fn = bg.instruction_fn

    clip_scorer = None
    if args.compute_clip:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading CLIP model {args.clip_model} ({args.clip_pretrained})...')
        clip_scorer = _make_clip_scorer(args.clip_model, args.clip_pretrained, device)
        print(f'LDraw path: {args.ldraw_path}')

    orig_ious   = {name: [] for name in cfg.views}
    masked_ious = {name: [] for name in cfg.views}
    orig_valid_count = 0
    masked_valid_count = 0
    orig_brick_counts:  list[int]   = []
    masked_brick_counts: list[int]  = []
    orig_clip_scores:   list[float] = []
    masked_clip_scores: list[float] = []

    for i, row in enumerate(data):
        caption = row['captions'][0] if isinstance(row.get('captions'), list) else row['caption']
        target_views = {
            'top':   parse_mask_string(row['top']),
            'front': parse_mask_string(row['front']),
            'side':  parse_mask_string(row['side']),
        }
        print(f"[{i+1:3d}/{len(data)}] {caption[:70]}")

        # ---- Original (text-only) ----
        bg.instruction_fn = orig_fn
        out_orig = bg(caption)
        bricks_txt_o = out_orig['bricks'].to_txt()
        iou_o, o_valid, o_n = per_view_iou(bricks_txt_o, target_views, cfg)
        for name in cfg.views:
            orig_ious[name].append(iou_o[name])
        if o_valid:
            orig_valid_count += 1
            orig_brick_counts.append(o_n)
        if clip_scorer and o_valid:
            cs = clip_scorer(out_orig['bricks'], caption, args.ldraw_path, args.render_samples)
            if not math.isnan(cs):
                orig_clip_scores.append(cs)

        # ---- Masked (strings in prompt) ----
        bg.instruction_fn = make_masked_instruction_fn(
            row['top'], row['front'], row['side']
        )
        out_masked = bg(caption)
        bg.instruction_fn = orig_fn
        bricks_txt_m = out_masked['bricks'].to_txt()
        iou_m, m_valid, m_n = per_view_iou(bricks_txt_m, target_views, cfg)
        for name in cfg.views:
            masked_ious[name].append(iou_m[name])
        if m_valid:
            masked_valid_count += 1
            masked_brick_counts.append(m_n)
        if clip_scorer and m_valid:
            cs = clip_scorer(out_masked['bricks'], caption, args.ldraw_path, args.render_samples)
            if not math.isnan(cs):
                masked_clip_scores.append(cs)

        # live progress
        iou_line = "  " + " | ".join(
            f"{name}: {iou_o[name]:.3f}->{iou_m[name]:.3f}" for name in cfg.views
        )
        meta_line = f"  valid:{'Y' if o_valid else 'N'}>{'Y' if m_valid else 'N'}  bricks:{o_n}>{m_n}"
        print(iou_line + meta_line)

    # ---- Summary table ----
    n = len(data)
    orig_valid_rate   = orig_valid_count  / n
    masked_valid_rate = masked_valid_count / n
    orig_avg_bricks   = float(np.mean(orig_brick_counts))   if orig_brick_counts   else 0.0
    masked_avg_bricks = float(np.mean(masked_brick_counts)) if masked_brick_counts else 0.0

    w = 52
    print()
    print("=" * w)
    print(f"{'':12} {'Original':>10} {'Masked':>10}  {'Δ':>8}")
    print("-" * w)
    for name in cfg.views:
        o = float(np.mean(orig_ious[name]))
        m = float(np.mean(masked_ious[name]))
        d = m - o
        print(f"{name.capitalize():<12} {o:>10.4f} {m:>10.4f}  {'+' if d>=0 else ''}{d:.4f}")
    print("-" * w)
    d_v = masked_valid_rate - orig_valid_rate
    print(f"{'Valid%':<12} {orig_valid_rate:>10.1%} {masked_valid_rate:>10.1%}  {'+' if d_v>=0 else ''}{d_v:.1%}")
    d_b = masked_avg_bricks - orig_avg_bricks
    print(f"{'AvgBricks':<12} {orig_avg_bricks:>10.1f} {masked_avg_bricks:>10.1f}  {'+' if d_b>=0 else ''}{d_b:.1f}")
    if clip_scorer:
        o_clip = float(np.mean(orig_clip_scores))   if orig_clip_scores   else float('nan')
        m_clip = float(np.mean(masked_clip_scores)) if masked_clip_scores else float('nan')
        d_clip = m_clip - o_clip
        print(f"{'CLIP':<12} {o_clip:>10.4f} {m_clip:>10.4f}  {'+' if d_clip>=0 else ''}{d_clip:.4f}")
    print("=" * w)
    print(f"(n = {n} samples)")


if __name__ == '__main__':
    main()
