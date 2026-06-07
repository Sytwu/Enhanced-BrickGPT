"""
Zero-shot *text-prompted* mask conditioning probe (no encoder, no training).

The trained Path-B route conditions generation on a silhouette via a learned mask-prefix encoder.
This script asks a cheaper question: **can the released BrickGPT (and the text-GRPO policy) follow a
mask given purely as text in the prompt?** It is the natural baseline for the encoder -- if a plain
text description of the target silhouette already lifts IoU, the encoder has a high bar to clear; if
it does nothing, the encoder route is better justified.

For each held-out ``(caption, GT bricks)`` we project the GT structure to **one** orthographic
silhouette (default ``top``; the projection fixes the coordinate frame so it matches the model's own
``(x,y,z)`` output), serialize the occupied cells into a plain-language instruction, and generate
twice through the *real* inference pipeline (logit masking + rejection sampling, ``use_gurobi=False``):

* **with mask**  -- the silhouette instruction is appended after the BrickGPT instruction.
* **no mask**    -- caption only (the baseline prior).

We report, per model, the mean single-view IoU of the generated structure's silhouette vs. the
target with and without the mask text, and the **lift** (with - without). A positive lift is direct
evidence the model conditions on the text mask. Because generation is temperature-sampled, each
prompt is averaged over ``samples_per_prompt`` completions to cut variance.

Run::

    # original BrickGPT only
    uv run python -m brickgpt.training.eval_zeroshot_mask --eval_grpo False
    # both original + text-GRPO policy
    uv run python -m brickgpt.training.eval_zeroshot_mask --adapter_dir output/grpo_text/adapter_final
"""
import logging
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from brickgpt.data import BrickStructure
from brickgpt.masking import MaskConditioningConfig, VIEW_AXES, three_view_masks
from brickgpt.models.brickgpt import BrickGPT, BrickGPTConfig, create_instruction
from brickgpt.models.llm import LLM
from brickgpt.visualize_sample import _label, _placeholder, mask_to_image, TILE

logger = logging.getLogger(__name__)

# Human-readable framing of each view: how it is seen, and what the two free cell axes are.
VIEW_DESC: dict[str, tuple[str, str]] = {
    'top':   ('viewed from above (looking down the z-axis)', '(x, y)'),
    'front': ('viewed from the front (looking down the y-axis)', '(x, z)'),
    'side':  ('viewed from the side (looking down the x-axis)', '(y, z)'),
}


@dataclass
class EvalArguments:
    base_model: str = field(default='AvaLovelace/BrickGPT')
    adapter_dir: str = field(default='output/grpo_text/adapter_final',
                             metadata={'help': 'Text-GRPO LoRA adapter (Path A) to also evaluate.'})
    eval_base: bool = field(default=True, metadata={'help': 'Evaluate the released BrickGPT.'})
    eval_grpo: bool = field(default=True, metadata={'help': 'Evaluate the text-GRPO policy in adapter_dir.'})

    dataset_name: str = field(default='AvaLovelace/StableText2Brick')
    split: str = field(default='train')
    num_prompts: int = field(default=20, metadata={'help': 'Distinct (caption, GT bricks) rows to probe.'})
    samples_per_prompt: int = field(default=4, metadata={'help': 'Completions averaged per prompt (cuts sampling noise).'})

    view: str = field(default='top', metadata={'help': "Which silhouette to condition on: 'top' | 'front' | 'side'."})
    world_dim: int = field(default=20)
    max_regenerations: int = field(default=10, metadata={'help': 'Stability rollback budget (connectivity, no Gurobi).'})
    temperature: float = field(default=0.6)
    seed: int = field(default=0)
    verbose: bool = field(default=False, metadata={'help': 'Also print the generated brick list per sample.'})

    save_dir: str = field(default='', metadata={'help': 'If set, save a per-prompt comparison strip PNG here.'})
    render: bool = field(default=False, metadata={'help': 'Add 3D renders to the comparison strip (needs LDraw/Blender).'})
    render_samples: int = field(default=32, metadata={'help': 'Cycles samples per render (low = fast).'})
    render_resolution: int = field(default=256)


# --- mask -> text -------------------------------------------------------------------------------

def build_mask_instruction(view: str, cells: np.ndarray) -> str:
    """Plain-language instruction describing the target silhouette as a list of occupied cells."""
    framing, axes = VIEW_DESC[view]
    cell_str = ', '.join(f'({int(a)},{int(b)})' for a, b in cells)
    return (f'Please match the following shape constraint. When the model is {framing}, its '
            f'silhouette should cover exactly these {axes} cells: {cell_str}.\n'
            f'Place and size the bricks so that, in that view, their combined footprint matches '
            f'these cells as closely as possible.')


def view_iou(structure: BrickStructure, target_2d: np.ndarray, axis: int) -> float:
    """Single-view pixel IoU between a generated structure's silhouette and the target mask."""
    pred = structure.top_down_mask(axis) > 0.5
    tgt = target_2d > 0.5
    inter = np.logical_and(pred, tgt).sum()
    union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 else float(inter / union)


# --- visualization ------------------------------------------------------------------------------

def render_structure(structure: BrickStructure, out_png: str, samples: int, resolution: int) -> Image.Image:
    """Render a structure to an image via the LDraw/Blender CLI in an isolated subprocess.

    Rendering runs in its own process with ``LD_LIBRARY_PATH`` unset so ``bpy`` does not clash with
    the torch/CUDA libs loaded by this eval process (see the bpy/Embree gotcha). Best-effort: any
    failure (no LDraw lib, no display, crash) yields a placeholder tile so the sheet still renders.
    """
    if not structure.bricks:
        return _placeholder('empty structure')
    try:
        with tempfile.TemporaryDirectory() as d:
            ldr_path = os.path.join(d, 'structure.ldr')
            with open(ldr_path, 'w') as f:
                f.write(structure.to_ldr())
            env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
            subprocess.run(
                ['uv', 'run', 'python', '-m', 'brickgpt.render_bricks', '--in_file', ldr_path,
                 '--out_file', out_png, '--samples', str(samples), '--img_resolution', str(resolution)],
                env=env, check=True, capture_output=True, timeout=300)
            return Image.open(out_png).convert('RGB').resize((TILE, TILE))
    except Exception as e:  # noqa: BLE001 -- rendering is best-effort
        logger.warning('Render failed (%s: %s); using placeholder.', type(e).__name__, e)
        return _placeholder('render failed')


def _silhouette_tile(structure: BrickStructure, view: str, axis: int) -> Image.Image:
    """Grayscale tile of a generated structure's silhouette in the conditioned view."""
    return mask_to_image(structure.top_down_mask(axis), height_axis=(view != 'top'))


def compose_row(panels: list[tuple[Image.Image, str]], tile: int = TILE) -> Image.Image:
    """Lays out labelled tiles in a single horizontal strip."""
    sheet = Image.new('RGB', (tile * len(panels), tile), color=(16, 16, 16))
    for i, (img, label) in enumerate(panels):
        sheet.paste(_label(img.resize((tile, tile)), label), (tile * i, 0))
    return sheet


def save_comparison(
        target_2d: np.ndarray, view: str, axis: int,
        s_no: BrickStructure, iou_no: float, s_yes: BrickStructure, iou_yes: float,
        out_png: str, render: bool, render_samples: int, render_resolution: int,
):
    """One comparison strip: target mask | no-mask (render+silhouette) | with-mask (render+silhouette)."""
    target_tile = mask_to_image(target_2d, height_axis=(view != 'top'))
    panels = [(target_tile, f'target {view} mask')]
    if render:
        base = os.path.splitext(out_png)[0]
        panels.append((render_structure(s_no, base + '_nomask.png', render_samples, render_resolution), 'no-mask render'))
        panels.append((_silhouette_tile(s_no, view, axis), f'no-mask IoU={iou_no:.2f}'))
        panels.append((render_structure(s_yes, base + '_mask.png', render_samples, render_resolution), 'with-mask render'))
        panels.append((_silhouette_tile(s_yes, view, axis), f'with-mask IoU={iou_yes:.2f}'))
    else:
        panels.append((_silhouette_tile(s_no, view, axis), f'no-mask IoU={iou_no:.2f}'))
        panels.append((_silhouette_tile(s_yes, view, axis), f'with-mask IoU={iou_yes:.2f}'))
    compose_row(panels).save(out_png)


# --- model loading ------------------------------------------------------------------------------

def _base_cfg(args, device: str) -> BrickGPTConfig:
    return BrickGPTConfig(
        model_name_or_path=args.base_model, world_dim=args.world_dim, use_gurobi=False,
        max_regenerations=args.max_regenerations, temperature=args.temperature, device=device)


def build_base_bg(args, device: str) -> BrickGPT:
    """The released BrickGPT, loaded the normal inference way (base + published adapter)."""
    return BrickGPT(_base_cfg(args, device))


def build_grpo_bg(args, device: str) -> BrickGPT:
    """Merged BrickGPT backbone + text-GRPO adapter, wrapped as a BrickGPT (mirrors eval_grpo_text)."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_cfg = PeftConfig.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(base, args.base_model).merge_and_unload().to(device)
    model = PeftModel.from_pretrained(backbone, args.adapter_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    llm = LLM.from_model(model, tokenizer, device)
    return BrickGPT(_base_cfg(args, device), llm=llm)


# --- eval ---------------------------------------------------------------------------------------

def _generate_structure(bg: BrickGPT, caption: str, mask_text: str | None) -> BrickStructure:
    """One generation; ``mask_text`` (if given) is appended after the standard BrickGPT instruction."""
    if mask_text is None:
        bg.instruction_fn = create_instruction
    else:
        bg.instruction_fn = lambda c: create_instruction(c) + '\n\n' + mask_text
    return bg(caption)['bricks']


def eval_model(name: str, bg: BrickGPT, examples: list[dict], args) -> dict:
    """Probe one model over all prompts; prints per-prompt prompt + IoU lift, returns aggregates."""
    axis = VIEW_AXES[args.view]
    mask_cfg = MaskConditioningConfig(world_dim=args.world_dim)
    per_nomask, per_mask = [], []

    print(f'\n{"=" * 88}\n  MODEL: {name}   (view={args.view}, {args.samples_per_prompt} samples/prompt)\n{"=" * 88}')
    for i, ex in enumerate(examples):
        caption = ex['captions'][0] if ex.get('captions') else ex['caption']
        target_2d = three_view_masks(ex['bricks'], mask_cfg)[args.view]
        cells = np.argwhere(target_2d > 0.5)
        mask_text = build_mask_instruction(args.view, cells)

        nomask_ious, mask_ious = [], []
        first_no = first_yes = None  # kept for the visualization (representative sample)
        for _ in range(args.samples_per_prompt):
            s_no = _generate_structure(bg, caption, None)
            s_yes = _generate_structure(bg, caption, mask_text)
            nomask_ious.append(view_iou(s_no, target_2d, axis))
            mask_ious.append(view_iou(s_yes, target_2d, axis))
            if first_no is None:
                first_no, first_yes = s_no, s_yes
            if args.verbose:
                print(f'    [no-mask bricks]\n{s_no.to_txt()}\n    [with-mask bricks]\n{s_yes.to_txt()}')
        iou_no, iou_yes = float(np.mean(nomask_ious)), float(np.mean(mask_ious))
        per_nomask.append(iou_no)
        per_mask.append(iou_yes)

        print(f'\n--- prompt {i + 1}/{len(examples)} ---')
        print(f'caption: {caption}   (target {args.view} silhouette: {len(cells)} cells)')
        print(f'>>> PROMPT SENT (with mask):\n{create_instruction(caption)}\n\n{mask_text}')
        print(f'>>> IoU  no_mask={iou_no:.3f}  with_mask={iou_yes:.3f}  lift={iou_yes - iou_no:+.3f}')

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out_png = os.path.join(args.save_dir, f'{name}_{args.view}_{i:02d}.png')
            save_comparison(target_2d, args.view, axis, first_no, view_iou(first_no, target_2d, axis),
                            first_yes, view_iou(first_yes, target_2d, axis), out_png,
                            args.render, args.render_samples, args.render_resolution)
            print(f'>>> saved comparison: {out_png}')

    return {
        'model': name, 'view': args.view, 'n': len(examples),
        'iou_nomask': float(np.mean(per_nomask)) if per_nomask else 0.0,
        'iou_mask': float(np.mean(per_mask)) if per_mask else 0.0,
        'iou_lift': float(np.mean(per_mask) - np.mean(per_nomask)) if per_mask else 0.0,
    }


def _print_table(rows: list[dict]):
    cols = [('model', 10, 's'), ('view', 7, 's'), ('n', 5, 'd'),
            ('iou_nomask', 12, '.3f'), ('iou_mask', 11, '.3f'), ('iou_lift', 11, '+.3f')]

    def cell(v, fmt: str) -> str:
        return str(v) if fmt == 's' else f'{v:{fmt}}'   # format first, then right-justify

    head = ''.join(f'{c:>{w}}' for c, w, _ in cols)
    print('\n' + head + '\n' + '-' * len(head))
    for r in rows:
        print(''.join(f'{cell(r[c], fmt):>{w}}' for c, w, fmt in cols))
    print()


def main():
    logging.basicConfig(level=logging.WARNING)  # quiet the per-brick rejection warnings
    from transformers import HfArgumentParser
    (args,) = HfArgumentParser(EvalArguments).parse_args_into_dataclasses()
    if args.view not in VIEW_AXES:
        raise ValueError(f'--view must be one of {list(VIEW_AXES)}, got {args.view!r}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from datasets import load_dataset
    data = load_dataset(args.dataset_name, split=args.split)
    idxs = random.sample(range(len(data)), min(args.num_prompts, len(data)))
    examples = [data[i] for i in idxs]
    logger.warning('Probing %d prompts x %d samples on view=%s.', len(examples), args.samples_per_prompt, args.view)

    rows = []
    if args.eval_base:
        rows.append(eval_model('base', build_base_bg(args, device), examples, args))
    if args.eval_grpo:
        rows.append(eval_model('grpo', build_grpo_bg(args, device), examples, args))

    _print_table(rows)


if __name__ == '__main__':
    main()
