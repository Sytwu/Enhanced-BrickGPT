# SFT Fine-tuning with Mask Conditioning (Text RLE)

This branch adds **text-mask SFT fine-tuning** on top of any BrickGPT-compatible model.  
Instead of using the CNN mask encoder, the target silhouette is injected directly into the
prompt as RLE-encoded binary strings (top / front / side views).

The colleague running GRPO RL can use this to fine-tune their trained model and evaluate
whether silhouette conditioning improves per-view IoU.

---

## Prerequisites

- Python 3.11
- `uv` package manager
- CUDA GPU (tested on RTX 2060 Super 8 GB)
- LDraw library at `ldraw/` (only needed for `--compute_clip`)

Install dependencies:
```bash
uv sync
```

---

## Workflow

### 1. Prepare the mask dataset

Converts the raw dataset into JSON rows with `top` / `front` / `side` binary mask strings.

```bash
uv run prepare_mask_dataset_text \
    --dataset AvaLovelace/StableText2Brick \
    --output_dir dataset01/mask01
```

Output: `dataset01/mask01/train_masks.json` and `test_masks.json`.

---

### 2. Fine-tune with LoRA

Pass your own model via `--model_name_or_path` (defaults to `AvaLovelace/BrickGPT`).

```bash
uv run train_sft_text_mask.py \
    --model_name_or_path <your_model_or_path> \
    --train_data dataset01/mask01/train_masks.json \
    --eval_data  dataset01/mask01/test_masks.json \
    --output_dir output/sft_text_mask \
    --max_steps 2000 \
    --lr 1e-4
```

| Argument | Default | Notes |
|---|---|---|
| `--model_name_or_path` | `AvaLovelace/BrickGPT` | HuggingFace repo or local path |
| `--max_steps` | `500` | Increase for better results |
| `--lr` | `2e-4` | Lower to `1e-4` when continuing from a pre-trained model |
| `--lora_r` | `16` | LoRA rank |
| `--output_dir` | `output/sft_text_mask` | Adapter and checkpoints saved here |

The final adapter is saved to `<output_dir>/adapter_final`.

---

### 3. Merge adapter into base model

```bash
uv run merge_adapter.py \
    --base <your_model_or_path> \
    --adapter output/sft_text_mask/adapter_final \
    --output  output/sft_text_mask/merged
```

This produces a standalone model at `output/sft_text_mask/merged` that can be loaded
with `AutoModelForCausalLM.from_pretrained`.

To continue fine-tuning from the merged result:
```bash
uv run train_sft_text_mask.py \
    --model_name_or_path output/sft_text_mask/merged \
    --output_dir output/sft_text_mask_v2 \
    --max_steps 2000 --lr 1e-4
```

---

### 4. Evaluate per-view IoU

Compares two conditions on the same model:

- **Original**: text-only prompt, no mask information
- **Masked**: prompt includes RLE-encoded top/front/side silhouettes

```bash
uv run eval_per_view_iou.py \
    --model output/sft_text_mask/merged \
    --data  dataset01/mask01/test_masks.json \
    --n_samples 50
```

**With CLIP score** (requires LDraw library at `ldraw/`, slow — ~20 min for 10 samples):
```bash
uv run eval_per_view_iou.py \
    --model output/sft_text_mask/merged \
    --data  dataset01/mask01/test_masks.json \
    --n_samples 50 \
    --compute_clip
```

Example output:
```
====================================================
               Original     Masked         Δ
----------------------------------------------------
Top              0.4429     0.7263  +0.2834
Front            0.3700     0.4822  +0.1122
Side             0.2974     0.4456  +0.1482
----------------------------------------------------
Valid%           100.0%     100.0%  +0.0%
AvgBricks         140.0      150.0  +10.0
====================================================
(n = 50 samples)
```

---

## Baseline comparison

To get the unmodified BrickGPT baseline (no fine-tuning, no mask):
```bash
uv run eval_per_view_iou.py \
    --model AvaLovelace/BrickGPT \
    --data  dataset01/mask01/test_masks.json \
    --n_samples 50
```

| | Top | Front | Side |
|---|---|---|---|
| BrickGPT original (no mask) | 0.4356 | 0.3702 | 0.3303 |
| BrickGPT original (with mask, no fine-tune) | 0.4643 | 0.3948 | 0.3423 |
| Fine-tuned v2 (no mask) | 0.4429 | 0.3700 | 0.2974 |
| **Fine-tuned v2 (with mask)** | **0.7263** | **0.4822** | **0.4456** |

---

## RLE mask format

Each row of the binary mask is encoded as:

```
A/B + run-length digits
```

- `A` = row starts with `0`, `B` = row starts with `1`
- `1`–`9` = run length 1–9
- `a`–`k` = run length 10–20

Example: `A1344341` → `0 111 0000 111 00001 1110` (the string from the project description)
