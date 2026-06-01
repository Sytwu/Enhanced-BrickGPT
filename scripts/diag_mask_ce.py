"""Teacher-forced diagnostic: does the mask prefix actually reduce CE?

Loads the trained SFT mask encoder and, for held-out (and train) examples, compares the cross-entropy
of the *same* GT brick sequence with the real mask present vs. a null (absent) mask. Zero sampling
variance, so it directly answers "is there any training signal for the encoder?" — unlike the IoU
probe. A clearly positive mean delta (null_ce - masked_ce > 0) means the mask helps; ~0 means the
frozen LLM already fits from the caption alone (=> escalate to LoRA-on-LLM, TODO §3.1).
"""
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from brickgpt.masking import MaskConditioningConfig, MaskBrickDataset

from brickgpt.models.masked_brickgpt import BrickGPTWithMask

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'output/sft_smoke2/mask_encoder_final.pt'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
MODEL = 'AvaLovelace/BrickGPT'


def per_example_ce(model, item, device, *, null):
    ids = item['input_ids'].unsqueeze(0).to(device)
    attn = item['attention_mask'].unsqueeze(0).to(device)
    labels = item['labels'].unsqueeze(0).to(device)
    mask = item['mask'].unsqueeze(0).float().to(device)
    has = item['has_mask'].unsqueeze(0).to(device)
    if null:
        mask = torch.zeros_like(mask)
        has = torch.zeros_like(has)
    out = model(input_ids=ids, attention_mask=attn, labels=labels, mask=mask, has_mask=has, use_cache=False)
    return out.loss.item()


def run_split(model, tokenizer, cfg, split, device):
    from datasets import load_dataset
    data = load_dataset('AvaLovelace/StableText2Brick', split=split)
    ds = MaskBrickDataset(data, tokenizer, cfg, train=False)  # train=False => all views present, no dropout
    idxs = np.linspace(0, len(ds) - 1, num=min(N, len(ds)), dtype=int)
    deltas, masked, null = [], [], []
    for i in idxs:
        item = ds[int(i)]
        m = per_example_ce(model, item, device, null=False)
        n = per_example_ce(model, item, device, null=True)
        masked.append(m); null.append(n); deltas.append(n - m)
    deltas = np.array(deltas)
    print(f'\n[{split}] n={len(deltas)}')
    print(f'  masked_ce mean = {np.mean(masked):.4f}')
    print(f'  null_ce   mean = {np.mean(null):.4f}')
    print(f'  delta (null-masked) mean = {deltas.mean():+.4f}  median = {np.median(deltas):+.4f}  std = {deltas.std():.4f}')
    print(f'  frac examples where mask helps (delta>0) = {(deltas > 0).mean():.2%}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = MaskConditioningConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = BrickGPTWithMask.from_pretrained(MODEL, cfg, torch_dtype=torch.bfloat16).to(device)
    sd = torch.load(CKPT, map_location=device)
    model.mask_prefix_encoder.load_state_dict(sd)
    model.eval()
    print(f'loaded encoder from {CKPT}')
    with torch.no_grad():
        run_split(model, tokenizer, cfg, 'test', device)
        run_split(model, tokenizer, cfg, 'train', device)


if __name__ == '__main__':
    main()
