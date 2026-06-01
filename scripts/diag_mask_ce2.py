"""2x2 teacher-forced diagnostic: separate the mask's effect WITH vs WITHOUT the caption.

For each example we build the prompt with the real caption or a blanked caption, and run the model
with the real mask or a null (absent) mask -> four CE numbers:

    cap+ / mask+ , cap+ / mask- , cap- / mask+ , cap- / mask-

Two lifts of interest (lift = null_ce - masked_ce, positive = mask reduces CE):
  * caption-present lift  = ce(cap+,mask-) - ce(cap+,mask+)   <- what the IoU probe sees
  * caption-absent  lift  = ce(cap-,mask-) - ce(cap-,mask+)   <- where caption-dropout training
                                                                 should create a strong signal

If the caption-absent lift is clearly positive, the (frozen) LLM CAN read the prefix and the encoder
learned shape extraction. If it stays ~0, the frozen LLM can't use the prefix -> need LoRA.
"""
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from brickgpt.masking import MaskConditioningConfig, stack_views
from brickgpt.models import create_instruction
from brickgpt.models.masked_brickgpt import BrickGPTWithMask

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'output/sft_cfgdrop/mask_encoder_final.pt'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
MODEL = 'AvaLovelace/BrickGPT'
SYS = 'You are a helpful assistant.'


def build_ids(tokenizer, caption, bricks_txt):
    prompt = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': create_instruction(caption)}]
    full = prompt + [{'role': 'assistant', 'content': bricks_txt}]
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(full, tokenize=True)
    labels = list(full_ids)
    for j in range(min(len(prompt_ids), len(labels))):
        labels[j] = -100
    return torch.tensor(full_ids)[None], torch.tensor(labels)[None]


def ce(model, ids, labels, mask, has, device):
    out = model(input_ids=ids.to(device), attention_mask=torch.ones_like(ids).to(device),
                labels=labels.to(device), mask=mask.to(device), has_mask=has.to(device), use_cache=False)
    return out.loss.item()


def run_split(model, tokenizer, cfg, split, device):
    from datasets import load_dataset
    data = load_dataset('AvaLovelace/StableText2Brick', split=split)
    idxs = np.linspace(0, len(data) - 1, num=min(N, len(data)), dtype=int)
    V = cfg.num_views
    cells = {k: [] for k in ('cap+mask+', 'cap+mask-', 'cap-mask+', 'cap-mask-')}
    for i in idxs:
        row = data[int(i)]
        caption = row['captions'][0]
        bricks = row['bricks']
        m_real = torch.from_numpy(stack_views(bricks, cfg)).unsqueeze(0).float()
        m_null = torch.zeros_like(m_real)
        present = torch.ones(1, V, dtype=torch.bool)
        absent = torch.zeros(1, V, dtype=torch.bool)
        ids_c, lbl_c = build_ids(tokenizer, caption, bricks)   # caption present
        ids_e, lbl_e = build_ids(tokenizer, '', bricks)        # caption blanked
        cells['cap+mask+'].append(ce(model, ids_c, lbl_c, m_real, present, device))
        cells['cap+mask-'].append(ce(model, ids_c, lbl_c, m_null, absent, device))
        cells['cap-mask+'].append(ce(model, ids_e, lbl_e, m_real, present, device))
        cells['cap-mask-'].append(ce(model, ids_e, lbl_e, m_null, absent, device))
    means = {k: float(np.mean(v)) for k, v in cells.items()}
    cap_present_lift = np.array(cells['cap+mask-']) - np.array(cells['cap+mask+'])
    cap_absent_lift = np.array(cells['cap-mask-']) - np.array(cells['cap-mask+'])
    print(f'\n[{split}] n={len(idxs)}   mean CE per cell:')
    for k in ('cap+mask+', 'cap+mask-', 'cap-mask+', 'cap-mask-'):
        print(f'    {k} = {means[k]:.4f}')
    print(f'  caption-PRESENT mask lift = {cap_present_lift.mean():+.4f}  (helps {100*(cap_present_lift>0).mean():.0f}% of ex)')
    print(f'  caption-ABSENT  mask lift = {cap_absent_lift.mean():+.4f}  (helps {100*(cap_absent_lift>0).mean():.0f}% of ex)  <-- key')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = MaskConditioningConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = BrickGPTWithMask.from_pretrained(MODEL, cfg, torch_dtype=torch.bfloat16).to(device)
    model.mask_prefix_encoder.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f'loaded encoder from {CKPT}')
    with torch.no_grad():
        run_split(model, tokenizer, cfg, 'test', device)
        run_split(model, tokenizer, cfg, 'train', device)


if __name__ == '__main__':
    main()
