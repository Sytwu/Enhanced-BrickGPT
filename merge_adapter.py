"""
Merge a saved LoRA adapter into the base model and save the result.

Usage:
    uv run merge_adapter.py \
        --base AvaLovelace/BrickGPT \
        --adapter output/sft_text_mask/adapter_final \
        --output  output/sft_text_mask/merged
"""
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='AvaLovelace/BrickGPT')
    parser.add_argument('--adapter', default='output/sft_text_mask/adapter_final')
    parser.add_argument('--output', default='output/sft_text_mask/merged')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    from peft import PeftModel

    logger.info('Loading base model from %s', args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    # BrickGPT loads as LlamaForCausalLM (already merged) — no double-PEFT issue.
    logger.info('Base model type: %s', type(base).__name__)

    logger.info('Loading adapter from %s', args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)

    logger.info('Merging LoRA weights...')
    merged = model.merge_and_unload()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained re-injects the 'base_model.model.' PEFT prefix even after merge_and_unload
    # because the model still carries PEFT metadata. Bypass it entirely: save config and
    # weights separately so the checkpoint has standard Llama key names.
    state_dict = merged.state_dict()
    peft_prefix = 'base_model.model.'
    if any(k.startswith(peft_prefix) for k in state_dict):
        logger.info('Stripping PEFT key prefix from state dict...')
        state_dict = {(k[len(peft_prefix):] if k.startswith(peft_prefix) else k): v
                      for k, v in state_dict.items()}

    merged.config.save_pretrained(out_dir)
    if hasattr(merged, 'generation_config'):
        try:
            merged.generation_config.save_pretrained(out_dir)
        except Exception:
            pass

    try:
        from safetensors.torch import save_file as _sf_save
        _sf_save({k: v.contiguous().cpu() for k, v in state_dict.items()},
                 str(out_dir / 'model.safetensors'))
        logger.info('Saved weights as model.safetensors')
    except ImportError:
        torch.save(state_dict, out_dir / 'pytorch_model.bin')
        logger.info('Saved weights as pytorch_model.bin (safetensors not available)')

    logger.info('Saving tokenizer...')
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out_dir)

    logger.info('Done. Run eval with:')
    logger.info('  uv run eval_per_view_iou.py --model %s --data dataset01/mask01/test_masks.json', out_dir)


if __name__ == '__main__':
    main()
