"""
Offline tests for the mask-conditioned generation path (Step B): ``LLM.prefill_with_embeds``.

These build a *real but tiny* ``LlamaForCausalLM`` from a config (no network / no download), so the
genuine ``DynamicCache`` + RoPE + ``generate`` continuation machinery is exercised, while staying
fully offline like the rest of the suite.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from brickgpt.models.llm import LLM


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return ''


def _tiny_llm(seed: int = 0) -> LLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config).eval()
    llm = LLM.__new__(LLM)        # bypass from_pretrained (no download)
    llm.device = 'cpu'
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.kv_cache = None
    llm.kv_cache_saved = None
    llm.input_ids_cache = None
    llm.input_ids_cache_saved = None
    return llm


def test_prefill_matches_full_forward():
    """Prefill(prefix + prompt[:-1]) then one step on the last prompt token == one full forward."""
    llm = _tiny_llm()
    model = llm.model
    embed = model.get_input_embeddings()
    torch.manual_seed(1)

    T, P = 6, 9
    prefix = torch.randn(1, T, model.config.hidden_size) * 0.02
    prompt_ids = torch.randint(2, 128, (1, P))

    with torch.no_grad():
        ref = model(inputs_embeds=torch.cat([prefix, embed(prompt_ids)], dim=1)).logits[:, -1]

    llm.prefill_with_embeds(prefix, prompt_ids)
    assert llm.kv_cache.get_seq_length() == T + P - 1          # cache holds prefix + prompt[:-1]
    assert llm.input_ids_cache.shape[1] == T + P               # placeholders(T) + prompt(P)

    with torch.no_grad():
        out = model(input_ids=prompt_ids[:, -1:], past_key_values=llm.kv_cache, use_cache=True)
    test = out.logits[:, -1]

    assert torch.allclose(ref, test, atol=1e-4)
    assert torch.equal(ref.argmax(-1), test.argmax(-1))


def test_prefill_then_generate_continues_through_cache():
    """After prefill, the standard continuation loop (prompt=None) generates tokens and grows the cache."""
    llm = _tiny_llm()
    T, P = 5, 7
    prefix = torch.randn(1, T, llm.model.config.hidden_size) * 0.02
    prompt_ids = torch.randint(2, 128, (1, P))

    llm.prefill_with_embeds(prefix, prompt_ids)

    for step in range(4):
        kv_before = llm.kv_cache.get_seq_length()
        tok = llm(None, return_as_ids=True, max_new_tokens=1)
        assert tok.shape == (1,)                               # exactly one new token
        # Invariant maintained: cache grows by one and stays one behind input_ids_cache.
        assert llm.kv_cache.get_seq_length() == kv_before + 1
        assert llm.kv_cache.get_seq_length() == llm.input_ids_cache.shape[1] - 1


def test_prefill_save_and_rollback():
    """Rejection sampling rewinds the cache; rollback must restore the prefilled state exactly."""
    llm = _tiny_llm()
    T, P = 4, 6
    prefix = torch.randn(1, T, llm.model.config.hidden_size) * 0.02
    prompt_ids = torch.randint(2, 128, (1, P))

    llm.prefill_with_embeds(prefix, prompt_ids)
    llm.save_state()
    len_before = llm.kv_cache.get_seq_length()

    llm(None, return_as_ids=True, max_new_tokens=1)
    assert llm.kv_cache.get_seq_length() == len_before + 1

    llm.rollback_to_saved_state()
    assert llm.kv_cache.get_seq_length() == len_before
    assert llm.input_ids_cache.shape[1] == T + P
