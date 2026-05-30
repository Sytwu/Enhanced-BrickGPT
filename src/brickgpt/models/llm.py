import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class LLM:
    """
    A small wrapper class for a language model.
    """

    def __init__(self, model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.kv_cache = None
        self.kv_cache_saved = None
        self.input_ids_cache = None
        self.input_ids_cache_saved = None

    @classmethod
    def from_model(cls, model, tokenizer, device: str) -> 'LLM':
        """
        Wraps an already-loaded model + tokenizer (no ``from_pretrained`` / disk load).

        Used to drive the constrained generation loop from an in-memory training model -- e.g. the
        SFT IoU probe and GRPO rollouts generate from ``BrickGPTWithMask.base``.
        """
        self = cls.__new__(cls)
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.kv_cache = None
        self.kv_cache_saved = None
        self.input_ids_cache = None
        self.input_ids_cache_saved = None
        return self

    def __call__(
            self,
            prompt: str | torch.Tensor | None = None,
            return_as_ids: bool = False,
            return_dict: bool = False,
            **kwargs,
    ):
        """
        Generates text, given a prompt.
        """

        # If prompt is None, continue generation from previously generated tokens
        if prompt is None:
            prompt = self.input_ids_cache
        else:
            self.reset_cache()

        # If prompt is a string, encode it into token ids
        if isinstance(prompt, str):
            encoded_input = self.tokenizer(prompt, return_tensors='pt')
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
        else:
            input_ids = prompt.to(self.device)
            attention_mask = torch.ones_like(input_ids)

        # Run generation
        output_dict = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            num_return_sequences=1,
            past_key_values=self.kv_cache,
            return_dict_in_generate=True,
            **kwargs,
        )
        self.input_ids_cache = output_dict['sequences']

        # Return result as token ids or as a string
        input_length = input_ids.shape[1]
        result_ids = output_dict['sequences'][0][input_length:]
        result = result_ids if return_as_ids else self.tokenizer.decode(result_ids)

        return (result, output_dict) if return_dict else result

    def prefill_with_embeds(
            self,
            prefix_embeds: torch.Tensor,
            prompt_ids: torch.Tensor,
    ) -> None:
        """
        Seeds the KV-cache with mask *prefix embeddings* followed by the text prompt, so that the
        existing token-by-token (logit-masking) generation loop can run with mask conditioning baked
        into the cache. After this call, generation continues with ``prompt=None`` (continuation).

        The cache is left in the same steady state the normal loop maintains, namely
        ``kv_cache_len == input_ids_cache_len - 1``: we process ``prefix + prompt[:-1]`` here (cache
        length ``T + P - 1``) and stash placeholder ids for the ``T`` prefix slots plus the full
        prompt in ``input_ids_cache`` (length ``T + P``). The *last* prompt token is therefore
        consumed by the first generation step -- exactly as a freshly-sampled token is consumed on
        each continuation step. The prefix occupies positions ``0..T-1``, matching training
        (``inputs_embeds = cat(prefix, text)``), so RoPE positions are consistent.

        :param prefix_embeds: Mask prefix embeddings of shape ``(1, T, hidden_size)``.
        :param prompt_ids: The tokenized prompt of shape ``(1, P)`` (or ``(P,)``).
        """
        self.reset_cache()
        prompt_ids = prompt_ids.to(self.device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        num_prefix = prefix_embeds.shape[1]

        embed = self.model.get_input_embeddings()
        prefix_embeds = prefix_embeds.to(device=self.device, dtype=embed.weight.dtype)
        text_embeds = embed(prompt_ids[:, :-1])
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)  # T + (P - 1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        self.kv_cache = out.past_key_values

        # Placeholder ids for the T prefix positions; never re-embedded (they live in the cache),
        # and sliced off output by `input_length` in __call__, so the exact value is irrelevant.
        placeholder = torch.zeros((1, num_prefix), dtype=prompt_ids.dtype, device=self.device)
        self.input_ids_cache = torch.cat([placeholder, prompt_ids], dim=1)  # T + P

    def reset_cache(self) -> None:
        self.kv_cache = DynamicCache()

    def save_state(self) -> None:
        self.kv_cache_saved = copy.deepcopy(self.kv_cache)
        self.input_ids_cache_saved = self.input_ids_cache

    def rollback_to_saved_state(self) -> None:
        self.kv_cache = self.kv_cache_saved
        self.input_ids_cache = self.input_ids_cache_saved
