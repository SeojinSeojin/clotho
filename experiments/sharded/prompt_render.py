"""Prompt rendering and output post-processing for the vLLM path.

This mirrors clotho/models/llama.py (and gemma.py) line by line, so vLLM sees exactly the
token ids that HF generate sees. check_prompt_parity.py verifies this against the real
llama.py code path (not against this file).

Kept free of torch/vllm imports so it runs in either venv.
"""
from clotho.exp_config import task2input_key, task2template

MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "gemma": "google/gemma-2-9b-it"}

# Generation settings of the HF path:
#   REPEAT, TEMPERATURE    -> experiments/data_extraction/2-1_generate_outputs_and_record_hidden_states.py
#   MAX_NEW_TOKENS         -> clotho/models/<model>.py
# check_prompt_parity.py fails if these drift.
GEN_REPEAT = 10
GEN_TEMPERATURE = 0.8
GEN_MAX_NEW_TOKENS = 512

LLAMA_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


def load_tokenizer(model):
    if model not in MODEL_IDS:
        raise NotImplementedError(
            f"vLLM path mirrors clotho/models/llama.py and gemma.py only; '{model}' renders prompts "
            f"and post-processes outputs differently (see clotho/models/{model}.py).")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_IDS[model], trust_remote_code=True)  # llama.py:34 / gemma.py:40
    if model == "llama":
        tok.add_special_tokens({"pad_token": "<pad>"})                             # llama.py:35
    tok.clotho_model = model
    return tok


def stop_token_id(tok):
    """Token the HF 2-1 loop stops on: llama tokenizer.eos_token_id (<|eot_id|>), gemma <end_of_turn>."""
    if tok.clotho_model == "gemma":
        return tok.convert_tokens_to_ids("<end_of_turn>")
    return tok.eos_token_id


def input_variables(task, data):
    key = task2input_key[task]
    return data if key is None else data[key]


def inline_system_as_user(message):
    """Copy of clotho/models/gemma.py:inline_system_as_user (that module loads the model on import)."""
    if not message:
        return message
    if len(message) >= 2 and message[0]["role"] == "system" and message[1]["role"] == "user":
        merged = f"{message[0]['content'].strip()}\n\n{message[1]['content'].strip()}"
        return [{"role": "user", "content": merged}] + list(message[2:])
    return message


def render(task, template_name, data, tok):
    """-> (prompt_string, input_ids) exactly as the HF wrapper's 2-1 path builds them."""
    messages = task2template[task][template_name](input_variables(task, data))
    if tok.clotho_model == "gemma":
        # gemma.py:98-101: system inlined into the user turn; the chat template already has <bos>
        # and gemma.py tokenizes with add_special_tokens=False (single BOS).
        prompt = tok.apply_chat_template(inline_system_as_user(messages), tokenize=False, add_generation_prompt=True)
        return prompt, tok(prompt, add_special_tokens=False).input_ids
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # NB: the chat template already starts with <|begin_of_text|> and llama.py
    # tokenizes with add_special_tokens=True, so the HF path has a *double* BOS.
    # We keep it deliberately so both paths condition on identical ids.
    ids = tok(prompt, add_special_tokens=True).input_ids
    return prompt, ids


def _gemma_decode_clean(tok, ids):
    """gemma.py:_decode_clean, line by line (decodes prompt + output, then splits at the first 'model\\n')."""
    text = tok.decode(ids, skip_special_tokens=True)
    if not text.strip():
        raw = tok.decode(ids, skip_special_tokens=False)
        for sp in tok.all_special_tokens:
            raw = raw.replace(sp, "")
        text = raw.strip()
    if "model\n" in text:
        text = text.split("model\n", 1)[1].strip()
    elif LLAMA_ASSISTANT_HEADER in text:
        text = text.split(LLAMA_ASSISTANT_HEADER, 1)[1].strip()
    return text


def postprocess(tok, prompt_ids, gen_ids):
    """Generated ids -> inference string exactly as the HF wrapper (llama.py:192-201, gemma.py:166-167)."""
    if tok.clotho_model == "gemma":
        return _gemma_decode_clean(tok, list(prompt_ids) + list(gen_ids))
    text = tok.decode(list(prompt_ids) + list(gen_ids), skip_special_tokens=False)
    if LLAMA_ASSISTANT_HEADER in text:
        text = text.split(LLAMA_ASSISTANT_HEADER)[-1]
        text = text.replace("<|eot_id|>", "").strip()
    else:
        print("Warning: Generated text does not contain assistant header.")
        print(f"*** Generated text: {text}")
    return text
