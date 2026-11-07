"""Prompt rendering and output post-processing for the vLLM path.

This mirrors clotho/models/llama.py line by line, so vLLM sees exactly the
token ids that HF generate sees. check_prompt_parity.py verifies this against the real
llama.py code path (not against this file).

Kept free of torch/vllm imports so it runs in either venv.
"""
from clotho.exp_config import task2input_key, task2template

MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct"}

# Generation settings of the HF path:
#   REPEAT, TEMPERATURE    -> experiments/data_extraction/2-1_generate_outputs_and_record_hidden_states.py
#   MAX_NEW_TOKENS         -> clotho/models/llama.py
# check_prompt_parity.py fails if these drift.
GEN_REPEAT = 10
GEN_TEMPERATURE = 0.8
GEN_MAX_NEW_TOKENS = 512

LLAMA_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


def load_tokenizer(model):
    if model not in MODEL_IDS:
        raise NotImplementedError(
            f"vLLM path only mirrors clotho/models/llama.py so far; '{model}' renders prompts "
            f"and post-processes outputs differently (see clotho/models/{model}.py).")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_IDS[model], trust_remote_code=True)  # llama.py:34
    tok.add_special_tokens({"pad_token": "<pad>"})                                 # llama.py:35
    return tok


def input_variables(task, data):
    key = task2input_key[task]
    return data if key is None else data[key]


def render(task, template_name, data, tok):
    """-> (prompt_string, input_ids) exactly as llama.py:119-126."""
    messages = task2template[task][template_name](input_variables(task, data))
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # NB: the chat template already starts with <|begin_of_text|> and llama.py
    # tokenizes with add_special_tokens=True, so the HF path has a *double* BOS.
    # We keep it deliberately so both paths condition on identical ids.
    ids = tok(prompt, add_special_tokens=True).input_ids
    return prompt, ids


def postprocess(tok, prompt_ids, gen_ids):
    """Generated ids -> inference string exactly as llama.py:192-201."""
    text = tok.decode(list(prompt_ids) + list(gen_ids), skip_special_tokens=False)
    if LLAMA_ASSISTANT_HEADER in text:
        text = text.split(LLAMA_ASSISTANT_HEADER)[-1]
        text = text.replace("<|eot_id|>", "").strip()
    else:
        print("Warning: Generated text does not contain assistant header.")
        print(f"*** Generated text: {text}")
    return text
