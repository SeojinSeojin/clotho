import os, sys
import json
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
import openai

import anthropic
from google import genai

from exp_config import supported_datasets, task2template, task2input_key, task2answer_key_list

NUM_INFERENCE_RUNS = 10

REPEAT = 10
TEMPERATURE = 0.8
MAX_TOKENS = 512
TOP_LOGPROBS = 5
RETRY_MAX = 6
RETRY_BASE_SLEEP = 1.5


target_testsuites = {
    'syntactic_bug_detection': ['syntactic_bug_injected'],
    'spell_check': ['misspell_injected_wordnet'],
    'github_typo_check': ['github_typo_corpus_cleaned'],
    'json_repair': ['invalid_json_dataset_2166', 'invalid_json_dataset_4397'],
    'pos_detection': ['cleaned_and_sampled_pos_tags', 'cleaned_and_sampled_pos_tags_trainset'],
    'topic_classification': ['ag_news_test'],
    'adding_odd_numbers': ['integer_sequences_length_1_to_10_uniform'],
    'model_name_extraction': ['ml_arxiv_papers_no_conflicting_labels', 'synthetic_abstracts_gpt4o_3600']
}


def _gemini_retry(fn, *args, **kwargs):
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            sleep = RETRY_BASE_SLEEP ** attempt
            time.sleep(sleep)
            if attempt == RETRY_MAX:
                raise

def pack_choice_gemini(response):
    text = response.text
    
    return {
        "text": text,
        "tokens": []
    }


def _openai_retry(fn, *args, **kwargs):
    """Simple exponential backoff wrapper."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError as e:
            sleep = RETRY_BASE_SLEEP ** attempt
            time.sleep(sleep)
            if attempt == RETRY_MAX:
                raise
        except openai.APIError as e:
            # Retry transient 5xx; re-raise others
            if getattr(e, "status", None) and int(e.status) >= 500:
                sleep = RETRY_BASE_SLEEP ** attempt
                time.sleep(sleep)
                if attempt == RETRY_MAX:
                    raise
            else:
                raise

def pack_choice_openai(choice):
    text = choice.message.content or ""
    tokens = []
    if choice.logprobs and getattr(choice.logprobs, "content", None):
        for item in choice.logprobs.content:
            if hasattr(item, "token"):
                tokens.append({
                    "token": item.token,
                    "logprob": item.logprob,
                })
    return {
        "text": text,
        "tokens": tokens
    }
    

def _claude_retry(fn, *args, **kwargs):
    """Simple exponential backoff wrapper for Claude API."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except anthropic.RateLimitError:
            sleep = RETRY_BASE_SLEEP ** attempt
            time.sleep(sleep)
            if attempt == RETRY_MAX:
                raise
        except Exception as e:
            sleep = RETRY_BASE_SLEEP ** attempt
            time.sleep(sleep)
            if attempt == RETRY_MAX:
                raise

def pack_choice_claude(block):
    """Pack a single Claude text block."""
    if block.type == "text":
        return {
            "text": block.text,
            "tokens": []  # Claude는 logprobs 없음
        }
    else:
        return {
            "text": "",
            "tokens": []
        }
    
def inline_system_as_user(message):
    """
    Convert [system, user, ...] to a Claude/Gemma supported format by 
    inlining the system content into the first user turn.
    """
    if not message:
        return message
    
    if len(message) >= 2 and message[0]["role"] == "system" and message[1]["role"] == "user":
        system_content = message[0]["content"].strip()
        user_content = message[1]["content"].strip()

        merged_content = f"{system_content}\n\n{user_content}"
        new_message = [{"role": "user", "content": merged_content}]
        new_message.extend(message[2:])
        return new_message

    return message

def stringify_message(message):
    return "\n\n".join([m['content'] for m in message])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI gpt-4o-mini inference with obtaining logprobs.")
    parser.add_argument("--dataset_type", "-t", type=str, required=True, help="e.g., spell_check")
    parser.add_argument("--prompt_template", "-p", type=str, required=True, help="e.g., messages_template")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for decoding reproducibility")
    parser.add_argument("--n", type=int, default=NUM_INFERENCE_RUNS, help="Completions per input")
    parser.add_argument("--max_records", type=int, default=None, help="Debug: only process first N items")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--model", type=str, default="gpt4o_mini")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of random samples to select from the dataset for inference")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DATASET_TYPE = args.dataset_type
    
    prompt_template_name = args.prompt_template
    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")

    # Load dataset
    dataset = []
    for dataset_name in target_testsuites[DATASET_TYPE]:
        with open(f'./dataset/{DATASET_TYPE}/{dataset_name}.jsonl') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
                
    dataset_random_indices = set(range(min(len(dataset), sample_size))) if args.sample_size is not None else set(range(len(dataset)))
    
    if args.max_records is not None:
        dataset = dataset[: args.max_records]
    
    output_result_path = f'../data/results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/concat_dataset_R{REPEAT}_T{TEMPERATURE}.json'
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    
    if 'gpt' in MODEL_NAME:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(f"[OpenAI] Running {MODEL_NAME} with {args.n} Inferences & logprobs")
        print(f"Writing to: {output_result_path}")
    elif 'claude' in MODEL_NAME:
        client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    elif 'gemini' in MODEL_NAME:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Unsupported model name: {MODEL_NAME}")

    if os.path.exists(output_result_path):
        with open(output_result_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)
    else:
        all_records = []

    if len(all_records) > 0:
        existing_results_max_index = max([record['input_index'] for record in all_records])
    else:
        existing_results_max_index = -1

    for input_index, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        if input_index <= existing_results_max_index:
            print('Skipping existing result for input index:', input_index)
            continue
        
        if input_index not in dataset_random_indices:
            continue
        
        if task2input_key[DATASET_TYPE] is None:
            input_vars = data
        else:
            input_vars = data[task2input_key[DATASET_TYPE]]
        
        answer_key_list = task2answer_key_list[DATASET_TYPE]
        GT = [data[answer_key] for answer_key in answer_key_list]
        
        messages = prompt_template_func(input_vars)
        
        if 'gpt' in MODEL_NAME:
            inferences = _openai_retry(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                logprobs=True,
                **({"seed": args.seed} if args.seed is not None else {})
            )
            inferences = [pack_choice_openai(choice) for choice in inferences.choices]
        elif 'claude' in MODEL_NAME:
            inferences = []
            for i in range(args.n):
                resp = _claude_retry(
                    client.messages.create,
                    model=MODEL_NAME,
                    messages=inline_system_as_user(messages),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    **({"seed": args.seed} if args.seed is not None else {})
                )
                packed = [pack_choice_claude(block) for block in resp.content if block.type == "text"]
                inferences.append(packed[0])
        elif 'gemini' in MODEL_NAME:
            inferences = []
            for i in range(args.n):
                resp = _gemini_retry(
                    client.models.generate_content,                    
                    model=MODEL_NAME,
                    contents=stringify_message(messages),
                    config=genai.types.GenerateContentConfig(
                        temperature=args.temperature,
                        max_output_tokens=args.max_tokens,
                        **({"seed": args.seed} if args.seed is not None else {})
                    )
                )
                packed = pack_choice_gemini(resp)
                inferences.append(packed)
        else:
            raise ValueError(f"Unsupported model name: {MODEL_NAME}")

        record = {
            "input_index": input_index,
            # "input_data": input_vars,
            # "prompt_messages": messages,
            "inferences": inferences,
            "gt": GT,
        }
        all_records.append(record)
    
        with open(output_result_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)