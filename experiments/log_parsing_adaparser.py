"""AdaParser's template corrector applied to our existing outputs (Experiment B, 2026-11-09).

  cd experiments
  CLOTHO_MODEL=llama ../.venv-vllm/bin/python log_parsing_adaparser.py correct --suite natural [--push]
  CLOTHO_MODEL=llama ../.venv-vllm/bin/python log_parsing_adaparser.py correct --suite capped  [--push]
  -> data/results_<model>/log_parsing/adaparser/rules_template/<ds>.json (+ __initial__.json, timing in each file)

Source: AdaParser (Wu, Yu, Li; ICPC 2025, arXiv 2406.03376), official code github.com/wuyifan18/AdaParser
@232951a (MIT), benchmark/logparser/AdaParser/AdaParser.py. Only the template corrector is used:
  1. template matching correction  (`post_process_nomatch`): if the template does not fullmatch the log message as a
     regex (each wildcard -> `(\\S.*){0,1}`, every other non-space run escaped), then: if the only template tokens
     missing from the message are wildcards, remove the wildcards without an LLM call (as the original does);
     otherwise one LLM call with `post_refine_prompt`.
  2. variable abstracting correction (`post_process_constant`): `get_constants` (the original heuristics: text
     captured by a wildcard that looks like a plain word, minus datetimes and `key=`/`key:` values) and, while
     constants remain, up to 3 LLM calls with `constant_refine_prompt` naming the first one (stops if the same
     constant comes back).
No parsing tree, no demonstrations, no re-parsing loop (the original's outer loop regenerates the template at
temperatures 0 / 0.5 / 1.0; that is parsing, not correction). The corrector sees exactly our current output.

Deviations from the original (also in reports/log_parsing_literature_baselines.md):
  - Placeholder: our outputs use `<*>`, AdaParser uses `{variables}`; the regexes and the two prompts use `<*>`.
  - Conversation: the original continues its own parsing conversation (system + parsing prompt + 3 demonstrations
    + log). Here the history is our rules_template prompt (the one that produced the output) and the assistant turn
    is the raw output text that produced the current template (the original writes "Log template: `T`").
  - Answer extraction: our parser (text after the last `Template:` line) first, then the original's (text between
    the first and last backtick, "Log template:" removed), else the whole response. The original's
    `post_process_template` (its own rule-based template clean-up) is not applied: it is not part of the corrector.
  - Parse failures of the original output enter the corrector with an empty template (no regex match -> LLM call).
  - Regex matching uses the `regex` module with a 2 s timeout (counted; a timeout counts as no match).
  - Model and decoding: the model that produced the outputs (vLLM, greedy = the original's first-round
    temperature 0.0, max 512 new tokens, seed 0).
Which samples: 5-9 (the samples behind the leak-free label) of every suite input, and of the 10 initial tests
(their labels enter Clotho's reference set too).
"""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
import sys
import time
from pathlib import Path

import regex as re

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import prompt_render as pr  # noqa: E402
import model_cfg as MC  # noqa: E402
from clotho.exp_config import supported_datasets, task2template  # noqa: E402
from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template  # noqa: E402

TASK, TEMPLATE, MODEL = "log_parsing", "rules_template", MC.MODEL
SUITES = EXP.parent / "clotho" / "dataset" / TASK
OUT = EXP / f"data/results_{MODEL}/log_parsing/adaparser/{TEMPLATE}"
SAMPLES = range(5, 10)
WILD = "<*>"
CAPPED = [d for d in supported_datasets[TASK] if d not in MC.dropped()]
NATURAL = [d.replace("_500", "_nat500") for d in CAPPED]
TIMEOUT = 2.0

# AdaParser.py prompts, placeholder {variables} -> <*>
CONSTANT_PROMPT = "The token {} may not be dynamic variables and do not need to be abstracted. Please provide a revised log template."
MATCH_PROMPT = ("The log template can not match the log message via regular expression. There may extra <*>, "
                "punctuations or spaces. If there are typos, do not fix it. Please provide a revised log template.")

_p_wild = re.compile(re.escape(WILD))
_p_tok = re.compile(r"[^ ]+")
_p3 = re.compile(r"^[A-Za-z\s/]*$")
_p4 = re.compile(r"(.*?:)")
_p5 = re.compile(r"/(<\*>|\w)+/?")
_KEYVAL = (r'(?<!Exception)(?<!interrupt)(?<!interrupted)(?<!thrown)(?<!failure)(?<!Error)(?<!read)(?<!Kickstart)(?<!install)(?<!because)'
           r'(?<!address)(?<!died)(?<!failed)(?<!Reason)(?<!Diagnostics)(?<!job)(?<!Responder)(?!<disconnected)(?<!answers)(?<!tftp)(?<!:)'
           r'([=:])\s*')
STATS = {"regex_timeouts": 0}


def _rex(template, wildcard):
    rex = _p_wild.sub("WILDCARD", template)
    rex = _p_tok.sub(lambda x: re.escape(x.group(0)), rex)
    return rex.replace("WILDCARD", wildcard)


def match_template(template, log):
    try:
        return re.fullmatch(_rex(template, r"(\S.*){0,1}"), log, timeout=TIMEOUT) is not None
    except TimeoutError:
        STATS["regex_timeouts"] += 1
        return False


def is_datetime_string(s):
    from dateutil import parser
    try:
        parser.parse(s)
        return True
    except (ValueError, OverflowError):
        return False
    except Exception:          # dateutil raises other errors on odd strings; the original would crash
        return False


def get_constants(template, log):
    """AdaParser.post_process_constant.get_constants, placeholder <*>."""
    if template == log:
        return []
    try:
        match = re.findall(_rex(template, "(.*)"), log, timeout=TIMEOUT)
    except TimeoutError:
        STATS["regex_timeouts"] += 1
        return []
    constants = []
    for tokens in match:
        tokens = tokens if isinstance(tokens, tuple) else [tokens]
        for token in tokens:
            if is_datetime_string(token):
                continue
            try:
                kv = re.search(_KEYVAL + re.escape(token), log, timeout=TIMEOUT)
            except TimeoutError:
                STATS["regex_timeouts"] += 1
                kv = None
            if kv and all(x not in token for x in ["exception", "Exception", "killed", "failed", "Failed", "connected", "Connection"]):
                continue
            tmp = []
            subtoken = token.split()
            for t in subtoken:
                if _p3.match(t) and not t.isupper() and not _p5.match(t):
                    tmp.append(t)
            if tmp:
                if tmp == subtoken:
                    tmp = _p4.split(' '.join(subtoken))
                    tmp = list(filter(lambda x: x != "", tmp))
                constants.extend(f"\"{t.strip()}\"" for t in tmp)
    return constants


def extract(text):
    """Our parser, then AdaParser's backtick extraction, else the whole response."""
    t, err = parse_result_messages_template(text)
    if not err:
        return t, "ours"
    s = text.replace("Log template:", "")
    a, b = s.find('`') + 1, s.rfind('`')
    if a != 0 and b != -1 and a < b:
        return s[a:b].strip("`").strip(), "backticks"
    return text.strip(), "whole"


def render(tok, messages):
    if tok.clotho_model == "gemma":
        p = tok.apply_chat_template(pr.inline_system_as_user(messages), tokenize=False, add_generation_prompt=True)
        return tok(p, add_special_tokens=False).input_ids
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=True).input_ids          # same (double BOS) convention as prompt_render.render


def load_items(ds):
    """[(input_idx, sample, log message, prompt messages, raw output)] for samples 5-9."""
    if ds == "__initial__":
        from log_parsing_initial_set import initial_dir
        from generate_initial_set_data import initial_tests
        j = json.load(open(EXP / initial_dir(MODEL, TEMPLATE) / "inference_results.json"))
        rows = [{"content": c["content"]} for c in initial_tests[TASK]]
        gts = [g if isinstance(g, list) else [g] for g in j["ground_truths"]]
        outs = j["raw_outputs"]
    else:
        rows = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
        g = json.load(open(EXP / f"data/results_{MODEL}/log_parsing/inference_results/{TEMPLATE}/{ds}_R10_T0.8.json"))
        assert len(g) == len(rows)
        gts, outs = [x["GT"] for x in g], [x["inferences"] for x in g]
    items = []
    for i, (r, o) in enumerate(zip(rows, outs)):
        msgs = task2template[TASK][TEMPLATE](r["content"])
        for k in SAMPLES:
            items.append({"i": i, "k": k, "log": r["content"].strip(), "msgs": msgs, "raw": o[k], "gt": gts[i][0]})
    return items


def correct(ds, llm, tok, push=False):
    from vllm import SamplingParams
    items = load_items(ds)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                        stop_token_ids=[pr.stop_token_id(tok)], seed=0)
    stats = {"llm_calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "regex_timeouts0": STATS["regex_timeouts"]}

    def call(batch, prompt_fn):
        """One LLM call per item in batch: history + assistant(last raw) + user(prompt). Updates T / raw."""
        if not batch:
            return
        ids = [render(tok, it["msgs"] + [{"role": "assistant", "content": it["cur_raw"]},
                                          {"role": "user", "content": prompt_fn(it)}]) for it in batch]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        for it, x, o in zip(batch, ids, outs):
            text = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True).strip()
            t, how = extract(text)
            it["calls"].append({"prompt": prompt_fn(it), "response": text, "extract": how, "template": t})
            it["T"], it["cur_raw"] = t, text
            stats["llm_calls"] += 1
            stats["prompt_tokens"] += len(x)
            stats["gen_tokens"] += len(o.outputs[0].token_ids)

    t0 = time.perf_counter()
    for it in items:
        p, err = parse_result_messages_template(it["raw"])
        it["orig"] = None if err else p
        it["T"], it["cur_raw"], it["calls"], it["steps"] = (p if not err else ""), it["raw"], [], []
    # 1. template matching correction
    need = []
    for it in items:
        if not match_template(it["T"], it["log"]):
            if set(it["T"].split()) - set(it["log"].split()) == {WILD}:
                it["T"] = it["T"].replace(WILD, "").strip()
                it["steps"].append("match_rule")
            else:
                it["steps"].append("match_llm")
                need.append(it)
    call(need, lambda it: MATCH_PROMPT)
    for it in items:
        it["flag_match"] = match_template(it["T"], it["log"])
    # 2. variable abstracting correction: up to 3 rounds
    for it in items:
        it["consts"], it["last_const"], it["retry"] = get_constants(it["T"], it["log"]), None, 3
    for _ in range(3):
        batch = []
        for it in items:
            if it["consts"] and it["retry"] > 0 and it["consts"][0] != it["last_const"]:
                it["last_const"] = it["consts"][0]
                it["retry"] -= 1
                it["steps"].append("constant_llm")
                batch.append(it)
        if not batch:
            break
        call(batch, lambda it: CONSTANT_PROMPT.format(it["last_const"]))
        for it in batch:
            it["consts"] = get_constants(it["T"], it["log"])
    for it in items:
        it["flag_constant"] = not get_constants(it["T"], it["log"]) and match_template(it["T"], it["log"])
    wall = time.perf_counter() - t0
    stats["regex_timeouts"] = STATS["regex_timeouts"] - stats.pop("regex_timeouts0")
    n_in = len({it["i"] for it in items})
    res = {"model": MODEL, "dataset": ds, "samples": list(SAMPLES), "n_inputs": n_in, "n_outputs": len(items),
           "wall_s": wall, "wall_s_per_input": wall / n_in, "wall_s_per_output": wall / len(items),
           "calls_per_output": stats["llm_calls"] / len(items), **stats,
           "items": [{k: it[k] for k in ("i", "k", "gt", "orig", "T", "steps", "calls", "flag_match", "flag_constant")}
                     for it in items]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{ds}.json"
    path.write_text(json.dumps(res))
    before = sum(it["orig"] == it["gt"] for it in items) / len(items)
    after = sum(it["T"] == it["gt"] for it in items) / len(items)
    print(f"{ds}: {len(items)} outputs, {stats['llm_calls']} calls, {wall:.0f}s, strict sample pass {before:.3f} -> {after:.3f}", flush=True)
    if push:
        import subprocess
        subprocess.run([sys.executable, str(EXP / "sharded" / "hub_sync.py"), "push", str(path.relative_to(EXP)),
                        "-m", f"adaparser corrector {MODEL} {ds}"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["correct"])
    ap.add_argument("--suite", choices=["natural", "capped"], required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--limit", type=int, help="pilot: first N datasets only")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    from vllm import LLM
    tok = pr.load_tokenizer(MODEL)
    llm = LLM(model=pr.MODEL_IDS[MODEL], dtype="bfloat16", seed=0, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm", enable_prefix_caching=True)
    ds_list = a.datasets or ((NATURAL if a.suite == "natural" else CAPPED) + ["__initial__"])
    for ds in ds_list[:a.limit] if a.limit else ds_list:
        if (OUT / f"{ds}.json").exists() and not a.redo:
            print(f"{ds}: exists", flush=True)
            continue
        correct(ds, llm, tok, push=a.push)


if __name__ == "__main__":
    main()
