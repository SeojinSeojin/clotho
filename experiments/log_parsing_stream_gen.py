"""LLM calls of the cached parser (TRIAGE / E, 2026-11-10): one per Drain group, on the group's first line.

  cd experiments && CLOTHO_MODEL=llama ../.venv-vllm/bin/python log_parsing_stream_gen.py [-s HDFS ...] [--push]
  -> data/stream_gen/<model>/<Sys>.json

Per first line: greedy output (rules_template, the gen_vllm renderer, max 512 tokens) with its token log-probs
(LLMLog confidence), 10 samples at T 0.8 (engine seed 1; output entropy), and the AdaParser corrector applied
to the greedy output (`log_parsing_adaparser.correct`, unchanged, fed these items). Wall-clock per template for
the greedy call and for the corrector are stored.
"""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import prompt_render as pr  # noqa: E402
import model_cfg as MC  # noqa: E402
import log_parsing_stream as ST  # noqa: E402
from clotho.exp_config import task2template  # noqa: E402
from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template  # noqa: E402

TASK, TEMPLATE, MODEL = "log_parsing", "rules_template", MC.MODEL
OUT = EXP / "data" / "stream_gen" / MODEL
N_SAMPLES = 10
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}[MODEL]
SYSTEMS = [s for s in ST.SYSTEMS if s not in DROP]


def run(system, llm, tok):
    from vllm import SamplingParams
    import log_parsing_adaparser as A
    d = ST.load(system)
    rows = [{"content": c} for c in d["first_content"]]
    gts = [d["templates"][d["gt"][i]] for i in d["first"]]
    ids = [pr.render(TASK, TEMPLATE, r, tok)[1] for r in rows]
    stop = [pr.stop_token_id(tok)]
    g = SamplingParams(n=1, temperature=0.0, max_tokens=pr.GEN_MAX_NEW_TOKENS, ignore_eos=True, stop_token_ids=stop,
                       detokenize=False, logprobs=0, seed=0)
    t0 = time.perf_counter()
    go = llm.generate([{"prompt_token_ids": x} for x in ids], g, use_tqdm=False)
    t_greedy = time.perf_counter() - t0
    s = SamplingParams(n=N_SAMPLES, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                       ignore_eos=True, stop_token_ids=stop, detokenize=False, seed=1)
    so = llm.generate([{"prompt_token_ids": x} for x in ids], s, use_tqdm=False)
    greedy, lps, samples = [], [], []
    for x, o, o2 in zip(ids, go, so):
        c = o.outputs[0]
        greedy.append(pr.postprocess(tok, x, c.token_ids))
        lps.append([float(next(iter(lp.values())).logprob) if len(lp) == 1 else float(lp[t].logprob)
                    for t, lp in zip(c.token_ids, c.logprobs)])
        samples.append([parse_result_messages_template(pr.postprocess(tok, x, cc.token_ids))[0] for cc in o2.outputs])
    parsed = [parse_result_messages_template(t)[0] for t in greedy]
    # AdaParser corrector on the greedy output (k = 0 is just an index here)
    items = [{"i": i, "k": 0, "log": r["content"].strip(), "msgs": task2template[TASK][TEMPLATE](r["content"]),
              "raw": greedy[i], "gt": gts[i]} for i, r in enumerate(rows)]
    A.load_items = lambda ds: items
    A.SAMPLES = [0]
    A.OUT = OUT / "adaparser"
    A.correct(system, llm, tok)
    ada = json.loads((A.OUT / f"{system}.json").read_text())
    corrected = [None] * len(rows)
    for it in ada["items"]:
        corrected[it["i"]] = it["T"]
    res = {"model": MODEL, "system": system, "n_groups": len(rows), "greedy_raw": greedy, "greedy": parsed,
           "token_logprobs": lps, "samples": samples, "adaparser": corrected,
           "wall_s_greedy_per_template": t_greedy / len(rows), "wall_s_adaparser_per_template": ada["wall_s_per_input"],
           "adaparser_calls_per_template": ada["calls_per_output"],
           "greedy_correct": sum(p == t for p, t in zip(parsed, gts)) / len(rows)}
    path = OUT / f"{system}.json"
    path.write_text(json.dumps(res))
    print(f"{system}: {len(rows)} groups, greedy correct {res['greedy_correct']:.3f}, "
          f"adaparser correct {sum(p == t for p, t in zip(corrected, gts)) / len(rows):.3f}, "
          f"{res['wall_s_greedy_per_template']*1000:.0f} ms/template greedy (batched)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    todo = [s for s in a.systems if not (OUT / f"{s}.json").exists()]
    if not todo:
        print("nothing to do")
        return
    from vllm import LLM
    tok = pr.load_tokenizer(MODEL)
    llm = LLM(model=pr.MODEL_IDS[MODEL], dtype="bfloat16", seed=1, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm", enable_prefix_caching=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in todo:
        p = run(s, llm, tok)
        if a.push:
            import hub_sync as hs
            files = {str(p.relative_to(EXP)): p, str((OUT / "adaparser" / f"{s}.json").relative_to(EXP)): OUT / "adaparser" / f"{s}.json"}
            hs.commit_files(hs.get_results_repo(), files, f"stream gen {MODEL} {s}")


if __name__ == "__main__":
    main()
