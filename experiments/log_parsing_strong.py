"""F: strong-parser rerun of the cached-parser simulation (2026-11-10). Protocol: PROGRESS.md Decisions (F).

  cd experiments
  CLOTHO_MODEL=llama ../.venv-vllm/bin/python log_parsing_strong.py gen --config C1 -s HDFS Spark Linux [--push]
  CLOTHO_MODEL=llama python log_parsing_strong.py choose                 # C1 / C2 rule on HDFS, Spark, Linux
  CLOTHO_MODEL=llama python log_parsing_strong.py sim [--push]          # F2 orderings, every system, seeds 0-4
  python log_parsing_strong.py report [--push]                          # both models -> reports/strong_details.{md,json}

-> data/stream_gen_strong/<model>/<config>/<Sys>.json, results_phase3{,_gemma}/strong/<Sys>.json

Parser: rules_template prompt + up to 3 kNN demonstrations from the stream's own cache (Drain position-wise token
agreement on LH2-preprocessed tokens), then the ported AdaParser corrector (log_parsing_adaparser helpers, greedy;
history = the ICL prompt + the raw output). Waves of min(32, cached groups, >= 1) groups in stream order; a wave
sees only groups cached before it. C2 adds 10 human-verified seed templates (the first 10 groups).
"""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import ast
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_stream as ST  # noqa: E402

MODEL = MC.MODEL
GEN = EXP / "data" / "stream_gen_strong" / MODEL
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "strong"
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}[MODEL]
SYSTEMS = [s for s in ST.SYSTEMS if s not in DROP]
TUNE = ("HDFS", "Spark", "Linux")
K_DEMO, WAVE_MAX, N_SEED, N_SAMPLES = 3, 32, 10, 10
TARGET = 0.75


def lh2_regex():
    """LH2 regex lists read from log_parsing_drain.py without importing it (pandas is not in .venv-vllm)."""
    import re
    tree = ast.parse((EXP / "log_parsing_drain.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LH2":
            lh2 = ast.literal_eval(node.value)
            return {s: [re.compile(r) for r in v["regex"]] for s, v in lh2.items()}
    raise KeyError("LH2")


def tokens(content, rex):
    for r in rex:
        content = r.sub("<*>", content)
    return content.strip().split()


def sim(a, b):
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b))


def icl_messages(content, demos):
    from clotho.exp_config import task2template
    msgs = task2template["log_parsing"]["rules_template"](content)
    if demos:
        block = "Examples of log messages from this system and their templates:\n\n" + "\n\n".join(
            f"Log message: {x}\nTemplate: {t}" for x, t in demos) + "\n\nNow parse this log message.\n\n"
        u = msgs[1]["content"]
        i = u.rindex("Log message:")
        msgs = [msgs[0], {"role": "user", "content": u[:i] + block + u[i:]}]
    return msgs


# ---------------------------------------------------------------- corrector (Experiment B code paths, items in memory)

def correct_items(items, llm, tok, stats):
    """items: dicts with log, msgs, raw. Sets it['T'] (corrected template) and it['orig']."""
    from vllm import SamplingParams
    import prompt_render as pr
    import log_parsing_adaparser as A
    from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                        stop_token_ids=[pr.stop_token_id(tok)], seed=0)

    def call(batch, prompt_fn):
        if not batch:
            return
        ids = [A.render(tok, it["msgs"] + [{"role": "assistant", "content": it["cur_raw"]},
                                            {"role": "user", "content": prompt_fn(it)}]) for it in batch]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        for it, o in zip(batch, outs):
            text = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True).strip()
            t, _ = A.extract(text)
            it["T"], it["cur_raw"] = t, text
            it["steps"].append(prompt_fn(it)[:20])
            stats["corrector_calls"] += 1

    for it in items:
        p, err = parse_result_messages_template(it["raw"])
        it["orig"] = None if err else p
        it["T"], it["cur_raw"], it["steps"] = (p if not err else ""), it["raw"], []
    need = []
    for it in items:
        if not A.match_template(it["T"], it["log"]):
            if set(it["T"].split()) - set(it["log"].split()) == {A.WILD}:
                it["T"] = it["T"].replace(A.WILD, "").strip()
            else:
                need.append(it)
    call(need, lambda it: A.MATCH_PROMPT)
    for it in items:
        it["consts"], it["last_const"], it["retry"] = A.get_constants(it["T"], it["log"]), None, 3
    for _ in range(3):
        batch = []
        for it in items:
            if it["consts"] and it["retry"] > 0 and it["consts"][0] != it["last_const"]:
                it["last_const"] = it["consts"][0]
                it["retry"] -= 1
                batch.append(it)
        if not batch:
            break
        call(batch, lambda it: A.CONSTANT_PROMPT.format(it["last_const"]))
        for it in batch:
            it["consts"] = A.get_constants(it["T"], it["log"])


# ---------------------------------------------------------------- generation

def gen_system(system, config, llm, tok, rex):
    from vllm import SamplingParams
    import prompt_render as pr
    import log_parsing_adaparser as A
    from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template
    d = ST.load(system)
    content = d["first_content"]
    G = len(content)
    gts = [d["templates"][d["gt"][i]] for i in d["first"]]
    toks = [tokens(c, rex) for c in content]
    stop = [pr.stop_token_id(tok)]
    gsp = SamplingParams(n=1, temperature=0.0, max_tokens=pr.GEN_MAX_NEW_TOKENS, stop_token_ids=stop,
                         logprobs=0, seed=0)
    cache = [None] * G
    raw, orig, lps, demos_of, prompts = [None] * G, [None] * G, [None] * G, [None] * G, [None] * G
    stats = {"corrector_calls": 0, "waves": 0}
    seeds = list(range(min(N_SEED, G))) if config == "C2" else []
    for g in seeds:
        cache[g] = gts[g]
    done = list(seeds)
    t0 = time.perf_counter()
    g = len(seeds)
    while g < G:
        w = min(WAVE_MAX, max(1, len(done)))
        wave = list(range(g, min(G, g + w)))
        items = []
        for q in wave:
            sc = sorted(((sim(toks[q], toks[j]), j) for j in done), key=lambda x: (-x[0], -x[1]))
            picks = [j for s, j in sc[:K_DEMO] if s > 0][::-1]          # least similar first
            demos_of[q] = picks
            msgs = icl_messages(content[q], [(content[j], cache[j]) for j in picks])
            prompts[q] = msgs
            items.append({"q": q, "log": content[q].strip(), "msgs": msgs})
        ids = [A.render(tok, it["msgs"]) for it in items]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], gsp, use_tqdm=False)
        for it, o in zip(items, outs):
            c = o.outputs[0]
            it["raw"] = tok.decode(c.token_ids, skip_special_tokens=True).strip()
            lps[it["q"]] = [float(lp[t].logprob) for t, lp in zip(c.token_ids, c.logprobs)]
        correct_items(items, llm, tok, stats)
        for it in items:
            q = it["q"]
            raw[q], orig[q], cache[q] = it["raw"], it["orig"], it["T"]
        done += wave
        g = wave[-1] + 1
        stats["waves"] += 1
    wall = time.perf_counter() - t0
    # 10 samples of the same ICL prompt (output entropy), after the sequential pass
    todo = [q for q in range(G) if prompts[q] is not None]
    ssp = SamplingParams(n=N_SAMPLES, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                         stop_token_ids=stop, seed=1)
    so = llm.generate([{"prompt_token_ids": A.render(tok, prompts[q])} for q in todo], ssp, use_tqdm=False)
    samples = [None] * G
    for q, o in zip(todo, so):
        samples[q] = [parse_result_messages_template(tok.decode(c.token_ids, skip_special_tokens=True).strip())[0]
                      for c in o.outputs]
    final_ok = [c == t for c, t in zip(cache, gts)]
    res = {"model": MODEL, "system": system, "config": config, "n_groups": G, "seeds": seeds,
           "final": cache, "greedy": [o if o is not None else "" for o in orig], "greedy_raw": raw,
           "token_logprobs": lps, "samples": samples, "demos": demos_of,
           "wall_s": wall, "wall_s_per_template": wall / max(1, len(todo)), **stats,
           "final_correct_groups": sum(final_ok) / G,
           "greedy_correct_groups": sum((o or "") == t for o, t in zip(orig, gts) if o is not None) / max(1, len(todo))}
    path = GEN / config / f"{system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    pa0 = sum(1 for i, gg in enumerate(d["group"]) if cache[gg] == d["templates"][d["gt"][i]]) / len(d["group"])
    print(f"{system} {config}: {G} groups, correct groups {res['final_correct_groups']:.3f} "
          f"(ICL before corrector {res['greedy_correct_groups']:.3f}), PA_0 {pa0:.3f}, "
          f"{wall:.0f}s, {stats['waves']} waves", flush=True)
    return path


def cmd_gen(a):
    from vllm import LLM
    import prompt_render as pr
    todo = [s for s in a.systems if not (GEN / a.config / f"{s}.json").exists()]
    if not todo:
        print("nothing to do")
        return
    tok = pr.load_tokenizer(MODEL)
    llm = LLM(model=pr.MODEL_IDS[MODEL], dtype="bfloat16", seed=1, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm", enable_prefix_caching=True)
    rex = lh2_regex()
    for s in sorted(todo, key=lambda s: len(ST.load(s)["first"])):
        p = gen_system(s, a.config, llm, tok, rex[s])
        if a.push:
            import hub_sync as hs
            hs.commit_files(hs.get_results_repo(), {str(p.relative_to(EXP)): p}, f"strong parser {MODEL} {a.config} {s}")


# ---------------------------------------------------------------- PA_0 and the C1/C2 rule

def pa0_of(system, config, model=None):
    d = ST.load(system)
    gen = GEN if model is None else EXP / "data" / "stream_gen_strong" / model
    cache = json.loads((gen / config / f"{system}.json").read_text())["final"]
    gts = [d["templates"][t] for t in d["gt"]]
    pa = sum(cache[g] == t for g, t in zip(d["group"], gts)) / len(gts)
    wrong = sum(cache[g] != d["templates"][d["gt"][i]] for g, i in enumerate(d["first"])) / len(d["first"])
    return pa, wrong


def cmd_choose(a):
    have = {c: all((GEN / c / f"{s}.json").exists() for s in TUNE) for c in ("C1", "C2")}
    m = {c: sum(pa0_of(s, c)[0] for s in TUNE) / len(TUNE) for c in have if have[c]}
    print("mean PA_0 on HDFS, Spark, Linux:", m)
    if "C1" not in m:
        print("run C1 on the tuning systems first")
        return
    if m["C1"] >= TARGET:
        ch = "C1"
    elif "C2" not in m:
        print("C1 below target: run C2 on the tuning systems")
        return
    else:
        ch = max(m, key=m.get)
    (GEN / "chosen.json").write_text(json.dumps({"mean_pa0_tune": m, "chosen": ch}, indent=1))
    print("chosen", ch)


def chosen():
    return json.loads((GEN / "chosen.json").read_text())["chosen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "choose", "sim", "report"])
    ap.add_argument("--config", choices=["C1", "C2"])
    ap.add_argument("-s", "--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("-w", "--workers", type=int, default=14)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        a.config = a.config or chosen()
        cmd_gen(a)
    elif a.cmd == "choose":
        cmd_choose(a)
    elif a.cmd == "sim":
        import log_parsing_strong_sim as S
        S.run(a.systems, a.workers, a.push)
    else:
        import log_parsing_strong_sim as S
        S.report(a.push)


if __name__ == "__main__":
    main()
