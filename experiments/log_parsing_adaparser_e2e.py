"""I (2026-11-10): AdaParser's official pipeline end to end as the system under test. Protocol: PROGRESS.md Decisions (H-M).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_adaparser_e2e.py prep
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_adaparser_e2e.py run  [--push]   # starts a local vLLM server (GPU)
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_adaparser_e2e.py hidden [--push] # D-layer states of each template's first line (GPU)
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_adaparser_e2e.py sim [--push]
  ../.venv/bin/python log_parsing_adaparser_e2e.py report [--push]                    -> reports/adaparser_e2e_details.{md,json}

Pipeline: github.com/wuyifan18/AdaParser @232951a (MIT), benchmark/logparser/AdaParser/AdaParser.py `LogParser.parse`
unchanged, run from a copy of the repo (~/.cache/adaparser_work/<model>), its README setting (log_ratio 20: the
repo's own sampled labelled candidates `full_dataset/sampled_examples_20%/<Sys>/32shot.json`, Loghub-2.0 samples of
the same system). Input `<Sys>_full.log_structured.csv` = our 50,000-line stream (LineId, Content). Its OpenAI client
reads the key and base URL from openai_key.txt; the base URL points at a local vLLM OpenAI-compatible server serving
the same model (bf16, seed 0, max_model_len 8192). Gemma's chat template has no system role: the server uses a chat
template that prepends the system text to the first user turn (our Gemma convention), the only change.
Review unit = one AdaParser output template (the lines it assigned); a review sets them to their majority ground truth.
Orderings: Random (seeds 0-4), Most frequent, D x size (seeds 0-4; zero-shot rules_template hidden state of the
template's first line, D as in F), Oracle. The pipeline exposes no log-probs or samples, so LLMLog / output entropy
are not run.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import shutil
import hashlib
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_stream as ST  # noqa: E402

MODEL = MC.MODEL
SRC = Path.home() / ".cache" / "adaparser_src" / "AdaParser"
WORK = Path.home() / ".cache" / "adaparser_work" / MODEL
DATA = EXP / "data" / "adaparser_e2e" / MODEL
RES = EXP / f"results_phase3{MC.SUFFIX}" / "adaparser_e2e"
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}[MODEL]
SYSTEMS = [s for s in ST.SYSTEMS if s not in DROP]
PORT = 8011
SERVED = f"local-{MODEL}"
GEMMA_TEMPLATE = (
    "{{ bos_token }}{% set ns = namespace(sys='', first=true) %}{% for m in messages %}{% if m['role'] == 'system' %}"
    "{% set ns.sys = m['content'] %}{% endif %}{% endfor %}{% for m in messages %}{% if m['role'] != 'system' %}"
    "{% set role = 'model' if m['role'] == 'assistant' else 'user' %}<start_of_turn>{{ role }}\n"
    "{% if ns.first and role == 'user' and ns.sys %}{{ ns.sys | trim }}\n\n{% endif %}{{ m['content'] | trim }}<end_of_turn>\n"
    "{% set ns.first = false %}{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")


def prep():
    import pandas as pd
    shutil.copytree(SRC, WORK, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
    (WORK / "openai_key.txt").write_text(f"EMPTY\nhttp://127.0.0.1:{PORT}/v1\n")
    for s in SYSTEMS:
        rows = ST.read_stretch(s)
        d = WORK / "full_dataset" / s
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"LineId": [r[0] for r in rows], "Content": [r[1] for r in rows]}).to_csv(d / f"{s}_full.log_structured.csv", index=False)
    if MODEL == "gemma":
        (WORK / "gemma_chat_template.jinja").write_text(GEMMA_TEMPLATE)
    print("prepared", WORK)


def start_server():
    import prompt_render as pr
    import urllib.request
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", HF_HUB_OFFLINE="1")
    cmd = [str(EXP.parent / ".venv-vllm" / "bin" / "vllm"), "serve", pr.MODEL_IDS[MODEL], "--served-model-name", SERVED,
           "--port", str(PORT), "--dtype", "bfloat16", "--max-model-len", "8192", "--seed", "0",
           "--gpu-memory-utilization", "0.90", "--generation-config", "vllm"]
    if MODEL == "gemma":
        cmd += ["--chat-template", str(WORK / "gemma_chat_template.jinja")]
    log = open(EXP / "logs" / f"vllm_server_{MODEL}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("vLLM server exited; see logs/vllm_server_*.log")
            time.sleep(5)
    proc.terminate()
    raise SystemExit("vLLM server did not come up")


RUNNER = """
import sys, time, json
sys.path.insert(0, {exp!r})
from adaparser_provenance import instrument
sys.path.insert(0, {bench!r})
from logparser.AdaParser.AdaParser import LogParser
p = LogParser({indir!r}, {outdir!r}, {model!r}, "20")
provenance = instrument(p)
t0 = time.perf_counter()
p.parse({log!r})
print("@@RESULT@@" + json.dumps({{"wall_s": time.perf_counter() - t0, "prompt_tokens": p.total_prompt_tokens,
                                   "completion_tokens": p.total_completion_tokens, "provenance": provenance()}}))
"""


def run(push):
    import pandas as pd
    proc = start_server()
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        for s in sorted(SYSTEMS, key=lambda s: len(ST.load(s)["first"])):
            out = DATA / f"{s}.json"
            if out.exists():
                old = json.loads(out.read_text())
                if old.get("provenance", {}).get("version") == 1:
                    continue
                backup = EXP / "data" / "adaparser_e2e_previous" / MODEL / out.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(out, backup)
                if push:
                    import hub_sync as hs
                    hs.commit_files(hs.get_results_repo(), {str(backup.relative_to(EXP)): backup}, f"preserve pre-provenance AdaParser {MODEL} {s}")
            # Prepare one input just before parsing, so the first system need not
            # wait for every raw ZIP to finish restoring.
            csv_path = WORK / "full_dataset" / s / f"{s}_full.log_structured.csv"
            if not csv_path.exists():
                for _ in range(60):
                    if (ST.RAW / f'{s}.zip').exists():
                        break
                    time.sleep(10)
                rows = ST.read_stretch(s)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({'LineId': [r[0] for r in rows], 'Content': [r[1] for r in rows]}).to_csv(csv_path, index=False)
            outdir = WORK / "result" / s
            code = RUNNER.format(exp=str(EXP), bench=str(WORK / "benchmark"), indir=str(WORK / "full_dataset" / s), outdir=str(outdir),
                                 model=SERVED, log=f"{s}_full.log")
            r = subprocess.run([str(EXP.parent / ".venv" / "bin" / "python"), "-I", "-c", code], cwd=WORK / "benchmark" / "evaluation",
                               capture_output=True, text=True)
            (EXP / "logs" / f"adaparser_e2e_{MODEL}_{s}.log").write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
            if r.returncode != 0 or "@@RESULT@@" not in r.stdout:
                print(f"{s}: FAILED (see logs/adaparser_e2e_{MODEL}_{s}.log)", flush=True)
                continue
            meta = json.loads(r.stdout.split("@@RESULT@@")[1].splitlines()[0])
            df = pd.read_csv(outdir / f"{s}_full.log_structured.csv")
            meta["queries"] = sum(1 for l in r.stdout.splitlines() if "(query times:" in l)
            meta["llm_calls"] = sum(1 for l in r.stdout.splitlines() if l.startswith(("pred (", "match (", "constant (")))
            res = {"model": MODEL, "system": s, **meta, "templates": df["EventTemplate"].fillna("").astype(str).tolist()}
            out.write_text(json.dumps(res))
            print(f"{s}: {meta['queries']} queries, {meta['wall_s']:.0f}s", flush=True)
            if push:
                import hub_sync as hs
                hs.commit_files(hs.get_results_repo(), {str(out.relative_to(EXP)): out}, f"AdaParser e2e {MODEL} {s}")
    finally:
        proc.terminate()
        proc.wait(timeout=120)


def groups(system):
    """AdaParser output templates as review groups, in first-appearance order."""
    t = json.loads((DATA / f"{system}.json").read_text())["templates"]
    ids, first, grp = {}, [], []
    for i, x in enumerate(t):
        if x not in ids:
            ids[x] = len(ids)
            first.append(i)
        grp.append(ids[x])
    return list(ids), first, grp


def hidden(push):
    import torch
    from clotho.exp_config import task2template
    D_LAYER = {"llama": 24, "gemma": 35}[MODEL]
    np.random.seed(0)
    torch.manual_seed(0)
    w = MC.wrapper()
    fn = task2template["log_parsing"]["rules_template"]
    for s in SYSTEMS:
        p = DATA / f"{s}_layer{D_LAYER}.npy"
        source = DATA / f"{s}.json"
        if not source.exists():
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        meta_path = p.with_suffix('.meta.json')
        if p.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get('source_sha256') == digest:
            continue
        rows = ST.read_stretch(s)
        _, first, _ = groups(s)
        V = []
        with torch.no_grad():
            for i in first:
                V.append(w.get_hidden_states(rows[i][1], fn, produce_inference_results=False)["hidden_states"][D_LAYER][0, -1].cpu().to(torch.float32).numpy())
        np.save(p, np.asarray(V, np.float32))
        meta_path.write_text(json.dumps({'source_sha256': digest, 'model': MODEL, 'layer': D_LAYER, 'n': len(V)}))
        print(s, len(first), flush=True)
        if push:
            import hub_sync as hs
            hs.commit_files(hs.get_results_repo(), {str(p.relative_to(EXP)): p, str(meta_path.relative_to(EXP)): meta_path}, f"AdaParser e2e hidden {MODEL} {s}")


# ---------------------------------------------------------------- simulation (log_parsing_triage machinery on AdaParser groups)

def install(system):
    """Replace the triage stream / cache / states with AdaParser's groups for `system`."""
    import log_parsing_triage as T
    from collections import Counter as C
    tmpls, first, grp = groups(system)
    base = ST.load(system)
    gt = np.asarray(base["gt"])
    G = len(tmpls)
    grp = np.asarray(grp)
    cnt = [C() for _ in range(G)]
    for g, t in zip(grp, gt):
        cnt[g][int(t)] += 1
    major = []
    for g in range(G):
        m = max(cnt[g].values())
        tied = [t for t, v in cnt[g].items() if v == m]
        major.append(next(int(t) for t in gt[grp == g] if t in tied) if len(tied) > 1 else tied[0])
    rows = ST.read_stretch(system)
    d = {"templates": base["templates"], "n": len(gt), "n_lines": len(gt), "G": G, "grp": grp, "gtl": gt, "cnt": cnt,
         "major": major, "first": np.asarray(first), "first_content": [rows[i][1] for i in first],
         "family": list(range(G)), "same": [C() for _ in range(G)]}
    D_LAYER = {"llama": 24, "gemma": 35}[MODEL]
    H = np.load(DATA / f"{system}_layer{D_LAYER}.npy")
    T.stream = lambda s: d
    T.cache_of = lambda s, v: tmpls
    T.states = lambda s, layer: H
    T.target_feats.cache_clear()
    T._SHARED.clear()
    return d, tmpls


def sim_job(system):
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    from log_parsing_adaparser_review import compute
    return compute(system, sys.modules[__name__])


def sim(push):
    from multiprocessing import Pool
    todo = [s for s in SYSTEMS if (DATA / f"{s}.json").exists() and (DATA / f"{s}_layer{24 if MODEL == 'llama' else 35}.npy").exists()]
    with Pool(min(4, len(todo)), maxtasksperchild=1) as p:      # each worker installs one system (fresh process per task)
        for r in p.imap_unordered(sim_job, todo, chunksize=1):
            print(*r, flush=True)
            if push:
                import hub_sync as hs
                q = Path(r[0])
                hs.commit_files(hs.get_results_repo(), {str(q.relative_to(EXP)): q}, f"AdaParser e2e sim {MODEL} {q.stem}")
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(), {str(p.relative_to(EXP)): p for p in RES.glob("*.json")}, f"AdaParser e2e sim {MODEL}")


def report(push=False):
    import statistics
    import log_parsing_stats as STT
    import log_parsing_strong_sim as SS
    L = ["# I: AdaParser's official pipeline as the parser under review (details)\n",
         "Generated by `log_parsing_adaparser_e2e.py report`. Protocol: PROGRESS.md Decisions (H-M). Share = (PA_N - PA_0) / "
         "(1 - PA_0), mean ± std over seeds of the macro mean over systems.\n"]
    J = {}
    for model, sfx in (("llama", ""), ("gemma", "_gemma")):
        rd = EXP / f"results_phase3{sfx}" / "adaparser_e2e"
        R = {p.stem: json.loads(p.read_text()) for p in sorted(rd.glob("*.json"))} if rd.exists() else {}
        dd = EXP / "data" / "adaparser_e2e" / model
        fails = [s for s in ST.SYSTEMS if (EXP / "logs" / f"adaparser_e2e_{model}_{s}.log").exists() and not (dd / f"{s}.json").exists()]
        if not R:
            if fails:
                L.append(f"\n## {model}: no system completed (failed: {', '.join(fails)})\n")
            continue
        J[model] = jm = {"systems": {}}
        L.append(f"\n## {model} ({len(R)} systems" + (f"; failed: {', '.join(fails)}" if fails else "") + ")\n")
        L.append("| system | groups (templates) | queries | LLM calls | wall s | PA_0 | wrong groups | gap D x size - Most frequent @20 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s, r in R.items():
            m = json.loads((dd / f"{s}.json").read_text())
            gap = SS.gap20(r) if r["pa0"] < 1 else float("nan")
            jm["systems"][s] = {"pa0": r["pa0"], "wrong": r["wrong_groups"], "G": r["G"], "queries": m["queries"], "gap20": gap}
            L.append(f"| {s} | {r['G']} | {m['queries']} | {m.get('llm_calls', '')} | {m['wall_s']:.0f} | {r['pa0']:.3f} | {r['wrong_groups']:.3f} | {gap:+.3f} |")
        L.append(f"\nMacro PA_0 {statistics.mean(r['pa0'] for r in R.values()):.3f}; macro wrong groups {statistics.mean(r['wrong_groups'] for r in R.values()):.3f}.\n")
        L.append("| ordering | " + " | ".join(f"share N={n}" for n in SS.NS) + " | " + " | ".join(f"lines/review N={n}" for n in SS.NS) + " | D x size minus row @20 |")
        L.append("|---|" + "---|" * (2 * len(SS.NS) + 1))
        d20 = SS.per_seed(R, "dsize", lambda x, r: SS.share(x, r, 20))
        for nm in ("random", "frequent", "dsize", "oracle"):
            sh = [SS.per_seed(R, nm, lambda x, r, N=N: SS.share(x, r, N)) for N in SS.NS]
            lp = [SS.per_seed(R, nm, lambda x, r, N=N: SS.lpr(x, r, N)) for N in SS.NS]
            jm[nm] = {"share": dict(zip(SS.NS, sh)), "lpr": dict(zip(SS.NS, lp))}
            mu, k = SS.signs(d20, sh[2])
            L.append(f"| {SS.NAMES[nm]} | " + " | ".join(SS.ms(v) for v in sh) + " | " + " | ".join(f"{statistics.mean(v):.0f}" for v in lp)
                     + (f" | {mu:+.3f} ({k}/5) |" if nm != "dsize" else " | |"))
        obs = {nm: [SS.share(SS.runs_of(r, nm)[k], r, 20) for r in R.values() if r["pa0"] < 1 for k in range(5)] for nm in ("random", "frequent", "dsize")}
        rk = STT.sk_esd(obs)
        L.append("\nSK-ESD @20: " + "; ".join(f"{rk[m]}. {SS.NAMES[m]} {np.mean(obs[m]):.3f}" for m in sorted(obs, key=lambda m: (rk[m], -np.mean(obs[m])))))
        a = STT.a12(obs["dsize"], obs["frequent"])
        L.append(f"\nA12 D x size vs Most frequent: {a:.2f} ({STT.magnitude(a)})")
        jm["sk"], jm["a12_vs_frequent"] = rk, a
    (EXP / "reports" / "adaparser_e2e_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "adaparser_e2e_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    from log_parsing_adaparser_review import report as review_report
    review_report(EXP, push)
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(), {f"reports/adaparser_e2e_details.{e}": EXP / "reports" / f"adaparser_e2e_details.{e}"
                                                for e in ("md", "json")}, "I AdaParser e2e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "hidden", "sim", "report", "describe"])
    ap.add_argument("--push", action="store_true")
    ap.add_argument('-s', '--systems', nargs='+')
    a = ap.parse_args()
    if a.systems:
        global SYSTEMS
        assert all(s in SYSTEMS for s in a.systems)
        SYSTEMS = a.systems
    def describe_all():
        from log_parsing_adaparser_review import describe
        for system in SYSTEMS:
            describe(system, sys.modules[__name__], a.push)
    {"describe": describe_all, "prep": lambda: prep(), "run": lambda: run(a.push), "hidden": lambda: hidden(a.push),
     "sim": lambda: sim(a.push), "report": lambda: report(a.push)}[a.cmd]()


if __name__ == "__main__":
    main()
