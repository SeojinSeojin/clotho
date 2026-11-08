"""Natural-distribution check: 500 messages per system sampled uniformly from the raw log lines (no template cap).

  cd experiments
  python log_parsing_natural.py sample                         # -> clotho/dataset/log_parsing/loghub_<sys>_nat500.jsonl
  ../.venv-vllm/bin/python log_parsing_natural.py gen          # 10 samples, vLLM, same settings as gen_vllm (seed 0)
  python log_parsing_layers.py -d <nat datasets> --layers 16 --skip_initial
  python log_parsing_postgen.py -d <nat datasets> --no_se
  python log_parsing_natural.py eval [--push]                  # -> reports/log_parsing_natural.{md,json}

Systems: HDFS, Hadoop, OpenStack, BGL. Sampling: every line of <System>_full.log_structured.csv (Loghub-2.0) whose
Content is not in the current 500-message suite is a candidate; 500 lines are drawn uniformly without
replacement (reservoir sampling, random.Random(4242)). No deduplication: repeated messages and frequent
templates appear at their natural rate, which is the point of the check. The suites (loghub_<sys>_500) instead
deduplicate and cap inputs per template.
These datasets are deliberately NOT registered in clotho.exp_config.supported_datasets (the Phase 3 scripts take
their system list from it); the generic scripts above accept any dataset name.

Eval: main setting fair_leakfree on the standard split (balanced sampling, layer 16, label from samples 5-9,
post-generation scores from samples 0-4), 5 seeds, ref 100/200; Clotho, MDSA, GMMb, Lookup (Drain, Loghub-2.0
settings, run on the 500 sampled messages), Lookup (oracle), output entropy, LOHSvar, Tok-Prob. Compared with
the same four systems' suite results (stored fair_leakfree jobs). A system is skipped if it has < 15 PASS or
< 15 FAIL inputs.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
import random
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
SUITES = EXP.parent / "clotho" / "dataset" / "log_parsing"
SYSTEMS = {"HDFS": "hdfs", "Hadoop": "hadoop", "OpenStack": "openstack", "BGL": "bgl"}
NAT = {k: f"loghub_{v}_nat500" for k, v in SYSTEMS.items()}
SUITE = {k: f"loghub_{v}_500" for k, v in SYSTEMS.items()}
TEMPLATE = "rules_template"
OUT = EXP / "results_phase3" / "natural"
REFS = (100, 200)
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["clotho", "MDSA", "GMMb", "lookup_drain", "lookup_oracle", "out_entropy", "loh_var", "tok_prob"]
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "GMMb": "GMMb", "lookup_drain": "Lookup (Drain)",
         "lookup_oracle": "Lookup (oracle)", "out_entropy": "Output entropy*", "loh_var": "LOHSvar*",
         "tok_prob": "Tok-Prob*"}


def sample():
    sys.path.insert(0, str(SUITES))
    import prepare_loghub as P
    stats = {}
    for name, ds in NAT.items():
        suite = {json.loads(l)["content"] for l in open(SUITES / f"{SUITE[name]}.jsonl")}
        rng = random.Random(4242)
        res, n = [], 0
        for row in P.stream_rows(P.fetch(name, Path.home() / ".cache" / "loghub2"), name):
            if row["Content"] in suite:
                continue
            n += 1
            item = (row["LineId"], row["Content"], row["EventTemplate"], row["EventId"])
            if len(res) < 500:
                res.append(item)
            else:
                j = rng.randrange(n)
                if j < 500:
                    res[j] = item
        rng.shuffle(res)
        with open(SUITES / f"{ds}.jsonl", "w") as f:
            for lid, c, t, e in res:
                f.write(json.dumps({"id": f"{name.lower()}-{lid}", "system": name, "content": c, "template": t,
                                    "template_id": e}) + "\n")
        tc = Counter(t for _, _, t, _ in res)
        stats[name] = {"candidate_lines": n, "templates": len(tc), "top_template_share": tc.most_common(1)[0][1] / 500,
                       "duplicate_messages": 500 - len({c for _, c, _, _ in res})}
        print(name, stats[name], flush=True)
    (SUITES / "natural_sample_stats.json").write_text(json.dumps(stats, indent=1))


def gen():
    from vllm import LLM, SamplingParams
    import prompt_render as pr
    from clotho.exp_config import task2answer_key_list
    tok = pr.load_tokenizer("llama")
    llm = LLM(model=pr.MODEL_IDS["llama"], dtype="bfloat16", seed=0, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm")
    sp = SamplingParams(n=pr.GEN_REPEAT, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                        ignore_eos=True, stop_token_ids=[tok.eos_token_id], detokenize=False)
    od = EXP / f"data/results_llama/log_parsing/inference_results/{TEMPLATE}"
    for ds in NAT.values():
        dest = od / f"{ds}_R10_T0.8.json"
        if dest.exists():
            continue
        rows = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
        ids = [pr.render("log_parsing", TEMPLATE, r, tok)[1] for r in rows]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=True)
        recs = [{"inferences": [pr.postprocess(tok, x, c.token_ids) for c in o.outputs],
                 "GT": [r[k] for k in task2answer_key_list["log_parsing"]]} for r, x, o in zip(rows, ids, outs)]
        dest.write_text(json.dumps(recs))
        print("wrote", dest, flush=True)


def build(ds, system):
    import numpy as np
    import pandas as pd
    import log_parsing_phase3_final as F
    import log_parsing_drain as D
    from clotho.dataset.log_parsing.output_parsers import score_output
    src = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
    g = json.load(open(EXP / f"data/results_llama/log_parsing/inference_results/{TEMPLATE}/{ds}_R10_T0.8.json"))
    sc_ = [[score_output(o, x["GT"], TEMPLATE) for o in x["inferences"]] for x in g]
    P = np.array([[s["strict_match"] for s in row] for row in sc_], dtype=bool)
    parsed = [[s["parsed"] for s in row] for row in sc_]
    H = F._hidden(ds, 16)
    Hi, Pi = F.initial_data(16)
    with tempfile.TemporaryDirectory() as d:
        drain = [f"drain:{x}" for x in D.drain_lh2([r["content"] for r in src], system, d)]
    pseudo = [f"__initial_{i}__" for i in range(len(Hi))]
    post = pd.read_pickle(EXP / f"data/results_llama/log_parsing/precalculated_metrics/{TEMPLATE}/{ds}_postgen.pkl").sort_values("input_index")
    n = len(H)
    return {"H": np.concatenate([H, Hi]), "sc": np.concatenate([P[:, 5:].mean(1), Pi[:, 5:].mean(1)]), "n": n,
            "init_idx": np.arange(n, n + len(Hi)), "templates": [r["template"] for r in src] + pseudo,
            "drain": drain + pseudo,
            "post": {"out_entropy": -np.array([F._entropy(p[:5]) for p in parsed]),
                     "loh_var": -post["loh_var_s04"].to_numpy(), "tok_prob": post["tok_prob"].to_numpy()}}


def job(args):
    import logging
    import numpy as np
    import log_parsing_phase3_lib as L
    system, seed = args
    ds = NAT[system]
    out = OUT / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    B = build(ds, system)
    H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
    y_all = sc[idx] > L.PASS_THRESH
    info = {"system": system, "seed": seed, "n_pass": int(y_all.sum())}
    if y_all.sum() < 15 or (~y_all).sum() < 15:
        info["skipped"] = f"{int(y_all.sum())} PASS / {int((~y_all).sum())} FAIL"
    else:
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init_idx, idx, None, seed, method="balanced")
        info["rows"], info["runs"] = [], {}
        for ref_n in REFS:
            r = runs[ref_n]
            t, ref = r["test"], r["ref"]
            y = sc[t] > L.PASS_THRESH
            s = {"clotho": r["clotho"], "MDSA": L.surprise_adequacy(H, ref, t, sc, "MDSA", seed),
                 "GMMb": L.surprise_adequacy(H, ref, t, sc, "MLSA", seed),
                 "lookup_drain": L.template_lookup(ref, t, sc, B["drain"], init_idx),
                 "lookup_oracle": L.template_lookup(ref, t, sc, B["templates"], init_idx),
                 **{k: v[t] for k, v in B["post"].items()}}
            for m in METHODS:
                auc = float("nan") if s[m] is None else L.metrics(y, s[m], seed)[0]
                info["rows"].append({"ref": ref_n, "method": m, "auc": auc})
            init = set(init_idx.tolist())
            seen_t = {B["templates"][i] for i in ref if i not in init}
            info["runs"][str(ref_n)] = {"n_test": int(len(t)), "n_test_pass": int(y.sum()),
                                        "template_seen_frac": float(np.mean([B["templates"][i] in seen_t for i in t]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), info.get("skipped", "ok")


def suite_seen_frac(ds, ref_n):
    """Fraction of suite test inputs whose Loghub template occurs among the non-initial references (recomputed)."""
    import numpy as np
    import log_parsing_phase3_final as F
    import log_parsing_phase3_lib as L
    out = []
    for seed in SEEDS:
        B = F.build(ds, "fair_leakfree")
        np.random.seed(seed)
        r = L.clotho_run(B["H"], B["sc"], B["init_idx"], np.arange(B["n"]), None, seed, ref_sizes=(ref_n,),
                         method="balanced")[ref_n]
        init = set(B["init_idx"].tolist())
        seen = {B["templates"][i] for i in r["ref"] if i not in init}
        out.append(float(np.mean([B["templates"][i] in seen for i in r["test"]])))
    return statistics.mean(out)


def evaluate(push=False, workers=10):
    from multiprocessing import Pool
    import numpy as np
    import log_parsing_phase3_final as F
    with Pool(workers) as p:
        for path, status in p.imap_unordered(job, [(s, k) for s in NAT for k in SEEDS]):
            print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)
    R = {}
    for f in sorted(OUT.glob("*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault(d["system"], {})[d["seed"]] = d
    FR = F.load_all()
    st = json.loads((SUITES / "natural_sample_stats.json").read_text())
    kept = [s for s in NAT if s in R and all("skipped" not in d for d in R[s].values())]
    J = {"sample_stats": st, "kept": kept}
    L_ = ["# Natural-distribution check: uniform samples of raw log lines vs the template-capped suites\n",
          "Generated by `experiments/log_parsing_natural.py` (protocol in its docstring). fair_leakfree, standard split, "
          "5 seeds. Natural = 500 lines drawn uniformly from all Loghub-2.0 lines not in the suite (no dedup, no template "
          "cap). Suite = the existing 500-message suite (dedup + template cap). * = needs the generations.\n",
          "## Template repetition\n",
          "| system | templates (suite / natural) | inputs per template (suite / natural) | most frequent template's share, natural | duplicate messages, natural | PASS rate (labels 5-9; suite / natural) | test inputs whose template is in the reference set @100 (suite / natural) |",
          "|---|---|---|---|---|---|---|"]
    for s in NAT:
        suite_rows = [json.loads(l) for l in open(SUITES / f"{SUITE[s]}.jsonl")]
        ts = len({r["template"] for r in suite_rows})
        pr_s = statistics.mean(d["n_pass"] for d in FR[("fair_leakfree", "standard", SUITE[s])].values()) / 500
        pr_n = statistics.mean(d["n_pass"] for d in R[s].values()) / 500 if s in R else float("nan")
        sf_n = (statistics.mean(d["runs"]["100"]["template_seen_frac"] for d in R[s].values())
                if s in kept else float("nan"))
        sf_s = suite_seen_frac(SUITE[s], 100)
        L_.append(f"| {s} | {ts} / {st[s]['templates']} | {500 / ts:.1f} / {500 / st[s]['templates']:.1f} | "
                  f"{st[s]['top_template_share']:.0%} | {st[s]['duplicate_messages']} | {pr_s:.1%} / {pr_n:.1%} | "
                  f"{sf_s:.0%} / {sf_n:.0%} |")
        J.setdefault("repetition", {})[s] = {"templates_suite": ts, "templates_nat": st[s]["templates"],
                                             "seen_frac_suite": sf_s, "seen_frac_nat": sf_n,
                                             "pass_suite": pr_s, "pass_nat": pr_n}
    skipped = {s: R[s][0]["skipped"] for s in NAT if s in R and s not in kept}
    if skipped:
        L_.append("\nSkipped (natural sample has < 15 PASS or < 15 FAIL inputs): " +
                  "; ".join(f"{s} ({v})" for s, v in skipped.items()) + "\n")
    for ref_n in REFS:
        L_ += [f"\n## ROC-AUC @{ref_n}, macro over {len(kept)} systems ({', '.join(kept)}): suite / natural\n",
               "| method | suite | natural | natural minus suite (paired per seed) |", "|---|---|---|---|"]
        for m in METHODS:
            sm = "loh_var" if m == "loh_var" else m
            a = [statistics.mean(F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], sm, ref_n, "auc") for s in kept) for k in SEEDS]
            b = [statistics.mean(F.val(R[s][k], m, ref_n, "auc") for s in kept) for k in SEEDS]
            d = [y - x for x, y in zip(a, b)]
            L_.append(f"| {NAMES[m]} | {F.fmt(a)} | {F.fmt(b)} | {statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} |")
            J.setdefault(f"auc@{ref_n}", {})[m] = {"suite": F.fmt(a), "natural": F.fmt(b)}
        for b in ("lookup_drain", "MDSA"):
            for lab, RR, key in (("suite", None, None), ("natural", R, None)):
                if RR is None:
                    dd = [statistics.mean(F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], "clotho", ref_n, "auc")
                                          - F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], b, ref_n, "auc")
                                          for s in kept) for k in SEEDS]
                else:
                    dd = [statistics.mean(F.val(R[s][k], "clotho", ref_n, "auc") - F.val(R[s][k], b, ref_n, "auc")
                                          for s in kept) for k in SEEDS]
                L_.append(f"\nPaired per seed, Clotho minus {NAMES[b]}, {lab}: {statistics.mean(dd):+.3f} ± "
                          f"{statistics.stdev(dd):.3f} (min {min(dd):+.3f}, max {max(dd):+.3f})")
                J.setdefault(f"paired@{ref_n}", {})[f"{b}_{lab}"] = [statistics.mean(dd), statistics.stdev(dd)]
        L_ += [f"\nPer system @{ref_n} (natural, mean over seeds):\n",
               "| system | " + " | ".join(NAMES[m] for m in METHODS) + " |", "|---" * (1 + len(METHODS)) + "|"]
        for s in kept:
            L_.append(f"| {s} | " + " | ".join(f"{statistics.mean(F.val(R[s][k], m, ref_n, 'auc') for k in SEEDS):.3f}"
                                               for m in METHODS) + " |")
    out = EXP / "reports" / "log_parsing_natural.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L_))
    if push:
        F._push([OUT, out, out.with_suffix(".json")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "gen", "eval"])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    {"sample": sample, "gen": gen, "eval": lambda: evaluate(a.push, a.workers)}[a.cmd]()
