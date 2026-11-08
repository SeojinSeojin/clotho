"""Natural-distribution check: 500 messages per system sampled uniformly from the raw log lines (no template cap).

  cd experiments
  python log_parsing_natural.py sample                         # -> clotho/dataset/log_parsing/loghub_<sys>_nat500.jsonl
  ../.venv-vllm/bin/python log_parsing_natural.py gen          # 10 samples, vLLM, same settings as gen_vllm (seed 0)
  python log_parsing_layers.py -d <nat datasets> --layers 16 --skip_initial
  python log_parsing_postgen.py -d <nat datasets> --no_se
  python log_parsing_natural.py eval [--push]                  # -> reports/log_parsing_natural.{md,json}

Systems: all 13 kept Loghub-2.0 systems (Stage 8 of 2026-11-08 used HDFS, Hadoop, OpenStack, BGL; their samples are
kept byte-identical: `sample` skips files that exist, and each system has its own random.Random(4242)). Sampling: every line of <System>_full.log_structured.csv (Loghub-2.0) whose
Content is not in the current 500-message suite is a candidate; 500 lines are drawn uniformly without
replacement (reservoir sampling, random.Random(4242)). No deduplication: repeated messages and frequent
templates appear at their natural rate, which is the point of the check. The suites (loghub_<sys>_500) instead
deduplicate and cap inputs per template.
These datasets are deliberately NOT registered in clotho.exp_config.supported_datasets (the Phase 3 scripts take
their system list from it); the generic scripts above accept any dataset name.

Eval: main setting fair_leakfree on the standard split (balanced sampling, fair layer (llama 16), label from samples
5-9, post-generation scores from samples 0-4), 5 seeds, ref 100/200; Clotho, MDSA, GMMb, Lookup (Drain, Loghub-2.0
settings, run on the 500 sampled messages), Lookup (oracle), output entropy (k=5: samples 0-4), Clotho + output
entropy (rank average over the test set, as log_parsing_followups.py). Per system: share of test inputs whose
Drain group / Loghub template occurs among the non-initial reference inputs. LOHSvar and Tok-Prob (Stage 8) are
dropped: they need the post-generation pass, which was run for the first 4 systems only.
CLOTHO_MODEL selects the model (model_cfg.py); outputs of a non-llama model get a suffix. Compared with
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
import model_cfg as MC  # noqa: E402
SYSTEMS = {"Apache": "apache", "BGL": "bgl", "HDFS": "hdfs", "HPC": "hpc", "Hadoop": "hadoop", "HealthApp": "healthapp",
           "Linux": "linux", "Mac": "mac", "OpenSSH": "openssh", "OpenStack": "openstack", "Spark": "spark",
           "Thunderbird": "thunderbird", "Zookeeper": "zookeeper"}
NAT = {k: f"loghub_{v}_nat500" for k, v in SYSTEMS.items()}
SUITE = {k: f"loghub_{v}_500" for k, v in SYSTEMS.items()}
TEMPLATE = "rules_template"
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "natural13"
REFS = (100, 200)
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["clotho", "MDSA", "GMMb", "lookup_drain", "lookup_oracle", "out_entropy", "clotho_oe"]
SUITE_METHODS = ["clotho", "MDSA", "GMMb", "lookup_drain", "lookup_oracle", "out_entropy"]   # stored suite jobs
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "GMMb": "GMMb", "lookup_drain": "Lookup (Drain)",
         "lookup_oracle": "Lookup (oracle)", "out_entropy": "Output entropy (k=5)*",
         "clotho_oe": "Clotho + output entropy*"}


def sample():
    sys.path.insert(0, str(SUITES))
    import prepare_loghub as P
    sp = SUITES / "natural_sample_stats.json"
    stats = json.loads(sp.read_text()) if sp.exists() else {}
    for name, ds in NAT.items():
        if (SUITES / f"{ds}.jsonl").exists() and name in stats:
            print(f"{name}: {ds}.jsonl exists, kept", flush=True)
            continue
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
    tok = pr.load_tokenizer(MC.MODEL)
    llm = LLM(model=pr.MODEL_IDS[MC.MODEL], dtype="bfloat16", seed=0, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm")
    sp = SamplingParams(n=pr.GEN_REPEAT, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                        ignore_eos=True, stop_token_ids=[pr.stop_token_id(tok)], detokenize=False)
    od = EXP / f"data/results_{MC.MODEL}/log_parsing/inference_results/{TEMPLATE}"
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
    g = json.load(open(EXP / f"data/results_{MC.MODEL}/log_parsing/inference_results/{TEMPLATE}/{ds}_R10_T0.8.json"))
    sc_ = [[score_output(o, x["GT"], TEMPLATE) for o in x["inferences"]] for x in g]
    P = np.array([[s["strict_match"] for s in row] for row in sc_], dtype=bool)
    parsed = [[s["parsed"] for s in row] for row in sc_]
    H = F._hidden(ds, F.FAIR_LAYER)
    Hi, Pi = F.initial_data(F.FAIR_LAYER)
    with tempfile.TemporaryDirectory() as d:
        drain = [f"drain:{x}" for x in D.drain_lh2([r["content"] for r in src], system, d)]
    pseudo = [f"__initial_{i}__" for i in range(len(Hi))]
    n = len(H)
    return {"H": np.concatenate([H, Hi]), "sc": np.concatenate([P[:, 5:].mean(1), Pi[:, 5:].mean(1)]), "n": n,
            "init_idx": np.arange(n, n + len(Hi)), "templates": [r["template"] for r in src] + pseudo,
            "drain": drain + pseudo,
            "post": {"out_entropy": -np.array([F._entropy(p[:5]) for p in parsed])}}


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
            s["clotho_oe"] = None if r["clotho"] is None else (F.rank01(s["clotho"]) + F.rank01(s["out_entropy"])) / 2
            for m in METHODS:
                auc = float("nan") if s[m] is None else L.metrics(y, s[m], seed)[0]
                info["rows"].append({"ref": ref_n, "method": m, "auc": auc})
            init = set(init_idx.tolist())
            seen_t = {B["templates"][i] for i in ref if i not in init}
            seen_d = {B["drain"][i] for i in ref if i not in init}
            info["runs"][str(ref_n)] = {"n_test": int(len(t)), "n_test_pass": int(y.sum()),
                                        "template_seen_frac": float(np.mean([B["templates"][i] in seen_t for i in t])),
                                        "drain_seen_frac": float(np.mean([B["drain"][i] in seen_d for i in t]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), info.get("skipped", "ok")


def evaluate(push=False, workers=10):
    from multiprocessing import Pool
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
    kept = [s for s in NAT if s in R and len(R[s]) == len(SEEDS) and all("skipped" not in d for d in R[s].values())]
    suite_ok = lambda s: all(("fair_leakfree", "standard", SUITE[s]) in FR and k in FR[("fair_leakfree", "standard", SUITE[s])]  # noqa: E731
                             and "runs" in FR[("fair_leakfree", "standard", SUITE[s])][k] for k in SEEDS)
    both = [s for s in kept if suite_ok(s)]
    mean = statistics.mean
    J = {"model": MC.MODEL, "layer": F.FAIR_LAYER, "sample_stats": st, "kept": kept, "kept_with_suite": both}
    suf = "" if MC.MODEL == "llama" else f" Model: {MC.MODEL}."
    L_ = ["# Natural-distribution check, 13 systems: uniform samples of raw log lines vs the template-capped suites\n",
          f"Generated by `experiments/log_parsing_natural.py eval` (protocol in its docstring).{suf} fair_leakfree "
          f"(balanced sampling, layer {F.FAIR_LAYER}, label from samples 5-9, PASS if >= 3 of 5), standard split, 5 seeds "
          "(0-4). Natural = 500 lines drawn uniformly from all Loghub-2.0 lines of the system that are not in its suite "
          "(no dedup, no template cap, seed 4242). Suite = the existing 500-message suite (dedup + template cap). "
          "Cells: mean ± std over seeds of the per-seed macro mean over systems. Output entropy uses samples 0-4 (k=5); "
          "Clotho + output entropy = rank average over the test set. * = needs the generations. "
          "`Drain seen` = share of test inputs whose Drain group occurs among the non-initial reference inputs.\n",
          "## Samples\n",
          "| system | templates (suite / natural) | most frequent template's share, natural | duplicate messages, natural | PASS rate, labels 5-9 (suite / natural) | Drain seen @100 (suite / natural) | Drain seen @200 (suite / natural) | template seen @100, natural |",
          "|---|---|---|---|---|---|---|---|"]
    for s in NAT:
        ts = len({json.loads(l)["template"] for l in open(SUITES / f"{SUITE[s]}.jsonl")})
        fr = FR.get(("fair_leakfree", "standard", SUITE[s]), {})
        pr_s = mean(d["n_pass"] for d in fr.values()) / 500 if fr else float("nan")
        pr_n = mean(d["n_pass"] for d in R[s].values()) / 500 if s in R else float("nan")
        cell = []
        for ref_n in REFS:
            ds_ = mean(d["runs"][str(ref_n)]["drain_seen_frac"] for d in fr.values() if "runs" in d) if s in both else float("nan")
            dn = mean(d["runs"][str(ref_n)]["drain_seen_frac"] for d in R[s].values()) if s in kept else float("nan")
            cell.append(f"{ds_:.0%} / {dn:.0%}")
            J.setdefault("drain_seen", {}).setdefault(s, {})[f"@{ref_n}"] = {"suite": ds_, "natural": dn}
        tn = mean(d["runs"]["100"]["template_seen_frac"] for d in R[s].values()) if s in kept else float("nan")
        L_.append(f"| {s} | {ts} / {st[s]['templates']} | {st[s]['top_template_share']:.0%} | "
                  f"{st[s]['duplicate_messages']} | {pr_s:.1%} / {pr_n:.1%} | {cell[0]} | {cell[1]} | {tn:.0%} |")
        J.setdefault("pass_rate", {})[s] = {"suite": pr_s, "natural": pr_n}
    skipped = {s: next(iter(R[s].values())).get("skipped") for s in NAT if s in R and s not in kept}
    if skipped:
        L_.append("\nSkipped on the natural sample (< 15 PASS or < 15 FAIL inputs under the 5-9 label): " +
                  "; ".join(f"{s} ({v})" for s, v in skipped.items()) + "\n")
    J["skipped"] = skipped
    combos = {}
    fu = EXP / "reports" / "log_parsing_followups.json"
    if fu.exists() and MC.MODEL == "llama":
        combos = json.loads(fu.read_text()).get("combos", {}).get("clotho+out_entropy", [])
    for ref_n in REFS:
        L_ += [f"\n## ROC-AUC @{ref_n}\n",
               f"Natural: macro over the {len(kept)} kept systems. Suite (same systems): macro over the {len(both)} systems "
               "that are kept on both. Clotho + output entropy on the suites is the 13-system value from "
               "`log_parsing_followups.md` (not paired).\n",
               "| method | natural (all kept) | natural (same systems) | suite (same systems) | natural minus suite (paired per seed) |",
               "|---|---|---|---|---|"]
        for m in METHODS:
            n_all = [mean(F.val(R[s][k], m, ref_n, "auc") for s in kept) for k in SEEDS]
            n_b = [mean(F.val(R[s][k], m, ref_n, "auc") for s in both) for k in SEEDS]
            if m in SUITE_METHODS:
                su = [mean(F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], m, ref_n, "auc") for s in both) for k in SEEDS]
                d = [y - x for x, y in zip(su, n_b)]
                sc, dc = F.fmt(su), f"{mean(d):+.3f} ± {statistics.stdev(d):.3f}"
            else:
                sc = (f"{combos[0 if ref_n == 100 else 1]} (13 systems)" if combos else "n/a")
                dc = "n/a"
            L_.append(f"| {NAMES[m]} | {F.fmt(n_all)} | {F.fmt(n_b)} | {sc} | {dc} |")
            J.setdefault(f"auc@{ref_n}", {})[m] = {"natural": F.fmt(n_all), "natural_same": F.fmt(n_b), "suite": sc,
                                                   "natural_seeds": n_all}
        L_.append("")
        for x, y in (("clotho", "lookup_drain"), ("clotho", "MDSA"), ("clotho", "out_entropy"),
                     ("clotho_oe", "out_entropy"), ("clotho_oe", "lookup_drain")):
            dd = [mean(F.val(R[s][k], x, ref_n, "auc") - F.val(R[s][k], y, ref_n, "auc") for s in kept) for k in SEEDS]
            wins = sum(mean(F.val(R[s][k], x, ref_n, "auc") - F.val(R[s][k], y, ref_n, "auc") for k in SEEDS) > 0 for s in kept)
            L_.append(f"Paired per seed, natural, {NAMES[x]} minus {NAMES[y]}: {mean(dd):+.3f} ± {statistics.stdev(dd):.3f} "
                      f"(min {min(dd):+.3f}, max {max(dd):+.3f}); ahead on {wins} of {len(kept)} systems (mean over seeds)  ")
            J.setdefault(f"paired@{ref_n}", {})[f"{x}-{y}"] = {"mean": mean(dd), "std": statistics.stdev(dd),
                                                             "min": min(dd), "max": max(dd), "systems_ahead": wins}
        for y in ("lookup_drain", "MDSA"):
            dd = [mean(F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], "clotho", ref_n, "auc")
                       - F.val(FR[("fair_leakfree", "standard", SUITE[s])][k], y, ref_n, "auc") for s in both) for k in SEEDS]
            L_.append(f"Paired per seed, suite (same {len(both)} systems), Clotho minus {NAMES[y]}: {mean(dd):+.3f} ± "
                      f"{statistics.stdev(dd):.3f}  ")
            J.setdefault(f"paired@{ref_n}", {})[f"suite_clotho-{y}"] = {"mean": mean(dd), "std": statistics.stdev(dd)}
        L_ += [f"\nPer system @{ref_n} (natural, mean over seeds):\n",
               "| system | PASS rate | Drain seen | " + " | ".join(NAMES[m] for m in METHODS) + " |",
               "|---" * (3 + len(METHODS)) + "|"]
        for s in kept:
            ds_ = mean(d["runs"][str(ref_n)]["drain_seen_frac"] for d in R[s].values())
            pr_n = mean(d["n_pass"] for d in R[s].values()) / 500
            L_.append(f"| {s} | {pr_n:.1%} | {ds_:.0%} | " +
                      " | ".join(f"{mean(F.val(R[s][k], m, ref_n, 'auc') for k in SEEDS):.3f}" for m in METHODS) + " |")
    out = EXP / "reports" / f"log_parsing_natural13{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L_))
    if push:
        F._push([OUT, out, out.with_suffix(".json"), SUITES / "natural_sample_stats.json"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "gen", "eval"])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    {"sample": sample, "gen": gen, "eval": lambda: evaluate(a.push, a.workers)}[a.cmd]()
