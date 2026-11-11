"""N5 (2026-11-10 night, CPU only): Drain's own templates vs the LLM cache on the 50,000-line streams.
Protocol: PROGRESS.md Decisions (N4-N5).

  cd experiments && ../.venv/bin/python log_parsing_drain_templates.py [--push]   -> reports/drain_templates.{md,json}

Drain = logparser3 Drain, Loghub-2.0 settings, the stream's short-message fix (same grouping as data/stream, asserted).
Template of a line = its cluster's final logTemplate joined by spaces. Strict line-level PA, Loghub-2.0 FTA, and a
split by group size (1-2 lines vs >= 3). Needs the raw Loghub-2.0 zips in ~/.cache/loghub2.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_stream as ST  # noqa: E402

MODELS = {"llama": "", "gemma": "_gemma"}
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}
CACHE = EXP / "data" / "stream_drain_templates"


def drain_templates(system):
    """-> (per-line cluster index in creation order, final template per cluster)."""
    p = CACHE / f"{system}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return j["group"], j["templates"]
    import pandas as pd
    from logparser.Drain import LogParser
    from log_parsing_drain import LH2
    rows = ST.read_stretch(system)
    messages = [r[1] for r in rows]
    s = LH2[system]
    holder = {}
    with tempfile.TemporaryDirectory() as wd:
        class P(LogParser):
            def load_data(self):
                self.df_log = pd.DataFrame({"LineId": range(1, len(messages) + 1), "Content": messages})

            def outputResult(self, logClustL):
                holder["cl"] = logClustL

            def addSeqToPrefixTree(self, rn, logClust):
                super().addSeqToPrefixTree(rn, logClust)
                seq = logClust.logTemplate
                node, d = rn.childD[len(seq)], 1
                for tok in seq:
                    if d >= self.depth or d > len(seq):
                        break
                    node = node.childD[tok] if tok in node.childD else node.childD["<*>"]
                    d += 1
                if isinstance(node.childD, dict):
                    assert not node.childD
                    node.childD = [logClust]
                elif logClust not in node.childD:
                    node.childD.append(logClust)
        P(log_format="<Content>", indir=wd, outdir=wd, depth=s["depth"], st=s["st"], rex=s["regex"],
          keep_para=False).parse(f"{system}.log")
    cl = sorted(holder["cl"], key=lambda c: min(c.logIDL))
    lab = [None] * len(messages)
    for g, c in enumerate(cl):
        for lid in c.logIDL:
            lab[lid - 1] = g
    tpl = [" ".join(c.logTemplate) for c in cl]
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"group": lab, "templates": tpl}))
    return lab, tpl


def fta(pred, gt):
    """Loghub-2.0 FTA: predicted templates grouped by string; correct iff same line set as a GT template and same string."""
    cp, cg, cj = Counter(pred), Counter(gt), Counter(zip(pred, gt))
    correct = sum(1 for p, n in cp.items() if p in cg and cg[p] == n and cj[(p, p)] == n)
    pta, rta = correct / len(cp), correct / len(cg)
    return {"PTA": pta, "RTA": rta, "FTA": 2 * pta * rta / (pta + rta) if pta + rta else 0.0, "n_pred": len(cp), "n_gt": len(cg)}


def caches(model, system):
    out = {}
    gb = EXP / "data" / "stream_gen" / model / f"{system}.json"
    if gb.exists():
        out["base"] = (json.loads(gb.read_text())["greedy"], [])
    gdir = EXP / "data" / "stream_gen_strong" / model
    ch = gdir / "chosen.json"
    if ch.exists():
        g = json.loads((gdir / json.loads(ch.read_text())["chosen"] / f"{system}.json").read_text())
        out["strong"] = (g["final"], [int(q) for q in g["seeds"]])
    return out


def score(pred, gtt, grp, sizes, exclude=()):
    ex = set(exclude)
    keep = np.array([g not in ex for g in grp])
    ok = np.array([p == t for p, t in zip(pred, gtt)])
    small = sizes[grp] <= 2
    r = {"PA": float(ok[keep].mean())}
    for b, m in (("small", small & keep), ("large", ~small & keep)):
        r[f"PA_{b}"] = float(ok[m].mean()) if m.any() else float("nan")
        r[f"lines_{b}"] = int(m.sum())
        r[f"groups_{b}"] = int(len(set(np.asarray(grp)[m])))
    r.update(fta([str(p) for p in pred], gtt) if not ex else {})
    return r


def run(push=False):
    systems = [s for s in ST.SYSTEMS if s not in DROP["llama"]]
    R = {}
    for s in systems:
        d = ST.load(s)
        lab, tpl = drain_templates(s)
        assert lab == d["group"], f"{s}: Drain grouping differs from the stored stream"
        grp = np.asarray(lab)
        sizes = np.bincount(grp).astype(float)
        gtt = [d["templates"][t] for t in d["gt"]]
        R[s] = {"drain": score([tpl[g] for g in grp], gtt, grp, sizes), "G": len(tpl)}
        for model in MODELS:
            if s in DROP[model]:
                continue
            for parser, (cache, seeds) in caches(model, s).items():
                pred = [cache[g] for g in grp]
                R[s][f"{model}/{parser}"] = score(pred, gtt, grp, sizes)
                if seeds:
                    R[s][f"{model}/{parser}_noseed"] = score(pred, gtt, grp, sizes, seeds)
                    R[s][f"drain_noseed_{model}"] = score([tpl[g] for g in grp], gtt, grp, sizes, seeds)
        print(s, {k: round(v["PA"], 3) for k, v in R[s].items() if isinstance(v, dict)}, flush=True)
    L = ["# N5: Drain templates vs the LLM cache (50,000-line streams)\n",
         "Generated by `log_parsing_drain_templates.py`. Protocol: PROGRESS.md Decisions (N4-N5). PA = share of lines whose "
         "template equals the Loghub-2.0 EventTemplate exactly (strict). FTA = Loghub-2.0 template-level F1 (strict strings). "
         "Small = lines of Drain groups with 1-2 lines in the stretch, large = groups with >= 3 lines (same groups for every "
         "parser). Strong includes 10 human-verified seed templates; 'excl. seeds' drops those groups' lines (Drain too).\n"]
    L.append("\n## Per system (PA; FTA in brackets)\n")
    cols = ["drain", "llama/base", "llama/strong", "gemma/base", "gemma/strong"]
    L.append("| system | groups | small-group lines | " + " | ".join(cols) + " |")
    L.append("|---|---|---|" + "---|" * len(cols))
    for s, r in R.items():
        L.append(f"| {s} | {r['G']} | {r['drain']['lines_small']} | " + " | ".join(
            f"{r[c]['PA']:.3f} ({r[c]['FTA']:.3f})" if c in r else "-" for c in cols) + " |")
    J = {"systems": R, "macro": {}}
    for model in MODELS:
        common = [s for s in R if f"{model}/base" in R[s] and f"{model}/strong" in R[s]]
        L.append(f"\n## {model}: macro over {len(common)} systems with both caches ({', '.join(s for s in R if s not in common and s not in DROP[model]) or 'none'} missing)\n")
        L.append("| parser | PA | FTA | PA small groups | PA large groups | LLM minus Drain, PA small (systems LLM wins) | LLM minus Drain, PA large (wins) |")
        L.append("|---|---|---|---|---|---|---|")
        rows = [("Drain", "drain"), ("LLM base", f"{model}/base"), ("LLM strong", f"{model}/strong"),
                ("LLM strong, excl. seeds", f"{model}/strong_noseed"), ("Drain, excl. strong seeds", f"drain_noseed_{model}")]
        J["macro"][model] = jm = {"systems": common}
        for name, k in rows:
            v = [R[s][k] for s in common]
            mean = lambda f: float(np.nanmean([f(x) for x in v]))  # noqa: E731
            cell = {"PA": mean(lambda x: x["PA"]), "FTA": mean(lambda x: x.get("FTA", np.nan)),
                    "PA_small": mean(lambda x: x["PA_small"]), "PA_large": mean(lambda x: x["PA_large"])}
            ref = "drain_noseed_" + model if k.endswith("noseed") else "drain"
            diffs = {}
            for b in ("small", "large"):
                dd = [(R[s][k][f"PA_{b}"] - R[s][ref][f"PA_{b}"]) for s in common
                      if not np.isnan(R[s][k][f"PA_{b}"])]
                diffs[b] = (float(np.mean(dd)), sum(x > 0 for x in dd), sum(x < 0 for x in dd), len(dd))
            cell["diff"] = diffs
            jm[name] = cell
            dtxt = (lambda b: f"{diffs[b][0]:+.3f} ({diffs[b][1]} win / {diffs[b][2]} lose of {diffs[b][3]})") if k.startswith(model) else (lambda b: "")
            L.append(f"| {name} | {cell['PA']:.3f} | " + ("-" if np.isnan(cell["FTA"]) else f"{cell['FTA']:.3f}")
                     + f" | {cell['PA_small']:.3f} | {cell['PA_large']:.3f} | {dtxt('small')} | {dtxt('large')} |")
        full = [s for s in R if f"{model}/strong" in R[s]]
        if len(full) != len(common):
            L.append(f"\nAll {len(full)} systems with the strong cache: Drain PA {np.mean([R[s]['drain']['PA'] for s in full]):.3f}, "
                     f"LLM strong PA {np.mean([R[s][f'{model}/strong']['PA'] for s in full]):.3f}; FTA "
                     f"{np.mean([R[s]['drain']['FTA'] for s in full]):.3f} vs {np.mean([R[s][f'{model}/strong']['FTA'] for s in full]):.3f}.")
    tot_small = sum(R[s]["drain"]["lines_small"] for s in R)
    L.append(f"\nSmall-group lines are {tot_small} of {sum(R[s]['drain']['lines_small'] + R[s]['drain']['lines_large'] for s in R)} stream lines in total "
             f"({100 * statistics.mean(R[s]['drain']['lines_small'] / (R[s]['drain']['lines_small'] + R[s]['drain']['lines_large']) for s in R):.1f}% per system on average).")
    (EXP / "reports" / "drain_templates.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "drain_templates.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        files = {f"reports/drain_templates.{e}": EXP / "reports" / f"drain_templates.{e}" for e in ("md", "json")}
        files.update({str(p.relative_to(EXP)): p for p in CACHE.glob("*.json")})
        hs.commit_files(hs.get_results_repo(), files, "N5 Drain templates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    run(ap.parse_args().push)


if __name__ == "__main__":
    main()
