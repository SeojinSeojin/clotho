"""L (2026-11-10): what makes a template hard. Protocol: PROGRESS.md Decisions (H-M).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_template_difficulty.py collect
  CLOTHO_MODEL=gemma ../.venv/bin/python log_parsing_template_difficulty.py collect
  ../.venv/bin/python log_parsing_template_difficulty.py fit [--push]      -> reports/template_difficulty.{md,json}

Unit = capped-suite input; y = FAIL of the main fair_leakfree label (PASS iff >= 3 of samples 5-9 pass). Features of the
ground-truth template, variable values from matching the template against the line. Logistic regression (statsmodels
GLM binomial, system fixed effects, cluster-robust SE by template), odds ratios of FAIL with 95% CI.
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
OUT = EXP / "data" / "template_difficulty"
FEATS = ["n_var", "len10", "share_var", "attached", "ip", "path", "hex", "number", "other_var"]
FNAMES = {"n_var": "variables (per variable)", "len10": "template length (per 10 tokens)",
          "share_var": "share of variable tokens (0 to 1)", "attached": "variable attached to text/punctuation",
          "ip": "IP address variable", "path": "path variable", "hex": "hex variable", "number": "pure-number variable",
          "other_var": "other variable (word-like, mixed)", "gemma": "model = Gemma"}
_IP = re.compile(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?")
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_HEX8 = re.compile(r"[0-9a-fA-F]{8,}")
_NUM = re.compile(r"[-+]?\d+(\.\d+)?")


def vtype(v):
    v = v.strip()
    if _IP.search(v):
        return "ip"
    if _HEX.search(v) or (_HEX8.fullmatch(v) and re.search(r"\d", v) and re.search(r"[a-fA-F]", v)):
        return "hex"
    if _NUM.fullmatch(v):
        return "number"
    if "/" in v:
        return "path"
    return "other_var"


def values(template, content):
    parts = template.split("<*>")
    rx = "(.*?)".join(r"\s+".join(re.escape(w) for w in p.split(" ")) for p in parts)
    try:
        m = re.fullmatch(rx, content.strip(), flags=re.S)
    except re.error:
        m = None
    return list(m.groups()) if m else None


def features(template, content):
    toks = template.split()
    nv = template.count("<*>")
    vt = [t for t in toks if "<*>" in t]
    f = {"n_var": nv, "len10": len(toks) / 10, "share_var": len(vt) / max(1, len(toks)),
         "attached": int(any(t != "<*>" for t in vt)), "matched": 1}
    vals = values(template, content)
    if vals is None:
        f["matched"] = 0
        vals = []
    types = {vtype(v) for v in vals if v.strip()}
    for k in ("ip", "path", "hex", "number", "other_var"):
        f[k] = int(k in types)
    return f


def collect():
    import model_cfg as MC
    import log_parsing_transfer as T
    import log_parsing_triage as TR
    rows = []
    for s in TR.SYSTEMS:
        ds = f"loghub_{s.lower()}_500"
        y = T.labels(ds)
        suite = [json.loads(l) for l in open(EXP.parent / "clotho" / "dataset" / "log_parsing" / f"{ds}.jsonl")]
        for r, ok in zip(suite, y):
            rows.append({"model": MC.MODEL, "system": s, "template": r["template"], "content": r["content"],
                         "fail": int(not ok), **features(r["template"], r["content"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{MC.MODEL}.json").write_text(json.dumps(rows))
    print(MC.MODEL, len(rows), "inputs, unmatched", sum(1 - r["matched"] for r in rows))


def glm(df, feats):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    f = "fail ~ " + " + ".join(feats) + " + C(system)"
    groups = df["system"] + "|" + df["template"]
    codes = groups.astype("category").cat.codes
    m = smf.glm(f, data=df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": codes})
    ci = m.conf_int()
    return {k: {"or": float(np.exp(m.params[k])), "lo": float(np.exp(ci.loc[k, 0])), "hi": float(np.exp(ci.loc[k, 1])),
                "p": float(m.pvalues[k])} for k in feats}, m


def fit(push=False):
    import pandas as pd
    rows = []
    for mdl in ("llama", "gemma"):
        p = OUT / f"{mdl}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    df = pd.DataFrame(rows)
    df["gemma"] = (df["model"] == "gemma").astype(int)
    J = {"n": len(df), "unmatched": int((1 - df["matched"]).sum()), "fail_rate": float(df["fail"].mean())}
    J["pooled"], _ = glm(df, FEATS + ["gemma"])
    for mdl in ("llama", "gemma"):
        J[mdl], _ = glm(df[df["model"] == mdl], FEATS)
    # prevalence and raw failure rate by factor
    J["prevalence"] = {k: float((df[k] > 0).mean()) for k in FEATS}
    sig = [k for k in FEATS if J["pooled"][k]["p"] < 0.05]
    top = sorted(sig, key=lambda k: -abs(np.log(J["pooled"][k]["or"])))[:3]
    J["top"] = top
    # template-level table (both models pooled)
    tt = df.groupby(["system", "template"]).agg(fail=("fail", "mean"), n=("fail", "size"),
                                                 **{k: (k, "max") for k in FEATS}).reset_index()
    ex = {}
    for k in top:
        if k in ("n_var", "len10", "share_var"):
            sel = tt[tt[k] >= tt[k].quantile(0.75)]
            if J["pooled"][k]["or"] < 1:
                sel = tt[tt[k] <= tt[k].quantile(0.25)]
        else:
            sel = tt[tt[k] == (1 if J["pooled"][k]["or"] > 1 else 0)]
        sel = sel.sort_values(["fail", "n", "system", "template"], ascending=[False, False, True, True]).head(10)
        ex[k] = sel[["system", "template", "fail", "n"]].to_dict("records")
    J["examples"] = ex
    L = ["# L: what makes a log template hard (details)\n",
         f"Generated by `log_parsing_template_difficulty.py fit`. Protocol: PROGRESS.md Decisions (H-M). {J['n']} capped-suite "
         f"inputs (both models, kept systems), FAIL rate {J['fail_rate']:.3f}; {J['unmatched']} inputs where the template did not "
         "match the line (variable types unknown, structural features kept). Odds ratio of FAIL (> 1 = harder), 95% CI "
         "cluster-robust by template, system fixed effects.\n",
         "| factor | prevalence | OR pooled [95% CI] | p | OR Llama | OR Gemma |", "|---|---|---|---|---|---|"]
    for k in FEATS + ["gemma"]:
        a = J["pooled"][k]
        L.append(f"| {FNAMES[k]} | {J['prevalence'].get(k, float('nan')):.2f} | {a['or']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}] | {a['p']:.3g} | "
                 + (f"{J['llama'][k]['or']:.2f} | {J['gemma'][k]['or']:.2f} |" if k != "gemma" else " | |"))
    L.append(f"\nTop factors (largest |log OR| with p < 0.05): {', '.join(FNAMES[k] for k in top)}.\n")
    for k in top:
        L.append(f"\n### {FNAMES[k]}: 10 illustrative templates (highest FAIL rate, both models pooled)\n")
        L.append("| system | template | FAIL rate | inputs |")
        L.append("|---|---|---|---|")
        for r in ex[k]:
            t = r["template"].replace("|", "\\|")
            L.append(f"| {r['system']} | `{t[:110]}` | {r['fail']:.2f} | {r['n']} |")
    (EXP / "reports" / "template_difficulty.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "template_difficulty.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        files = {f"reports/template_difficulty.{e}": EXP / "reports" / f"template_difficulty.{e}" for e in ("md", "json")}
        files.update({str(p.relative_to(EXP)): p for p in OUT.glob("*.json")})
        hs.commit_files(hs.get_results_repo(), files, "L template difficulty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit"])
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    collect() if a.cmd == "collect" else fit(a.push)


if __name__ == "__main__":
    main()
