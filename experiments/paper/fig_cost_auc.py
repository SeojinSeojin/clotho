"""Figure 1: cost per test input vs ROC-AUC, both models, standard split, @100 and @200.

  cd experiments && python paper/fig_cost_auc.py   -> paper/figures/cost_auc.pdf
Quality: per-job JSON of log_parsing_cost_quality.py (fair_leakfree; output entropy from samples 0..k-1,
Clotho + output entropy(k) = rank average). Cost: log_parsing_timing.py on one NVIDIA L40S, mean over HDFS,
Hadoop, BGL, OpenStack, seconds per input, offline batch (the same definitions as log_parsing_cost_quality.costs):
  Clotho = batched HF forward pass through layers 1..L (L = 16 Llama, 21 Gemma; batch 32) + PCA/GMM scoring;
  MDSA = same forward pass + PCA/Mahalanobis; Lookup (Drain) = Drain parse; output entropy(k) = vLLM, k samples,
  all 500 prompts at once; Clotho + output entropy(k) = the sum.
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REFS = (100, 200)
KS = [2, 3, 4, 5]


def costs(model):
    T = C.rp(model) / "timing"
    hf, vl = json.loads((T / "hf.json").read_text()), json.loads((T / "vllm.json").read_text())
    mean = lambda d, k: statistics.mean(v[k] for v in d["systems"].values())  # noqa: E731
    fwd, gmm = mean(hf, "batched_l16"), hf["gmm_bs1_nf50_nc10"]
    c = {"clotho": fwd + gmm, "MDSA": fwd + hf["mdsa_bs1"], "lookup_drain": hf["drain_lh2_per_input"]}
    for k in KS:
        c[f"oe{k}"] = mean(vl, f"gen_{k}")
        c[f"c+oe{k}"] = c["clotho"] + c[f"oe{k}"]
    return c


def quality(model, ref):
    R = {}
    for f in (C.rp(model) / "cost_quality" / "standard").glob("*/seed*.json"):
        d = json.loads(f.read_text())
        if C.short(d["system"]) not in C.DROPPED[model]:
            R.setdefault(d["system"], {})[d["seed"]] = d
    keys = next(iter(next(iter(R.values())).values()))["refs"][str(ref)].keys()
    return {m: [statistics.mean(R[s][k]["refs"][str(ref)][m] for s in R) for k in C.SEEDS] for m in keys}, len(R)


def main():
    plt = C.style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for i, model in enumerate(C.MODELS):
        c = costs(model)
        for j, ref in enumerate(REFS):
            ax = axes[i, j]
            Q, n = quality(model, ref)
            mu = lambda m: statistics.mean(Q[m])  # noqa: E731
            sd = lambda m: statistics.stdev(Q[m])  # noqa: E731
            for key, name in (("oe", "Output entropy (k samples)"), ("c+oe", "Clotho + output entropy (k)")):
                xs, ys = [c[f"{key}{k}"] for k in KS], [mu(f"{key}{k}") for k in KS]
                ax.errorbar(xs, ys, yerr=[sd(f"{key}{k}") for k in KS], color=C.COL[key], lw=2, marker="o", ms=5,
                            capsize=2, label=name, zorder=3)
                for k, x, y in zip(KS, xs, ys):
                    if key == "oe" or k in (2, 5):
                        ax.annotate(f"k={k}", (x, y), textcoords="offset points",
                                    xytext=(5, -10) if key == "oe" else (-6, 6), ha="left" if key == "oe" else "right",
                                    fontsize=7, color=C.INK2)
            for key, name in (("clotho", "Clotho"), ("MDSA", "MDSA"), ("lookup_drain", "Lookup (Drain)")):
                ax.errorbar([c[key]], [mu(key)], yerr=[sd(key)], color=C.COL[key], marker="s", ms=7, capsize=2,
                            ls="none", label=name, zorder=5, mec="white", mew=1)
            ax.set_xscale("log")
            ax.set_title(f"{C.LABEL[model]} ({n} systems), reference size {ref}", color=C.INK)
    fig.supxlabel("cost per test input, seconds (log scale; one L40S, offline batch)", fontsize=9, y=0.07)
    fig.supylabel("ROC-AUC, standard split (mean ± std over 5 seeds)", fontsize=9)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = C.PAPER / "figures" / "cost_auc.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
