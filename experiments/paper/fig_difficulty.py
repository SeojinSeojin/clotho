"""Figure (P1): what makes a template hard (L). Odds ratios of FAIL with 95% CI, pooled over both models (system fixed
effects, cluster-robust SE by template) and per model.

  cd experiments && python paper/fig_difficulty.py   -> paper/figures/difficulty.pdf (+ .png, caption difficulty.tex)
Source: reports/template_difficulty.json (log_parsing_template_difficulty.py).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FEATS = [("attached", "variable attached to text / punctuation"), ("other_var", "word-like variable"), ("path", "path variable"),
         ("n_var", "per extra variable"), ("len10", "template length (per 10 tokens)"), ("share_var", "share of variable tokens"),
         ("ip", "IP variable"), ("hex", "hex variable"), ("number", "number variable")]
SER = [("pooled", "pooled (both models)", C.INK, "o"), ("llama", C.LABEL["llama"], "#2a78d6", "s"), ("gemma", C.LABEL["gemma"], "#1baf7a", "^")]


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "template_difficulty.json").read_text())
    feats = [(k, n) for k, n in FEATS if k in J["pooled"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for j, (key, name, col, mk) in enumerate(SER):
        for i, (f, _) in enumerate(feats):
            r = J[key].get(f)
            if not r:
                continue
            y = i + (j - 1) * 0.22
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.4)
            ax.plot(r["or"], y, mk, color=col, ms=5.5 if j == 0 else 4.5, mfc=col if r["p"] < 0.05 else "white",
                    label=name if i == 0 else None)
    ax.axvline(1, color=C.INK2, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([n for _, n in feats])
    ax.invert_yaxis()
    ax.set_xlabel("odds ratio of FAIL (log scale; filled = p < 0.05)")
    ax.grid(axis="x", color=C.GRID, lw=0.8)
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout()
    out = C.PAPER / "figures" / "difficulty.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("difficulty", "fig_difficulty.py",
              "What makes a template hard. Logistic regression of FAIL (main label: fewer than 3 of samples 5--9 match the "
              "ground truth exactly) on features of the ground-truth template, over the capped suites of all kept systems "
              f"({J['n']:,} inputs), with system fixed effects and standard errors clustered by template. Odds ratios with "
              "95\\% confidence intervals, pooled over both models (with a model indicator) and per model; filled markers: "
              "$p < 0.05$.")
    print(out)


if __name__ == "__main__":
    main()
