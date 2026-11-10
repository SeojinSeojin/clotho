"""Figure 3: within-template ROC-AUC against the noisy 5-sample label vs the clean 40-sample label (Llama).

  cd experiments && python paper/fig_within_template.py   -> paper/figures/within_template.pdf
Within-template AUC uses only PASS/FAIL pairs of test inputs with the same Loghub template. Noisy label =
samples 5-9 (log_parsing_followups.py jobs); clean label = > 20 of 40 extra samples (engine seed 1;
log_parsing_label_noise.py jobs). Same systems (>= 20 same-template pairs on every seed) and test sets
(fair_leakfree, standard split). Ceiling = a perfect predictor of the original 10-sample pass rate, scored
against the clean label. Gemma has no 40-sample run.
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REFS = (100, 200)
METHODS = [("clotho", "Clotho"), ("MDSA", "MDSA"), ("loh_var", "LOHSvar"), ("out_entropy", "Output\nentropy")]
MIN_PAIRS = 20


def load(sub):
    R = {}
    for f in (C.rp("llama") / sub).glob("**/seed*.json"):
        d = json.loads(f.read_text())
        if sub == "followups" and d["split"] != "standard":
            continue
        R.setdefault(d["system"], {})[d["seed"]] = d
    return R


def main():
    clean, noisy = load("label_noise"), load("followups")
    plt = C.style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    w = 0.38
    for ax, ref in zip(axes, REFS):
        elig = [s for s in clean if len(clean[s]) == 5 and
                all(clean[s][k]["refs"][str(ref)]["clotho"]["within_pairs"] >= MIN_PAIRS for k in C.SEEDS)]
        ms = lambda R, m: [statistics.mean(R[s][k]["refs"][str(ref)][m]["within_auc"] for s in elig) for k in C.SEEDS]  # noqa: E731
        for i, (m, name) in enumerate(METHODS):
            for off, R, lab, col in ((-w / 2, noisy, "noisy label (5 samples)", "#9ec5f4"),
                                     (w / 2, clean, "clean label (40 samples)", C.COL["clotho"])):
                v = ms(R, m)
                ax.bar(i + off, statistics.mean(v) - 0.5, bottom=0.5, width=w - 0.04, color=col,
                       yerr=statistics.stdev(v), error_kw={"lw": 0.8, "capsize": 2, "ecolor": C.INK2},
                       label=lab if i == 0 else None)
                ax.text(i + off, statistics.mean(v) + statistics.stdev(v) + 0.004, f"{statistics.mean(v):.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=C.INK2)
        ceil = statistics.mean(ms(clean, "p10"))
        ax.axhline(ceil, color=C.INK2, lw=0.9, ls="--")
        ax.text(len(METHODS) - 0.5, ceil + 0.004, f"ceiling {ceil:.3f}", ha="right", va="bottom", fontsize=7, color=C.INK2)
        ax.axhline(0.5, color=C.INK2, lw=0.8)
        ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS])
        ax.set_title(f"reference size {ref} ({len(elig)} systems)", color=C.INK)
        ax.set_ylim(0.45, 0.75)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("within-template ROC-AUC")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = C.PAPER / "figures" / "within_template.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("within_template", "fig_within_template.py", "Within-template ROC-AUC (only PASS/FAIL pairs of test inputs with the same Loghub template), "
              "Llama-3.1-8B, against the noisy label used everywhere else (samples 5--9, PASS if $\\geq$3 of 5) and a "
              "clean label (more than 20 of 40 extra samples, vLLM engine seed 1, never seen by any method). Same "
              "systems (at least 20 same-template pairs on every seed: 7 at @100, 6 at @200) and same test sets "
              "(fair\\_leakfree, standard split, balanced sampling, layer 16). Bars: mean $\\pm$ std over 5 seeds of "
              "the macro mean. Dashed: a perfect predictor of the original 10-sample pass rate, scored against the clean "
              "label. Template lookup is 0.5 by construction. No 40-sample run exists for Gemma.")
    print("wrote", out)


if __name__ == "__main__":
    main()
