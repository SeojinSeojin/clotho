"""Figure (b): failures found vs labelling budget, testing strategies (Experiment C); both models, capped and natural.

  cd experiments && python paper/fig_budget.py   -> paper/figures/budget.pdf (+ .png, caption budget.tex)
Source: reports/log_parsing_strategy_details.json (log_parsing_strategy.py report; per-seed totals over systems).
Top row: failures found with 2B human checks (B labels + top B of the ranking), summed over systems, mean over 5 seeds.
Bottom row: difference to Clotho (same seeds), mean ± std over seeds.
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BUDGETS = (50, 100, 200)
STRATS = [("clotho", "Clotho", C.COL["clotho"], "-", "o"),
          ("drain_clotho", "Drain-propagate + Clotho", C.COL["lookup_drain"], "--", "s"),
          ("drain_random", "Drain-propagate, no Clotho", C.COL["MDSA"], ":", "^"),
          ("random", "Random", C.INK2, "-.", "D")]
COLS = [("llama", "capped"), ("llama", "natural"), ("gemma", "capped"), ("gemma", "natural")]


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "log_parsing_strategy_details.json").read_text())
    fig, ax = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True)
    for j, (mo, su) in enumerate(COLS):
        d = J[f"{mo}/{su}"]
        for st, name, col, ls, mk in STRATS:
            ys = [statistics.mean(d["found"][str(b)][st]["seeds"]) for b in BUDGETS]
            ax[0, j].plot(BUDGETS, ys, ls, marker=mk, ms=4.5, lw=1.6, color=col, label=name)
            if st != "clotho":
                diff = [[x - y for x, y in zip(d["found"][str(b)][st]["seeds"], d["found"][str(b)]["clotho"]["seeds"])] for b in BUDGETS]
                ax[1, j].errorbar(BUDGETS, [statistics.mean(v) for v in diff], yerr=[statistics.stdev(v) for v in diff],
                                  fmt=ls, marker=mk, ms=4.5, lw=1.6, capsize=2.5, color=col)
        ax[1, j].axhline(0, color=C.COL["clotho"], lw=1.2)
        ax[0, j].set_title(f"{C.LABEL[mo].split('-')[0]}, {su} ({len(d['systems'])} sys.)")
        ax[1, j].set_xticks(BUDGETS)
    ax[0, 0].set_ylabel("failures found\n(2B checks, all systems)")
    ax[1, 0].set_ylabel("minus Clotho")
    fig.supxlabel("labelling budget B (inputs labelled per system)", fontsize=9)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    out = C.PAPER / "figures" / "budget.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("budget", "fig_budget.py",
              "Failures found per labelling budget. Each strategy labels $B$ inputs per system, then a human checks the "
              "top $B$ of its ranking of the remaining inputs ($2B$ checks). Clotho: its own active reference sampling "
              "(plus its 10 synthetic initial tests), ranking by Clotho. Drain-propagate: one input per Drain group "
              "(Loghub-2.0 settings), largest groups first; unlabelled inputs inherit their group's label and are ranked "
              "inherited-FAIL, never-labelled groups, inherited-PASS, within each tier by Clotho fitted on the collected "
              "labels (+ Clotho) or at random (no Clotho). Top: failures found summed over systems, mean over 5 seeds. "
              "Bottom: difference to Clotho, mean $\\pm$ std over seeds. Label from samples 5--9 (PASS if $\\geq$3 of 5 "
              "strict matches).")
    print(out)


if __name__ == "__main__":
    main()
