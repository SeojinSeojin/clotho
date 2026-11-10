"""Figure: parsing accuracy vs number of human reviews in the cached-parser simulation (TRIAGE, E1); both models.

  cd experiments && python paper/fig_triage.py   -> paper/figures/triage_reviews.pdf (+ .png, caption triage_reviews.tex)
Source: reports/triage_details.json (log_parsing_triage.py report). Macro mean PA over systems, mean over 5 seeds;
band = ± std over seeds (seeded orderings only).
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NS = (0, 5, 10, 20, 50, 100)
ORD = [("triage", "TRIAGE", C.COL["clotho"], "-", "o"),
       ("dsize", "TRIAGE without active selection", C.COL["c+oe"], (0, (3, 1, 1, 1)), "P"),
       ("frequent", "Most frequent", C.COL["MDSA"], "--", "s"),
       ("llmlog", "LLMLog confidence x size", C.COL["lookup_drain"], "-.", "^"),
       ("clotho", "Clotho alone", C.COL["oe"], ":", "D"),
       ("random", "Random", C.INK2, (0, (1, 1.5)), "v"),
       ("oracle", "Oracle", C.INK, (0, (5, 2)), None)]


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "triage_details.json").read_text())
    models = [m for m in C.MODELS if m in J]
    fig, ax = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), sharey=True, squeeze=False)
    for j, mo in enumerate(models):
        P = J[mo]["e1/base/all/pa"]
        a = ax[0, j]
        for key, name, col, ls, mk in ORD:
            ys = [statistics.mean(P[key][str(n)]) for n in NS]
            sd = [statistics.stdev(P[key][str(n)]) for n in NS]
            a.plot(NS, ys, linestyle=ls, marker=mk, ms=4, lw=1.6, color=col, label=name)
            if max(sd) > 0:
                a.fill_between(NS, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)], color=col, alpha=0.15, lw=0)
        a.set_xscale("symlog", linthresh=5)
        a.set_xticks(NS)
        a.set_xticklabels([str(n) for n in NS])
        a.set_title(f"{C.LABEL[mo]} ({len(J[mo]['systems'])} systems)")
        a.set_xlabel("human reviews (one cached template each)")
    ax[0, 0].set_ylabel("parsing accuracy (all lines)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    out = C.PAPER / "figures" / "triage_reviews.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("triage_reviews", "fig_triage.py",
              "Parsing accuracy of a cached LLM log parser versus the number of human reviews. Per system, the first "
              "50,000 Loghub-2.0 lines in log order; Drain (Loghub-2.0 settings) groups lines, the LLM parses the first "
              "line of each new group (rules\\_template, greedy) and the template is cached for the group. A review "
              "replaces one cached template with the group's correct template. Macro mean over systems, mean over 5 "
              "seeds (band: $\\pm$ std over seeds). TRIAGE: cross-system prior refit with verdicts, Clotho-style "
              "uncertainty and diversity selection among the top-$M$ by risk $\\times$ lines; without active selection: top risk $\\times$ lines; Oracle: wrong templates by "
              "lines they break.")
    print(out)


if __name__ == "__main__":
    main()
