"""Figure (P6): same-group validation (W) in the cached-parser streams: recall, precision and share of mis-parsed lines
in flagged groups at a group's 2nd, 3rd and 5th line; base and strong parser, both models.

  cd experiments && python paper/fig_groupcheck.py   -> paper/figures/groupcheck.pdf (+ .png, caption groupcheck.tex)
Source: reports/groupcheck_details.json (log_parsing_groupcheck.py report).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KS = (2, 3, 5)
CELLS = [("llama/base", "#2a78d6", "-", "o"), ("llama/strong", "#eb6834", "--", "s"),
         ("gemma/base", "#1baf7a", "-", "^"), ("gemma/strong", "#eda100", "--", "D")]
PANELS = [("recall", "recall (wrong groups flagged)"), ("precision", "precision (flagged groups wrong)"),
          ("bad_lines_flagged", "share of mis-parsed lines\nin flagged groups")]


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "groupcheck_details.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True)
    for key, col, ls, mk in CELLS:
        if key not in J:
            continue
        m, p = key.split("/")
        for a, (f, lab) in zip(ax, PANELS):
            a.plot(KS, [J[key]["w2"][str(k)][f] for k in KS], ls, marker=mk, color=col, lw=1.6, ms=5,
                   label=f"{C.LABEL[m].split('-')[0]}, {p}")
    for a, (f, lab) in zip(ax, PANELS):
        a.set_title(lab)
        a.set_xticks(KS)
        a.set_ylim(0, 1.02)
    ax[1].set_xlabel("line of the group at which the check runs")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = C.PAPER / "figures" / "groupcheck.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    sg = {k: J[k]["w2"]["single_share"] for k, *_ in CELLS if k in J}
    C.caption("groupcheck", "fig_groupcheck.py",
              "Same-group validation of cached templates. When a group's 2nd, 3rd or 5th line arrives, the group is flagged "
              "if a line seen so far does not match the cached template with \\texttt{<*>} as a wildcard, or a token the "
              "template keeps constant differs from the first line's (flags are cumulative; no look-ahead). Wrong = cached "
              "template differs from the first line's ground truth. Macro means over systems. Groups that never get a 2nd "
              f"line ({min(sg.values()):.0%}--{max(sg.values()):.0%} of groups) cannot be checked.")
    print(out)


if __name__ == "__main__":
    main()
