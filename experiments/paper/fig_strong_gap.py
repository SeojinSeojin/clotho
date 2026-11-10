"""Figure (F3): gap between D x size and Most frequent (share of mis-parsed lines fixed at 20 reviews) against each
system's share of wrong groups, base and strong cached parser; both models.

  cd experiments && ../.venv/bin/python paper/fig_strong_gap.py   -> paper/figures/strong_gap.pdf (+ .png, caption .tex)
Source: reports/strong_details.json (log_parsing_strong.py report). One point per (system, parser), seed mean;
a thin line joins a system's base and strong points.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "strong_details.json").read_text())
    models = [m for m in C.MODELS if m in J]
    fig, ax = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), sharey=True, squeeze=False)
    for j, mo in enumerate(models):
        a = ax[0, j]
        S = J[mo]["systems"]
        for s, v in S.items():
            a.plot([v["wrong_base"], v["wrong"]], [v["gap20_base"], v["gap20"]], color=C.GRID, lw=0.9, zorder=1)
        a.scatter([v["wrong_base"] for v in S.values()], [v["gap20_base"] for v in S.values()], s=26, marker="o",
                  facecolor="white", edgecolor=C.COL["MDSA"], lw=1.4, label="base parser (zero-shot)", zorder=3)
        a.scatter([v["wrong"] for v in S.values()], [v["gap20"] for v in S.values()], s=26, marker="s",
                  color=C.COL["clotho"], edgecolor="white", lw=0.6, label="strong parser (ICL + corrector)", zorder=3)
        lo = min(min(v["gap20"], v["gap20_base"]) for v in S.values())
        for s, v in S.items():
            if abs(v["gap20"]) > 0.02 or abs(v["gap20_base"]) > 0.02:
                x, y = (v["wrong"], v["gap20"]) if abs(v["gap20"]) >= abs(v["gap20_base"]) else (v["wrong_base"], v["gap20_base"])
                a.annotate(s, (x, y), xytext=(4, -3), textcoords="offset points", fontsize=7, color=C.INK2)
        a.axhline(0, color=C.INK2, lw=0.8)
        f3 = J[mo]["f3"]
        a.set_title(f"{C.LABEL[mo]} (Spearman {f3['spearman']:+.2f}, p = {f3['p']:.2f})")
        a.set_xlabel("share of groups with a wrong cached template")
        a.set_xlim(0, 1)
        a.set_ylim(min(-0.05, lo - 0.03), 0.08)
    ax[0, 0].set_ylabel("share fixed @20: D x size minus Most frequent")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = C.PAPER / "figures" / "strong_gap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("strong_gap", "fig_strong_gap.py",
              "Does risk prediction pay off when the cached parser is more accurate? Per system, the share of "
              "mis-parsed lines fixed after 20 reviews by our transfer prior (risk $\\times$ lines) minus that of "
              "reviewing the most frequent groups first (mean over 5 seeds), against the share of Drain groups whose "
              "cached template is wrong. Circles: zero-shot parser; squares: the same model with in-context "
              "demonstrations from its own cache, 10 verified seed templates and AdaParser's corrector. Lines join a "
              "system's two parsers. Points above zero favour risk prediction.")
    print(out)


if __name__ == "__main__":
    main()
