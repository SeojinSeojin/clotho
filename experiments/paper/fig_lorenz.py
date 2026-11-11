"""Figure: group-size concentration (N1). Lorenz curves of Drain group sizes per system, and the share of wrong groups
among the 20 largest (Most frequent's first 20 reviews) vs the overall wrong-group share, per model x parser.

  cd experiments && python paper/fig_lorenz.py   -> paper/figures/lorenz.pdf (+ .png, caption lorenz.tex)
Source: results_phase3{,_gemma}/cpu/n1.json (log_parsing_cpu_analyses.py n1).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CELLS = [("llama", "base", "#2a78d6"), ("llama", "strong", "#eb6834"), ("gemma", "base", "#1baf7a"), ("gemma", "strong", "#eda100")]


def main():
    plt = C.style()
    from matplotlib.gridspec import GridSpec
    N = {m: json.loads((C.rp(m) / "cpu" / "n1.json").read_text()) for m in C.MODELS}
    systems = list(N["llama"]["systems"])
    fig = plt.figure(figsize=(10, 5.8))
    gs = GridSpec(3, 7, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.55, wspace=0.25)
    for i, s in enumerate(systems):
        ax = fig.add_subplot(gs[i // 7, i % 7])
        r = N["llama"]["systems"][s]
        x, y = r["lorenz"]
        ax.plot([0, 1], [0, 1], color=C.GRID, lw=1)
        ax.plot(x, y, color=C.INK, lw=1.6)
        ax.set_title(s, pad=2)
        ax.text(0.04, 0.93, f"Gini {r['gini']:.2f}\n{r['G']} groups\ntop 20: {r['top_share']['20']:.0%}",
                transform=ax.transAxes, va="top", fontsize=7, color=C.INK2)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7, length=2)
        if i % 7:
            ax.set_yticklabels([])
        if i // 7 == 0:
            ax.set_xticklabels([])
    fig.text(0.075, 0.70, "cumulative share of lines", rotation=90, va="center", fontsize=8, color=C.INK2)
    fig.text(0.50, 0.405, "cumulative share of groups (smallest first)", ha="center", fontsize=8, color=C.INK2)
    ax = fig.add_subplot(gs[2, :])
    w = 0.18
    for j, (m, p, col) in enumerate(CELLS):
        xs, top, allw = [], [], []
        for i, s in enumerate(systems):
            if p not in N[m]["systems"].get(s, {}).get("cells", {}):
                continue
            c = N[m]["systems"][s]["cells"][p]
            xs.append(i + (j - 1.5) * w); top.append(c["top20_wrong"]); allw.append(c["wrong_share"])
        ax.vlines(xs, allw, top, color=col, lw=1, alpha=0.6)
        ax.scatter(xs, allw, s=26, facecolors="white", edgecolors=col, lw=1.3, zorder=3)
        ax.scatter(xs, top, s=38, color=col, edgecolors="white", lw=0.6, zorder=4,
                   label=f"{C.LABEL[m].split('-')[0]}, {p} parser")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("share of wrong groups")
    ax.grid(axis="y", color=C.GRID, lw=0.8)
    ax.set_axisbelow(True)
    h, lab = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="o", ls="", color=C.INK2, ms=5), Line2D([], [], marker="o", ls="", mfc="white", mec=C.INK2, ms=5)]
    lab += ["among the 20 largest groups", "among all groups"]
    ax.legend(h, lab, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7.5)
    out = C.PAPER / "figures" / "lorenz.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    C.caption("lorenz", "fig_lorenz.py",
              "Why reviewing the most frequent groups first nearly matches the oracle. Top: Lorenz curves of Drain group "
              "sizes over each system's first 50,000 lines (Loghub-2.0 settings); the 20 largest groups hold most lines. "
              "Bottom: share of wrong groups (cached template differs from the first line's ground truth) among the 20 "
              "largest groups (filled) and among all groups (hollow), per model and parser (base: zero-shot greedy; strong: "
              "kNN demonstrations + AdaParser corrector + 10 verified seed groups, which are excluded from the 20).")
    print(out)


if __name__ == "__main__":
    main()
