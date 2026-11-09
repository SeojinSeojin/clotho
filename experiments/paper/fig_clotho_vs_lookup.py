"""Figure 2: Clotho minus Lookup (Drain) per system vs the share of test inputs whose Drain group is already in
the reference set; template-capped vs natural suites, both models, @100 and @200.

  cd experiments && python paper/fig_clotho_vs_lookup.py   -> paper/figures/clotho_vs_lookup.pdf
Per system: mean over 5 seeds of (Clotho AUC - Lookup (Drain) AUC) and of the Drain-seen share
(runs[ref].drain_seen_frac: test inputs whose Drain group occurs among the non-initial reference inputs).
Capped: log_parsing_phase3_final.py jobs; natural: log_parsing_natural.py jobs. fair_leakfree, standard split.
"""
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REFS = (100, 200)


def points(R, ref):
    out = {}
    for s in C.kept_systems(R):
        d = [C.row_val(R[s][k], "clotho", ref) - C.row_val(R[s][k], "lookup_drain", ref) for k in C.SEEDS]
        x = [R[s][k]["runs"][str(ref)]["drain_seen_frac"] for k in C.SEEDS]
        out[C.short(s) if s.startswith("loghub") else s.lower()] = (statistics.mean(x), statistics.mean(d))
    return out


def main():
    plt = C.style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for i, model in enumerate(C.MODELS):
        cap, nat = C.capped(model), C.natural(model)
        for j, ref in enumerate(REFS):
            ax = axes[i, j]
            pc, pn = points(cap, ref), points(nat, ref)
            for s in pc:
                if s in pn:
                    ax.plot([pc[s][0], pn[s][0]], [pc[s][1], pn[s][1]], color=C.GRID, lw=1, zorder=1)
            ax.axhline(0, color=C.INK2, lw=0.8, zorder=2)
            for P, name, col, mk in ((pc, "capped suite", C.COL["clotho"], "o"), (pn, "natural sample", C.COL["MDSA"], "s")):
                xs, ys = [v[0] for v in P.values()], [v[1] for v in P.values()]
                ax.scatter(xs, ys, s=30, marker=mk, color=col, edgecolor="white", linewidth=1, zorder=3,
                           label=f"{name} ({len(P)} systems)")
                mx, my = statistics.mean(xs), statistics.mean(ys)
                ax.scatter([mx], [my], s=90, marker=mk, facecolor="none", edgecolor=col, linewidth=1.8, zorder=4)
            ax.set_title(f"{C.LABEL[model]}, reference size {ref}", color=C.INK)
            ax.set_xlim(-0.03, 1.03)
            if i == 0 and j == 0:
                ax.legend(loc="lower left", frameon=False)
    fig.supylabel("Clotho minus Lookup (Drain), ROC-AUC", fontsize=9)
    fig.supxlabel("share of test inputs whose Drain group is in the reference set", fontsize=9, y=0.02)
    fig.text(0.01, -0.02, "Each small marker is one system (mean over 5 seeds); large hollow markers are the means "
             "over systems; grey lines join a system's capped and natural points.", fontsize=7.5, color=C.INK2)
    fig.tight_layout()
    out = C.PAPER / "figures" / "clotho_vs_lookup.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
