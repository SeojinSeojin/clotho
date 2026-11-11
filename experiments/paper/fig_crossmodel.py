"""Figure (P2): per-template pass rates, Llama vs Gemma (J). Rate = mean over the template's capped-suite inputs of the
share of samples 5-9 that match the ground truth exactly; the 12 systems both models keep.

  cd experiments && python paper/fig_crossmodel.py   -> paper/figures/crossmodel_rates.pdf (+ .png, caption)
Source: recomputed with log_parsing_crossmodel.overlap's code path; asserted equal to reports/crossmodel_details.json.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402


def rates():
    import log_parsing_crossmodel as X
    R = {}
    for s in X.SYSTEMS:
        ds = X.ds_name(s, "capped")
        tm = [json.loads(l)["template"] for l in open(C.EXP.parent / "clotho" / "dataset" / X.TASK / f"{ds}.jsonl")]
        for m in ("llama", "gemma"):
            P = X.pass_matrix(m, ds)[:, 5:].mean(1)
            for t, v in zip(tm, P):
                R.setdefault((s, t), {}).setdefault(m, []).append(v)
    a = np.array([np.mean(v["llama"]) for v in R.values()])
    b = np.array([np.mean(v["gemma"]) for v in R.values()])
    n = np.array([len(v["llama"]) for v in R.values()])
    return a, b, n


def main():
    from scipy.stats import spearmanr
    plt = C.style()
    a, b, n = rates()
    st = json.loads((C.EXP / "reports" / "crossmodel_details.json").read_text())["overlap"]
    rho = float(spearmanr(a, b).correlation)
    assert abs(rho - st["spearman"]) < 1e-12 and len(a) == st["n_templates"]
    rng = np.random.RandomState(0)
    j = lambda x: np.clip(x + rng.uniform(-0.012, 0.012, len(x)), -0.02, 1.02)  # noqa: E731  (jitter: rates are multiples of 1/5n)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(j(a), j(b), s=6 + 2 * np.sqrt(n), color="#2a78d6", alpha=0.35, lw=0)
    ax.axhline(0.5, color=C.GRID, lw=0.9); ax.axvline(0.5, color=C.GRID, lw=0.9)
    ax.plot([0, 1], [0, 1], color=C.INK2, lw=0.8, ls="--")
    ax.set_xlabel(f"{C.LABEL['llama']} pass rate"); ax.set_ylabel(f"{C.LABEL['gemma']} pass rate")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.text(0.02, 0.98, f"{len(a):,} templates\nSpearman {rho:.2f}\nagreement (rate ≥ 0.5) {st['agree']:.2f}", va="top",
            transform=ax.transAxes, fontsize=7.5, color=C.INK2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C.GRID, alpha=0.95), zorder=5)
    fig.tight_layout()
    out = C.PAPER / "figures" / "crossmodel_rates.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("crossmodel_rates", "fig_crossmodel.py",
              "Per-template pass rates of the two parsers. Each point is a ground-truth template of the capped suites of the "
              "12 systems both models keep; rate = mean over its inputs of the share of samples 5--9 that match the ground "
              f"truth exactly (point size grows with the number of inputs; small jitter added). {st['both_fail']:.0%} of templates "
              f"fail for both models, {st['llama_only_fail']:.0%} only for Llama, {st['gemma_only_fail']:.0%} only for Gemma.")
    print(out)


if __name__ == "__main__":
    main()
