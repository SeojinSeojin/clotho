"""Figure (c): cross-system transfer, ROC-AUC vs number of target labels k (Experiment D, paired); both models, capped
and natural, held-out targets.

  cd experiments && python paper/fig_transfer.py   -> paper/figures/transfer.pdf (+ .png, caption transfer.tex)
Source: reports/log_parsing_transfer_paired.json (log_parsing_transfer_paired.py report). All lines at a given k are
scored on the same target inputs (excluding the k labels and the 90 inputs within-system Clotho @100 labels).
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KS = (0, 10, 25, 50)
LINES = [("refit", "Source model refit with $k$ target labels (protocol)", C.COL["MDSA"], "-", "o"),
         ("refit_lr", "Logistic probe refit (not protocol-chosen)", C.COL["oe"], "--", "s"),
         ("within_k", "Within-system Clotho, $k$ labels", C.COL["clotho"], ":", "^"),
         ("within100", "Within-system Clotho @100", C.COL["clotho"], "-", None)]
COLS = [("llama", "capped"), ("llama", "natural"), ("gemma", "capped"), ("gemma", "natural")]


def main():
    plt = C.style()
    J = json.loads((C.EXP / "reports" / "log_parsing_transfer_paired.json").read_text())
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
    for j, (mo, su) in enumerate(COLS):
        d = J[f"{mo}/{su}"]
        for m, name, col, ls, mk in LINES:
            ks = [k for k in KS if not (m == "within_k" and k == 0)]
            mu = [statistics.mean(d[m][str(k)]) for k in ks]
            sd = [statistics.stdev(d[m][str(k)]) for k in ks]
            if m == "within100":
                ax[j].plot(ks, mu, ls, lw=1.2, color=col, alpha=0.9, label=name)
                ax[j].fill_between(ks, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], color=col, alpha=0.12, lw=0)
            else:
                ax[j].errorbar(ks, mu, yerr=sd, fmt=ls, marker=mk, ms=4.5, lw=1.6, capsize=2.5, color=col, label=name)
        ax[j].axhline(0.5, color=C.INK2, lw=0.8, ls=(0, (2, 3)))
        ax[j].set_title(f"{C.LABEL[mo].split('-')[0]}, {su} ({len(d['targets'])} targets)")
        ax[j].set_xticks(KS)
    ax[0].set_ylabel("ROC-AUC on the held-out system")
    fig.supxlabel("target labels $k$", fontsize=9)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    out = C.PAPER / "figures" / "transfer.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    C.caption("transfer", "fig_transfer.py",
              "Failure prediction on a new log system (leave-one-system-out; held-out targets only, i.e. not HDFS, Spark "
              "or Linux, which chose every hyperparameter). Source model: trained on the pass/fail labels of all other "
              "systems (protocol-chosen: MLP probe for Llama, logistic probe + generic log features for Gemma), refit with "
              "$k$ random target labels; the logistic probe is shown for comparison but was not the protocol's choice. "
              "Within-system Clotho is fitted on the same $k$ labels, or on its own 90 actively sampled labels (@100, band "
              "= $\\pm$ std). At each $k$ all lines are scored on the same target inputs. Label from samples 5--9; ROC-AUC, "
              "mean $\\pm$ std over 5 seeds of the macro mean over targets.")
    print(out)


if __name__ == "__main__":
    main()
