"""Table (d): cross-system transfer, both models: reference set from system A, test = all 500 inputs of B.

  cd experiments && python paper/table_cross.py   -> paper/tables/cross.tex
Per-job JSON from log_parsing_cross_fair.py (fair_leakfree config); within-system Clotho on B from
log_parsing_phase3_final.py (fair_leakfree, standard split).
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PAIRS = ["hdfs__spark", "spark__hdfs", "hadoop__spark", "hadoop__hdfs", "bgl__thunderbird"]
PRE = ["clotho", "MDSA", "lookup_drain"]
REFS = (100, 200)


def per_seed(model, pair, m, ref):
    out = []
    for k in C.SEEDS:
        d = json.loads((C.rp(model) / "cross_fair" / pair / f"seed{k}.json").read_text())
        out.append(C.row_val(d, m, ref))
    return out


def main():
    body = []
    for model in C.MODELS:
        within = C.capped(model)
        body.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{C.LABEL[model]}}}}} \\\\")
        rows = {}
        for pair in PAIRS:
            a, b = pair.split("__")
            rows[f"{a.upper() if a in ('hdfs', 'bgl') else a.capitalize()} $\\to$ "
                 f"{b.upper() if b in ('hdfs', 'bgl') else b.capitalize()}"] = {
                **{(m, ref): per_seed(model, pair, m, ref) for m in PRE + ["out_entropy"] for ref in REFS},
                "within": [C.row_val(within[f"loghub_{b}_500"][k], "clotho", 200) for k in C.SEEDS]}
        keys = list(next(iter(rows.values())).keys())
        rows["Mean over pairs"] = {key: [statistics.mean(r[key][k] for r in rows.values()) for k in range(len(C.SEEDS))]
                                   for key in keys}
        for name, r in rows.items():
            if name == "Mean over pairs":
                body.append("\\cmidrule(lr){2-9}")
            cells = []
            for ref in REFS:
                b = C.best_key({m: r[(m, ref)] for m in PRE})
                cells += [C.fmt(r[(m, ref)], b == m) for m in PRE]
            cells += [C.fmt(r[("out_entropy", 200)]), C.fmt(r["within"])]
            body.append(name + "".join(f" & {c}" for c in cells) + " \\\\")
        if model != C.MODELS[-1]:
            body.append("\\midrule")
    header = ["& \\multicolumn{3}{c}{@100} & \\multicolumn{3}{c}{@200} & & \\\\",
              "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
              "A $\\to$ B & Clotho & MDSA & Lookup (Drain) & Clotho & MDSA & Lookup (Drain) & Output entropy$^*$ & "
              "Within B (Clotho @200) \\\\"]
    caption = ("Cross-system transfer: the reference set (and its labels) comes from system A, the test set is all "
               "500 inputs of system B. Lookup (Drain) uses each system's own Drain run, so a B input matches "
               "only an identical Drain template from A. Output entropy needs no reference set (its value does "
               "not depend on the reference size). Within B = Clotho trained and tested on B (standard split, "
               "@200). " + C.SETTING_TEXT + " Bold: within each reference size, the best of Clotho, MDSA and "
               "Lookup (Drain) in that row, only if it beats the runner-up on every seed.")
    C.table(C.PAPER / "tables" / "cross.tex", "table_cross.py", caption, "tab:cross", "lcccccccc", header, body)


if __name__ == "__main__":
    main()
