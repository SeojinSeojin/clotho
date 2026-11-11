"""Table (P3): downstream anomaly detection F1 vs reviews (H count-based detectors, N3 template-word detectors), base
parser, both models.

  cd experiments && python paper/table_downstream.py   -> paper/tables/downstream.tex
Sources: reports/downstream_details.json (H, log_parsing_downstream.py) and results_phase3{,_gemma}/cpu/n3.json
(N3, log_parsing_cpu_analyses.py n3). Seeded rows (D x size; Isolation Forest seeds) are means over seeds 0-4.
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DETS = [("PCA", "PCA (counts)", "H"), ("IM", "Invariant Mining (counts)", "H"), ("IF", "Isolation Forest (template words)", "N3"),
        ("OCSVM", "One-class SVM (template words)", "N3")]


def h_cell(H, system, model, o, N, det):
    try:
        return statistics.mean(x[det]["f1"] for x in H[system]["runs"][f"{model}/base"][o][str(N)])
    except KeyError:
        return None


def n3_cell(R, system, row, det):
    r = R.get(system) if R else None
    if not r:
        return None
    x = r[row]
    if row == "dsize20":
        return statistics.mean(v[det]["f1"] for v in x.values())
    v = x[det]
    return statistics.mean(y["f1"] for y in v.values()) if det == "IF" else v["f1"]


def main():
    H = json.loads((C.EXP / "reports" / "downstream_details.json").read_text())
    N3 = {m: json.loads((C.rp(m) / "cpu" / "n3.json").read_text())["systems"] for m in C.MODELS if (C.rp(m) / "cpu" / "n3.json").exists()}
    f = lambda v: "--" if v is None else f"{v:.3f}"  # noqa: E731
    body = []
    for system in ("HDFS", "BGL"):
        for det, name, src in DETS:
            if src == "H":
                gt = H[system]["gt"][det]["f1"]
            else:
                gt = next((n3_cell(N3[m], system, "gt", det) for m in N3 if system in N3[m]), None)
            cells = [f(gt)]
            for m in C.MODELS:
                if src == "H":
                    cells += [f(h_cell(H, system, m, "frequent", 0, det)), f(h_cell(H, system, m, "frequent", 20, det)),
                              f(h_cell(H, system, m, "dsize", 20, det))]
                else:
                    cells += [f(n3_cell(N3.get(m), system, k, det)) for k in ("n0", "frequent20", "dsize20")]
            body.append(f"{system} & {name} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    header = [" & & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{C.LABEL[m]}, base parser}}" for m in C.MODELS) + " \\\\",
              "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
              "system & detector & GT templates" + " & $N=0$ & MF @20 & D$\\times$size @20" * len(C.MODELS) + " \\\\"]
    C.table(C.PAPER / "tables" / "downstream.tex", "table_downstream.py",
            "Downstream anomaly detection on the cached parser's templates: F1 (anomaly = positive) on the second half of "
            "sessions (HDFS: blocks; BGL: windows of 20 lines), detectors fit unsupervised on the first half. Count-based "
            "detectors (loglizer PCA, Invariant Mining) see only the line partition and are unchanged by template text; "
            "template-word detectors (TF-IDF of template words, Isolation Forest and one-class SVM) depend on it. GT = "
            "ground-truth templates; MF = Most frequent; D$\\times$size and Isolation Forest: means over 5 seeds. -- = "
            "Llama base cache for HDFS not available.",
            "tab:downstream", "llr" + "rrr" * len(C.MODELS), header, body[:-1])


if __name__ == "__main__":
    main()
