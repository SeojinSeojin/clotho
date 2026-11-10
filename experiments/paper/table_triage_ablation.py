"""Table: TRIAGE ablation (E2), share of mis-parsed lines fixed after 20 and 50 reviews; both models.

  cd experiments && python paper/table_triage_ablation.py   -> paper/tables/triage_ablation.tex
Source: reports/triage_details.json (log_parsing_triage.py report), greedy cache, all systems.
Cells: mean ± std over 5 seeds of the macro mean over systems; Δ = row minus TRIAGE (seeds with that sign).
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROWS = [("triage", "TRIAGE"), ("no_prior", "$-$ prior (own verdicts only)"), ("dsize", "$-$ active selection"),
        ("no_prop", "$-$ propagation"), ("no_freq", "$-$ frequency weighting"), ("oracle", "Oracle")]
NS = ("20", "50")


def main():
    J = json.loads((C.EXP / "reports" / "triage_details.json").read_text())
    models = [m for m in C.MODELS if m in J]
    body = []
    for key, name in ROWS:
        cells = []
        for mo in models:
            E = J[mo]["e1/base/all"]
            for n in NS:
                v, t = E[key][n], E["triage"][n]
                if key == "triage":
                    cells.append(C.fmt(v))
                else:
                    d = [a - b for a, b in zip(v, t)]
                    mu = statistics.mean(d)
                    k = sum((x > 0) == (mu > 0) and x != 0 for x in d)
                    cells.append(f"{C.fmt(v)} ({mu:+.3f}, {k}/5)")
        body.append(name + "".join(f" & {c}" for c in cells) + " \\\\")
    header = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{C.LABEL[m]} ({len(J[m]['systems'])} systems)}}" for m in models) + " \\\\",
              "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))),
              "variant" + " & 20 reviews & 50 reviews" * len(models) + " \\\\"]
    C.table(C.PAPER / "tables" / "triage_ablation.tex", "table_triage_ablation.py",
            "TRIAGE ablation in the cached-parser simulation: share of mis-parsed lines fixed after 20 and 50 reviews "
            "(mean $\\pm$ std over 5 seeds of the macro mean over systems; in brackets the difference to TRIAGE and the "
            "seeds with that sign). One component removed at a time: prior = cross-system transfer model (replaced by a "
            "probe on the stream's own verdicts); active selection = Clotho-style uncertainty and diversity among the "
            "top-$M$ (replaced by top risk $\\times$ lines); propagation = a correction covers the whole Drain group "
            "(replaced by lines identical to the reviewed one); frequency weighting = risk $\\times$ lines (replaced by risk).",
            "tab:triage-ablation", "l" + "rr" * len(models), header, body)


if __name__ == "__main__":
    main()
