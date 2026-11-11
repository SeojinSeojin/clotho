"""Table (P5): zero-extra-call failure prediction on unseen templates (U + V), template-disjoint split, @100 and @200,
both models.

  cd experiments && python paper/table_unseen.py   -> paper/tables/unseen.tex
Source: reports/unseen_details.json (log_parsing_unseen.py report). Cells: mean AUC over 5 seeds of the macro mean
over systems; in brackets the paired difference to Clotho (seeds with that sign out of 5).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROWS = [("clotho", "Clotho"), ("MDSA", "MDSA"), ("lookup_drain", "Lookup (Drain)"),
        ("U:S1", "S1 hidden states, discriminative"), ("U:S2", "S2 line structure"), ("U:S3", "S3 checks on the parser's answer"),
        ("U:S1+S2", "S1+S2"), ("U:S1+S3", "S1+S3"), ("U:S2+S3", "S2+S3"), ("U:S1+S2+S3", "S1+S2+S3"),
        ("V:V1ens", "V1 multi-layer probe ensemble"), ("V:V5", "V5 Drain disagreement"),
        ("U+V1", "U best + V1 multi-layer"), ("U+V3", "U best + V3 reasoning"), ("U+V4", "U best + V4 same-group"),
        ("U+V5", "U best + V5 Drain disagreement"), ("U+V6", "U best + V6 soft lookup"), ("U+V7", "U best + V7 cross-system"),
        ("U+V8a", "U best, template-level targets"), ("U+V8b", "U best, template-grouped CV"),
        ("out_entropy", "Output entropy (5 extra samples)$^*$"), ("random", "Random")]


def main():
    J = json.loads((C.EXP / "reports" / "unseen_details.json").read_text())
    models = [m for m in C.MODELS if m in J]
    body = []
    for key, name in ROWS:
        cells = []
        for m in models:
            for ref in (100, 200):
                d = J[m].get(f"{m}/disjoint/@{ref}/all", {}).get(key)
                if not d:
                    cells.append("--")
                    continue
                c = f"{d['auc']:.3f}"
                if key != "clotho" and d.get("minus_clotho") is not None:
                    c += f" ({d['minus_clotho']:+.3f})"
                cells.append(c)
        best = " (U best)" if any(key == f"U:{J[m]['U_best']}" for m in models) else ""
        body.append(name + best + "".join(f" & {c}" for c in cells) + " \\\\")
    header = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{C.LABEL[m]}}}" for m in models) + " \\\\",
              "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))),
              "signal" + " & @100 & @200" * len(models) + " \\\\"]
    C.table(C.PAPER / "tables" / "unseen.tex", "table_unseen.py",
            "Failure prediction on unseen templates (template-disjoint split), ROC-AUC with PASS as positive, mean over 5 "
            "seeds of the macro mean over systems; in brackets the paired difference to Clotho. Every row except output "
            "entropy needs no extra LLM call (S3, V3--V5 read the parser's own answer, sample 0; labels come from samples "
            "5--9). Signal sets and hyperparameters tuned on HDFS, Spark and Linux only. U best = the set chosen on those "
            "systems. $^*$ needs 5 extra samples per input.",
            "tab:unseen", "l" + "rr" * len(models), header, body)


if __name__ == "__main__":
    main()
