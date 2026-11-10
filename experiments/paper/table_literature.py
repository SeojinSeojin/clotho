"""Table (e): literature baselines. LLMLog's selection score as a failure ranking (A) and predictors against the
label after AdaParser's template corrector (B); both models, capped and natural suites, @100 and @200.

  cd experiments && python paper/table_literature.py   -> paper/tables/literature.tex
Source: reports/log_parsing_literature_details.json (log_parsing_literature.py report; per-seed macro means).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COLS = [("llama", "capped"), ("llama", "natural"), ("gemma", "capped"), ("gemma", "natural")]
A_ROWS = [("clotho", "Clotho"), ("lookup_drain", "Lookup (Drain)"), ("out_entropy", "Output entropy$^*$"),
          ("llmlog", "LLMLog score$^*$"), ("llmlog_conf", "\\quad confidence only$^*$"),
          ("llmlog_rep", "\\quad representativeness only"), ("random", "Random")]
B_ROWS = [("clotho", "Clotho"), ("lookup_drain", "Lookup (Drain)"), ("out_entropy", "Output entropy$^*$"), ("random", "Random")]


def main():
    J = json.loads((C.EXP / "reports" / "log_parsing_literature_details.json").read_text())
    body = []
    for ri, ref in enumerate((100, 200)):
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{(A) original label, reference size {ref}}}}} \\\\")
        cols = [{m: J[f"{mo}/{su}"]["A"][m]["auc_seeds"][ri] for m, _ in A_ROWS} for mo, su in COLS]
        best = [C.best_key(c) for c in cols]
        for m, name in A_ROWS:
            body.append(name + " & " + " & ".join(C.fmt(c[m], bold=(b == m)) for c, b in zip(cols, best)) + " \\\\")
        body.append("\\midrule")
    for ri, ref in enumerate((100, 200)):
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{(B) label after AdaParser's corrector, reference size {ref}}}}} \\\\")
        cols = [{m: J[f"{mo}/{su}"]["B"][m]["corrected_seeds"][ri] for m, _ in B_ROWS} for mo, su in COLS]
        best = [C.best_key(c) for c in cols]
        for m, name in B_ROWS:
            body.append(name + " & " + " & ".join(C.fmt(c[m], bold=(b == m)) for c, b in zip(cols, best)) + " \\\\")
        body.append("\\midrule")
    fx = [J[f"{mo}/{su}"]["corrector"] for mo, su in COLS]
    body.append("Corrector: failing inputs fixed & " + " & ".join(f"{c['fixed']} / {c['fail_before']} ({100 * c['fixed'] / c['fail_before']:.1f}\\%)" for c in fx) + " \\\\")
    body.append("Corrector: passing inputs broken & " + " & ".join(f"{c['broken']} / {c['pass_before']}" for c in fx) + " \\\\")
    body.append("Corrector: LLM calls per output & " + " & ".join(f"{c['calls_per_output']:.2f}" for c in fx) + " \\\\")
    header = ["& \\multicolumn{2}{c}{Llama-3.1-8B} & \\multicolumn{2}{c}{Gemma-2-9B} \\\\",
              "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}", "& capped & natural & capped & natural \\\\"]
    cap = ("Literature baselines. (A) LLMLog's log-selection score (official code, greedy pick order of "
           "$\\lambda C + (1-\\lambda)\\,$coverage, $\\lambda = 0.5$; confidence $C$ from generation samples 0--4) used as a "
           "failure ranking, which is not its purpose (it picks logs to annotate as demonstrations). (B) The pass label "
           "recomputed after AdaParser's template corrector (official prompts and checks, greedy, same model) rewrote "
           "label samples 5--9; Clotho is re-run under the corrected label, output entropy uses the uncorrected samples "
           "0--4. Gemma natural (B) has 11 systems (Linux has 10 passing inputs after correction). " + C.SETTING_TEXT +
           " Bold: best in its column and ahead of the runner-up on every seed.")
    C.table(C.PAPER / "tables" / "literature.tex", "table_literature.py", cap, "tab:literature", "lcccc", header, body[:-1] if body[-1] == "\\midrule" else body)


if __name__ == "__main__":
    main()
