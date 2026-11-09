"""Table (a): main result, both models, template-capped vs natural suites, standard split, @100 and @200.

  cd experiments && python paper/table_main.py   -> paper/tables/main.tex
Capped = the 500-message Loghub-2.0 suites (dedup, per-template cap; log_parsing_phase3_final.py, Clotho +
output entropy from log_parsing_followups.py). Natural = 500 raw lines drawn uniformly (log_parsing_natural.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METHODS = ["clotho", "MDSA", "GMMb", "lookup_drain", "out_entropy", "loh_var", "clotho_oe", "random"]
REFS = (100, 200)


def column(model, suite, ref):
    if suite == "capped":
        R, FU = C.capped(model), C.followups(model)
        get = lambda m: C.macro(FU, "clotho+out_entropy", ref, C.fu_val) if m == "clotho_oe" else C.macro(R, m, ref)  # noqa: E731
        n = len(C.kept_systems(R))
    else:
        R = C.natural(model)
        get = lambda m: C.macro(R, m, ref)  # noqa: E731
        n = len(C.kept_systems(R))
    return {m: get(m) for m in METHODS}, n


def main():
    cols, ns = [], {}
    for model in C.MODELS:
        for suite in ("capped", "natural"):
            for ref in REFS:
                col, n = column(model, suite, ref)
                cols.append(col)
                ns[(model, suite)] = n
    bold = [C.best_key(c) for c in cols]
    header = ["& \\multicolumn{4}{c}{" + C.LABEL["llama"] + "} & \\multicolumn{4}{c}{" + C.LABEL["gemma"] + "} \\\\",
              "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
              "& \\multicolumn{2}{c}{capped (" + str(ns[("llama", "capped")]) + ")} & \\multicolumn{2}{c}{natural ("
              + str(ns[("llama", "natural")]) + ")} & \\multicolumn{2}{c}{capped (" + str(ns[("gemma", "capped")])
              + ")} & \\multicolumn{2}{c}{natural (" + str(ns[("gemma", "natural")]) + ")} \\\\",
              "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
              "Method" + " & @100 & @200" * 4 + " \\\\"]
    body = [C.NAMES[m] + "".join(f" & {C.fmt(c[m], bold=(b == m))}" for c, b in zip(cols, bold)) + " \\\\"
            for m in METHODS]
    caption = ("Main result: ROC-AUC for predicting log-parsing failures, standard split (Clotho's protocol), "
               "reference sizes 100 and 200. Capped = the 500-message Loghub-2.0 suites (deduplicated, inputs per "
               "template capped); natural = 500 log lines drawn uniformly from each system's raw log. Number of "
               "systems in parentheses (Llama: Proxifier dropped, natural Apache kept; Gemma: Apache and Proxifier "
               "dropped). Lookup (Drain) = failure rate of reference inputs in the same Drain group (Loghub-2.0 "
               "settings). Clotho + output entropy = rank average of the two scores. " + C.SETTING_TEXT +
               " Bold: best in its column and ahead of the runner-up on every seed (no bold otherwise).")
    C.table(C.PAPER / "tables" / "main.tex", "table_main.py", caption, "tab:main", "l" + "c" * 8, header, body)


if __name__ == "__main__":
    main()
