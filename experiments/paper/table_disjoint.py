"""Table (b): template-disjoint split, both models, template-capped suites, @100 and @200.

  cd experiments && python paper/table_disjoint.py   -> paper/tables/disjoint.tex
Disjoint split (log_parsing_phase3_final.py): templates shuffled per seed and assigned to the reference pool
until it holds >= 50% of the inputs; test = inputs of the remaining templates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METHODS = ["clotho", "MDSA", "GMMb", "lookup_drain", "out_entropy", "loh_var", "clotho_oe", "random"]
REFS = (100, 200)


def main():
    cols, ns = [], {}
    for model in C.MODELS:
        R, FU = C.capped(model, split="disjoint"), C.followups(model, "disjoint")
        ns[model] = len(C.kept_systems(R))
        for ref in REFS:
            cols.append({m: (C.macro(FU, "clotho+out_entropy", ref, C.fu_val) if m == "clotho_oe" else C.macro(R, m, ref))
                         for m in METHODS})
    bold = [C.best_key(c) for c in cols]
    header = [f"& \\multicolumn{{2}}{{c}}{{{C.LABEL['llama']} ({ns['llama']})}} & "
              f"\\multicolumn{{2}}{{c}}{{{C.LABEL['gemma']} ({ns['gemma']})}} \\\\",
              "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}", "Method & @100 & @200 & @100 & @200 \\\\"]
    body = [C.NAMES[m] + "".join(f" & {C.fmt(c[m], bold=(b == m))}" for c, b in zip(cols, bold)) + " \\\\"
            for m in METHODS]
    caption = ("Template-disjoint split: no test input shares its Loghub template with a reference input (per "
               "seed, templates are shuffled and assigned to the reference pool until it holds at least half of "
               "the inputs; test = the rest). Template-capped suites, number of systems in parentheses. "
               + C.SETTING_TEXT + " Bold: best in its column and ahead of the runner-up on every seed.")
    C.table(C.PAPER / "tables" / "disjoint.tex", "table_disjoint.py", caption, "tab:disjoint", "lcccc", header, body)


if __name__ == "__main__":
    main()
