"""Table (c): label leakage, leaky vs leak-free, post-generation methods (pre-generation rows for reference).

  cd experiments && python paper/table_leakage.py   -> paper/tables/leakage.tex
leaky (fair_leaky, the original protocol): label = > 5 of 10 samples pass, post-generation scores from the same
10 samples. leak-free (fair_leakfree): label from samples 5-9, scores from samples 0-4. Same reference sampling
and layer; template-capped suites, standard split (log_parsing_phase3_final.py).
"""
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METHODS = ["clotho", "MDSA", "tok_prob", "tok_ent", "sem_ent", "loh_var", "out_entropy"]
REFS = (100, 200)


def main():
    body = []
    for model in C.MODELS:
        Rk, Rf = C.capped(model, "fair_leaky"), C.capped(model, "fair_leakfree")
        n = len(C.kept_systems(Rf))
        cols = {}
        for ref in REFS:
            cols[("leaky", ref)] = {m: C.macro(Rk, m, ref) for m in METHODS}
            cols[("leakfree", ref)] = {m: C.macro(Rf, m, ref) for m in METHODS}
        bold = {k: C.best_key(v) for k, v in cols.items()}
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{C.LABEL[model]} ({n} systems)}}}} \\\\")
        for m in METHODS:
            cells = []
            for ref in REFS:
                a, b = cols[("leaky", ref)][m], cols[("leakfree", ref)][m]
                d = None if a is None or b is None else [x - y for x, y in zip(a, b)]
                cells += [C.fmt(a, bold[("leaky", ref)] == m), C.fmt(b, bold[("leakfree", ref)] == m),
                          "--" if d is None else f"{statistics.mean(d):+.3f} $\\pm$ {statistics.stdev(d):.3f}"]
            body.append(C.NAMES[m] + "".join(f" & {c}" for c in cells) + " \\\\")
        if model != C.MODELS[-1]:
            body.append("\\midrule")
    header = ["& \\multicolumn{3}{c}{@100} & \\multicolumn{3}{c}{@200} \\\\",
              "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
              "Method & leaky & leak-free & $\\Delta$ & leaky & leak-free & $\\Delta$ \\\\"]
    caption = ("Label leakage in the original protocol. Leaky: the pass label (more than 5 of 10 samples pass) and "
               "the post-generation scores come from the same 10 generations. Leak-free: label from samples 5--9 "
               "(PASS if $\\geq$3 of 5), scores from samples 0--4. $\\Delta$ = leaky minus leak-free, paired per "
               "seed. Clotho and MDSA never see generations; their $\\Delta$ is the effect of the noisier "
               "5-sample label alone. Template-capped suites, standard split, balanced reference sampling, layer "
               "16 (Llama) / 21 (Gemma), rules\\_template, strict match; mean $\\pm$ std over 5 seeds of the macro "
               "mean over systems. Sem-Ent: semantic entropy (DeBERTa-v2-xlarge-mnli). $^*$ needs the generated "
               "outputs. Bold: best in its column (within a model) and ahead of the runner-up on every seed; "
               "$\\Delta$ columns are not marked.")
    C.table(C.PAPER / "tables" / "leakage.tex", "table_leakage.py", caption, "tab:leakage", "lcccccc", header, body)


if __name__ == "__main__":
    main()
