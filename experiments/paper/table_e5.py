"""Table: E5, leakage of the paper's post-generation baselines on the original Clotho tasks (Llama).

  cd experiments && python paper/table_e5.py   -> paper/tables/e5_leakage.tex
Source: reports/e5_leakage.json (log_parsing_e5_leakage.py). Spearman with the test score (the paper's RQ3 measure),
scores oriented so higher = PASS. Leaky = paper protocol; leak-free = label from 5 samples, score from the other 5
(5 seeded splits, mean ± std).
"""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

M = [("tokprob", "Tok-Prob$^*$"), ("tokent", "Tok-Ent$^*$"), ("ansent", "Output entropy$^*$"), ("mdsa", "MDSA")]
ALIAS = {"syntactic_bug_detection": "SYN-BUG", "spell_check": "SPELL-CHK", "github_typo_check": "GH-TYPO",
         "json_repair": "JSON-FIX", "pos_detection": "POS-TAG", "topic_classification": "TOPIC-CLS",
         "adding_odd_numbers": "ODD-ADD", "model_name_extraction": "MODEL-EX"}


def main():
    J = json.loads((C.EXP / "reports" / "e5_leakage.json").read_text())
    body = []
    for t, r in J["tasks"].items():
        cells = []
        for m, _ in M:
            lf = [x[m]["spearman"] for x in r["leakfree"]]
            cells.append(f"{r['leaky'][m]['spearman']:.3f} & {statistics.mean(lf):.3f} $\\pm$ {statistics.stdev(lf):.3f}")
        body.append(ALIAS[t] + " & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")
    d = J["drops"]["spearman"]
    body.append("$\\Delta$ (mean over tasks) & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{statistics.mean(d[m]):+.3f} $\\pm$ {statistics.stdev(d[m]):.3f}}}" for m, _ in M) + " \\\\")
    f = J["flips"]
    header = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in M) + " \\\\",
              "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(M))),
              "task" + " & leaky & leak-free" * len(M) + " \\\\"]
    C.table(C.PAPER / "tables" / "e5_leakage.tex", "table_e5.py",
            "Leakage beyond logs: the paper's post-generation baselines on the original Clotho tasks (Llama-3.1-8B), "
            "Spearman correlation with the test score (scores oriented so higher means PASS). Leaky: the original "
            "protocol (test score from all 10 samples; Tok-Prob and Tok-Ent from sample 0, output entropy from all 10). "
            "Leak-free: test score from 5 samples, score from the other 5 (5 seeded splits, mean $\\pm$ std). MDSA is "
            "pre-generation, so its change is label noise only. "
            f"Pairwise method comparisons that change sign: {f['spearman']['n_flip']} of {f['spearman']['n_pairs']} "
            f"(Spearman), {f['auc']['n_flip']} of {f['auc']['n_pairs']} (ROC-AUC). $^*$ needs the generated outputs.",
            "tab:e5-leakage", "l" + "rr" * len(M), header, body)


if __name__ == "__main__":
    main()
