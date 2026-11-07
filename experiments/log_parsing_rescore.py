"""Rescore every log_parsing generation (local + RESULTS_REPO) under all scorers, without regenerating.

  python experiments/log_parsing_rescore.py                 # table -> experiments/reports/log_parsing_rescore.{md,json}
  python experiments/log_parsing_rescore.py --no_pull       # local files only
  python experiments/log_parsing_rescore.py --samples loghub_hdfs_500 --template messages_template \
      --scorer lenient_match -n 20                          # random failing outputs for manual inspection

Scorers (clotho/dataset/log_parsing/output_parsers.py): strict_match, normalized_match,
lenient_match and its numbered extensions. Input PASS = more than 5 of 10 samples match
(the threshold Clotho's reports use). The rescore report is pushed to RESULTS_REPO.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from clotho.dataset.log_parsing.output_parsers import SCORERS, lenient_for, score_output, normalize

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / "sharded"))
DATA = EXP / "data"
SUITES = EXP.parent / "clotho" / "dataset" / "log_parsing"
PASS_THRESH = 0.5


def gen_files(model):
    """(task, template, dataset, path) for every merged generation. task is log_parsing, or the
    legacy log_parsing_rules results (old, uncommitted rules prompt) kept for reference."""
    out = []
    for p in sorted(DATA.glob(f"results_{model}/log_parsing*/inference_results/*/*_R10_T0.8.json")):
        out.append((p.parts[-4], p.parts[-2], p.name.replace("_R10_T0.8.json", ""), p))
    return out


def pull(model):
    import hub_sync as hs
    repo = hs.get_results_repo()
    hs.download_prefix(repo, [f"data/results_{model}/log_parsing*/inference_results/*/*_R10_T0.8.json"], EXP)


def rescore(path, template):
    res = json.loads(path.read_text())
    n_out = 0
    tot = {k: 0 for k in SCORERS + ("parse_failure",)}
    inp = {k: 0 for k in SCORERS}
    for r in res:
        sc = [score_output(o, r["GT"], template) for o in r["inferences"]]
        n_out += len(sc)
        for k in tot:
            tot[k] += sum(s[k] for s in sc)
        for k in SCORERS:
            inp[k] += sum(s[k] for s in sc) / len(sc) > PASS_THRESH
    return {"n_inputs": len(res), "n_outputs": n_out,
            "parse_fail": tot["parse_failure"] / n_out,
            **{f"out_{k}": tot[k] / n_out for k in SCORERS},
            **{f"inp_{k}": inp[k] / len(res) for k in SCORERS}}


def samples(a):
    path = DATA / f"results_{a.model}/log_parsing/inference_results/{a.template}/{a.dataset}_R10_T0.8.json"
    res = json.loads(path.read_text())
    src = [json.loads(l) for l in open(SUITES / f"{a.dataset}.jsonl")]
    assert all(s["template"] == r["GT"][0] for s, r in zip(src, res)), "suite / generation order mismatch"
    fails = []
    for i, (s, r) in enumerate(zip(src, res)):
        for o in r["inferences"]:
            sc = score_output(o, r["GT"], a.template)
            if not sc[a.scorer] and not sc["parse_failure"]:
                fails.append((i, s["content"], r["GT"][0], sc["parsed"]))
    print(f"{len(fails)} failing parsed outputs under {a.scorer}")
    lf = lenient_for(a.scorer)
    for k, (i, c, g, p) in enumerate(random.Random(a.seed).sample(fails, min(a.n, len(fails)))):
        print(f"[{k + 1}] input #{i}\n  IN : {c}\n  GT : {g}\n  OUT: {p}")
        if lf:
            print(f"  GT': {lf(g)}\n  OUT': {lf(p)}")
        else:
            print(f"  GT': {normalize(g)}\n  OUT': {normalize(p)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="llama")
    ap.add_argument("--no_pull", action="store_true")
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--samples", dest="dataset")
    ap.add_argument("--template", default="messages_template")
    ap.add_argument("--scorer", default="lenient_match")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dataset:
        return samples(a)
    if not a.no_pull:
        pull(a.model)
    rows = {}
    for task, template, ds, path in gen_files(a.model):
        rows[f"{task}/{template}/{ds}"] = {"task": task, "template": template, "dataset": ds,
                                           **rescore(path, template)}
    L = ["# log_parsing rescoring (no regeneration)\n",
         "Input PASS = more than 5 of 10 samples match. Cells: input PASS rate / output match rate. "
         "`log_parsing_rules` = legacy results from an uncommitted rules prompt (reference only).\n",
         "| task | template | system | " + " | ".join(SCORERS) + " | parse fail |",
         "|---" * (4 + len(SCORERS)) + "|"]
    for r in rows.values():
        L.append(f"| {r['task']} | {r['template']} | {r['dataset'].replace('loghub_', '').replace('_500', '')} | "
                 + " | ".join(f"{r['inp_' + k]:.1%} / {r['out_' + k]:.1%}" for k in SCORERS)
                 + f" | {r['parse_fail']:.1%} |")
    out = EXP / "reports" / "log_parsing_rescore.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (EXP / "reports" / "log_parsing_rescore.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(L))
    if not (a.no_pull or a.no_push):
        import hub_sync as hs
        repo = hs.get_results_repo()
        hs.commit_files(repo, {f"reports/{p.name}": p for p in (out, out.with_suffix(".json"))},
                        "log_parsing rescore under strict/normalized/lenient scorers")


if __name__ == "__main__":
    main()
