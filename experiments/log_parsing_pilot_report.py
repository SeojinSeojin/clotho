"""Pilot report for the log_parsing task (run from experiments/ after the sharded runs).

  python log_parsing_pilot_report.py --dataset loghub_hdfs_500 [--model llama]

Reads the merged generation file (data/results_<model>/log_parsing/inference_results/
messages_template/<dataset>_R10_T0.8.json), the input jsonl, shard DONE.json files
(runtime per shard) and, if present, Clotho result pickles (ROC-AUC). Writes
reports/log_parsing_pilot_<dataset>.md and prints the summary.
"""
import argparse
import glob
import html
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from clotho.dataset.log_parsing.output_parsers import score_output

PASS_THRESH = 0.5  # same as run_clotho_iterations.py
EXP = Path(__file__).resolve().parent


def cell(s, limit=300):
    s = s if len(s) <= limit else s[:limit] + " …"
    return html.escape(s).replace("|", "\\|").replace("\n", "<br>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="loghub_hdfs_500")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--n_failures", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    inputs = [json.loads(l) for l in
              (EXP.parent / "clotho/dataset/log_parsing" / f"{a.dataset}.jsonl").read_text().splitlines()]
    gen_path = (EXP / f"data/results_{a.model}/log_parsing/inference_results/messages_template/"
                f"{a.dataset}_R10_T0.8.json")
    results = json.loads(gen_path.read_text())
    assert len(results) == len(inputs), f"{len(results)} results vs {len(inputs)} inputs"
    prov = gen_path.with_name(gen_path.stem + ".provenance.json")
    generator = json.loads(prov.read_text())["stage"] if prov.exists() else "unknown"

    outs, input_scores, failures = [], [], []
    per_template = defaultdict(list)
    for inp, res in zip(inputs, results):
        assert res["GT"] == [inp["template"]], f"GT mismatch for {inp['id']}"
        scores = [score_output(o, res["GT"]) for o in res["inferences"]]
        outs += scores
        sc = sum(s["normalized_match"] for s in scores) / len(scores)
        input_scores.append(sc)
        per_template[inp["template_id"]].append(sc)
        for o, s in zip(res["inferences"], scores):
            if not s["normalized_match"]:
                failures.append((inp, o, s))

    n = len(outs)
    rate = lambda k: sum(s[k] for s in outs) / n
    L = []
    L.append(f"# log_parsing pilot: {a.dataset} ({a.model}, generator: {generator})\n")
    L.append(f"{len(inputs)} inputs x {n // len(inputs)} samples = {n} outputs\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| output-level normalized match (Clotho label) | {rate('normalized_match'):.1%} |")
    L.append(f"| output-level strict match | {rate('strict_match'):.1%} |")
    L.append(f"| output-level parse_failure | {rate('parse_failure'):.1%} |")
    L.append(f"| input-level mean test score | {statistics.mean(input_scores):.3f} |")
    L.append(f"| inputs labelled PASS (score > {PASS_THRESH}) | {sum(s > PASS_THRESH for s in input_scores) / len(input_scores):.1%} |")
    L.append(f"| inputs with 10/10 pass / 0/10 pass | {sum(s == 1 for s in input_scores)} / {sum(s == 0 for s in input_scores)} |")

    # runtime per shard
    L.append("\n## Runtime per shard\n\n| stage | shard | inputs | seconds | s/input |\n|---|---|---|---|---|")
    for done in sorted(glob.glob(str(EXP / f"sharded_work/shards/{a.model}/log_parsing/{a.dataset}/*/*/size*/shard_*/DONE.json"))):
        d = json.loads(Path(done).read_text())
        L.append(f"| {d['stage']} | {d['shard']} | {d['count']} | {d['seconds']:.0f} | {d['seconds'] / d['count']:.2f} |")

    # hardest templates
    tmpl_text = {i["template_id"]: i["template"] for i in inputs}
    L.append("\n## Pass rate by template (10 hardest)\n\n| template_id | n | mean score | template |\n|---|---|---|---|")
    for tid, scs in sorted(per_template.items(), key=lambda kv: statistics.mean(kv[1]))[:10]:
        L.append(f"| {tid} | {len(scs)} | {statistics.mean(scs):.2f} | {cell(tmpl_text[tid], 150)} |")

    # random failures
    rng = random.Random(a.seed)
    sample = rng.sample(failures, min(a.n_failures, len(failures)))
    kinds = Counter("parse_failure" if s["parse_failure"] else "mismatch" for _, _, s in failures)
    L.append(f"\n## {len(sample)} random failures (of {len(failures)}: {dict(kinds)}; seed {a.seed})\n")
    L.append("| # | id | input | ground truth | model output (parsed; raw if parse failure) | type |\n|---|---|---|---|---|---|")
    for j, (inp, raw, s) in enumerate(sample, 1):
        shown = s["parsed"] if not s["parse_failure"] else raw
        L.append(f"| {j} | {inp['id']} | {cell(inp['content'])} | {cell(inp['template'])} | {cell(shown)} | "
                 f"{'parse_failure' if s['parse_failure'] else 'mismatch'} |")

    # Clotho ROC-AUC
    pkls = sorted(glob.glob(str(EXP / f"results_clotho/LIHS_refset_iter_*_{a.model}/log_parsing/layer_*_seed_*/*_pca.pkl")))
    if pkls:
        import pandas as pd
        L.append("\n## Clotho ROC-AUC (on inputs not yet in the reference set)\n")
        for p in pkls:
            df = pd.read_pickle(p)
            L.append(f"**{Path(p).parent.name} / {Path(p).stem}** (columns: {', '.join(map(str, df.columns))})\n")
            cols = [c for c in df.columns if "roc" in str(c).lower()]
            if cols:
                show = df[[c for c in df.columns if any(k in str(c).lower() for k in ("iter", "refset", "size", "roc", "spearman"))]]
                L.append("```\n" + show.iloc[[0, 1, 2, 4, 9, 19, -1] if len(show) > 20 else slice(None)].to_string() + "\n```")
                L.append("")
    else:
        L.append("\n## Clotho ROC-AUC\n\n(no results_clotho pickles for log_parsing yet)")

    out = EXP / "reports" / f"log_parsing_pilot_{a.dataset}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
