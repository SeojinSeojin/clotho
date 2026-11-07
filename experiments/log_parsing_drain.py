"""Drain group IDs for the realistic template-lookup baseline (log_parsing), plus grouping accuracy.

  cd experiments && python log_parsing_drain.py [--push]
  -> data/results_llama/log_parsing/precalculated_metrics/drain_groups.json
     reports/log_parsing_drain.md

Two variants, each run separately per system on the 500 suite messages (the `content` field, i.e. the
message after the Loghub header) in suite file order. Label-free: Drain never sees the templates.
  drain_lh2      logparser3 Drain (logpai) with the Loghub-2.0 benchmark settings per system (regex, st,
                 depth): logpai/loghub-2.0 benchmark/old_benchmark/Drain_benchmark.py at commit ac4aad2,
                 which Loghub-2.0's benchmark/evaluation/Drain_eval.py imports. Copied below (only Spark
                 differs from logparser3's own loghub_2k settings: no regex). logparser3's Drain.py differs
                 from Loghub-2.0's only in log_to_dataframe (file loading), which is bypassed here.
  drain3_default drain3 TemplateMiner() with its default config (sim_th 0.4, depth 4, max_children 100,
                 no masking); group = cluster id returned when the message is added.
Grouping accuracy (GA, Loghub definition): fraction of messages whose predicted group contains exactly
the same messages as their ground-truth template.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / "sharded"))
from clotho.exp_config import supported_datasets  # noqa: E402

TASK = "log_parsing"
SUITES = EXP.parent / "clotho" / "dataset" / TASK
OUT = EXP / "data/results_llama/log_parsing/precalculated_metrics/drain_groups.json"
REPORT = EXP / "reports" / "log_parsing_drain.md"

# Loghub-2.0 benchmark settings for Drain (regex, st, depth), commit ac4aad2ea86f561e5a0acbc6587fa28b487259f9.
LH2 = {
    "HDFS": {"regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"], "st": 0.5, "depth": 4},
    "Hadoop": {"regex": [r"(\d+\.){3}\d+"], "st": 0.5, "depth": 4},
    "Spark": {"regex": [], "st": 0.5, "depth": 4},
    "Zookeeper": {"regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"], "st": 0.5, "depth": 4},
    "BGL": {"regex": [r"core\.\d+"], "st": 0.5, "depth": 4},
    "HPC": {"regex": [r"=\d+"], "st": 0.5, "depth": 4},
    "Thunderbird": {"regex": [r"(\d+\.){3}\d+"], "st": 0.5, "depth": 4},
    "Linux": {"regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"], "st": 0.39, "depth": 6},
    "HealthApp": {"regex": [], "st": 0.2, "depth": 4},
    "Apache": {"regex": [r"(\d+\.){3}\d+"], "st": 0.5, "depth": 4},
    "Proxifier": {"regex": [r"<\d+\ssec", r"([\w-]+\.)+[\w-]+(:\d+)?", r"\d{2}:\d{2}(:\d{2})*", r"[KGTM]B"],
                  "st": 0.6, "depth": 3},
    "OpenSSH": {"regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"], "st": 0.6, "depth": 5},
    "OpenStack": {"regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"], "st": 0.5, "depth": 5},
    "Mac": {"regex": [r"([\w-]+\.){2,}[\w-]+"], "st": 0.7, "depth": 6},
}


def load_suite(ds):
    rows = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
    return [r["content"] for r in rows], [r["template"] for r in rows], rows[0]["system"]


def drain_lh2(messages, system, workdir):
    from logparser.Drain import LogParser
    s = LH2[system]

    class P(LogParser):
        def load_data(self):            # bypass log-file parsing: messages are already header-free
            self.df_log = pd.DataFrame({"LineId": range(1, len(messages) + 1), "Content": messages})

    p = P(log_format="<Content>", indir=str(workdir), outdir=str(workdir), depth=s["depth"], st=s["st"],
          rex=s["regex"], keep_para=False)
    p.parse(f"{system}.log")
    return [str(e) for e in p.df_log["EventId"]]


def drain3_default(messages):
    from drain3 import TemplateMiner
    tm = TemplateMiner()
    return [str(tm.add_log_message(m)["cluster_id"]) for m in messages]


def grouping_accuracy(pred, gt):
    pg, gg = defaultdict(set), defaultdict(set)
    for i, (p, g) in enumerate(zip(pred, gt)):
        pg[p].add(i)
        gg[g].add(i)
    return sum(pg[p] == gg[g] for p, g in zip(pred, gt)) / len(gt)


def groups(ds, variant, path=OUT):
    """Per-input group labels for `ds` (for template_lookup); prefixed so they never equal a real template."""
    d = json.loads(Path(path).read_text())[ds][variant]
    return [f"{variant}:{g}" for g in d["groups"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datasets", nargs="+",
                    default=[d for d in supported_datasets[TASK] if "proxifier" not in d])
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    import logging
    import tempfile
    logging.disable(logging.WARNING)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as wd:
        for ds in a.datasets:
            msgs, gt, system = load_suite(ds)
            res[ds] = {"system": system, "n": len(msgs), "n_templates": len(set(gt))}
            for name, g in (("drain_lh2", drain_lh2(msgs, system, wd)), ("drain3_default", drain3_default(msgs))):
                res[ds][name] = {"groups": g, "n_groups": len(set(g)), "GA": grouping_accuracy(g, gt)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    L = ["# log_parsing: Drain grouping on the 500-message suites\n",
         "Generated by `experiments/log_parsing_drain.py`. GA = grouping accuracy against Loghub-2.0 templates. "
         "`Drain (LH2)` = logparser3 Drain with Loghub-2.0 per-system benchmark settings; `drain3 default` = "
         "drain3 TemplateMiner defaults. Each run on one system's 500 suite messages.\n",
         "| system | GT templates | Drain (LH2) groups | Drain (LH2) GA | drain3 default groups | drain3 default GA |",
         "|---|---|---|---|---|---|"]
    for ds in a.datasets:
        r = res[ds]
        L.append(f"| {r['system']} | {r['n_templates']} | {r['drain_lh2']['n_groups']} | {r['drain_lh2']['GA']:.3f} | "
                 f"{r['drain3_default']['n_groups']} | {r['drain3_default']['GA']:.3f} |")
    m = lambda k: sum(res[ds][k]["GA"] for ds in a.datasets) / len(a.datasets)  # noqa: E731
    L.append(f"| **mean** | | | {m('drain_lh2'):.3f} | | {m('drain3_default'):.3f} |")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.push:
        import hub_sync as hs
        files = {str(p.relative_to(EXP)): p for p in (OUT, REPORT)}
        hs.commit_files(hs.get_results_repo(), files, "log_parsing drain groups")
        print(f"pushed {len(files)} files")


if __name__ == "__main__":
    main()
