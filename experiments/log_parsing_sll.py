"""SLL baseline (input sequence log-likelihood) for log_parsing, as run_baseline_SLL.py but for one
template and the log_parsing suites only. No generation: one forward pass per input prompt.

  cd experiments && python log_parsing_sll.py -p rules_template
  -> data/results_llama/log_parsing/precalculated_metrics/<template>/<dataset>_input_logprobs.pkl
"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from clotho.exp_config import supported_datasets, task2input_key, task2template
from clotho.metrics.logprobs import calc_confidence_scores_input

TASK = "log_parsing"
import model_cfg as MC  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--template", default="rules_template")
    ap.add_argument("-d", "--datasets", nargs="+", default=supported_datasets[TASK])
    a = ap.parse_args()
    model_wrapper = MC.wrapper()
    fn = task2template[TASK][a.template]
    out_dir = f"./data/results_{MC.MODEL}/{TASK}/precalculated_metrics/{a.template}"
    os.makedirs(out_dir, exist_ok=True)
    for ds in a.datasets:
        path = f"{out_dir}/{ds}_input_logprobs.pkl"
        if os.path.exists(path):
            print(f"skip {path}")
            continue
        data = [json.loads(l) for l in open(f"../clotho/dataset/{TASK}/{ds}.jsonl")]
        rows = []
        for i, d in enumerate(tqdm(data, desc=ds)):
            r = calc_confidence_scores_input(model_wrapper=model_wrapper, prompt=fn(d[task2input_key[TASK]]))
            rows.append({"input_index": i, "average_log_probs": r["average_log_probs"], "perplexity": r["perplexity"]})
        pd.DataFrame(rows).to_pickle(path)
        print(f"wrote {path}")
