"""Extract input hidden states at extra layers for log_parsing (suites + initial tests), no generation.

  cd experiments && python log_parsing_layers.py -p rules_template --layers 11 16 21 24 [-d <datasets>]

Same vector as 1-1_extract_input_hidden_states.py (last prompt token, hidden_states[i], i = output of
decoder layer i, 0 = embeddings) and as log_parsing_initial_set.py (initial tests via
inference.from_input). One forward pass per prompt yields every layer.
Writes (existing files are skipped):
  data/results_llama/log_parsing/input_hidden_states/<template>/<ds>/layer_<L>/hidden_vectors.pt
  initial_tests_llama/log_parsing@<template>/hidden_vectors_layer_<L>.pt
Layer 21 is already stored by the original pipeline; if requested it is recomputed and compared
(max abs diff printed) as a parity check, never overwritten.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from clotho.exp_config import supported_datasets, task2input_key, task2template
from generate_initial_set_data import initial_tests
from log_parsing_initial_set import initial_dir

TASK = "log_parsing"


def save_or_check(vecs, path):
    vecs = np.array(vecs, dtype=np.float32)
    if os.path.exists(path):
        old = np.asarray(torch.load(path, weights_only=False), dtype=np.float32)
        diff = float(np.abs(old - vecs).max()) if old.shape == vecs.shape else float("inf")
        rel = diff / (float(np.abs(old).max()) + 1e-12)
        print(f"  parity {path}: shape {old.shape} vs {vecs.shape}, max abs diff {diff:.4g} (rel {rel:.2g})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(vecs, path)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--template", default="rules_template")
    ap.add_argument("-d", "--datasets", nargs="+", default=[d for d in supported_datasets[TASK] if "proxifier" not in d])
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 16, 21, 24])
    ap.add_argument("--skip_initial", action="store_true")
    a = ap.parse_args()
    import clotho.models.llama as llama
    fn = task2template[TASK][a.template]

    def lasttok(text):
        hs = llama.get_hidden_states(text, fn, produce_inference_results=False)["hidden_states"]
        return {l: hs[l][0, -1].cpu().to(torch.float32).numpy() for l in a.layers}

    if not a.skip_initial:
        out = initial_dir("llama", a.template)
        vecs = {l: [] for l in a.layers}
        for case in initial_tests[TASK]:
            for l, v in lasttok(case[task2input_key[TASK]]).items():
                vecs[l].append(v)
        for l in a.layers:
            save_or_check(vecs[l], f"{out}/hidden_vectors_layer_{l}.pt")

    for ds in a.datasets:
        base = f"./data/results_llama/{TASK}/input_hidden_states/{a.template}/{ds}"
        if all(os.path.exists(f"{base}/layer_{l}/hidden_vectors.pt") for l in a.layers):
            print(f"{ds}: all layers present, skip")
            continue
        data = [json.loads(l) for l in open(f"../clotho/dataset/{TASK}/{ds}.jsonl")]
        vecs = {l: [] for l in a.layers}
        for d in tqdm(data, desc=ds):
            for l, v in lasttok(d[task2input_key[TASK]]).items():
                vecs[l].append(v)
        for l in a.layers:
            save_or_check(vecs[l], f"{base}/layer_{l}/hidden_vectors.pt")


if __name__ == "__main__":
    main()
