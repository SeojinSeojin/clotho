"""Clotho initial tests for log_parsing under a given prompt template (HF path).

  cd experiments && python log_parsing_initial_set.py -p rules_template

Same procedure as generate_initial_set_data.py (10 synthetic inputs from its
`initial_tests['log_parsing']`, 10 HF inference runs each, layer 21), but for any
template. Output:
  messages_template -> initial_tests_llama/log_parsing/              (what Clotho reads by default)
  other templates   -> initial_tests_llama/log_parsing@<template>/   (run_clotho_log_parsing.py maps it)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import clotho.inference as inference
from generate_initial_set_data import initial_tests

TASK = "log_parsing"


def initial_dir(model, template):
    sub = TASK if template == "messages_template" else f"{TASK}@{template}"
    return f"./initial_tests_{model}/{sub}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--template", required=True)
    ap.add_argument("-m", "--model", default="llama")
    ap.add_argument("--layers", type=int, nargs="+", default=[21])
    a = ap.parse_args()

    LIH, scores, labels = defaultdict(list), [], []
    for i, case in enumerate(initial_tests[TASK]):
        outs, layer2hidden = inference.from_input(
            input_variables=case, task=TASK, prompt_template_name=a.template,
            num_inference_runs=10, target_layers=a.layers, produce_inference_results=True, model=a.model)
        for l in layer2hidden:
            LIH[l].append(layer2hidden[l])
        s, lab = inference.evaluate(outs, case["label"], task=TASK, prompt_template_name=a.template)
        scores.append(s)
        labels.append(lab)
        print(f"initial test {i + 1}: score {s}", flush=True)

    out = initial_dir(a.model, a.template)
    os.makedirs(out, exist_ok=True)
    for l, vecs in LIH.items():
        torch.save(np.array(vecs), f"{out}/hidden_vectors_layer_{l}.pt")
    with open(f"{out}/inference_results.json", "w") as f:
        json.dump({"test_scores": scores, "test_labels": labels}, f, indent=2)
    print(f"wrote {out} | scores {scores}")


if __name__ == "__main__":
    main()
