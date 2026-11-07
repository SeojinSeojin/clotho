"""Clotho initial tests for log_parsing under a given prompt template (HF path).

  cd experiments && python log_parsing_initial_set.py -p rules_template

Same procedure as generate_initial_set_data.py (10 synthetic inputs from its
`initial_tests['log_parsing']`, 10 HF inference runs each, layer 21), but for any
template. Output:
  messages_template -> initial_tests_llama/log_parsing/              (what Clotho reads by default)
  other templates   -> initial_tests_llama/log_parsing@<template>/   (run_clotho_log_parsing.py maps it)

inference_results.json also keeps the raw outputs and ground truths, so the labels can be
recomputed under another scorer without the GPU:
  cd experiments && python log_parsing_initial_set.py -p rules_template --rescore
(test_scores / test_labels always follow output_parsers.PASS_SCORER.)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import clotho.inference as inference
from clotho.dataset.log_parsing.output_parsers import PASS_SCORER
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
    ap.add_argument("--rescore", action="store_true", help="relabel saved raw outputs, no generation")
    a = ap.parse_args()
    out = initial_dir(a.model, a.template)
    if a.rescore:
        path = f"{out}/inference_results.json"
        d = json.load(open(path))
        if "raw_outputs" not in d:
            raise SystemExit(f"{path} has no raw_outputs (old format); regenerate without --rescore")
        res = [inference.evaluate(o, gt, task=TASK, prompt_template_name=a.template)
               for o, gt in zip(d["raw_outputs"], d["ground_truths"])]
        d["test_scores"], d["test_labels"] = [r[0] for r in res], [r[1] for r in res]
        d["scorer"] = PASS_SCORER
        json.dump(d, open(path, "w"), indent=2)
        print(f"rescored {path} with {PASS_SCORER}: {d['test_scores']}")
        return

    LIH, scores, labels, raws = defaultdict(list), [], [], []
    for i, case in enumerate(initial_tests[TASK]):
        outs, layer2hidden = inference.from_input(
            input_variables=case, task=TASK, prompt_template_name=a.template,
            num_inference_runs=10, target_layers=a.layers, produce_inference_results=True, model=a.model)
        for l in layer2hidden:
            LIH[l].append(layer2hidden[l])
        s, lab = inference.evaluate(outs, case["label"], task=TASK, prompt_template_name=a.template)
        scores.append(s)
        raws.append(outs)
        labels.append(lab)
        print(f"initial test {i + 1}: score {s}", flush=True)

    os.makedirs(out, exist_ok=True)
    for l, vecs in LIH.items():
        torch.save(np.array(vecs), f"{out}/hidden_vectors_layer_{l}.pt")
    with open(f"{out}/inference_results.json", "w") as f:
        json.dump({"test_scores": scores, "test_labels": labels, "scorer": PASS_SCORER, "raw_outputs": raws,
                   "ground_truths": [c["label"] for c in initial_tests[TASK]]}, f, indent=2)
    print(f"wrote {out} | scores {scores}")


if __name__ == "__main__":
    main()
