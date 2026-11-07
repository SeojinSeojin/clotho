"""Post-generation baselines for log_parsing, teacher-forced on the EXISTING gen_vllm outputs (no regeneration).

  cd experiments && python log_parsing_postgen.py -p rules_template [-d <datasets>] [--push] [--no_se]

One fused HF forward pass per input (its 10 samples batched, right-padded) over prompt + output + EOS,
the same token sequence as clotho.models.llama.prepare_inputs_for_token_prob_calculation (3-2, 3-3). Records:
  * per-token output log-probs (incl. EOS, as calc_confidence_scores) -> Tok-Prob
  * per-token full-vocab entropy (EOS step excluded, as calc_average_entropy) -> Tok-Ent
  * layer-23 hidden state at the position that predicts EOS (= 2-1's 'IE' state) -> LOHSvar
    (trace of the covariance of the 10 states, 3-1 calculate_variance_repeated_generations)
Then semantic entropy (3-3): DeBERTa-v2-xlarge-mnli strict bidirectional entailment over the 10 raw outputs,
greedy clustering exactly as clotho.metrics.semantic_entropy.get_semantic_ids, but with all ordered pairs of an
input classified in one batch; then predictive_entropy_rao(logsumexp_by_id(ids, mean log-likelihoods)).

Off-by-one in the repo helper: prompt_ids = tokenizer.encode(chat_text) adds a second <|begin_of_text|> (the chat
template already has one), so prompt_len is one too long and 3-2/3-3 drop the first output token. Here the
prompt length is counted correctly; *_paper columns reproduce the repo's slice (first output token dropped).

The wrapper's added <pad> token (random lm_head row) is excluded from the softmax; the repo helpers include it,
so the parity check shows a small difference from that alone.

Paper conventions (RQ/analysis_utils.py): Tok-Prob and Tok-Ent use sample 0 only.
Outputs (data/results_llama/log_parsing/precalculated_metrics/<template>/):
  <ds>_postgen_samples.pkl  one row per (input, sample): avg log-prob / entropy (correct and paper), per-token arrays
  <ds>_postgen.pkl          one row per input: tok_prob, tok_ent (sample 0), loh_var, sem_ent, n_sem_clusters,
                            plus *_paper and mean-over-10-samples variants
Higher tok_prob = more confident; higher tok_ent, loh_var, sem_ent = more uncertain.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / "sharded"))
from clotho.exp_config import supported_datasets, task2input_key, task2template  # noqa: E402

TASK = "log_parsing"
LOH_LAYER = 23          # RQ/analysis_utils.load_loh_variance_df: llama -> layer 23, token 'IE'
NLI = "microsoft/deberta-v2-xlarge-mnli"   # clotho.metrics.semantic_entropy.EntailmentDeberta


def out_dir(template):
    return EXP / f"data/results_llama/{TASK}/precalculated_metrics/{template}"


@torch.no_grad()
def forward_input(llama, msgs, outputs):
    tok, model, dev = llama.tokenizer, llama.model, llama.device
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    P = len(tok.encode(chat, add_special_tokens=False))          # correct prompt length
    seqs = [tok.encode(chat + o, add_special_tokens=False) + [tok.eos_token_id] for o in outputs]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.tensor(s)
        att[k, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
    hs = out.hidden_states[LOH_LAYER]
    res = []
    for k, s in enumerate(seqs):
        n = len(s)
        assert s[:P] == seqs[0][:P], "prompt tokens changed when the output was appended"
        # predicts tokens P..n-1 (output + EOS). Drop the <pad> row that clotho.models.llama appends
        # (resize_token_embeddings, randomly initialised): vLLM sampled from the original vocabulary.
        lg = out.logits[k, P - 1:n - 1, :tok.pad_token_id].float()
        lp = F.log_softmax(lg, dim=-1)
        tgt = ids[k, P:n]
        tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
        ent = -(lp.exp() * lp).sum(-1)[:-1]                  # EOS step excluded (include_eos=False)
        res.append({"lp": tok_lp.cpu().numpy().astype(np.float32),
                    "ent": ent.cpu().numpy().astype(np.float32),
                    "ie": hs[k, n - 2].float().cpu()})        # last output token: the step that emits EOS
    return res


def loh_variance(states):
    X = torch.stack(states)                                   # 3-1 calculate_variance_repeated_generations
    Xc = X - X.mean(0, keepdim=True)
    return float(torch.trace(Xc.T @ Xc / (X.shape[0] - 1)))


class BatchedNLI:
    def __init__(self, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI).to(device).eval()
        self.device = device

    @torch.no_grad()
    def semantic_ids(self, strings, bs=48):
        n = len(strings)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        pred = {}
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            enc = self.tok([strings[i] for i, _ in chunk], [strings[j] for _, j in chunk], return_tensors="pt",
                           truncation=True, max_length=512, padding=True).to(self.device)
            p = self.model(**enc).logits.argmax(-1).tolist()    # argmax(softmax) as check_implication
            pred.update(zip(chunk, p))
        ids, nxt = [-1] * n, 0                                   # get_semantic_ids, strict_entailment=True
        for i in range(n):
            if ids[i] == -1:
                ids[i] = nxt
                for j in range(i + 1, n):
                    if ids[j] == -1 and pred[(i, j)] == 2 and pred[(j, i)] == 2:
                        ids[j] = nxt
                nxt += 1
        return ids


def run_system(ds, template, llama, nli, push):
    od = out_dir(template)
    od.mkdir(parents=True, exist_ok=True)
    per_input_path = od / f"{ds}_postgen.pkl"
    if per_input_path.exists():
        print(f"skip {per_input_path.relative_to(EXP)}")
        return
    from clotho.metrics.semantic_entropy import logsumexp_by_id, predictive_entropy_rao
    fn = task2template[TASK][template]
    data = [json.loads(l) for l in open(EXP.parent / f"clotho/dataset/{TASK}/{ds}.jsonl")]
    gen = json.load(open(EXP / f"data/results_llama/{TASK}/inference_results/{template}/{ds}_R10_T0.8.json"))
    assert len(data) == len(gen)
    srows, irows = [], []
    t0 = time.time()
    for i, (d, g) in enumerate(tqdm(list(zip(data, gen)), desc=ds)):
        outs = g["inferences"]
        r = forward_input(llama, fn(d[task2input_key[TASK]]), outs)
        for j, x in enumerate(r):
            srows.append({"input_index": i, "inference_index": j, "n_tokens": len(x["lp"]),
                          "avg_logprob": float(x["lp"].mean()),
                          "avg_logprob_paper": float(x["lp"][1:].mean()) if len(x["lp"]) > 1 else float("nan"),
                          "avg_entropy": float(x["ent"].mean()) if len(x["ent"]) else 0.0,
                          "avg_entropy_paper": float(x["ent"][1:].mean()) if len(x["ent"]) > 1 else 0.0,
                          "token_logprobs": x["lp"].astype(np.float16), "token_entropies": x["ent"].astype(np.float16)})
        s = srows[-len(r):]
        row = {"input_index": i,
               "tok_prob": s[0]["avg_logprob"], "tok_ent": s[0]["avg_entropy"],
               "tok_prob_paper": s[0]["avg_logprob_paper"], "tok_ent_paper": s[0]["avg_entropy_paper"],
               "tok_prob_mean10": float(np.mean([x["avg_logprob"] for x in s])),
               "tok_ent_mean10": float(np.mean([x["avg_entropy"] for x in s])),
               "loh_var": loh_variance([x["ie"] for x in r])}
        if nli is not None:
            sid = nli.semantic_ids(outs)
            row["sem_ent"] = float(predictive_entropy_rao(logsumexp_by_id(sid, [x["avg_logprob"] for x in s])))
            row["n_sem_clusters"] = len(set(sid))
            row["semantic_ids"] = sid
        irows.append(row)
    pd.DataFrame(srows).to_pickle(od / f"{ds}_postgen_samples.pkl")
    pd.DataFrame(irows).to_pickle(per_input_path)
    print(f"wrote {per_input_path.relative_to(EXP)} ({time.time() - t0:.0f} s)", flush=True)
    if push:
        import hub_sync as hs
        rel = [str((od / f).relative_to(EXP)) for f in (f"{ds}_postgen_samples.pkl", f"{ds}_postgen.pkl")]
        hs.commit_files(hs.get_results_repo(), {r: EXP / r for r in rel}, f"log_parsing postgen {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--template", default="rules_template")
    ap.add_argument("-d", "--datasets", nargs="+",
                    default=[d for d in supported_datasets[TASK] if "proxifier" not in d])
    ap.add_argument("--no_se", action="store_true", help="skip semantic entropy")
    ap.add_argument("--limit_check", type=int, default=0,
                    help="parity check only: compare against the repo helpers on N inputs of the first dataset")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    os.chdir(EXP)
    import clotho.models.llama as llama
    if a.limit_check:
        parity(llama, a.datasets[0], a.template, a.limit_check)
        return
    nli = None if a.no_se else BatchedNLI(llama.device)
    for ds in a.datasets:
        run_system(ds, a.template, llama, nli, a.push)


def parity(llama, ds, template, n):
    """Compare fused batched values against the repo's unbatched helpers (3-2) on n inputs x 10 samples."""
    from clotho.metrics.logprobs import calc_average_entropy, calc_confidence_scores
    fn = task2template[TASK][template]
    data = [json.loads(l) for l in open(EXP.parent / f"clotho/dataset/{TASK}/{ds}.jsonl")][:n]
    gen = json.load(open(EXP / f"data/results_llama/{TASK}/inference_results/{template}/{ds}_R10_T0.8.json"))[:n]
    dlp, dent = [], []
    for d, g in zip(data, gen):
        msgs = fn(d[task2input_key[TASK]])
        r = forward_input(llama, msgs, g["inferences"])
        for o, x in zip(g["inferences"], r):
            c = calc_confidence_scores(llama, msgs, o)["average_log_probs"]
            e = calc_average_entropy(llama, msgs, o)["average_entropy"]
            dlp.append(abs(c - float(x["lp"][1:].mean())))
            dent.append(abs(e - float(x["ent"][1:].mean())))
    print(f"parity vs repo helpers (paper slice), {len(dlp)} samples: "
          f"max |d avg_logprob| {max(dlp):.4f}, max |d avg_entropy| {max(dent):.4f}")


if __name__ == "__main__":
    main()
