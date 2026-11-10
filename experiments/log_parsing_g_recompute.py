"""G (2026-11-10): leak-free LOHSvar and Sem-Ent on the 8 original Clotho tasks (Llama). Protocol: PROGRESS.md Decisions (G).

  cd experiments
  ../.venv/bin/python log_parsing_g_recompute.py check --data ~/.cache/clotho_zenodo/x/data      # parity vs stored LOHSvar
  ../.venv/bin/python log_parsing_g_recompute.py loh   --data ... [--push]                       # Llama pass (GPU)
  ../.venv/bin/python log_parsing_g_recompute.py se    --data ... [--push]                       # DeBERTa pass (GPU)
  -> data/g_recompute/<task>/<ds>_loh.npz, <ds>_se.npz (in RESULTS_REPO); E5's report then adds LOHSvar and Sem-Ent.

Upstream definitions (coinse/clotho, this repo's copy):
  LOHSvar (RQ/analysis_utils.load_loh_variance_df, data_extraction/3-1, clotho/models/llama.py 2-1): layer 23 of
  Llama-3.1-8B-Instruct (hidden_states[23], before the final norm), token 'IE' = the hidden state at the generation
  step that emits EOS, i.e. the state at the position of the second-to-last generated token (the last one, EOS or the
  token sampled at the length limit, is never fed); variance = trace of the sample covariance (n - 1) of the states.
  Rebuilt here teacher-forced from the archive's stored `generated_tokens` (exact ids), prompt ids as the HF 2-1 path
  (prompt_render.render: chat template + add_special_tokens=True, double BOS), bf16. Stored per input: the 10 x 10
  matrix of squared distances D between the IE states, so any subset S gives trace(cov) = sum_{i,j in S} D_ij / (2 n (n-1)).
  Prompt encoded once (KV cache reused for the 10 outputs); only the first 24 decoder layers are run.
  Sem-Ent (data_extraction/3-3, clotho/metrics/semantic_entropy.pipeline): DeBERTa-v2-xlarge-mnli strict bidirectional
  entailment over the raw inferences, greedy clustering in sample order (get_semantic_ids), log-likelihoods =
  per-sample `average_log_probs` (the archive's `_log_probs.pkl`, the same calc_confidence_scores value the pipeline
  computes), predictive_entropy_rao(logsumexp_by_id(ids, ll)). Pair verdicts are cached per input (10 x 10, -1 = not
  needed) so leaky (all 10) and leak-free subsets (E5's score halves, in sample order) cluster exactly as the upstream
  greedy loop would on that subset. DeBERTa in bf16 autocast as log_parsing_postgen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_e5_leakage as E5  # noqa: E402

OUT = EXP / "data" / "g_recompute"
LAYER = 23


def inputs(task, ds):
    from clotho.exp_config import task2input_key
    rows = [json.loads(l) for l in open(EXP.parent / "clotho" / "dataset" / task / f"{ds}.jsonl")]
    return rows


def load_llama():
    import torch
    from transformers import AutoModelForCausalLM
    import prompt_render as pr
    tok = pr.load_tokenizer("llama")
    m = AutoModelForCausalLM.from_pretrained(pr.MODEL_IDS["llama"], torch_dtype=torch.bfloat16).cuda().eval()
    m.model.layers = m.model.layers[:LAYER + 1]
    m.config.num_hidden_layers = LAYER + 1
    return tok, m


def ie_states(tok, m, prompt_ids, gens):
    """IE states (10, d) float32, prompt encoded once."""
    import torch
    from transformers import DynamicCache
    with torch.no_grad():
        p = torch.tensor([prompt_ids], device="cuda")
        cache = DynamicCache()
        o = m.model(input_ids=p, past_key_values=cache, use_cache=True, output_hidden_states=True)
        last_prompt = o.hidden_states[LAYER][0, -1].float()
        feed = [g[:-1] for g in gens]
        L = max(len(f) for f in feed)
        out = torch.empty(len(gens), last_prompt.shape[0], device="cuda")
        if L == 0:
            return last_prompt.expand(len(gens), -1).cpu().numpy()
        cache.batch_repeat_interleave(len(gens))
        P = len(prompt_ids)
        ids = torch.full((len(gens), L), tok.pad_token_id or 0, dtype=torch.long, device="cuda")
        att = torch.zeros((len(gens), P + L), dtype=torch.long, device="cuda")
        att[:, :P] = 1
        for k, f in enumerate(feed):
            if f:
                ids[k, :len(f)] = torch.tensor(f, device="cuda")
                att[k, P:P + len(f)] = 1
        o2 = m.model(input_ids=ids, attention_mask=att, past_key_values=cache, use_cache=True,
                     output_hidden_states=True, cache_position=torch.arange(P, P + L, device="cuda"))
        hs = o2.hidden_states[LAYER]
        for k, f in enumerate(feed):
            out[k] = hs[k, len(f) - 1].float() if f else last_prompt
        return out.cpu().numpy()


def ie_states_full(tok, m, prompt_ids, gens):
    """Reference: one full forward per sample, no cache (parity of the cached path)."""
    import torch
    res = []
    with torch.no_grad():
        for g in gens:
            ids = torch.tensor([prompt_ids + g[:-1]], device="cuda")
            res.append(m.model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float().cpu().numpy())
    return np.stack(res)


def sqdist(X):
    X = X.astype(np.float64)
    g = X @ X.T
    n = np.diag(g)
    return np.maximum(n[:, None] + n[None, :] - 2 * g, 0).astype(np.float32)


def var_from_D(D, S):
    S = list(S)
    n = len(S)
    return float(D[np.ix_(S, S)].sum() / (2 * n * (n - 1)))


def loh_ds(root, task, ds, tok, m, limit=None):
    import prompt_render as pr
    res = json.load(open(root / task / "inference_results" / "messages_template" / f"{ds}_R10_T0.8.json"))
    rows = inputs(task, ds)
    assert len(rows) == len(res)
    n = len(res) if limit is None else limit
    D = np.zeros((n, 10, 10), np.float32)
    t0 = time.time()
    for i in range(n):
        _, pids = pr.render(task, "messages_template", rows[i], tok)
        gens = [tok.convert_tokens_to_ids(g) for g in res[i]["generated_tokens"]]
        D[i] = sqdist(ie_states(tok, m, pids, gens))
        if i % 2000 == 0:
            print(f"  {task}/{ds} {i}/{n} {time.time() - t0:.0f}s", flush=True)
    return D, time.time() - t0


def stored_var(root, task, ds):
    import pandas as pd
    p = root / task / "precalculated_metrics" / "messages_template" / f"{ds}_R10_T0.8_output_state_variances.pkl"
    if not p.exists():
        return None
    d = pd.read_pickle(p)
    d = d[(d["layer_index"] == LAYER) & (d["token_index"] == "IE")].drop_duplicates("input_index").set_index("input_index")["variance"]
    return d


def cmd_check(a):
    import prompt_render as pr
    from scipy.stats import spearmanr
    tok, m = load_llama()
    root = a.data / "results_llama"
    out = {}
    for task, dss in E5.TASKS.items():
        ds = dss[0]
        res = json.load(open(root / task / "inference_results" / "messages_template" / f"{ds}_R10_T0.8.json"))
        rows = inputs(task, ds)
        _, pids = pr.render(task, "messages_template", rows[0], tok)
        gens = [tok.convert_tokens_to_ids(g) for g in res[0]["generated_tokens"]]
        A, B = ie_states(tok, m, pids, gens), ie_states_full(tok, m, pids, gens)
        cache_err = float(np.abs(A - B).max() / (np.abs(B).max() + 1e-9))
        D, wall = loh_ds(root, task, ds, tok, m, limit=a.n)
        mine = np.array([var_from_D(D[i], range(10)) for i in range(a.n)])
        st = stored_var(root, task, ds)
        if st is not None:
            idx = [i for i in range(a.n) if i in st.index]
            s = st.loc[idx].to_numpy()
            mm = mine[idx]
            rho = float(spearmanr(mm, s).correlation)
            rel = float(np.median(np.abs(mm - s) / np.maximum(s, 1e-6)))
        else:
            rho = rel = float("nan")
        out[task] = {"ds": ds, "cache_vs_full_max_rel": cache_err, "spearman_vs_stored": rho, "median_rel_err": rel,
                     "s_per_input": wall / a.n}
        print(task, out[task], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps(out, indent=1))


def push(paths, msg):
    import hub_sync as hs
    hs.commit_files(hs.get_results_repo(), {str(p.relative_to(EXP)): p for p in paths}, msg)


def cmd_loh(a):
    tok, m = load_llama()
    root = a.data / "results_llama"
    for task, dss in E5.TASKS.items():
        for ds in dss:
            p = OUT / task / f"{ds}_loh.npz"
            if p.exists():
                continue
            D, wall = loh_ds(root, task, ds, tok, m)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, D=D, wall_s=wall)
            print(f"{task}/{ds}: {len(D)} inputs, {wall:.0f}s", flush=True)
            if a.push:
                push([p], f"G LOHSvar {task} {ds}")


def cmd_se(a):
    import torch
    from log_parsing_postgen import BatchedNLI
    nli = BatchedNLI("cuda")
    root = a.data / "results_llama"
    for task, dss in E5.TASKS.items():
        for ds in dss:
            p = OUT / task / f"{ds}_se.npz"
            if p.exists():
                continue
            res = json.load(open(root / task / "inference_results" / "messages_template" / f"{ds}_R10_T0.8.json"))
            subsets = [list(range(10))] + [sorted(np.random.RandomState(s).permutation(10)[:5].tolist()) for s in E5.SEEDS]
            V = np.full((len(res), 10, 10), -1, np.int8)
            t0 = time.time()
            for i, r in enumerate(res):
                outs = r["inferences"]
                memo = {}
                # greedy clustering per subset; classify lazily in rounds (one batch per representative step)
                for S in subsets:
                    ids = {j: -1 for j in S}
                    nxt = 0
                    for ii in S:
                        if ids[ii] != -1:
                            continue
                        ids[ii] = nxt
                        js = [j for j in S if j > ii and ids[j] == -1]
                        pairs = [(outs[ii], outs[j]) for j in js] + [(outs[j], outs[ii]) for j in js]
                        todo = list({q for q in pairs if q not in memo})
                        if todo:
                            strings = [s for q in todo for s in q]
                            pred = nli._classify([(2 * t, 2 * t + 1) for t in range(len(todo))], strings)
                            for t, q in enumerate(todo):
                                memo[q] = pred[(2 * t, 2 * t + 1)]
                        for j in js:
                            V[i, ii, j] = memo[(outs[ii], outs[j])]
                            V[i, j, ii] = memo[(outs[j], outs[ii])]
                            if V[i, ii, j] == 2 and V[i, j, ii] == 2:
                                ids[j] = nxt
                        nxt += 1
                if i % 2000 == 0:
                    print(f"  {task}/{ds} {i}/{len(res)} {time.time() - t0:.0f}s", flush=True)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, V=V, wall_s=time.time() - t0)
            print(f"{task}/{ds}: {len(res)} inputs, {time.time() - t0:.0f}s", flush=True)
            if a.push:
                push([p], f"G Sem-Ent verdicts {task} {ds}")
            torch.cuda.empty_cache()


def semantic_entropy(V, S, ll):
    """Upstream greedy clustering on subset S (sample order) from cached verdicts, then Rao entropy."""
    from clotho.metrics.semantic_entropy import logsumexp_by_id, predictive_entropy_rao
    ids = {j: -1 for j in S}
    nxt = 0
    for ii in S:
        if ids[ii] != -1:
            continue
        ids[ii] = nxt
        for j in S:
            if j > ii and ids[j] == -1 and V[ii, j] == 2 and V[j, ii] == 2:
                ids[j] = nxt
        nxt += 1
    sid = [ids[j] for j in S]
    return float(predictive_entropy_rao(logsumexp_by_id(sid, [ll[j] for j in S])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "loh", "se"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "loh": cmd_loh, "se": cmd_se}[a.cmd](a)


if __name__ == "__main__":
    main()
