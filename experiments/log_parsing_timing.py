"""Wall-clock cost per input on this GPU (L40S): Clotho scoring vs generating k samples.

  cd experiments
  python log_parsing_timing.py hf                    # .venv: HF forward pass (+ GMM, MDSA, Drain on CPU)
  ../.venv-vllm/bin/python log_parsing_timing.py vllm # .venv-vllm: k samples with vLLM, k = 1..5, and prefill only
  -> results_phase3/timing/{hf,vllm}.json

Systems: HDFS, Hadoop, BGL, OpenStack (500 suite inputs each), rules_template prompts.
hf:
  pipeline_bs1   the extraction code the results came from (clotho.models.llama.get_hidden_states: one prompt,
                 every layer), one input at a time, N_BS1 inputs per system. Online latency.
  batched_l16    batch of 32 prompts (left padded) through decoder layers 1-16 only (the other layers are
                 not needed for layer-16 hidden states); amortised per input over all 500. Parity of the
                 layer-16 vector vs the stored one is recorded (bf16).
  gmm / mdsa     CPU: PCA transform + GMM score_samples (Clotho), PCA + Mahalanobis (MDSA), per input, from a
                 fitted model (fit cost is a one-off per reference set and excluded).
  drain          Drain (Loghub-2.0 settings) parse of the 500 messages, per input. Lookup needs no model.
vllm (same engine settings and SamplingParams as sharded/vllm_generate.py; engine start excluded):
  gen_k          n=k samples for all 500 prompts submitted at once; amortised seconds per input.
  gen_k_bs1      n=k samples, one prompt at a time, N_BS1 prompts per system. Online latency.
  prefill        max_tokens=1 (prompt forward pass only) in vLLM, amortised; a lower bound for the forward
                 pass Clotho needs (vLLM does not expose hidden states).
Times use time.perf_counter() with torch.cuda.synchronize() around GPU work; one warm-up batch first.
"""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import prompt_render as pr  # noqa: E402

TASK, TEMPLATE, MODEL = "log_parsing", "rules_template", "llama"
SYSTEMS = ["loghub_hdfs_500", "loghub_hadoop_500", "loghub_bgl_500", "loghub_openstack_500"]
OUT = EXP / "results_phase3" / "timing"
N_BS1 = 30
KS = [1, 2, 3, 4, 5]
LAYER = 16


def suite(ds):
    return [json.loads(l) for l in open(EXP.parent / "clotho" / "dataset" / TASK / f"{ds}.jsonl")]


def run_hf():
    import numpy as np
    import torch
    import clotho.models.llama as llama
    from clotho.exp_config import task2template, task2input_key
    fn = task2template[TASK][TEMPLATE]
    model, tok = llama.model, llama.tokenizer
    res = {"device": torch.cuda.get_device_name(0), "systems": {}}
    sync = torch.cuda.synchronize
    for ds in SYSTEMS:
        rows = suite(ds)
        texts = [r[task2input_key[TASK]] for r in rows]
        r = {}
        # pipeline, batch size 1
        llama.get_hidden_states(texts[0], fn)                           # warm-up
        sync(); t0 = time.perf_counter()
        for x in texts[:N_BS1]:
            llama.get_hidden_states(x, fn)["hidden_states"][LAYER][0, -1].float().cpu()
        sync(); r["pipeline_bs1"] = (time.perf_counter() - t0) / N_BS1
        # batched, layers 1-16 only
        prompts = [pr.render(TASK, TEMPLATE, row, tok)[1] for row in rows]
        layers = model.model.layers
        model.model.layers = layers[:LAYER + 1]          # hidden_states[16] = output of layer 16, before norm
        tok.padding_side = "left"
        bs, vecs = 32, []

        def batch(ids_list):
            m = max(len(x) for x in ids_list)
            ids = torch.tensor([[tok.pad_token_id] * (m - len(x)) + x for x in ids_list], device=model.device)
            att = torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in ids_list], device=model.device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
            return out.hidden_states[LAYER][:, -1].float().cpu()
        batch(prompts[:bs])
        sync(); t0 = time.perf_counter()
        for i in range(0, len(prompts), bs):
            vecs.append(batch(prompts[i:i + bs]))
        sync(); r["batched_l16"] = (time.perf_counter() - t0) / len(prompts)
        model.model.layers = layers
        V = torch.cat(vecs).numpy()
        stored = np.asarray(torch.load(EXP / f"data/results_llama/{TASK}/input_hidden_states/{TEMPLATE}/{ds}/layer_{LAYER}/hidden_vectors.pt",
                                       weights_only=False), dtype=np.float32)
        cos = (V * stored).sum(1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(stored, axis=1))
        r["batched_parity_min_cos"] = float(cos.min())
        r["prompt_tokens_mean"] = float(np.mean([len(p) for p in prompts]))
        res["systems"][ds] = r
        print(ds, r, flush=True)
    # CPU scoring
    import log_parsing_phase3_lib as L
    from sklearn.mixture import GaussianMixture
    import log_parsing_drain as D
    H = np.asarray(torch.load(EXP / f"data/results_llama/{TASK}/input_hidden_states/{TEMPLATE}/{SYSTEMS[0]}/layer_{LAYER}/hidden_vectors.pt",
                              weights_only=False), dtype=np.float32)
    for nf, nc in ((10, 5), (50, 10)):
        dr = L.PCAFeatureReducer(n_features=nf); dr.fit(H)
        g = GaussianMixture(n_components=nc, covariance_type="full", random_state=0, reg_covar=1e-3).fit(dr.transform(H[:200]))
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            for i in range(500):
                g.score_samples(dr.transform(H[i:i + 1]))
        res[f"gmm_bs1_nf{nf}_nc{nc}"] = (time.perf_counter() - t0) / (reps * 500)
    sa = L.SA(mode="MDSA", dimension_reducer=dr, random_state=0)
    sa.register_reference_set(H[:100])
    t0 = time.perf_counter()
    for i in range(500):
        sa.calculate(H[i:i + 1])
    res["mdsa_bs1"] = (time.perf_counter() - t0) / 500
    import tempfile
    dts = []
    for ds in SYSTEMS:
        msgs, _, system = D.load_suite(ds)
        with tempfile.TemporaryDirectory() as d:
            t0 = time.perf_counter()
            D.drain_lh2(msgs, system, d)
            dts.append((time.perf_counter() - t0) / len(msgs))
    res["drain_lh2_per_input"] = float(np.mean(dts))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hf.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def run_vllm():
    from vllm import LLM, SamplingParams
    tok = pr.load_tokenizer(MODEL)
    llm = LLM(model=pr.MODEL_IDS[MODEL], dtype="bfloat16", seed=2, max_model_len=8192,
              gpu_memory_utilization=0.90, generation_config="vllm")

    def sp(n, max_tokens=pr.GEN_MAX_NEW_TOKENS):
        return SamplingParams(n=n, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=max_tokens,
                              ignore_eos=True, stop_token_ids=[tok.eos_token_id], detokenize=False)
    res = {"systems": {}}
    for ds in SYSTEMS:
        ids = [{"prompt_token_ids": pr.render(TASK, TEMPLATE, r, tok)[1]} for r in suite(ds)]
        llm.generate(ids[:32], sp(2), use_tqdm=False)                   # warm-up
        r = {}
        t0 = time.perf_counter()
        llm.generate(ids, sp(1, max_tokens=1), use_tqdm=False)
        r["prefill"] = (time.perf_counter() - t0) / len(ids)
        for k in KS:
            t0 = time.perf_counter()
            outs = llm.generate(ids, sp(k), use_tqdm=False)
            r[f"gen_{k}"] = (time.perf_counter() - t0) / len(ids)
            r[f"gen_{k}_out_tokens_mean"] = sum(len(c.token_ids) for o in outs for c in o.outputs) / (len(ids) * k)
            t0 = time.perf_counter()
            for x in ids[:N_BS1]:
                llm.generate([x], sp(k), use_tqdm=False)
            r[f"gen_{k}_bs1"] = (time.perf_counter() - t0) / N_BS1
        res["systems"][ds] = r
        print(ds, r, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vllm.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hf", "vllm"])
    a = ap.parse_args()
    run_hf() if a.mode == "hf" else run_vllm()
