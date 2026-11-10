"""Extra generations per input (same prompt, sampling and decoding as gen_vllm), for label-noise analysis.

  .venv-vllm/bin/python experiments/log_parsing_extra_samples.py [-n 40] [--seed 1] [-d <datasets>] [--push]
  -> experiments/data/results_llama/log_parsing/extra_samples/rules_template/<ds>_R<n>_T0.8_seed<seed>.json

Uses sharded/prompt_render.py (the renderer the HF/vLLM parity check covers) and the SamplingParams of
sharded/vllm_generate.py, with a different engine seed than the original run (seed 0), so these samples are
new draws from the same distribution. The original R10 files are not touched. Resumable per system.
"""
import argparse
import json
import os
import sys

# Same as sharded/vllm_generate.py: FlashInfer's sampler JIT-compiles CUDA code and needs nvcc, which these
# servers lack; vLLM's PyTorch sampler draws from the same softmax(logits / T).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / "sharded"))
import prompt_render as pr  # noqa: E402
sys.path.insert(0, str(EXP))
import model_cfg as MC  # noqa: E402

TASK, TEMPLATE, MODEL = "log_parsing", "rules_template", MC.MODEL      # CLOTHO_MODEL=gemma (2026-11-09 night)
SYSTEMS = ["loghub_apache_500", "loghub_bgl_500", "loghub_hdfs_500", "loghub_hpc_500", "loghub_hadoop_500",
           "loghub_healthapp_500", "loghub_linux_500", "loghub_mac_500", "loghub_openssh_500",
           "loghub_openstack_500", "loghub_spark_500", "loghub_thunderbird_500", "loghub_zookeeper_500"]
SYSTEMS = [d for d in SYSTEMS if d not in MC.dropped()]
OUT = EXP / f"data/results_{MODEL}/{TASK}/extra_samples/{TEMPLATE}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-d", "--datasets", nargs="+", default=SYSTEMS)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    assert a.seed != 0, "seed 0 is the original gen_vllm run"
    todo = [ds for ds in a.datasets if not (OUT / f"{ds}_R{a.n}_T{pr.GEN_TEMPERATURE}_seed{a.seed}.json").exists()]
    if not todo:
        print("nothing to do")
        return
    from vllm import LLM, SamplingParams
    from clotho.exp_config import task2answer_key_list
    tok = pr.load_tokenizer(MODEL)
    llm = LLM(model=pr.MODEL_IDS[MODEL], dtype="bfloat16", seed=a.seed, max_model_len=8192,
              gpu_memory_utilization=a.gpu_memory_utilization, generation_config="vllm")
    sp = SamplingParams(n=a.n, temperature=pr.GEN_TEMPERATURE, top_p=1.0, max_tokens=pr.GEN_MAX_NEW_TOKENS,
                        ignore_eos=True, stop_token_ids=[pr.stop_token_id(tok)], detokenize=False)   # llama: eos_token_id (unchanged)
    keys = task2answer_key_list[TASK]
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in todo:
        rows = [json.loads(l) for l in open(EXP.parent / "clotho" / "dataset" / TASK / f"{ds}.jsonl")]
        ids = [pr.render(TASK, TEMPLATE, r, tok)[1] for r in rows]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=True)
        recs = [{"inferences": [pr.postprocess(tok, x, c.token_ids) for c in o.outputs], "GT": [r[k] for k in keys]}
                for r, x, o in zip(rows, ids, outs)]
        assert all(len(r["inferences"]) == a.n for r in recs)
        dest = OUT / f"{ds}_R{a.n}_T{pr.GEN_TEMPERATURE}_seed{a.seed}.json"
        dest.write_text(json.dumps(recs))
        print(f"wrote {dest.relative_to(EXP)}", flush=True)
        if a.push:
            import hub_sync as hs
            hs.commit_files(hs.get_results_repo(), {dest.relative_to(EXP).as_posix(): dest},
                            f"extra samples {ds} n={a.n} seed={a.seed}")


if __name__ == "__main__":
    main()
