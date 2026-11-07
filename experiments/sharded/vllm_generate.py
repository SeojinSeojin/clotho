"""Sharded vLLM generation for pass/fail labels (stage gen_vllm). Runs in .venv-vllm.

Normally started via:
  python experiments/sharded/run_sharded.py --stage gen_vllm -t <task> -d <dataset>
(which re-execs into .venv-vllm). Same flags as run_sharded.py, plus a few
vLLM-only flags.

Matches 2-1 + clotho/models/llama.py: identical prompt token ids (see
prompt_render.py; verified by check_prompt_parity.py, which runs first),
REPEAT=10 samples at temperature 0.8, pure temperature sampling (no top-p/top-k),
max 512 new tokens, stopping only on tokenizer.eos_token_id, and the same output
post-processing. The output file has the same format and relative path as 2-1's
(<dataset>_R10_T0.8.json: [{"inferences": [...10 strings], "GT": [...]}, ...]).
Output hidden states are NOT produced: use the HF scripts for hidden states.
The samples are drawn by a different RNG, so individual strings differ
from an HF run; the sampling distribution is the same.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shardlib  # noqa: E402
import prompt_render as pr  # noqa: E402


def make_vllm_runner_factory(args):
    def factory(job):
        from clotho.exp_config import task2answer_key_list
        from vllm import LLM, SamplingParams

        tok = pr.load_tokenizer(job.model)
        llm = LLM(model=pr.MODEL_IDS[job.model], dtype="bfloat16", seed=args.seed,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  generation_config="vllm")  # ignore the model's generation_config.json defaults
        sp = SamplingParams(
            n=pr.GEN_REPEAT, temperature=pr.GEN_TEMPERATURE, top_p=1.0,
            max_tokens=pr.GEN_MAX_NEW_TOKENS,
            # HF loop stops only on tokenizer.eos_token_id (<|eot_id|>); ignore_eos
            # disables vLLM's extra generation_config eos ids, stop_token_ids restores eot.
            ignore_eos=True, stop_token_ids=[tok.eos_token_id],
            detokenize=False,  # we decode ourselves, exactly like llama.py
        )
        vocab = llm.llm_engine.model_config.get_vocab_size()
        answer_keys = task2answer_key_list[job.task]

        def run_fn(k, lines, out_dir):
            rows = [json.loads(l) for l in lines]
            prompt_ids = [pr.render(job.task, job.template, r, tok)[1] for r in rows]
            for i, ids in enumerate(prompt_ids):
                if max(ids) >= vocab:
                    raise SystemExit(f"input {k * job.shard_size + i}: token id {max(ids)} >= vocab {vocab} "
                                     f"(literal '<pad>' in input?)")
                if len(ids) + pr.GEN_MAX_NEW_TOKENS > args.max_model_len:
                    raise SystemExit(f"input {k * job.shard_size + i}: prompt has {len(ids)} tokens; "
                                     f"raise --max_model_len")
            outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sp, use_tqdm=True)
            recs = []
            for r, ids, o in zip(rows, prompt_ids, outs):
                infs = [pr.postprocess(tok, ids, c.token_ids) for c in o.outputs]
                assert len(infs) == pr.GEN_REPEAT
                recs.append({"inferences": infs, "GT": [r[key] for key in answer_keys]})
            dest = (out_dir / f"results_{job.model}" / job.task / "inference_results" / job.template
                    / f"{job.dataset}_R{pr.GEN_REPEAT}_T{pr.GEN_TEMPERATURE}.json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(recs, indent=4))

        return run_fn
    return factory


def main():
    p = shardlib.build_arg_parser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--skip_parity_check", action="store_true")
    args = p.parse_args()
    args.stage = "gen_vllm"

    if not (args.status or args.merge_only or args.skip_parity_check):
        rc = subprocess.run([sys.executable, str(HERE / "check_prompt_parity.py"),
                             "--tasks", args.task, "-m", args.model, "-p", args.template]).returncode
        if rc != 0:
            raise SystemExit("prompt parity check failed; refusing to generate (see above).")
    return shardlib.main_with_runner(args, make_vllm_runner_factory(args))


if __name__ == "__main__":
    sys.exit(main())
