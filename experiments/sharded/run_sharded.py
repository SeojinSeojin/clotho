"""Sharded, resumable wrapper around the original data_extraction scripts.

Examples (from the repo root, with .venv active):

  # input hidden states (1-1), 2 shards as a smoke test
  python experiments/sharded/run_sharded.py --stage hidden -t spell_check -d misspell_injected_wordnet --limit_shards 2

  # generation for pass/fail labels with vLLM (re-execs into .venv-vllm)
  python experiments/sharded/run_sharded.py --stage gen_vllm -t spell_check -d misspell_injected_wordnet

  # original HF generation + output hidden states (2-1); very slow
  python experiments/sharded/run_sharded.py --stage gen_hf -t spell_check -d misspell_injected_wordnet

  # what's done?
  python experiments/sharded/run_sharded.py --stage hidden -t spell_check -d misspell_injected_wordnet --status

For `hidden`/`gen_hf` each shard runs the UNMODIFIED original script in its own
working directory, whose ./dataset/<task>/<dataset>.jsonl contains only that
shard's lines; outputs land in <workdir>/../data/results_<model>/... exactly as
in a normal run, and are then moved into the shard directory. Note: every
shard re-loads the model (~30s), so for the fast `hidden` stage a larger
--shard_size is more efficient.
"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shardlib  # noqa: E402

DATA_EXTRACTION = shardlib.EXP_DIR / "data_extraction"
SCRIPTS = {
    "hidden": DATA_EXTRACTION / "1-1_extract_input_hidden_states.py",
    "gen_hf": DATA_EXTRACTION / "2-1_generate_outputs_and_record_hidden_states.py",
}
VLLM_PYTHON = shardlib.REPO_ROOT / ".venv-vllm" / "bin" / "python"


def make_original_script_runner(job):
    script = SCRIPTS[job.stage]

    def run_fn(k, lines, out_dir):
        work = job.local / "work" / job.shard_name(k)
        cwd = work / "cwd"
        ds = cwd / "dataset" / job.task / f"{job.dataset}.jsonl"
        content = "\n".join(lines) + "\n"
        if not ds.exists() or ds.read_text() != content:
            shutil.rmtree(work, ignore_errors=True)  # stale work dir from different inputs
            ds.parent.mkdir(parents=True, exist_ok=True)
            ds.write_text(content)
        # (gen_hf: if a previous attempt in this session died mid-shard, 2-1
        # resumes from the partial json still in work/data/.)
        cmd = [sys.executable, "-u", str(script), "-m", job.model, "-t", job.task,
               "-d", job.dataset, "-p", job.template]
        print(f"[shard] {job.shard_name(k)}: {len(lines)} inputs -> {' '.join(cmd[2:])}", flush=True)
        with open(work / "run.log", "a") as log:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                if "Skipping input" not in line and "Memory Status" not in line:
                    sys.stdout.write(line)
            rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"[shard] {script.name} failed with exit code {rc} (log: {work / 'run.log'})")

        results = work / "data" / f"results_{job.model}" / job.task
        n = len(lines)
        if job.stage == "hidden":
            src = results / "input_hidden_states"
            pts = sorted(src.rglob("hidden_vectors.pt"))
            if not pts:
                raise SystemExit(f"[shard] no hidden_vectors.pt produced under {src}")
            import torch
            for p in pts:
                a = torch.load(p, weights_only=False)
                assert a.shape[0] == n, f"{p}: {a.shape[0]} rows, expected {n}"
                dest = out_dir / p.relative_to(work / "data")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        else:  # gen_hf
            import json
            jsons = sorted((results / "inference_results").rglob("*.json"))
            assert len(jsons) == 1, f"expected one inference json, got {jsons}"
            recs = json.loads(jsons[0].read_text())
            assert len(recs) == n, f"{jsons[0]}: {len(recs)} records, expected {n}"
            dest = out_dir / jsons[0].relative_to(work / "data")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jsons[0], dest)
            # thousands of small .pt files -> one uncompressed tar per shard
            with tarfile.open(out_dir / "output_hidden_states.tar", "w") as tf:
                for p in sorted((results / "output_hidden_states").rglob("*.pt")):
                    tf.add(p, arcname=p.relative_to(work / "data").as_posix())
        shutil.rmtree(work, ignore_errors=True)

    return run_fn


def main():
    args = shardlib.build_arg_parser().parse_args()
    if args.stage == "gen_vllm":
        if not VLLM_PYTHON.exists():
            raise SystemExit(f"{VLLM_PYTHON} not found: run `INSTALL_VLLM=1 bash setup.sh`")
        script = Path(__file__).resolve().parent / "vllm_generate.py"
        os.execv(str(VLLM_PYTHON), [str(VLLM_PYTHON), str(script), *sys.argv[1:]])
    return shardlib.main_with_runner(args, make_original_script_runner)


if __name__ == "__main__":
    sys.exit(main())
