"""Verify that the vLLM path feeds the model exactly what the HF path does.

  python experiments/sharded/check_prompt_parity.py [--tasks spell_check ...] [--n 5]

Step 1 (in .venv, the HF env): imports the REAL clotho/models/llama.py with
  model loading stubbed out. The stub model records the input ids it receives
  and "generates" a scripted answer, so llama.py's own code produces, for each
  sample: the rendered prompt string, the input ids seen by the forward pass
  in get_hidden_states (1-1) and in get_hidden_states_during_generation (2-1),
  and the post-processed output string.
Step 2 (in .venv-vllm): renders the same samples with prompt_render.py (the
  code vllm_generate.py uses) and post-processes the same generated ids.
Then it compares prompt strings, token ids, output strings and generation
constants, and exits non-zero on any mismatch. Needs HF_TOKEN (gated
tokenizer) and a GPU (llama.py exits without CUDA); does not load weights.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
HF_PY = REPO_ROOT / ".venv" / "bin" / "python"
VLLM_PY = REPO_ROOT / ".venv-vllm" / "bin" / "python"

# Scripted "model output": format lines plus non-ASCII text to exercise decoding.
SCRIPTED_ANSWER = "Reasoning: \"tings\" → \"things\" (café, naïve, 日本)\nMisspelled word: tings\nCorrected word: things"


def pick_samples(tasks, n, seed=0):
    from clotho.exp_config import supported_datasets
    out = []
    for task in tasks:
        ds = supported_datasets[task][0]
        lines = (REPO_ROOT / "clotho" / "dataset" / task / f"{ds}.jsonl").read_text().splitlines()
        idx = list(range(min(3, len(lines))))
        rng = random.Random(seed)
        idx += rng.sample(range(3, len(lines)), max(0, n - len(idx)))
        out += [{"task": task, "dataset": ds, "index": i, "data": json.loads(lines[i])} for i in idx[:n]]
    return out


# ------------------------------------------------------------------------ HF side
def hf_side(samples, template):
    import types
    import torch
    import transformers
    import huggingface_hub

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(pad_token_id=None)
            self.calls, self.script, self.vocab = [], [], None

        def resize_token_embeddings(self, n, *a, **k):
            self.vocab = n

        def forward(self, input_ids=None, output_hidden_states=False, **kw):
            self.calls.append(input_ids[0].tolist())
            step = len(self.calls) - 1
            nxt = self.script[step] if step < len(self.script) else self.eos
            logits = torch.full((1, 1, self.vocab), float("-inf"), device=input_ids.device)
            logits[0, -1, nxt] = 0.0
            hs = tuple(torch.zeros(1, 1, 4, device=input_ids.device) for _ in range(33))
            return types.SimpleNamespace(logits=logits, hidden_states=hs)

    stub = StubModel()
    transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **k: stub
    huggingface_hub.login = lambda *a, **k: None
    from clotho.models import llama  # the real module, model load stubbed
    from clotho.exp_config import task2template, task2input_key

    stub.eos = llama.tokenizer.eos_token_id
    rendered = []
    orig_act = llama.tokenizer.apply_chat_template

    def recording_act(*a, **k):
        r = orig_act(*a, **k)
        rendered.append(r)
        return r
    llama.tokenizer.apply_chat_template = recording_act

    gen_ids = llama.tokenizer.encode(SCRIPTED_ANSWER, add_special_tokens=False) + [stub.eos]
    results = []
    for s in samples:
        key = task2input_key[s["task"]]
        iv = s["data"] if key is None else s["data"][key]
        tmpl = task2template[s["task"]][template]

        stub.calls, rendered[:] = [], []
        llama.get_hidden_states(iv, tmpl, produce_inference_results=False)          # 1-1 path
        ids_1_1, prompt_1_1 = stub.calls[0], rendered[0]

        stub.calls, rendered[:], stub.script = [], [], gen_ids
        res = llama.get_hidden_states_during_generation(                            # 2-1 path
            iv, tmpl, repeat=1, target_layers=[16, 23], temperature=0.8, template_keywords=None)
        ids_2_1, prompt_2_1 = stub.calls[0], rendered[0]
        assert stub.calls[-1][len(ids_2_1):] == gen_ids[:-1], "stub generation went off script"
        results.append(s | {"prompt": prompt_2_1, "ids": ids_2_1, "gen_ids": gen_ids,
                            "output": res["inference_results"][0],
                            "hf_1_1_vs_2_1_same": ids_1_1 == ids_2_1 and prompt_1_1 == prompt_2_1})

    src = (REPO_ROOT / "experiments/data_extraction/2-1_generate_outputs_and_record_hidden_states.py").read_text()
    consts = {"REPEAT": int(re.search(r"^REPEAT\s*=\s*(\d+)", src, re.M).group(1)),
              "TEMPERATURE": float(re.search(r"^TEMPERATURE\s*=\s*([\d.]+)", src, re.M).group(1)),
              "MAX_NEW_TOKENS": llama.MAX_NEW_TOKENS,
              "eos_token_id": llama.tokenizer.eos_token_id}
    return {"transformers": transformers.__version__, "constants": consts, "samples": results}


# ---------------------------------------------------------------------- vLLM side
def vllm_side(hf_dump, model):
    import transformers
    import prompt_render as pr
    tok = pr.load_tokenizer(model)
    consts = {"REPEAT": pr.GEN_REPEAT, "TEMPERATURE": pr.GEN_TEMPERATURE,
              "MAX_NEW_TOKENS": pr.GEN_MAX_NEW_TOKENS, "eos_token_id": tok.eos_token_id}
    problems = []
    for name, v in consts.items():
        if hf_dump["constants"][name] != v:
            problems.append(f"constant {name}: HF={hf_dump['constants'][name]} vLLM={v}")
    for s in hf_dump["samples"]:
        tag = f"{s['task']}[{s['index']}]"
        prompt, ids = pr.render(s["task"], hf_dump["template"], s["data"], tok)
        out = pr.postprocess(tok, ids, s["gen_ids"])
        if not s["hf_1_1_vs_2_1_same"]:
            problems.append(f"{tag}: HF 1-1 and 2-1 paths disagree with each other")
        if prompt != s["prompt"]:
            problems.append(f"{tag}: prompt strings differ")
        if ids != s["ids"]:
            problems.append(f"{tag}: token ids differ (len {len(s['ids'])} vs {len(ids)})")
        if out != s["output"]:
            problems.append(f"{tag}: post-processed outputs differ: {s['output']!r} vs {out!r}")
        if max(ids) >= 128256:
            problems.append(f"{tag}: prompt contains the added <pad> token (id {max(ids)}), "
                            f"which the base vLLM model's vocab doesn't have")
    print(f"[parity] transformers: HF venv {hf_dump['transformers']} | vLLM venv {transformers.__version__}")
    print(f"[parity] {len(hf_dump['samples'])} samples, constants {consts}")
    s0 = hf_dump["samples"][0]
    print(f"[parity] example ({s0['task']}[{s0['index']}], {len(s0['ids'])} ids, first ids {s0['ids'][:3]}):")
    print("   " + s0["prompt"][:400].replace("\n", "\n   ") + (" ..." if len(s0["prompt"]) > 400 else ""))
    if problems:
        print("[parity] FAILED:\n  - " + "\n  - ".join(problems))
        return 1
    print("[parity] OK: prompt strings, token ids, post-processed outputs and constants are identical.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["hf", "vllm"])
    p.add_argument("--tasks", nargs="+", default=None, help="default: all tasks")
    p.add_argument("--n", type=int, default=5, help="samples per task")
    p.add_argument("-m", "--model", default="llama")
    p.add_argument("-p", "--template", default="messages_template")
    p.add_argument("--dump", type=Path)
    a = p.parse_args()

    if a.side == "hf":
        from clotho.exp_config import supported_datasets
        samples = pick_samples(a.tasks or list(supported_datasets), a.n)
        d = hf_side(samples, a.template) | {"template": a.template}
        a.dump.write_text(json.dumps(d))
        return 0
    if a.side == "vllm":
        return vllm_side(json.loads(a.dump.read_text()), a.model)

    # orchestrate both sides
    if a.model != "llama":
        raise SystemExit("parity check only implemented for llama")
    extra = (["--tasks", *a.tasks] if a.tasks else []) + ["--n", str(a.n), "-p", a.template]
    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "hf.json"
        subprocess.run([str(HF_PY), __file__, "--side", "hf", "--dump", str(dump), *extra], check=True)
        return subprocess.run([str(VLLM_PY), __file__, "--side", "vllm", "--dump", str(dump),
                               "-m", a.model]).returncode


if __name__ == "__main__":
    sys.exit(main())
