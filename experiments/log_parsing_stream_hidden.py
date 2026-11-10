"""Hidden states of the cached parser's LLM calls (TRIAGE / E, 2026-11-10): the first line of each Drain group.

  cd experiments && CLOTHO_MODEL=llama python log_parsing_stream_hidden.py [--push]
  -> data/stream_hidden/<model>/<Sys>_layer<L>.npy  (fair layer and D's tuned layer)

Same vector as log_parsing_layers.py (last prompt token, hidden_states[L]); one forward pass per prompt gives every
layer. In a deployment this comes from the parser's own prefill; here it is a separate HF pass, timed per template.
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
import model_cfg as MC  # noqa: E402
import log_parsing_stream as ST  # noqa: E402

D_LAYER = {"llama": 24, "gemma": 35}[MC.MODEL]
LAYERS = sorted({MC.fair_layer(), D_LAYER})
OUT = EXP / "data" / "stream_hidden" / MC.MODEL
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}[MC.MODEL]
SYSTEMS = [s for s in ST.SYSTEMS if s not in DROP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    todo = [s for s in a.systems if not all((OUT / f"{s}_layer{L}.npy").exists() for L in LAYERS)]
    if not todo:
        print("nothing to do")
        return
    import torch
    from clotho.exp_config import task2template
    w = MC.wrapper()
    fn = task2template["log_parsing"]["rules_template"]
    OUT.mkdir(parents=True, exist_ok=True)
    timing = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    for s in todo:
        msgs = ST.load(s)["first_content"]
        vecs = {L: [] for L in LAYERS}
        t0 = time.perf_counter()
        with torch.no_grad():
            for m in msgs:
                hs = w.get_hidden_states(m, fn, produce_inference_results=False)["hidden_states"]
                for L in LAYERS:
                    vecs[L].append(hs[L][0, -1].cpu().to(torch.float32).numpy())
        torch.cuda.synchronize()
        timing[s] = {"n": len(msgs), "s_per_template": (time.perf_counter() - t0) / len(msgs)}
        files = {}
        for L in LAYERS:
            p = OUT / f"{s}_layer{L}.npy"
            np.save(p, np.asarray(vecs[L], dtype=np.float32))
            files[str(p.relative_to(EXP))] = p
        (OUT / "timing.json").write_text(json.dumps(timing, indent=1))
        files[str((OUT / "timing.json").relative_to(EXP))] = OUT / "timing.json"
        print(f"{s}: {len(msgs)} templates, {timing[s]['s_per_template']*1000:.1f} ms each", flush=True)
        if a.push:
            import hub_sync as hs
            hs.commit_files(hs.get_results_repo(), files, f"stream hidden {MC.MODEL} {s}")


if __name__ == "__main__":
    main()
