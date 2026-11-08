"""Which LLM the log_parsing scripts run on, and the per-model settings they need.

Select with the environment variable CLOTHO_MODEL (default "llama"):
  CLOTHO_MODEL=gemma python log_parsing_phase3_final.py run

Data paths keep the original layout: data/results_<model>/..., initial_tests_<model>/...
Phase 3 job/report outputs for a model other than llama get a suffix (results_phase3_gemma/,
reports/..._gemma.md) so the Llama results are never overwritten.
"""
import os

MODEL = os.environ.get("CLOTHO_MODEL", "llama")

CFG = {
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "candidate_layers": [11, 16, 21, 24],
        "fair_layer": 16,      # chosen by layer tuning on HDFS, Spark, Linux (Phase 3b)
        "paper_layer": 21,     # run_clotho_iterations.target_layer_map
        "loh_layer": 23,       # RQ/analysis_utils.load_loh_variance_df
        "dropped": ["loghub_proxifier_500"],   # outside 10-90% PASS (> 5 of 10 strict), 2026-11-07
    },
    "gemma": {
        "hf_id": "google/gemma-2-9b-it",
        "n_layers": 42,
        # same relative depths as Llama's 11/16/21/24 of 32 (0.34/0.5/0.66/0.75), with 35 for the deep end;
        # 28 is the paper's Gemma layer for both Clotho and LOHSvar
        "candidate_layers": [14, 21, 28, 35],
        "fair_layer": 21,      # layer tuning 2026-11-08 (HDFS, Spark, Linux; balanced): 14 .758, 21 .773, 28 .765, 35 .769
        "paper_layer": 28,
        "loh_layer": 28,
        # same 10-90% rule (> 5 of 10 strict): Apache 94.4% PASS (reports/log_parsing_pass_rates_gemma.md);
        # Proxifier was not generated (excluded up front, as in the Llama study)
        "dropped": ["loghub_apache_500", "loghub_proxifier_500"],
    },
}[MODEL]

def dropped():
    """Systems excluded for this model (Proxifier for llama; Gemma: same rule, its own list)."""
    return set(CFG["dropped"] or ["loghub_proxifier_500"])


SUFFIX = "" if MODEL == "llama" else f"_{MODEL}"


def fair_layer():
    v = os.environ.get("CLOTHO_FAIR_LAYER")
    if v:
        return int(v)
    if CFG["fair_layer"] is None:
        raise SystemExit(f"no tuned layer for {MODEL} yet: set CLOTHO_FAIR_LAYER or fill model_cfg.CFG")
    return CFG["fair_layer"]


def wrapper():
    """The repo's HF model module (loads the model on import)."""
    import importlib
    return importlib.import_module(f"clotho.models.{MODEL}")


def chat_messages(msgs):
    """Messages exactly as the wrapper feeds them to apply_chat_template."""
    if MODEL == "gemma":   # Gemma-2 has no system role: clotho/models/gemma.py inlines it into the user turn
        from clotho.models.gemma import inline_system_as_user
        return inline_system_as_user(msgs)
    return msgs


def stop_token_id(tok):
    """The token that ends a generation (HF 2-1 loop and vLLM)."""
    return tok.convert_tokens_to_ids("<end_of_turn>") if MODEL == "gemma" else tok.eos_token_id


def vocab_limit(tok):
    """Logit columns of the real vocabulary. Llama's wrapper appends a randomly initialised <pad> row
    (resize_token_embeddings); Gemma has a native pad token, so its whole vocabulary is real."""
    return tok.pad_token_id if MODEL == "llama" else None
