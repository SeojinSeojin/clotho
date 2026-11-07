"""Phase 3 harness: Clotho with an explicit reference pool and test set, plus baselines on the same splits.

Clotho logic is the core loop of run_clotho_iterations.simulate_iterations (random extension,
n_features chosen by CV via rci.get_next_n_features, n_components adapted from the GMM
perplexity, GMM fitted on passing references only, skip while < 10 passing references), with
three changes so that splits can be controlled:
  * reference extension draws only from `pool` (core: every non-reference input),
  * scores are evaluated on `test` (core: every non-reference input),
  * extension uses np.random.RandomState(seed) (core: the unseeded global np.random).
With pool = all suite inputs and test = None (complement of the reference set) this is the core
protocol; check_core_equivalence() compares it against rci.simulate_iterations on one system.

Score convention for every method: higher = more likely to PASS. y = 1 for PASS (test score > 0.5).
"""
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import utils  # noqa: E402
import run_clotho_iterations as rci  # noqa: E402
from clotho.preprocessing.feature_reduction import PCAFeatureReducer  # noqa: E402
from clotho.metrics.reference_based.density_estimation import GMMScorerConfig  # noqa: E402
from clotho.metrics.sa import SA  # noqa: E402
from clotho.dataset.log_parsing.output_parsers import score_output  # noqa: E402
from log_parsing_initial_set import initial_dir  # noqa: E402

TASK = "log_parsing"
TEMPLATE = "rules_template"
LAYER = 21
MODEL = "llama"
PASS_THRESH = 0.5
REF_SIZES = (50, 100, 200)
SUITES = EXP.parent / "clotho" / "dataset" / "log_parsing"


# ---------------------------------------------------------------- data

def load_system(ds, template=TEMPLATE):
    """Hidden states, test scores, templates, entropy of the 10 parsed outputs, for one suite."""
    utils.target_testsuites[TASK] = [ds]
    utils.prompt_templates[TASK] = template
    cwd = Path.cwd()
    import os
    os.chdir(EXP)
    try:
        H = utils.load_input_hidden_states(TASK, LAYER, model=MODEL).numpy().astype(np.float32)
        res, labels, scores = utils.get_test_results(MODEL, TASK)
    finally:
        os.chdir(cwd)
    src = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
    assert len(src) == len(res) == len(H)
    assert all(s["template"] == r["GT"][0] for s, r in zip(src, res))
    ent = []
    for r in res:
        parsed = [score_output(o, r["GT"], template)["parsed"] for o in r["inferences"]]
        c = Counter("<PARSE_FAILURE>" if p is None else p for p in parsed)
        n = sum(c.values())
        ent.append(-sum(v / n * math.log(v / n) for v in c.values()))
    return {"ds": ds, "H": H, "scores": np.array(scores, dtype=float),
            "templates": [s["template"] for s in src], "entropy": np.array(ent)}


def load_initial(template=TEMPLATE):
    d = EXP / initial_dir(MODEL, template)
    H = torch.load(d / f"hidden_vectors_layer_{LAYER}.pt", weights_only=False).astype(np.float32)
    scores = np.array(json.load(open(d / "inference_results.json"))["test_scores"], dtype=float)
    return H, scores


def assemble(systems, init):
    """Stack suites + initial tests. Returns (H, scores, templates, entropy, offsets, init_idx).
    Initial tests get unique pseudo-templates (never match a Loghub template)."""
    Hs, sc, tp, en, offsets = [], [], [], [], {}
    n = 0
    for s in systems:
        offsets[s["ds"]] = np.arange(n, n + len(s["H"]))
        n += len(s["H"])
        Hs.append(s["H"]); sc.append(s["scores"]); tp += s["templates"]; en.append(s["entropy"])
    Hi, si = init
    init_idx = np.arange(n, n + len(Hi))
    tp += [f"__initial_{i}__" for i in range(len(Hi))]
    return (np.concatenate(Hs + [Hi]), np.concatenate(sc + [si]), tp,
            np.concatenate(en + [np.full(len(Hi), np.nan)]), offsets, init_idx)


# ---------------------------------------------------------------- Clotho

def clotho_run(H, scores, init_idx, pool, test, seed, ref_sizes=REF_SIZES, num_add=10, global_rng=False):
    """Returns {ref_size: {"ref": ref indices, "test": test indices, "clotho": scores on test or None}}.
    test=None -> every input of `pool` not in the reference set (core protocol)."""
    labels = scores
    max_n_components = max_n_features = 50
    rng = np.random if global_rng else np.random.RandomState(seed)  # global: equivalence check only
    ref = list(init_idx)
    pool = list(pool)
    LIH_rep = {}
    current_n_features, current_n_components = 10, 5
    scorer_config = GMMScorerConfig(n_components=current_n_components, random_state=seed)
    out = {}
    while len(ref) <= max(ref_sizes):
        num_eff = int((labels[ref] > PASS_THRESH).sum())
        ref_lab = (labels[ref] > PASS_THRESH).astype(int)
        skip = ref_lab.sum() < 10
        if current_n_features not in LIH_rep:
            LIH_rep[current_n_features] = PCAFeatureReducer(n_features=current_n_features).fit_transform(H)
        scorer = None
        if not skip:
            current_n_features = rci.get_next_n_features(H, LIH_rep, ref, labels, current_n_features,
                                                         max_n_features, "pca", scorer_config)
            _s = rci.construct_scorer(replace(scorer_config, n_init=10))
            _s.fit(LIH_rep[current_n_features][ref], ref_lab)
            perp = utils.get_gmm_components_perplexity(_s.model.weights_)
            if perp > current_n_components - 0.5:
                nxt = min(current_n_components + 1, max_n_components)
                if num_eff // 10 > nxt:
                    current_n_components = nxt
                    scorer_config = replace(scorer_config, n_components=nxt)
            elif perp < current_n_components - 1.5:
                current_n_components = max(current_n_components - 1, 5)
                scorer_config = replace(scorer_config, n_components=current_n_components)
            scorer = rci.construct_scorer(scorer_config)
            scorer.fit(LIH_rep[current_n_features][ref], ref_lab)
        if len(ref) in ref_sizes:
            rs = set(ref)
            t = np.array([i for i in pool if i not in rs]) if test is None else np.asarray(test)
            out[len(ref)] = {"ref": np.array(ref), "test": t,
                             "clotho": None if scorer is None else scorer.score(LIH_rep[current_n_features][t])}
        # extension, same draw as core extend_reference_set_random (but from `pool`, seeded)
        rs = set(ref)
        cand = [i for i in pool if i not in rs]
        if len(cand) < num_add:
            break
        ref.extend(rng.choice(cand, size=num_add, replace=False).tolist())
    return out


# ---------------------------------------------------------------- baselines (same ref / test)

def template_lookup(ref, test, scores, templates, init_idx):
    """Failure rate of reference inputs with the same template; unseen -> overall reference failure
    rate. Initial tests (synthetic) are not used. Returned negated (higher = more likely pass)."""
    init = set(init_idx.tolist())
    r = [i for i in ref if i not in init]
    fail = {}
    for i in r:
        fail.setdefault(templates[i], []).append(scores[i] <= PASS_THRESH)
    overall = float(np.mean([scores[i] <= PASS_THRESH for i in r]))
    return -np.array([np.mean(fail[templates[i]]) if templates[i] in fail else overall for i in test])


def surprise_adequacy(H, ref, test, scores, mode, seed, n_features=10):
    """run_baseline_OOD.py: SA (clotho.metrics.sa) on passing references, PCA(10) on all inputs."""
    dr = PCAFeatureReducer(n_features=n_features)
    dr.fit(H)
    passing = [i for i in ref if scores[i] > PASS_THRESH]
    sa = SA(mode=mode, dimension_reducer=dr, random_state=seed)
    sa.register_reference_set(H[passing])
    return -np.asarray(sa.calculate(H[test]))


def metrics(y_pass, s, seed, k=50):
    """ROC-AUC (y = PASS) and #failures among the k least pass-like test inputs (ties broken at random)."""
    y_pass = np.asarray(y_pass).astype(int)
    auc = roc_auc_score(y_pass, s) if 0 < y_pass.sum() < len(y_pass) else float("nan")
    tie = np.random.RandomState(seed + 7919).rand(len(s))
    order = np.lexsort((tie, s))            # ascending score: least pass-like first
    return auc, int((y_pass[order[:k]] == 0).sum())


def check_core_equivalence(ds="loghub_hdfs_500", seed=0, iters=10):
    """Run rci.simulate_iterations (global np.random seeded) and clotho_run on the standard split;
    compare ROC-AUC at each reference size. Returns list of (ref, core_auc, harness_auc)."""
    import pandas as pd  # noqa: F401
    s = load_system(ds)
    H, sc, tp, en, off, init_idx = assemble([s], load_initial())
    np.random.seed(seed)
    n = len(s["H"])  # signature: (task, layer, LIH_init, LIH_target, labels_init, labels_target, ...)
    df, _ = rci.simulate_iterations(TASK, LAYER, H[n:], H[:n], sc[n:], sc[:n],
                                    refset_extension_method="random", random_seed=seed,
                                    n_iterations=iters, plot_intermediate_results=False,
                                    LIH_vis=np.zeros((len(H), 2)))  # skip t-SNE: it would consume the global RNG
    sizes = tuple(10 + 10 * i for i in range(iters))
    np.random.seed(seed)
    mine = clotho_run(H, sc, init_idx, off[ds], None, seed, ref_sizes=sizes, global_rng=True)
    rows = []
    for it, g in df.groupby("iteration"):
        t = g[g.label == "test"]
        y = (t.test_score > PASS_THRESH).astype(int)
        ref_n = int((g.label == "ref").sum())
        m = mine.get(ref_n)
        a2 = metrics((sc[m["test"]] > PASS_THRESH), m["clotho"], seed)[0] if m and m["clotho"] is not None else None
        rows.append((ref_n, roc_auc_score(y, t.logprob), a2))
    return rows
