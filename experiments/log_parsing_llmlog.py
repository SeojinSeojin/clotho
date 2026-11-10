"""LLMLog's log-selection score as a failure ranking (Experiment A, 2026-11-09).

Source: LLMLog (arXiv 2508.09594, "LLMLog: Advanced Log Template Generation via LLM-driven Multi-Round
Annotation"), official code github.com/XinTT/LLMLog @648553d (LLMLog.py: `annotation`, `max_selection`,
`edit_distance_on_real_sequences`; README run: --selection max_influence --prompt adaptive --lambda 0.5).

LLMLog picks logs to annotate by greedily maximising (paper Eq. 8, code `max_selection`)
    lambda * C(s) + (1 - lambda) * |I_s minus already covered| / |U|
  C(s)  confidence (Eq. 7): a * (1 - P) + (1 - a) * I, P = mean token probability of the predicted template,
        I = word-consistency indicator (code: 1 if some predicted template word, with `<*>` removed and split on
        spaces, is not a substring of any space-separated token of the log).
  I_s   logs represented by s (Eq. 4): SED(s, s') <= delta * min(len(s), len(s')), SED = word-level edit distance
        over the words not already covered by labelled logs (code: word cost 0 iff the words are equal, since the
        README run (`--prompt adaptive`) loads no word-similarity table; distance averaged over both directions).
  Here: U = the test set (the inputs still unlabelled), labelled logs = the non-initial reference inputs (their
  text only; no labels are used). lambda = 0.5, delta = 0.5 (paper Sec. 4.1 and the code's hard-coded 0.5),
  a = 0.3 (code `weighted` mode: 0.3 * probability part + 0.7 * consistency part; the paper gives no value).
The greedy pick order is the ranking: picked earlier = flagged earlier as a likely failure. The score used for
ROC-AUC (higher = PASS) is the pick position. Inputs left when the greedy gain reaches 0 follow, by C descending.
  llmlog       lambda = 0.5 (the full selection score)
  llmlog_conf  lambda = 1 (confidence alone: order by C)
  llmlog_rep   lambda = 0 (representativeness alone: greedy max coverage, the paper's LLMLog\\PC ablation)
  *_nokw       the same with SED over all words of each log. With 90-190 labelled reference inputs, the faithful s^r
               is often empty for every test log (all its words already occur in the reference set), so every log
               represents every other one and the coverage term vanishes after the first pick; the _nokw rows show
               representativeness as LLMLog's first round sees it (nothing labelled yet).

Deviations (also listed in reports/log_parsing_literature_details.md):
  - Leak-free: C is the mean over generation samples 0-4 of the per-sample C (LLMLog uses its one prediction).
  - P is the mean token probability over the whole generated output (reasoning line + template + end token),
    from the stored teacher-forced token log-probs (log_parsing_postgen.py), not over template words only.
  - Words: LLMLog reads pre-processed word lists (processedLogs/parsing_<sys>.json, only for 8 Loghub-2k systems,
    no generator released). Here: split on whitespace and on []()<>{}:;,=|"' , strip, drop tokens with a digit
    (their `parsing_parameters` drop variable-like tokens).
  - The code normalises coverage by 2000 (its dataset size); the paper by |U|. Here |U| (= test-set size).
  - No multi-round loop, no DPP initialisation, no adaptive budget: the ranking covers the whole test set once,
    given the reference set as the labelled set.
"""
import math
import re

import numpy as np

LAMBDA, DELTA, A_PROB = 0.5, 0.5, 0.3
_SPLIT = re.compile(r"[\s\[\]()<>{}:;,=|\"']+")
_DIGIT = re.compile(r"\d")


def words(content):
    """Constant-like words of a log message (stand-in for LLMLog's `parsing_parameters`)."""
    return [w for w in (x.strip() for x in _SPLIT.split(content)) if w and not _DIGIT.search(w)]


def consistency(parsed, content):
    """LLMLog.annotation word-consistency check: 1 = inconsistent (or no template)."""
    if parsed is None:
        return 1
    toks = content.split(' ')
    for item in parsed.replace('<*>', '').split(' '):
        if not any(item in t for t in toks):
            return 1
    return 0


def confidence(token_logprobs, parsed, content):
    """Eq. 7 for one sample: a * (1 - mean token prob) + (1 - a) * inconsistency."""
    p = float(np.mean(np.exp(token_logprobs))) if len(token_logprobs) else 0.0
    return A_PROB * (1 - p) + (1 - A_PROB) * consistency(parsed, content)


def _ed(a, b):
    """Word-level Levenshtein distance (edit_distance_on_real_sequences with exact word match)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def represented(seqs, delta=DELTA):
    """Boolean matrix R[i, j] = j is represented by i (Eq. 4), diagonal True (the code adds s itself)."""
    uniq = {}
    key = [uniq.setdefault(tuple(s), len(uniq)) for s in seqs]
    us = list(uniq)
    m = len(us)
    U = np.zeros((m, m), dtype=bool)
    for a in range(m):
        U[a, a] = True                          # identical sequences: distance 0 <= threshold
        for b in range(a + 1, m):
            d = _ed(us[a], us[b])               # symmetric, so 0.5 * (d(a,b) + d(b,a)) = d
            U[a, b] = U[b, a] = d <= delta * min(len(us[a]), len(us[b]))
    k = np.array(key)
    R = U[np.ix_(k, k)]
    np.fill_diagonal(R, True)
    return R


def greedy_order(C, R, lam, seed=0):
    """max_selection: repeatedly pick argmax lam * C + (1 - lam) * |R[i] minus covered| / n (first max wins).
    When no pick has a positive gain, the rest follow by the static score lam * C + (1 - lam) * |R[i]| / n,
    ties in a seeded random order (so the representativeness-only order never uses C)."""
    n = len(C)
    C = np.asarray(C, dtype=float)
    covered = np.zeros(n, dtype=bool)
    picked = np.zeros(n, dtype=bool)
    order = []
    Ri = R.astype(np.int32)
    for _ in range(n):
        gain = Ri @ (~covered).astype(np.int32)
        w = lam * C + (1 - lam) * gain / n
        w[picked] = -math.inf
        b = int(np.argmax(w))
        if not w[b] > 0:
            break
        order.append(b)
        picked[b] = True
        covered |= R[b]
    static = lam * C + (1 - lam) * R.sum(1) / n
    tie = np.random.RandomState(seed + 31337).rand(n)
    rest = [int(i) for i in np.lexsort((tie, -static)) if not picked[i]]
    return order + rest


VARIANTS = (("llmlog", LAMBDA, True), ("llmlog_conf", 1.0, True), ("llmlog_rep", 0.0, True),
            ("llmlog_nokw", LAMBDA, False), ("llmlog_rep_nokw", 0.0, False))


def scores(test, ref_noninit, contents, conf, seed=0, variants=VARIANTS):
    """-> {name: score over test (higher = PASS = picked later)}. contents: messages by input index; conf: per-input
    C (mean over samples 0-4). kw=True: SED over the words not covered by the reference inputs (LLMLog's s^r);
    kw=False: over all words (as if nothing were labelled yet, LLMLog's first round)."""
    kw = set()
    for i in ref_noninit:
        kw.update(words(contents[i]))
    C = np.asarray([conf[i] for i in test])
    Rs = {}
    out = {}
    for name, lam, use_kw in variants:
        if lam == 1.0:
            order = [int(i) for i in np.lexsort((np.random.RandomState(seed + 31337).rand(len(C)), -C))]
        else:
            if use_kw not in Rs:
                Rs[use_kw] = represented([[w for w in words(contents[i]) if not use_kw or w not in kw] for i in test])
            order = greedy_order(C, Rs[use_kw], lam, seed)
        pos = np.empty(len(test))
        pos[np.asarray(order, dtype=int)] = np.arange(len(test))
        out[name] = pos
    return out
