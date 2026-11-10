# TRIAGE: risk-informed review of a cached LLM log parser (E1-E5)

Setup (PROGRESS.md Decisions): first 50,000 Loghub-2.0 lines per system; Drain (Loghub-2.0 settings) groups lines; the LLM parses each group's first line (greedy) and the template is cached. Llama 13 systems, Gemma 12. Share fixed = share of mis-parsed lines fixed; mean ± std over 5 seeds; (k/5) = seeds keeping the sign. Details: `triage_details.md`.

**Baseline.** Cached PA is 0.321 (Llama) / 0.364 (Gemma); on average 59% / 54% of groups (11-97% per system) get a wrong template.

**E1, share fixed after 20 reviews.**

| ordering | Llama | Gemma |
|---|---|---|
| Oracle | 0.878 | 0.863 |
| Most frequent | 0.861 | 0.833 |
| TRIAGE without active selection | 0.851 ± 0.013 | 0.814 |
| LLMLog confidence x size | 0.845 | 0.796 |
| TRIAGE | 0.707 ± 0.036 | 0.719 |
| Clotho alone | 0.364 ± 0.017 | 0.314 ± 0.033 |
| Random | 0.323 ± 0.052 | 0.248 ± 0.054 |

TRIAGE reaches 0.80 / 0.83 of the oracle, and 0.97 / 0.94 without active selection. It trails Most frequent on every seed (-0.154 / -0.115, 5/5). SK-ESD: Llama puts TRIAGE below Most frequent; Gemma puts both in the top group (A12 0.33 medium / 0.39 small). Held-out systems give the same order. Ours (active + lookup) equals prior x size, because coarse Drain families rarely hold two groups. AdaParser raises PA_0 by only +0.022 / +0.045; on top of it, Most frequent is again best (0.863 / 0.841).

**E2, ablation at 20 reviews.** Removing propagation costs -0.581 / -0.611 and removing frequency weighting -0.387 / -0.439. Removing active selection improves the share (+0.144 / +0.095), and so does removing the prior (+0.129 / +0.059); all 5/5.

**E3, cold start.** The prior alone loses to frequency-first (N = 1: -0.271 / -0.241, 5/5). Prior x size wins only the first review (+0.092 / +0.020, 5/5) and ties by N = 10.

**E4, drift.** On lines of templates first seen in the second half, at 20 reviews, TRIAGE beats Most frequent for Llama (+0.156, 5/5) but not for Gemma (-0.006). At 100 reviews it trails prior x size (-0.026 / -0.015). There is no consistent advantage.

**Cost per new template.** TRIAGE CPU: 85 ms (Llama MLP) / 2 ms (Gemma logistic). Hidden state: 42 / 59 ms as a separate pass, free from the parser's prefill. AdaParser corrector: 74 / 56 ms of vLLM time.

**E5, leakage beyond logs (Llama, 8 Clotho tasks).** Spearman, leak-free minus leaky: Tok-Prob -0.024 ± 0.003, Tok-Ent -0.021 ± 0.004, output entropy -0.151 ± 0.001, pre-generation MDSA control -0.011 ± 0.001 (all 5/5). The paper's Tok-Prob and Tok-Ent use one sample, so they leak little. Changes: the best method on 1 of 8 tasks; pairwise comparisons 1 of 48 (Spearman) and 4 of 48 (AUC). LOHSvar and Sem-Ent need recomputation (LOHSvar about 3.5-5 GPU hours).

**Answer.** Group size and propagation decide which templates are worth reviewing. A zero-label prior x size is a sound default, level with frequency-first. TRIAGE's active selection, as specified, loses 0.10-0.15 of the fixable lines at 20 reviews.
