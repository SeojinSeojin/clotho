# TRIAGE: risk-informed review of a cached LLM log parser (E1-E5, F)

Setup (PROGRESS.md Decisions): first 50,000 Loghub-2.0 lines per system; Drain (Loghub-2.0 settings) groups lines; the LLM parses each group's first line and the template is cached. Llama 13 systems, Gemma 12. Share fixed = share of mis-parsed lines fixed; mean ± std over 5 seeds; (k/5) = seeds keeping the sign. Details: `triage_details.md`, `strong_details.md`.

**Baseline.** Cached PA is 0.321 (Llama) / 0.364 (Gemma); 59% / 54% of groups get a wrong template.

**E1, share fixed after 20 reviews.**

| ordering | Llama | Gemma |
|---|---|---|
| Oracle | 0.878 | 0.863 |
| Most frequent | 0.861 | 0.833 |
| TRIAGE without active selection (D x size) | 0.851 ± 0.013 | 0.814 |
| LLMLog confidence x size | 0.845 | 0.796 |
| TRIAGE | 0.707 ± 0.036 | 0.719 |
| Clotho alone | 0.364 ± 0.017 | 0.314 ± 0.033 |
| Random | 0.323 ± 0.052 | 0.248 ± 0.054 |

TRIAGE trails Most frequent on every seed (-0.154 / -0.115, 5/5; A12 0.33 / 0.39). Held-out systems give the same order. AdaParser's corrector raises PA_0 by only +0.022 / +0.045.

**E2, ablation.** Removing propagation costs -0.581 / -0.611, removing frequency weighting -0.387 / -0.439. Removing active selection improves the share (+0.144 / +0.095), as does removing the prior (+0.129 / +0.059); all 5/5.

**E3, E4.** The prior alone loses to frequency-first (N = 1: -0.271 / -0.241). Prior x size wins only the first review. On templates first seen in the stream's second half there is no consistent advantage.

**Cost per new template.** TRIAGE CPU 85 / 2 ms; hidden state 42 / 59 ms (free from the parser's prefill); AdaParser corrector 74 / 56 ms.

**F, a stronger parser.** Same model with up to 3 kNN demonstrations from its own cache, AdaParser's corrector and 10 verified seed templates (C2; chosen on HDFS, Spark, Linux over cache-only demonstrations). PA_0 rises to 0.639 / 0.630 and wrong groups fall to 42% / 42%. The 0.75 target is missed: Spark's two largest groups need `<*> KB` written as one `<*>`, which no cached demonstration teaches. At 20 reviews D x size still trails Most frequent (-0.020 / -0.007, 5/5; A12 0.49 / 0.49, negligible); SK-ESD puts both in the top group with LLMLog and output entropy. F3: the per-system gap does not grow as the parser improves (Spearman between wrong-group share and gap -0.27, p 0.18 / +0.17, p 0.42; figure `strong_gap.pdf`). On the most accurate system, OpenStack (4% wrong groups for Llama), D x size falls 0.24 behind because its few wrong groups are small.

**E5, leakage beyond logs (Llama, 8 Clotho tasks).** See `e5_leakage.md`: Tok-Prob and Tok-Ent use one sample and leak little (-0.024 / -0.021 Spearman); output entropy over all samples drops -0.151; the pre-generation control drops -0.011.

**Answer.** Group size and propagation decide which templates are worth reviewing, for weak and strong parsers alike. A zero-label prior x size is a sound default, level with frequency-first; TRIAGE's active selection, as specified, loses 0.10-0.15 of the fixable lines. Risk prediction did not overtake frequency as the parser became more accurate.
