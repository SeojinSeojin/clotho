# CPU round 3: reviewer robustness checks

CPU only: stored outputs, hidden states and stream caches; no new LLM calls or model/tokenizer downloads. Protocols and deviations are in PROGRESS Decisions. Seeds 0–4; mean ± sample std of seed macro means. Held-out excludes HDFS, Spark and Linux. Llama base uses 11 systems (HDFS/OpenSSH caches missing); strong Llama 13, Gemma 12.

## R1. Does scoring convention explain the conclusions?

**The 20-review frequency claim changes under lenient scoring:** D x size beats Most frequent on all five seeds for gemma/base/L1, gemma/base/L2.

“Close to Oracle” is quantified below, rather than assumed. L1/L2 are sensitivity analyses, not a new ground truth.

| model/parser | scorer | systems | PA_0 | Most frequent / Oracle @20 | D minus frequency @20 (positive seeds) |
|---|---|---:|---:|---:|---|
| llama/base | strict | 11 | 0.255 | 0.977 | -0.011 ± 0.016 (2/5) |
| llama/base | L1 | 11 | 0.362 | 0.962 | -0.026 ± 0.035 (2/5) |
| llama/base | L2 | 11 | 0.363 | 0.962 | -0.013 ± 0.028 (1/5) |
| llama/strong | strict | 13 | 0.639 | 0.964 | -0.020 ± 0.004 (0/5) |
| llama/strong | L1 | 13 | 0.696 | 0.954 | -0.022 ± 0.004 (0/5) |
| llama/strong | L2 | 13 | 0.696 | 0.954 | -0.024 ± 0.004 (0/5) |
| gemma/base | strict | 12 | 0.364 | 0.966 | -0.019 ± 0.000 (0/5) |
| gemma/base | L1 | 12 | 0.424 | 0.960 | 0.008 ± 0.000 (5/5) |
| gemma/base | L2 | 12 | 0.438 | 0.950 | 0.005 ± 0.000 (5/5) |
| gemma/strong | strict | 12 | 0.630 | 0.960 | -0.007 ± 0.000 (0/5) |
| gemma/strong | L1 | 12 | 0.672 | 0.954 | -0.003 ± 0.000 (0/5) |
| gemma/strong | L2 | 12 | 0.673 | 0.953 | -0.002 ± 0.000 (0/5) |

Suite PASS rates increase under L1/L2. The disjoint U-best advantage over Clotho remains; full AUC, variance-share, N1 and N5 results are in the details.

## R2. Template metrics and the tail

**At 20 reviews, gains over frequency on every seed occur in:** llama/base/full/groups: urisk 0.089 ± 0.003; llama/base/full/FTA: llmlog_size 0.025 ± 0.000; llama/base/tail/groups: urisk 0.119 ± 0.002; llama/base/tail/FTA: urisk 0.077 ± 0.002; llama/base/tail/lines: usize 0.057 ± 0.002; llama/strong/full/groups: urisk 0.128 ± 0.004; llama/strong/full/FTA: llmlog_size 0.016 ± 0.000; llama/strong/full/lines: usize 0.003 ± 0.001; llama/strong/tail/groups: urisk 0.050 ± 0.001; llama/strong/tail/FTA: usize 0.073 ± 0.001; llama/strong/tail/lines: usize 0.024 ± 0.001; gemma/base/full/groups: urisk 0.066 ± 0.008; gemma/base/full/FTA: usize 0.020 ± 0.001; gemma/base/full/lines: usize 0.007 ± 0.002; gemma/base/tail/groups: urisk 0.071 ± 0.013; gemma/base/tail/FTA: llmlog_size 0.025 ± 0.000; gemma/base/tail/lines: usize 0.054 ± 0.005; gemma/strong/full/groups: urisk 0.053 ± 0.006; gemma/strong/full/FTA: wfirst 0.014 ± 0.000; gemma/strong/tail/groups: drisk 0.090 ± 0.000; gemma/strong/tail/FTA: llmlog_size 0.018 ± 0.000; gemma/strong/tail/lines: usize 0.061 ± 0.006.

U in R2 uses 100 target labels; U-zero in R3 uses none. Output entropy requires extra samples and is excluded from this headline.

Only completed systems are shown; the system count below makes partial coverage explicit. R2 details also compare frequency-20 then risk with uninterrupted frequency at equal total budgets.

| model/parser | systems | scope | metric | largest observed predictor gain | minus frequency (positive seeds) |
|---|---:|---|---|---|---|
| llama/base | 11 | full | groups | urisk | 0.089 ± 0.003 (5/5) |
| llama/base | 11 | full | FTA | llmlog_size | 0.025 ± 0.000 (5/5) |
| llama/base | 11 | full | lines | dsize | -0.011 ± 0.016 (2/5) |
| llama/base | 11 | tail | groups | urisk | 0.119 ± 0.002 (5/5) |
| llama/base | 11 | tail | FTA | urisk | 0.077 ± 0.002 (5/5) |
| llama/base | 11 | tail | lines | usize | 0.057 ± 0.002 (5/5) |
| llama/strong | 13 | full | groups | urisk | 0.128 ± 0.004 (5/5) |
| llama/strong | 13 | full | FTA | llmlog_size | 0.016 ± 0.000 (5/5) |
| llama/strong | 13 | full | lines | usize | 0.003 ± 0.001 (5/5) |
| llama/strong | 10 | tail | groups | urisk | 0.050 ± 0.001 (5/5) |
| llama/strong | 10 | tail | FTA | usize | 0.073 ± 0.001 (5/5) |
| llama/strong | 10 | tail | lines | usize | 0.024 ± 0.001 (5/5) |
| gemma/base | 12 | full | groups | urisk | 0.066 ± 0.008 (5/5) |
| gemma/base | 12 | full | FTA | usize | 0.020 ± 0.001 (5/5) |
| gemma/base | 12 | full | lines | usize | 0.007 ± 0.002 (5/5) |
| gemma/base | 11 | tail | groups | urisk | 0.071 ± 0.013 (5/5) |
| gemma/base | 11 | tail | FTA | llmlog_size | 0.025 ± 0.000 (5/5) |
| gemma/base | 11 | tail | lines | usize | 0.054 ± 0.005 (5/5) |
| gemma/strong | 12 | full | groups | urisk | 0.053 ± 0.006 (5/5) |
| gemma/strong | 12 | full | FTA | wfirst | 0.014 ± 0.000 (5/5) |
| gemma/strong | 12 | full | lines | usize | 0.001 ± 0.004 (4/5) |
| gemma/strong | 10 | tail | groups | drisk | 0.090 ± 0.000 (5/5) |
| gemma/strong | 10 | tail | FTA | llmlog_size | 0.018 ± 0.000 (5/5) |
| gemma/strong | 10 | tail | lines | usize | 0.061 ± 0.006 (5/5) |

## R3. U without target-system labels

**At 20 reviews, gains over frequency on every seed occur in:** llama/base/review/groups: U-zero risk 0.108 ± 0.000; llama/base/review/FTA: U+V5-zero x size 0.018 ± 0.000; llama/base/review/lines: U+V5-zero x size 0.001 ± 0.000; llama/strong/review/groups: U+V5-zero risk 0.134 ± 0.000; llama/strong/review/FTA: U-zero x size 0.013 ± 0.000; llama/strong/review/lines: U+V5-zero x size 0.003 ± 0.000; gemma/base/review/groups: U+V5-zero risk 0.089 ± 0.011; gemma/base/review/FTA: U-zero x size 0.016 ± 0.003; gemma/strong/review/groups: U-zero risk 0.067 ± 0.006; gemma/strong/review/FTA: U+V5-zero x size 0.007 ± 0.004.

U in R2 uses 100 target labels; U-zero in R3 uses none. Output entropy requires extra samples and is excluded from this headline.

Only completed systems are shown; the system count below makes partial coverage explicit. R2 details also compare frequency-20 then risk with uninterrupted frequency at equal total budgets.

| model/parser | systems | scope | metric | largest observed predictor gain | minus frequency (positive seeds) |
|---|---:|---|---|---|---|
| llama/base | 11 | review | groups | U-zero risk | 0.108 ± 0.000 (5/5) |
| llama/base | 11 | review | FTA | U+V5-zero x size | 0.018 ± 0.000 (5/5) |
| llama/base | 11 | review | lines | U+V5-zero x size | 0.001 ± 0.000 (5/5) |
| llama/strong | 13 | review | groups | U+V5-zero risk | 0.134 ± 0.000 (5/5) |
| llama/strong | 13 | review | FTA | U-zero x size | 0.013 ± 0.000 (5/5) |
| llama/strong | 13 | review | lines | U+V5-zero x size | 0.003 ± 0.000 (5/5) |
| gemma/base | 12 | review | groups | U+V5-zero risk | 0.089 ± 0.011 (5/5) |
| gemma/base | 12 | review | FTA | U-zero x size | 0.016 ± 0.003 (5/5) |
| gemma/base | 12 | review | lines | U-zero x size | 0.000 ± 0.008 (4/5) |
| gemma/strong | 12 | review | groups | U-zero risk | 0.067 ± 0.006 (5/5) |
| gemma/strong | 12 | review | FTA | U+V5-zero x size | 0.007 ± 0.004 (5/5) |
| gemma/strong | 12 | review | lines | U+V5-zero x size | -0.003 ± 0.002 (0/5) |

## R4. Full-length concentration

Queued only if R1–R3 finish with at least 30 minutes before the CPU cutoff. No wrongness is measured by R4.

All per-system artifacts and statistics are in [cpu_round3_details.md](cpu_round3_details.md). Tail budgets are additional reviews after the 20 largest eligible groups; FTA uses only remaining lines. “Largest observed gain” describes results, not a held-out method-selection rule.
