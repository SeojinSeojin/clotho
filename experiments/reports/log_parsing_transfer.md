# Failure prediction on a new log system

`log_parsing_transfer.py`; details in `log_parsing_transfer_details.md`. Leave-one-system-out, 5 seeds, tuned on HDFS, Spark, Linux only. ROC-AUC on held-out targets (Llama 10, Gemma 9).

| method | Llama capped | Llama natural | Gemma capped | Gemma natural |
|---|---|---|---|---|
| 1. Pooled Clotho, fair layer | 0.588 | 0.499 | 0.619 | 0.664 |
| 1. Pooled Clotho, tuned layer | 0.632 | 0.573 | 0.572 | 0.444 |
| 2. + z-score | 0.591 | 0.449 | 0.567 | 0.429 |
| 2. + z-score + CORAL | 0.593 | 0.534 | 0.550 | 0.459 |
| 3. Logistic probe | 0.714 | 0.686 | 0.672 | 0.691 |
| 4. MLP probe | 0.697 | 0.640 | 0.668 | 0.659 |
| 5. Log features | 0.652 | 0.680 | 0.668 | 0.622 |
| 5. Probe + features | 0.711 | 0.667 | 0.661 | 0.727 |
| 6. Refit, k = 10, 25, 50 | 0.734, 0.775, 0.797 | 0.823, 0.846, 0.886 | 0.709, 0.748, 0.784 | 0.854, 0.898, 0.927 |
| 6. Rank average, k = 10, 25, 50 | 0.695, 0.720, 0.742 | 0.726, 0.747, 0.782 | 0.698, 0.725, 0.749 | 0.793, 0.805, 0.832 |
| Within Clotho, k = 10, 25, 50 | 0.602, 0.666, 0.706 | 0.700, 0.781, 0.836 | 0.660, 0.700, 0.729 | 0.717, 0.798, 0.842 |
| Within Clotho @100 | 0.737 | 0.849 | 0.785 | 0.889 |

Refit: best source model (Llama MLP, Gemma probe + features) retrained with the k target labels upweighted. Rank average: with within-system Clotho.

**(1) Which step closes most of the gap?** The logistic probe (step 3): it closes 78%, 41%, 32% and 12% of the gap between the best pooled Clotho and within Clotho @100. Normalisation, CORAL and the MLP add nothing.

**(2) How many target labels match within-system Clotho @100?** With refit, about 25 (Llama capped, Gemma natural) or 50 (Llama natural; Gemma capped ties). Within Clotho with the same k is lower at every k. Not paired: test sets differ.

**(3) Do generic log features transfer better than hidden states?** No, but nearly as well (0.622 to 0.680): they tie the probe on Llama natural and Gemma capped and beat pooled Clotho on three of four suites. Adding them to the probe helps only on Gemma natural.
