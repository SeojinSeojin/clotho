# U: unseen templates at zero extra LLM calls

Details: `unseen_details.md`. Template-disjoint split, 100 references, AUC, mean over 5 seeds of the macro mean (Llama 13 systems, Gemma 12); gains paired per seed (k/5 keeping the sign). Tuned on HDFS, Spark, Linux.

| signal set | Llama | minus Clotho | Gemma | minus Clotho |
|---|---|---|---|---|
| Clotho | 0.645 | | 0.659 | |
| MDSA | 0.656 | +0.011 (3/5) | 0.655 | -0.004 (3/5) |
| S1 states | 0.749 | +0.104 (5/5) | 0.762 | +0.102 (5/5) |
| S2 line | 0.710 | +0.065 (5/5) | 0.698 | +0.039 (5/5) |
| S3 answer checks | 0.669 | +0.024 (4/5) | 0.709 | +0.050 (5/5) |
| S1+S2 | 0.745 | +0.100 (5/5) | 0.748 | +0.089 (5/5) |
| S1+S3 | **0.776** | +0.132 ± 0.046 (5/5) | 0.773 | +0.113 (5/5) |
| S2+S3 | 0.745 | +0.100 (5/5) | 0.755 | +0.096 (5/5) |
| S1+S2+S3 | 0.773 | +0.128 (5/5) | **0.773** | +0.114 ± 0.039 (5/5) |
| Output entropy* | 0.796 | +0.151 (5/5) | 0.824 | +0.164 (5/5) |

Bold = U's best. * 5 extra calls; every U row needs 0.

Gap to output entropy: Llama -0.019 ± 0.023 (3/5), same Scott-Knott group (A12 0.41, small); Gemma -0.050 ± 0.012 (5/5), one group lower. Held-out systems: +0.129 / +0.101 over Clotho. Standard split: +0.033 / +0.031 at 100 references, +0.001 / +0.015 at 200. Natural suites: ties Clotho; Lookup (Drain) is level (Llama) or better (Gemma, 0.916 vs 0.874).

Streams (U + V5, best on the tuning systems; trained on 100 labelled suite inputs of the same system, which D never sees):

- Review, risk x size minus Most frequent at 5 / 10 reviews: Llama strong +0.044 / +0.022, Gemma base +0.027 / +0.019, Gemma strong +0.015 / +0.033 (5/5 each); level at 20; Llama base -0.051 / -0.013.
- Creation-time gate minus random: precision +0.14 to +0.30 everywhere. Share fixed is higher for Gemma at every q (+0.015 to +0.062) and Llama strong at q = 20 (+0.044, 5/5); lower for Llama base.

GPU estimate: token-level states (mean / max pooling) need one forward pass per input, about 26,000 inputs per model at 0.02-0.04 s, so 10-20 min per model plus 6 min loading.
