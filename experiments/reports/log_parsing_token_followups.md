# G5: input-token pooling and decision-point confidence

Status: partial; only complete five-seed systems are shown.

Fixed ablations; no retuning, winner selection or new generation calls. Mean ± sample std of five seed macros over matched complete systems. Held-out excludes HDFS/Spark/Linux. Reviews retain nominal reference100 (90 real target-suite labels, plus online verdicts) and the original stream layer. Nominal suite references100/200 likewise train U on 90/190 real inputs after dropping ten synthetic initial tests. V2 uses only sample0/initial greedy output; ambiguous stored-token alignments are missing, not guessed. Suite alignment reproduces teacher-forcing tokenization; stream checks use the declared whitespace/stop candidate family because original token IDs are unavailable. Full per-system predictions and review sequences are in `results_phase3{,_gemma}/token_followups/`.

## llama: suites 13/13, review systems 13/13

| System | Aligned suite share | Aligned base share | Aligned strong share |
|---|---|---|---|
| Apache | 1.000 | 0.034 | 0.034 |
| BGL | 0.992 | 0.098 | 0.039 |
| HDFS | 1.000 | 0.000 | 0.000 |
| HPC | 1.000 | 0.015 | 0.000 |
| Hadoop | 0.998 | 0.061 | 0.054 |
| HealthApp | 1.000 | 0.006 | 0.006 |
| Linux | 0.994 | 0.116 | 0.140 |
| Mac | 0.992 | 0.062 | 0.023 |
| OpenSSH | 1.000 | 0.000 | 0.000 |
| OpenStack | 1.000 | 0.000 | 0.000 |
| Spark | 0.998 | 0.194 | 0.144 |
| Thunderbird | 0.998 | 0.176 | 0.122 |
| Zookeeper | 0.996 | 0.182 | 0.143 |

### Disjoint AUC: all

| Reference | Systems | Variant | AUC | Baseline | Δ AUC | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|
| 100 | 13 | S1 | 0.749 ± 0.032 | S1 | 0.000 ± 0.000 | +0/−0/5 | 2 | 0.500 |
| 100 | 13 | S1 mean | 0.707 ± 0.034 | S1 | -0.042 ± 0.010 | +0/−5/5 | 3 | 0.347 |
| 100 | 13 | S1 max | 0.682 ± 0.027 | S1 | -0.067 ± 0.027 | +0/−5/5 | 3 | 0.298 |
| 100 | 13 | U | 0.776 ± 0.028 | U | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 100 | 13 | U mean | 0.745 ± 0.033 | U | -0.032 ± 0.013 | +0/−5/5 | 2 | 0.370 |
| 100 | 13 | U max | 0.715 ± 0.027 | U | -0.062 ± 0.025 | +0/−5/5 | 3 | 0.307 |
| 100 | 13 | U+V2 | 0.777 ± 0.029 | U | 0.001 ± 0.002 | +2/−3/5 | 1 | 0.516 |
| 100 | 13 | U+V5 | 0.783 ± 0.027 | U+V5 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 100 | 13 | U+V5+V2 | 0.782 ± 0.027 | U+V5 | -0.001 ± 0.002 | +1/−4/5 | 1 | 0.502 |
| 200 | 13 | S1 | 0.755 ± 0.028 | S1 | 0.000 ± 0.000 | +0/−0/5 | 2 | 0.500 |
| 200 | 13 | S1 mean | 0.718 ± 0.034 | S1 | -0.037 ± 0.014 | +0/−5/5 | 3 | 0.355 |
| 200 | 13 | S1 max | 0.693 ± 0.032 | S1 | -0.063 ± 0.022 | +0/−5/5 | 3 | 0.302 |
| 200 | 13 | U | 0.779 ± 0.024 | U | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 200 | 13 | U mean | 0.751 ± 0.032 | U | -0.028 ± 0.019 | +1/−4/5 | 2 | 0.385 |
| 200 | 13 | U max | 0.727 ± 0.026 | U | -0.052 ± 0.017 | +0/−5/5 | 3 | 0.312 |
| 200 | 13 | U+V2 | 0.781 ± 0.025 | U | 0.001 ± 0.001 | +5/−0/5 | 1 | 0.517 |
| 200 | 13 | U+V5 | 0.786 ± 0.023 | U+V5 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 200 | 13 | U+V5+V2 | 0.786 ± 0.024 | U+V5 | -0.001 ± 0.001 | +2/−3/5 | 1 | 0.502 |

### Review share of wrong lines fixed: all

base: 11 stored baseline systems reproduced to 1e-12; newly reconstructed: HDFS, OpenSSH.

strong: 13 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

| Parser | Reviews | Systems | Variant | Share fixed | Δ baseline | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|
| base | 5 | 13 | baseline | 0.706 ± 0.022 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 5 | 13 | mean | 0.716 ± 0.016 | 0.009 ± 0.007 | +4/−1/5 | 1 | 0.515 |
| base | 5 | 13 | max | 0.714 ± 0.010 | 0.007 ± 0.028 | +4/−1/5 | 1 | 0.503 |
| base | 5 | 13 | V2 | 0.732 ± 0.007 | 0.026 ± 0.020 | +4/−1/5 | 1 | 0.521 |
| base | 10 | 13 | baseline | 0.799 ± 0.004 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 10 | 13 | mean | 0.796 ± 0.002 | -0.003 ± 0.004 | +1/−4/5 | 1 | 0.493 |
| base | 10 | 13 | max | 0.805 ± 0.014 | 0.005 ± 0.017 | +2/−3/5 | 1 | 0.478 |
| base | 10 | 13 | V2 | 0.801 ± 0.007 | 0.002 ± 0.006 | +3/−2/5 | 1 | 0.502 |
| base | 20 | 13 | baseline | 0.837 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 20 | 13 | mean | 0.843 ± 0.012 | 0.006 ± 0.012 | +2/−3/5 | 1 | 0.497 |
| base | 20 | 13 | max | 0.853 ± 0.011 | 0.016 ± 0.011 | +4/−1/5 | 1 | 0.505 |
| base | 20 | 13 | V2 | 0.848 ± 0.013 | 0.011 ± 0.014 | +5/−0/5 | 1 | 0.502 |
| base | 50 | 13 | baseline | 0.904 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 50 | 13 | mean | 0.907 ± 0.001 | 0.003 ± 0.001 | +5/−0/5 | 1 | 0.496 |
| base | 50 | 13 | max | 0.906 ± 0.001 | 0.002 ± 0.001 | +5/−0/5 | 1 | 0.482 |
| base | 50 | 13 | V2 | 0.904 ± 0.001 | 0.000 ± 0.001 | +4/−1/5 | 1 | 0.495 |
| strong | 5 | 13 | baseline | 0.641 ± 0.003 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 5 | 13 | mean | 0.647 ± 0.008 | 0.006 ± 0.010 | +4/−1/5 | 1 | 0.500 |
| strong | 5 | 13 | max | 0.644 ± 0.006 | 0.003 ± 0.007 | +2/−3/5 | 1 | 0.499 |
| strong | 5 | 13 | V2 | 0.524 ± 0.033 | -0.117 ± 0.035 | +0/−5/5 | 2 | 0.413 |
| strong | 10 | 13 | baseline | 0.742 ± 0.005 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 10 | 13 | mean | 0.733 ± 0.003 | -0.008 ± 0.004 | +0/−5/5 | 1 | 0.477 |
| strong | 10 | 13 | max | 0.723 ± 0.005 | -0.019 ± 0.009 | +0/−5/5 | 1 | 0.466 |
| strong | 10 | 13 | V2 | 0.696 ± 0.005 | -0.045 ± 0.007 | +0/−5/5 | 1 | 0.458 |
| strong | 20 | 13 | baseline | 0.794 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 20 | 13 | mean | 0.790 ± 0.003 | -0.004 ± 0.002 | +0/−5/5 | 1 | 0.487 |
| strong | 20 | 13 | max | 0.790 ± 0.003 | -0.004 ± 0.004 | +0/−5/5 | 1 | 0.492 |
| strong | 20 | 13 | V2 | 0.789 ± 0.002 | -0.005 ± 0.002 | +0/−5/5 | 1 | 0.487 |
| strong | 50 | 13 | baseline | 0.841 ± 0.000 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 50 | 13 | mean | 0.843 ± 0.001 | 0.002 ± 0.001 | +5/−0/5 | 1 | 0.499 |
| strong | 50 | 13 | max | 0.843 ± 0.001 | 0.002 ± 0.001 | +5/−0/5 | 1 | 0.500 |
| strong | 50 | 13 | V2 | 0.836 ± 0.001 | -0.005 ± 0.001 | +0/−5/5 | 1 | 0.489 |

### Disjoint AUC: held-out

| Reference | Systems | Variant | AUC | Baseline | Δ AUC | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|
| 100 | 10 | S1 | 0.733 ± 0.038 | S1 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 100 | 10 | S1 mean | 0.695 ± 0.043 | S1 | -0.038 ± 0.011 | +0/−5/5 | 2 | 0.370 |
| 100 | 10 | S1 max | 0.667 ± 0.039 | S1 | -0.066 ± 0.041 | +0/−5/5 | 2 | 0.311 |
| 100 | 10 | U | 0.764 ± 0.036 | U | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 100 | 10 | U mean | 0.736 ± 0.042 | U | -0.027 ± 0.015 | +0/−5/5 | 1 | 0.397 |
| 100 | 10 | U max | 0.701 ± 0.035 | U | -0.063 ± 0.038 | +0/−5/5 | 2 | 0.316 |
| 100 | 10 | U+V2 | 0.764 ± 0.037 | U | 0.001 ± 0.002 | +3/−2/5 | 1 | 0.516 |
| 100 | 10 | U+V5 | 0.770 ± 0.035 | U+V5 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 100 | 10 | U+V5+V2 | 0.769 ± 0.036 | U+V5 | -0.001 ± 0.001 | +1/−4/5 | 1 | 0.503 |
| 200 | 10 | S1 | 0.738 ± 0.035 | S1 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 200 | 10 | S1 mean | 0.708 ± 0.038 | S1 | -0.030 ± 0.013 | +0/−5/5 | 2 | 0.394 |
| 200 | 10 | S1 max | 0.677 ± 0.037 | S1 | -0.061 ± 0.027 | +0/−5/5 | 2 | 0.328 |
| 200 | 10 | U | 0.766 ± 0.032 | U | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 200 | 10 | U mean | 0.743 ± 0.036 | U | -0.022 ± 0.018 | +1/−4/5 | 1 | 0.422 |
| 200 | 10 | U max | 0.717 ± 0.029 | U | -0.049 ± 0.022 | +0/−5/5 | 2 | 0.347 |
| 200 | 10 | U+V2 | 0.767 ± 0.033 | U | 0.001 ± 0.002 | +3/−2/5 | 1 | 0.515 |
| 200 | 10 | U+V5 | 0.772 ± 0.031 | U+V5 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| 200 | 10 | U+V5+V2 | 0.771 ± 0.033 | U+V5 | -0.001 ± 0.002 | +1/−4/5 | 1 | 0.498 |

### Review share of wrong lines fixed: held-out

base: 9 stored baseline systems reproduced to 1e-12; newly reconstructed: OpenSSH.

strong: 10 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

| Parser | Reviews | Systems | Variant | Share fixed | Δ baseline | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|
| base | 5 | 10 | baseline | 0.653 ± 0.028 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 5 | 10 | mean | 0.665 ± 0.021 | 0.012 ± 0.009 | +4/−1/5 | 1 | 0.520 |
| base | 5 | 10 | max | 0.663 ± 0.013 | 0.010 ± 0.036 | +4/−1/5 | 1 | 0.503 |
| base | 5 | 10 | V2 | 0.687 ± 0.009 | 0.034 ± 0.026 | +4/−1/5 | 1 | 0.527 |
| base | 10 | 10 | baseline | 0.760 ± 0.006 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 10 | 10 | mean | 0.758 ± 0.002 | -0.002 ± 0.005 | +1/−4/5 | 1 | 0.490 |
| base | 10 | 10 | max | 0.767 ± 0.017 | 0.007 ± 0.022 | +2/−3/5 | 1 | 0.475 |
| base | 10 | 10 | V2 | 0.762 ± 0.009 | 0.002 ± 0.008 | +3/−2/5 | 1 | 0.501 |
| base | 20 | 10 | baseline | 0.805 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 20 | 10 | mean | 0.813 ± 0.015 | 0.009 ± 0.015 | +2/−3/5 | 1 | 0.501 |
| base | 20 | 10 | max | 0.825 ± 0.014 | 0.021 ± 0.014 | +4/−1/5 | 1 | 0.506 |
| base | 20 | 10 | V2 | 0.819 ± 0.017 | 0.014 ± 0.018 | +5/−0/5 | 1 | 0.508 |
| base | 50 | 10 | baseline | 0.886 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| base | 50 | 10 | mean | 0.890 ± 0.001 | 0.004 ± 0.001 | +5/−0/5 | 1 | 0.498 |
| base | 50 | 10 | max | 0.889 ± 0.001 | 0.003 ± 0.001 | +5/−0/5 | 1 | 0.488 |
| base | 50 | 10 | V2 | 0.886 ± 0.001 | 0.000 ± 0.001 | +4/−1/5 | 1 | 0.495 |
| strong | 5 | 10 | baseline | 0.564 ± 0.004 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 5 | 10 | mean | 0.572 ± 0.010 | 0.008 ± 0.012 | +4/−1/5 | 1 | 0.505 |
| strong | 5 | 10 | max | 0.566 ± 0.009 | 0.002 ± 0.009 | +2/−3/5 | 1 | 0.495 |
| strong | 5 | 10 | V2 | 0.513 ± 0.023 | -0.051 ± 0.026 | +0/−5/5 | 1 | 0.462 |
| strong | 10 | 10 | baseline | 0.684 ± 0.007 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 10 | 10 | mean | 0.675 ± 0.003 | -0.009 ± 0.005 | +0/−5/5 | 1 | 0.486 |
| strong | 10 | 10 | max | 0.660 ± 0.006 | -0.024 ± 0.011 | +0/−5/5 | 1 | 0.461 |
| strong | 10 | 10 | V2 | 0.631 ± 0.007 | -0.052 ± 0.008 | +0/−5/5 | 1 | 0.461 |
| strong | 20 | 10 | baseline | 0.747 ± 0.001 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 20 | 10 | mean | 0.743 ± 0.004 | -0.005 ± 0.003 | +0/−5/5 | 1 | 0.479 |
| strong | 20 | 10 | max | 0.743 ± 0.003 | -0.005 ± 0.004 | +0/−5/5 | 1 | 0.484 |
| strong | 20 | 10 | V2 | 0.742 ± 0.003 | -0.006 ± 0.003 | +0/−5/5 | 1 | 0.483 |
| strong | 50 | 10 | baseline | 0.805 ± 0.000 | 0.000 ± 0.000 | +0/−0/5 | 1 | 0.500 |
| strong | 50 | 10 | mean | 0.807 ± 0.001 | 0.003 ± 0.001 | +5/−0/5 | 1 | 0.497 |
| strong | 50 | 10 | max | 0.807 ± 0.002 | 0.003 ± 0.002 | +5/−0/5 | 1 | 0.504 |
| strong | 50 | 10 | V2 | 0.798 ± 0.001 | -0.006 ± 0.001 | +0/−5/5 | 1 | 0.481 |

## gemma: suites 0/12, review systems 0/12

| System | Aligned suite share | Aligned base share | Aligned strong share |
|---|---|---|---|
| BGL | 1.000 | 0.961 | 0.784 |
| HDFS | 1.000 | 1.000 | 0.286 |
| HPC | 1.000 | 1.000 | 0.853 |
| Hadoop | 1.000 | 0.981 | 0.950 |
| HealthApp | 1.000 | 1.000 | 0.999 |
| Linux | 1.000 | 0.986 | 0.965 |
| Mac | 1.000 | 0.986 | 0.989 |
| OpenSSH | 1.000 | 1.000 | 0.615 |
| OpenStack | 1.000 | 1.000 | 0.996 |
| Spark | 1.000 | 0.978 | 0.914 |
| Thunderbird | 1.000 | 0.944 | 0.924 |
| Zookeeper | 1.000 | 1.000 | 0.870 |

### Disjoint AUC: all

| Reference | Systems | Variant | AUC | Baseline | Δ AUC | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|

### Review share of wrong lines fixed: all

base: 0 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

strong: 0 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

| Parser | Reviews | Systems | Variant | Share fixed | Δ baseline | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|

### Disjoint AUC: held-out

| Reference | Systems | Variant | AUC | Baseline | Δ AUC | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|

### Review share of wrong lines fixed: held-out

base: 0 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

strong: 0 stored baseline systems reproduced to 1e-12; newly reconstructed: none.

| Parser | Reviews | Systems | Variant | Share fixed | Δ baseline | Seed signs | SK-ESD | A12 |
|---|---|---|---|---|---|---|---|---|
