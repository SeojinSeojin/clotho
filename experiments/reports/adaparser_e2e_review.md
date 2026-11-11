# Real AdaParser: provenance, concentration and review outcomes

The official parser preloads 32 labelled Loghub-2.0 candidates per system; provenance shares count only templates with stream lines. Strict scoring; seeds 0–4; mean ± sample std of per-seed system macro means. Template failures use the first-line verdict, and a majority correction fixes one only when it equals that first-line GT. Line and template Oracles optimize their own metric. Parser cache seed 0; seeds 0–4 apply to review policies. Candidate/LLM/mixed/unknown provenance tracks final trie ancestry. LLM-only excludes mixed groups; LLM-involved includes them. Both restrictions exclude unknowns and refit within their own queue. Budgets are capped at the cohort’s available templates; actual review counts are saved per run.

## llama: complete with protocol skips: 12/13 successful; skipped: Mac

| System | Disposition | Reason | Failed line |
|---|---|---|---|
| Mac | unsupported_context | Correction dialogue exceeds the fixed local context; original client exhausted its retries. | 394 |

| System | Output templates | Candidate insertions | PA_0 | Wrong first-line | Wrong majority | Line share candidate / LLM / mixed / unknown | Template share candidate / LLM / mixed / unknown |
|---|---|---|---|---|---|---|---|
| Apache | 29 | 32 | 0.999 | 0.069 | 0.069 | 0.998 / 0.002 / 0.000 / 0.000 | 0.828 / 0.172 / 0.000 / 0.000 |
| BGL | 66 | 32 | 0.995 | 0.273 | 0.273 | 0.131 / 0.869 / 0.000 / 0.000 | 0.212 / 0.788 / 0.000 / 0.000 |
| HDFS | 13 | 32 | 1.000 | 0.000 | 0.000 | 1.000 / 0.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 / 0.000 |
| HPC | 22 | 32 | 0.965 | 0.136 | 0.136 | 0.856 / 0.144 / 0.000 / 0.000 | 0.727 / 0.273 / 0.000 / 0.000 |
| Hadoop | 228 | 32 | 0.939 | 0.246 | 0.246 | 0.136 / 0.864 / 0.000 / 0.000 | 0.140 / 0.860 / 0.000 / 0.000 |
| HealthApp | 182 | 32 | 0.807 | 0.379 | 0.379 | 0.188 / 0.801 / 0.012 / 0.000 | 0.165 / 0.830 / 0.005 / 0.000 |
| Linux | 365 | 32 | 0.876 | 0.384 | 0.384 | 0.166 / 0.833 / 0.001 / 0.000 | 0.077 / 0.921 / 0.003 / 0.000 |
| OpenSSH | 31 | 32 | 1.000 | 0.161 | 0.161 | 0.708 / 0.003 / 0.290 / 0.000 | 0.677 / 0.290 / 0.032 / 0.000 |
| OpenStack | 45 | 32 | 0.989 | 0.022 | 0.022 | 0.823 / 0.177 / 0.000 / 0.000 | 0.711 / 0.289 / 0.000 / 0.000 |
| Spark | 141 | 32 | 0.985 | 0.206 | 0.206 | 0.012 / 0.988 / 0.000 / 0.000 | 0.149 / 0.851 / 0.000 / 0.000 |
| Thunderbird | 577 | 32 | 0.644 | 0.489 | 0.490 | 0.041 / 0.959 / 0.000 / 0.000 | 0.035 / 0.965 / 0.000 / 0.000 |
| Zookeeper | 98 | 32 | 0.955 | 0.224 | 0.224 | 0.781 / 0.219 / 0.000 / 0.000 | 0.327 / 0.673 / 0.000 / 0.000 |

| System | Gini | Top 1 / 5 / 10 / 20 line share | Top20 wrong first | Top20 wrong majority | ρ(size, first wrong) | ρ(size, majority wrong) |
|---|---|---|---|---|---|---|
| Apache | 0.815 | 0.386 / 0.843 / 0.981 / 0.997 | 0.000 | 0.000 | -0.32630621299906587 | -0.32630621299906587 |
| BGL | 0.963 | 0.756 / 0.977 / 0.990 / 0.993 | 0.250 | 0.250 | -0.21950489674524018 | -0.21950489674524018 |
| HDFS | 0.656 | 0.244 / 0.983 / 1.000 / 1.000 | 0.000 | 0.000 | None | None |
| HPC | 0.769 | 0.267 / 0.880 / 0.979 / 1.000 | 0.100 | 0.100 | -0.2924509453959643 | -0.2924509453959643 |
| Hadoop | 0.855 | 0.265 / 0.492 / 0.625 / 0.755 | 0.100 | 0.100 | -0.23134461280721527 | -0.23134461280721527 |
| HealthApp | 0.926 | 0.138 / 0.654 / 0.860 / 0.913 | 0.350 | 0.350 | -0.24589711330745997 | -0.24589711330745997 |
| Linux | 0.902 | 0.435 / 0.761 / 0.825 / 0.868 | 0.300 | 0.300 | -0.18589620211514787 | -0.18589620211514787 |
| OpenSSH | 0.863 | 0.290 / 0.915 / 0.994 / 0.999 | 0.000 | 0.000 | -0.5168933211417037 | -0.5168933211417037 |
| OpenStack | 0.631 | 0.467 / 0.622 / 0.707 / 0.813 | 0.050 | 0.050 | 0.029412171797582633 | 0.029412171797582633 |
| Spark | 0.961 | 0.493 / 0.943 / 0.956 / 0.968 | 0.200 | 0.200 | 0.08625984726840735 | 0.08625984726840735 |
| Thunderbird | 0.780 | 0.227 / 0.403 / 0.489 / 0.563 | 0.300 | 0.300 | -0.2645863415768722 | -0.26091942661191475 |
| Zookeeper | 0.889 | 0.140 / 0.695 / 0.845 / 0.981 | 0.150 | 0.150 | -0.36830216732123067 | -0.36830216732123067 |

N5 comparison uses the same 12 systems for every parser, all stream lines; strong retains its 10 verified seed templates.

| Parser | Strict PA | PTA | RTA | FTA |
|---|---|---|---|---|
| AdaParser | 0.929 | 0.763 | 0.829 | 0.793 |
| Drain | 0.347 | 0.305 | 0.364 | 0.312 |
| base | 0.327 | 0.346 | 0.350 | 0.347 |
| strong | 0.661 | 0.579 | 0.572 | 0.574 |

### all, all systems: share of initially wrong lines fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.066 ± 0.052 | 0.051 ± 0.000 | 0.115 ± 0.010 | 0.737 ± 0.000 | 0.063 ± 0.010 (+5/−0/5) | 0.685 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.550 (negligible) |
| 10 | 0.145 ± 0.069 | 0.372 ± 0.000 | 0.319 ± 0.047 | 0.825 ± 0.000 | -0.053 ± 0.047 (+1/−4/5) | 0.453 ± 0.000 | oracle:1; frequent:2; dsize:2; random:3 | 0.444 (negligible) |
| 20 | 0.310 ± 0.080 | 0.578 ± 0.000 | 0.557 ± 0.035 | 0.881 ± 0.000 | -0.021 ± 0.035 (+1/−4/5) | 0.303 ± 0.000 | oracle:1; frequent:2; dsize:2; random:3 | 0.464 (negligible) |
| 50 | 0.526 ± 0.054 | 0.840 ± 0.000 | 0.846 ± 0.010 | 0.909 ± 0.000 | 0.007 ± 0.010 (+3/−2/5) | 0.069 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.529 (negligible) |

### all, all systems: share of initially wrong templates fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.069 ± 0.035 | 0.001 ± 0.000 | 0.003 ± 0.001 | 0.445 ± 0.000 | 0.002 ± 0.001 (+5/−0/5) | 0.443 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.554 (negligible) |
| 10 | 0.164 ± 0.085 | 0.054 ± 0.000 | 0.039 ± 0.016 | 0.526 ± 0.000 | -0.016 ± 0.016 (+2/−3/5) | 0.472 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.466 (negligible) |
| 20 | 0.313 ± 0.103 | 0.220 ± 0.000 | 0.161 ± 0.019 | 0.678 ± 0.000 | -0.059 ± 0.019 (+0/−5/5) | 0.458 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.483 (negligible) |
| 50 | 0.578 ± 0.011 | 0.537 ± 0.000 | 0.581 ± 0.006 | 0.832 ± 0.000 | 0.043 ± 0.006 (+5/−0/5) | 0.295 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.540 (negligible) |

### all, held-out: share of initially wrong lines fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.076 ± 0.057 | 0.042 ± 0.000 | 0.120 ± 0.012 | 0.784 ± 0.000 | 0.078 ± 0.012 (+5/−0/5) | 0.742 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.562 (small) |
| 10 | 0.167 ± 0.082 | 0.371 ± 0.000 | 0.303 ± 0.061 | 0.855 ± 0.000 | -0.067 ± 0.061 (+0/−5/5) | 0.485 ± 0.000 | oracle:1; frequent:2; dsize:2; random:3 | 0.446 (negligible) |
| 20 | 0.364 ± 0.099 | 0.593 ± 0.000 | 0.547 ± 0.044 | 0.887 ± 0.000 | -0.046 ± 0.044 (+0/−5/5) | 0.294 ± 0.000 | oracle:1; frequent:2; dsize:2; random:3 | 0.431 (small) |
| 50 | 0.589 ± 0.063 | 0.848 ± 0.000 | 0.862 ± 0.008 | 0.907 ± 0.000 | 0.014 ± 0.008 (+5/−0/5) | 0.059 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.536 (negligible) |

### all, held-out: share of initially wrong templates fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.081 ± 0.043 | 0.001 ± 0.000 | 0.003 ± 0.001 | 0.521 ± 0.000 | 0.003 ± 0.001 (+5/−0/5) | 0.520 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.562 (small) |
| 10 | 0.193 ± 0.105 | 0.058 ± 0.000 | 0.036 ± 0.020 | 0.597 ± 0.000 | -0.022 ± 0.020 (+0/−3/5) | 0.538 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.452 (negligible) |
| 20 | 0.366 ± 0.129 | 0.248 ± 0.000 | 0.163 ± 0.025 | 0.736 ± 0.000 | -0.086 ± 0.025 (+0/−5/5) | 0.488 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.468 (negligible) |
| 50 | 0.651 ± 0.015 | 0.588 ± 0.000 | 0.633 ± 0.005 | 0.866 ± 0.000 | 0.044 ± 0.005 (+5/−0/5) | 0.278 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.536 (negligible) |

### llm_only, all systems: share of initially wrong lines fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.262 ± 0.057 | 0.457 ± 0.000 | 0.558 ± 0.009 | 0.781 ± 0.000 | 0.101 ± 0.009 (+5/−0/5) | 0.324 ± 0.000 | oracle:1; dsize:2; frequent:3; random:4 | 0.577 (small) |
| 10 | 0.359 ± 0.044 | 0.695 ± 0.000 | 0.707 ± 0.007 | 0.869 ± 0.000 | 0.012 ± 0.007 (+5/−0/5) | 0.174 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.506 (negligible) |
| 20 | 0.455 ± 0.027 | 0.793 ± 0.000 | 0.806 ± 0.002 | 0.925 ± 0.000 | 0.013 ± 0.002 (+5/−0/5) | 0.132 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.512 (negligible) |
| 50 | 0.628 ± 0.070 | 0.903 ± 0.000 | 0.910 ± 0.003 | 0.953 ± 0.000 | 0.008 ± 0.003 (+5/−0/5) | 0.050 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.537 (negligible) |

### llm_only, all systems: share of initially wrong templates fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.268 ± 0.052 | 0.187 ± 0.000 | 0.279 ± 0.001 | 0.445 ± 0.000 | 0.092 ± 0.001 (+5/−0/5) | 0.258 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.577 (small) |
| 10 | 0.374 ± 0.062 | 0.406 ± 0.000 | 0.411 ± 0.002 | 0.526 ± 0.000 | 0.004 ± 0.002 (+5/−0/5) | 0.120 ± 0.000 | oracle:1; dsize:1; frequent:1; random:1 | 0.516 (negligible) |
| 20 | 0.466 ± 0.014 | 0.459 ± 0.000 | 0.474 ± 0.002 | 0.678 ± 0.000 | 0.015 ± 0.002 (+5/−0/5) | 0.220 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.530 (negligible) |
| 50 | 0.641 ± 0.013 | 0.594 ± 0.000 | 0.632 ± 0.005 | 0.832 ± 0.000 | 0.038 ± 0.005 (+5/−0/5) | 0.238 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.552 (negligible) |

### llm_only, held-out: share of initially wrong lines fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.315 ± 0.069 | 0.516 ± 0.000 | 0.640 ± 0.011 | 0.838 ± 0.000 | 0.124 ± 0.011 (+5/−0/5) | 0.321 ± 0.000 | oracle:1; dsize:2; frequent:3; random:4 | 0.601 (small) |
| 10 | 0.429 ± 0.052 | 0.753 ± 0.000 | 0.759 ± 0.009 | 0.909 ± 0.000 | 0.006 ± 0.009 (+2/−0/5) | 0.155 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.502 (negligible) |
| 20 | 0.536 ± 0.033 | 0.838 ± 0.000 | 0.842 ± 0.004 | 0.940 ± 0.000 | 0.004 ± 0.004 (+4/−1/5) | 0.102 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.504 (negligible) |
| 50 | 0.709 ± 0.079 | 0.921 ± 0.000 | 0.932 ± 0.004 | 0.961 ± 0.000 | 0.011 ± 0.004 (+5/−0/5) | 0.040 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.543 (negligible) |

### llm_only, held-out: share of initially wrong templates fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.323 ± 0.067 | 0.227 ± 0.000 | 0.339 ± 0.001 | 0.521 ± 0.000 | 0.112 ± 0.001 (+5/−0/5) | 0.293 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.596 (small) |
| 10 | 0.447 ± 0.075 | 0.485 ± 0.000 | 0.486 ± 0.002 | 0.597 ± 0.000 | 0.001 ± 0.002 (+2/−0/5) | 0.112 ± 0.000 | oracle:1; dsize:1; frequent:1; random:1 | 0.505 (negligible) |
| 20 | 0.549 ± 0.019 | 0.529 ± 0.000 | 0.539 ± 0.004 | 0.736 ± 0.000 | 0.010 ± 0.004 (+5/−0/5) | 0.208 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.514 (negligible) |
| 50 | 0.718 ± 0.013 | 0.651 ± 0.000 | 0.688 ± 0.009 | 0.866 ± 0.000 | 0.037 ± 0.009 (+5/−0/5) | 0.215 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.544 (negligible) |

### llm_involved, all systems: share of initially wrong lines fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.233 ± 0.073 | 0.435 ± 0.000 | 0.509 ± 0.009 | 0.737 ± 0.000 | 0.074 ± 0.009 (+5/−0/5) | 0.302 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.561 (small) |
| 10 | 0.315 ± 0.044 | 0.651 ± 0.000 | 0.663 ± 0.007 | 0.825 ± 0.000 | 0.012 ± 0.007 (+5/−0/5) | 0.174 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.506 (negligible) |
| 20 | 0.411 ± 0.027 | 0.747 ± 0.000 | 0.763 ± 0.003 | 0.881 ± 0.000 | 0.015 ± 0.003 (+5/−0/5) | 0.134 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.512 (negligible) |
| 50 | 0.585 ± 0.070 | 0.858 ± 0.000 | 0.866 ± 0.003 | 0.909 ± 0.000 | 0.008 ± 0.003 (+5/−0/5) | 0.050 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.533 (negligible) |

### llm_involved, all systems: share of initially wrong templates fixed

Eligible systems: 11/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.257 ± 0.054 | 0.187 ± 0.000 | 0.260 ± 0.001 | 0.445 ± 0.000 | 0.073 ± 0.001 (+5/−0/5) | 0.258 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.518 (negligible) |
| 10 | 0.374 ± 0.062 | 0.406 ± 0.000 | 0.411 ± 0.002 | 0.526 ± 0.000 | 0.004 ± 0.002 (+5/−0/5) | 0.120 ± 0.000 | oracle:1; dsize:1; frequent:1; random:1 | 0.516 (negligible) |
| 20 | 0.466 ± 0.014 | 0.457 ± 0.000 | 0.474 ± 0.002 | 0.678 ± 0.000 | 0.017 ± 0.002 (+5/−0/5) | 0.221 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.535 (negligible) |
| 50 | 0.641 ± 0.013 | 0.594 ± 0.000 | 0.633 ± 0.007 | 0.832 ± 0.000 | 0.039 ± 0.007 (+5/−0/5) | 0.238 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.555 (negligible) |

### llm_involved, held-out: share of initially wrong lines fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.279 ± 0.088 | 0.490 ± 0.000 | 0.580 ± 0.011 | 0.784 ± 0.000 | 0.090 ± 0.011 (+5/−0/5) | 0.295 ± 0.000 | oracle:1; dsize:2; frequent:3; random:4 | 0.579 (small) |
| 10 | 0.376 ± 0.052 | 0.700 ± 0.000 | 0.706 ± 0.009 | 0.855 ± 0.000 | 0.006 ± 0.009 (+2/−0/5) | 0.155 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.502 (negligible) |
| 20 | 0.483 ± 0.033 | 0.782 ± 0.000 | 0.789 ± 0.005 | 0.887 ± 0.000 | 0.007 ± 0.005 (+4/−1/5) | 0.105 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.504 (negligible) |
| 50 | 0.656 ± 0.079 | 0.868 ± 0.000 | 0.879 ± 0.004 | 0.907 ± 0.000 | 0.012 ± 0.004 (+5/−0/5) | 0.040 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.537 (negligible) |

### llm_involved, held-out: share of initially wrong templates fixed

Eligible systems: 9/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.310 ± 0.068 | 0.227 ± 0.000 | 0.317 ± 0.001 | 0.521 ± 0.000 | 0.090 ± 0.001 (+5/−0/5) | 0.293 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.527 (negligible) |
| 10 | 0.447 ± 0.075 | 0.485 ± 0.000 | 0.486 ± 0.002 | 0.597 ± 0.000 | 0.001 ± 0.002 (+2/−0/5) | 0.112 ± 0.000 | oracle:1; dsize:1; frequent:1; random:1 | 0.505 (negligible) |
| 20 | 0.549 ± 0.019 | 0.527 ± 0.000 | 0.539 ± 0.003 | 0.736 ± 0.000 | 0.012 ± 0.003 (+5/−0/5) | 0.209 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.520 (negligible) |
| 50 | 0.718 ± 0.014 | 0.651 ± 0.000 | 0.689 ± 0.011 | 0.866 ± 0.000 | 0.038 ± 0.011 (+5/−0/5) | 0.215 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.548 (negligible) |

### Per-system D − frequency gains (mean gap, number of strictly positive seeds / 5)

| System | Subset | Metric | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|---|
| Apache | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.617 (4/5) | +0.000 (0/5) |
| Apache | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.400 (4/5) | +0.000 (0/5) |
| Apache | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| BGL | all | lines | +0.000 (0/5) | +0.000 (0/5) | -0.056 (0/5) | +0.015 (5/5) |
| BGL | all | templates | +0.000 (0/5) | +0.000 (0/5) | -0.056 (0/5) | +0.211 (5/5) |
| BGL | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | -0.013 (2/5) | +0.004 (5/5) |
| BGL | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.056 (4/5) | +0.056 (5/5) |
| BGL | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | -0.013 (2/5) | +0.004 (5/5) |
| BGL | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.056 (4/5) | +0.056 (5/5) |
| HDFS | all | lines | undefined | undefined | undefined | undefined |
| HDFS | all | templates | undefined | undefined | undefined | undefined |
| HPC | all | lines | +0.000 (0/5) | -0.599 (0/5) | -0.000 (0/5) | +0.000 (0/5) |
| HPC | all | templates | +0.000 (0/5) | -0.200 (0/5) | -0.133 (0/5) | +0.000 (0/5) |
| HPC | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Hadoop | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.090 (5/5) |
| Hadoop | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.054 (5/5) |
| Hadoop | llm_only | lines | +0.000 (0/5) | +0.032 (1/5) | +0.054 (5/5) | +0.094 (5/5) |
| Hadoop | llm_only | templates | +0.000 (0/5) | +0.004 (1/5) | +0.029 (5/5) | +0.054 (5/5) |
| Hadoop | llm_involved | lines | +0.000 (0/5) | +0.032 (1/5) | +0.054 (5/5) | +0.094 (5/5) |
| Hadoop | llm_involved | templates | +0.000 (0/5) | +0.004 (1/5) | +0.029 (5/5) | +0.054 (5/5) |
| HealthApp | all | lines | +0.700 (5/5) | +0.000 (0/5) | -0.042 (0/5) | -0.003 (3/5) |
| HealthApp | all | templates | +0.023 (5/5) | +0.000 (0/5) | -0.026 (0/5) | +0.017 (5/5) |
| HealthApp | llm_only | lines | +0.115 (3/5) | +0.000 (0/5) | -0.027 (0/5) | -0.002 (0/5) |
| HealthApp | llm_only | templates | +0.009 (3/5) | +0.000 (0/5) | -0.017 (0/5) | -0.017 (0/5) |
| HealthApp | llm_involved | lines | +0.077 (2/5) | +0.000 (0/5) | -0.003 (0/5) | -0.002 (1/5) |
| HealthApp | llm_involved | templates | +0.006 (2/5) | +0.000 (0/5) | +0.000 (0/5) | -0.006 (2/5) |
| Linux | all | lines | -0.002 (0/5) | -0.085 (0/5) | +0.003 (1/5) | +0.027 (5/5) |
| Linux | all | templates | +0.000 (0/5) | -0.006 (0/5) | +0.001 (1/5) | +0.071 (5/5) |
| Linux | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.016 (5/5) | +0.038 (5/5) |
| Linux | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.006 (4/5) | +0.072 (5/5) |
| Linux | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.018 (5/5) | +0.038 (5/5) |
| Linux | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.007 (5/5) | +0.073 (5/5) |
| OpenSSH | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.052 (1/5) | +0.000 (0/5) |
| OpenSSH | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.040 (1/5) | +0.000 (0/5) |
| OpenSSH | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_involved | lines | -0.261 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_involved | templates | -0.200 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenStack | all | lines | +0.000 (0/5) | +0.000 (0/5) | -1.000 (0/5) | +0.000 (0/5) |
| OpenStack | all | templates | +0.000 (0/5) | +0.000 (0/5) | -1.000 (0/5) | +0.000 (0/5) |
| OpenStack | llm_only | lines | +1.000 (5/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenStack | llm_only | templates | +1.000 (5/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenStack | llm_involved | lines | +1.000 (5/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenStack | llm_involved | templates | +1.000 (5/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Spark | all | lines | +0.000 (0/5) | +0.107 (5/5) | +0.182 (5/5) | -0.080 (0/5) |
| Spark | all | templates | +0.000 (0/5) | +0.034 (5/5) | +0.124 (5/5) | +0.007 (2/5) |
| Spark | llm_only | lines | +0.000 (0/5) | +0.076 (5/5) | +0.093 (5/5) | -0.057 (0/5) |
| Spark | llm_only | templates | +0.000 (0/5) | +0.034 (5/5) | +0.069 (5/5) | +0.014 (3/5) |
| Spark | llm_involved | lines | +0.000 (0/5) | +0.076 (5/5) | +0.093 (5/5) | -0.057 (0/5) |
| Spark | llm_involved | templates | +0.000 (0/5) | +0.034 (5/5) | +0.069 (5/5) | +0.014 (3/5) |
| Thunderbird | all | lines | +0.000 (0/5) | -0.008 (0/5) | +0.014 (3/5) | +0.009 (4/5) |
| Thunderbird | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.004 (3/5) | +0.009 (5/5) |
| Thunderbird | llm_only | lines | -0.003 (0/5) | +0.000 (0/5) | +0.022 (5/5) | +0.006 (5/5) |
| Thunderbird | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.006 (5/5) | +0.005 (5/5) |
| Thunderbird | llm_involved | lines | -0.003 (0/5) | +0.000 (0/5) | +0.022 (5/5) | +0.006 (5/5) |
| Thunderbird | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.006 (5/5) | +0.005 (5/5) |
| Zookeeper | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.015 (5/5) |
| Zookeeper | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.109 (5/5) |
| Zookeeper | llm_only | lines | +0.000 (0/5) | +0.023 (1/5) | +0.000 (2/5) | +0.002 (5/5) |
| Zookeeper | llm_only | templates | +0.000 (0/5) | +0.009 (1/5) | +0.018 (2/5) | +0.236 (5/5) |
| Zookeeper | llm_involved | lines | +0.000 (0/5) | +0.023 (1/5) | +0.000 (2/5) | +0.002 (5/5) |
| Zookeeper | llm_involved | templates | +0.000 (0/5) | +0.009 (1/5) | +0.018 (2/5) | +0.236 (5/5) |

Headline: with the real parser, Most frequent achieves 65.6% of Oracle’s macro gain after 20 reviews: 0.578 versus 0.881 of wrong lines fixed (gap 0.303). D × size beats frequency on every seed in: all/all systems/lines/5, all/all systems/templates/5, all/all systems/templates/50, all/held-out/lines/5, all/held-out/lines/50, all/held-out/templates/5, all/held-out/templates/50, llm_only/all systems/lines/5, llm_only/all systems/lines/10, llm_only/all systems/lines/20, llm_only/all systems/lines/50, llm_only/all systems/templates/5, llm_only/all systems/templates/10, llm_only/all systems/templates/20, llm_only/all systems/templates/50, llm_only/held-out/lines/5, llm_only/held-out/lines/50, llm_only/held-out/templates/5, llm_only/held-out/templates/20, llm_only/held-out/templates/50, llm_involved/all systems/lines/5, llm_involved/all systems/lines/10, llm_involved/all systems/lines/20, llm_involved/all systems/lines/50, llm_involved/all systems/templates/5, llm_involved/all systems/templates/10, llm_involved/all systems/templates/20, llm_involved/all systems/templates/50, llm_involved/held-out/lines/5, llm_involved/held-out/lines/50, llm_involved/held-out/templates/5, llm_involved/held-out/templates/20, llm_involved/held-out/templates/50, BGL/all/lines/50, BGL/all/templates/50, BGL/llm_only/lines/50, BGL/llm_only/templates/50, BGL/llm_involved/lines/50, BGL/llm_involved/templates/50, Hadoop/all/lines/50, Hadoop/all/templates/50, Hadoop/llm_only/lines/20, Hadoop/llm_only/lines/50, Hadoop/llm_only/templates/20, Hadoop/llm_only/templates/50, Hadoop/llm_involved/lines/20, Hadoop/llm_involved/lines/50, Hadoop/llm_involved/templates/20, Hadoop/llm_involved/templates/50, HealthApp/all/lines/5, HealthApp/all/templates/5, HealthApp/all/templates/50, Linux/all/lines/50, Linux/all/templates/50, Linux/llm_only/lines/20, Linux/llm_only/lines/50, Linux/llm_only/templates/50, Linux/llm_involved/lines/20, Linux/llm_involved/lines/50, Linux/llm_involved/templates/20, Linux/llm_involved/templates/50, OpenStack/llm_only/lines/5, OpenStack/llm_only/templates/5, OpenStack/llm_involved/lines/5, OpenStack/llm_involved/templates/5, Spark/all/lines/10, Spark/all/lines/20, Spark/all/templates/10, Spark/all/templates/20, Spark/llm_only/lines/10, Spark/llm_only/lines/20, Spark/llm_only/templates/10, Spark/llm_only/templates/20, Spark/llm_involved/lines/10, Spark/llm_involved/lines/20, Spark/llm_involved/templates/10, Spark/llm_involved/templates/20, Thunderbird/all/templates/50, Thunderbird/llm_only/lines/20, Thunderbird/llm_only/lines/50, Thunderbird/llm_only/templates/20, Thunderbird/llm_only/templates/50, Thunderbird/llm_involved/lines/20, Thunderbird/llm_involved/lines/50, Thunderbird/llm_involved/templates/20, Thunderbird/llm_involved/templates/50, Zookeeper/all/lines/50, Zookeeper/all/templates/50, Zookeeper/llm_only/lines/50, Zookeeper/llm_only/templates/50, Zookeeper/llm_involved/lines/50, Zookeeper/llm_involved/templates/50.

## gemma: complete

| System | Output templates | Candidate insertions | PA_0 | Wrong first-line | Wrong majority | Line share candidate / LLM / mixed / unknown | Template share candidate / LLM / mixed / unknown |
|---|---|---|---|---|---|---|---|
| BGL | 57 | 32 | 0.996 | 0.123 | 0.123 | 0.131 / 0.869 / 0.000 / 0.000 | 0.246 / 0.754 / 0.000 / 0.000 |
| HDFS | 13 | 32 | 1.000 | 0.000 | 0.000 | 1.000 / 0.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 / 0.000 |
| HPC | 20 | 32 | 1.000 | 0.000 | 0.000 | 0.856 / 0.144 / 0.000 / 0.000 | 0.800 / 0.200 / 0.000 / 0.000 |
| Hadoop | 224 | 32 | 0.922 | 0.196 | 0.196 | 0.140 / 0.860 / 0.000 / 0.000 | 0.143 / 0.857 / 0.000 / 0.000 |
| HealthApp | 140 | 32 | 0.798 | 0.171 | 0.171 | 0.199 / 0.801 / 0.000 / 0.000 | 0.221 / 0.779 / 0.000 / 0.000 |
| Linux | 350 | 32 | 0.380 | 0.409 | 0.409 | 0.166 / 0.833 / 0.001 / 0.000 | 0.080 / 0.917 / 0.003 / 0.000 |
| Mac | 466 | 32 | 0.707 | 0.367 | 0.367 | 0.083 / 0.909 / 0.008 / 0.000 | 0.067 / 0.931 / 0.002 / 0.000 |
| OpenSSH | 30 | 32 | 0.999 | 0.167 | 0.167 | 0.708 / 0.003 / 0.289 / 0.000 | 0.700 / 0.267 / 0.033 / 0.000 |
| OpenStack | 45 | 32 | 1.000 | 0.000 | 0.000 | 0.823 / 0.177 / 0.000 / 0.000 | 0.711 / 0.289 / 0.000 / 0.000 |
| Spark | 134 | 32 | 0.986 | 0.157 | 0.157 | 0.012 / 0.988 / 0.000 / 0.000 | 0.157 / 0.843 / 0.000 / 0.000 |
| Thunderbird | 545 | 32 | 0.646 | 0.378 | 0.376 | 0.039 / 0.961 / 0.000 / 0.000 | 0.035 / 0.965 / 0.000 / 0.000 |
| Zookeeper | 87 | 32 | 0.951 | 0.069 | 0.069 | 0.781 / 0.219 / 0.000 / 0.000 | 0.368 / 0.632 / 0.000 / 0.000 |

| System | Gini | Top 1 / 5 / 10 / 20 line share | Top20 wrong first | Top20 wrong majority | ρ(size, first wrong) | ρ(size, majority wrong) |
|---|---|---|---|---|---|---|
| BGL | 0.958 | 0.756 / 0.977 / 0.991 / 0.994 | 0.100 | 0.100 | -0.17990644910385414 | -0.17990644910385414 |
| HDFS | 0.656 | 0.244 / 0.983 / 1.000 / 1.000 | 0.000 | 0.000 | None | None |
| HPC | 0.746 | 0.267 / 0.880 / 0.979 / 1.000 | 0.000 | 0.000 | None | None |
| Hadoop | 0.853 | 0.265 / 0.492 / 0.625 / 0.755 | 0.100 | 0.100 | -0.09512662547343127 | -0.09512662547343127 |
| HealthApp | 0.907 | 0.138 / 0.654 / 0.862 / 0.915 | 0.250 | 0.250 | -0.02098975486773534 | -0.02098975486773534 |
| Linux | 0.897 | 0.435 / 0.763 / 0.814 / 0.870 | 0.550 | 0.550 | 0.022190811287220367 | 0.022190811287220367 |
| Mac | 0.779 | 0.098 / 0.201 / 0.297 / 0.419 | 0.200 | 0.200 | -0.09151035832860416 | -0.09151035832860416 |
| OpenSSH | 0.858 | 0.289 / 0.915 / 0.994 / 0.999 | 0.000 | 0.000 | -0.4606187608610215 | -0.4606187608610215 |
| OpenStack | 0.631 | 0.467 / 0.622 / 0.707 / 0.813 | 0.000 | 0.000 | None | None |
| Spark | 0.959 | 0.493 / 0.943 / 0.956 / 0.969 | 0.250 | 0.250 | -0.010960445768136758 | -0.010960445768136758 |
| Thunderbird | 0.772 | 0.227 / 0.435 / 0.512 / 0.577 | 0.350 | 0.350 | -0.01965712186205011 | -0.025666984618592802 |
| Zookeeper | 0.875 | 0.140 / 0.695 / 0.845 / 0.981 | 0.100 | 0.100 | 0.0919019615892372 | 0.0919019615892372 |

N5 comparison uses the same 12 systems for every parser, all stream lines; strong retains its 10 verified seed templates.

| Parser | Strict PA | PTA | RTA | FTA |
|---|---|---|---|---|
| AdaParser | 0.865 | 0.822 | 0.836 | 0.828 |
| Drain | 0.318 | 0.267 | 0.342 | 0.277 |
| base | 0.364 | 0.415 | 0.441 | 0.424 |
| strong | 0.630 | 0.566 | 0.572 | 0.568 |

### all, all systems: share of initially wrong lines fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.050 ± 0.035 | 0.179 ± 0.000 | 0.272 ± 0.000 | 0.715 ± 0.000 | 0.094 ± 0.000 (+5/−0/5) | 0.536 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.617 (small) |
| 10 | 0.088 ± 0.026 | 0.501 ± 0.000 | 0.558 ± 0.000 | 0.802 ± 0.000 | 0.057 ± 0.000 (+5/−0/5) | 0.301 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.543 (negligible) |
| 20 | 0.174 ± 0.028 | 0.614 ± 0.000 | 0.670 ± 0.000 | 0.852 ± 0.000 | 0.056 ± 0.000 (+5/−0/5) | 0.238 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.562 (small) |
| 50 | 0.404 ± 0.020 | 0.810 ± 0.000 | 0.821 ± 0.000 | 0.901 ± 0.000 | 0.011 ± 0.000 (+5/−0/5) | 0.090 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.506 (negligible) |

### all, all systems: share of initially wrong templates fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.060 ± 0.029 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.355 ± 0.000 | 0.007 ± 0.000 (+5/−0/5) | 0.346 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.636 (small) |
| 10 | 0.111 ± 0.018 | 0.078 ± 0.000 | 0.097 ± 0.000 | 0.477 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.400 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.519 (negligible) |
| 20 | 0.210 ± 0.008 | 0.131 ± 0.000 | 0.175 ± 0.000 | 0.622 ± 0.000 | 0.044 ± 0.000 (+5/−0/5) | 0.490 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.617 (small) |
| 50 | 0.408 ± 0.022 | 0.424 ± 0.000 | 0.413 ± 0.000 | 0.760 ± 0.000 | -0.010 ± 0.000 (+0/−5/5) | 0.336 ± 0.000 | oracle:1; frequent:2; dsize:2; random:2 | 0.512 (negligible) |

### all, held-out: share of initially wrong lines fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.056 ± 0.051 | 0.074 ± 0.000 | 0.194 ± 0.000 | 0.701 ± 0.000 | 0.120 ± 0.000 (+5/−0/5) | 0.627 ± 0.000 | oracle:1; dsize:2; frequent:3; random:3 | 0.653 (medium) |
| 10 | 0.103 ± 0.051 | 0.458 ± 0.000 | 0.534 ± 0.000 | 0.773 ± 0.000 | 0.076 ± 0.000 (+5/−0/5) | 0.315 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.561 (small) |
| 20 | 0.194 ± 0.032 | 0.562 ± 0.000 | 0.634 ± 0.000 | 0.825 ± 0.000 | 0.072 ± 0.000 (+5/−0/5) | 0.262 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.602 (small) |
| 50 | 0.449 ± 0.021 | 0.779 ± 0.000 | 0.793 ± 0.000 | 0.883 ± 0.000 | 0.014 ± 0.000 (+5/−0/5) | 0.104 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.500 (negligible) |

### all, held-out: share of initially wrong templates fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.070 ± 0.039 | 0.002 ± 0.000 | 0.011 ± 0.000 | 0.418 ± 0.000 | 0.009 ± 0.000 (+5/−0/5) | 0.416 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.684 (medium) |
| 10 | 0.129 ± 0.026 | 0.087 ± 0.000 | 0.113 ± 0.000 | 0.536 ± 0.000 | 0.025 ± 0.000 (+5/−0/5) | 0.449 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.541 (negligible) |
| 20 | 0.245 ± 0.003 | 0.131 ± 0.000 | 0.187 ± 0.000 | 0.643 ± 0.000 | 0.056 ± 0.000 (+5/−0/5) | 0.512 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.643 (medium) |
| 50 | 0.462 ± 0.020 | 0.461 ± 0.000 | 0.448 ± 0.000 | 0.791 ± 0.000 | -0.013 ± 0.000 (+0/−5/5) | 0.329 ± 0.000 | oracle:1; random:2; frequent:2; dsize:2 | 0.520 (negligible) |

### llm_only, all systems: share of initially wrong lines fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.126 ± 0.008 | 0.524 ± 0.000 | 0.589 ± 0.000 | 0.752 ± 0.000 | 0.065 ± 0.000 (+5/−0/5) | 0.228 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.556 (negligible) |
| 10 | 0.194 ± 0.017 | 0.682 ± 0.000 | 0.701 ± 0.000 | 0.840 ± 0.000 | 0.020 ± 0.000 (+5/−0/5) | 0.158 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.512 (negligible) |
| 20 | 0.292 ± 0.080 | 0.752 ± 0.000 | 0.779 ± 0.000 | 0.890 ± 0.000 | 0.027 ± 0.000 (+5/−0/5) | 0.138 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.519 (negligible) |
| 50 | 0.508 ± 0.029 | 0.860 ± 0.000 | 0.877 ± 0.000 | 0.938 ± 0.000 | 0.017 ± 0.000 (+5/−0/5) | 0.078 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.549 (negligible) |

### llm_only, all systems: share of initially wrong templates fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.126 ± 0.009 | 0.127 ± 0.000 | 0.135 ± 0.000 | 0.355 ± 0.000 | 0.007 ± 0.000 (+5/−0/5) | 0.228 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.549 (negligible) |
| 10 | 0.203 ± 0.005 | 0.220 ± 0.000 | 0.240 ± 0.000 | 0.478 ± 0.000 | 0.020 ± 0.000 (+5/−0/5) | 0.258 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.519 (negligible) |
| 20 | 0.280 ± 0.009 | 0.296 ± 0.000 | 0.300 ± 0.000 | 0.622 ± 0.000 | 0.004 ± 0.000 (+5/−0/5) | 0.326 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.525 (negligible) |
| 50 | 0.478 ± 0.010 | 0.489 ± 0.000 | 0.509 ± 0.000 | 0.760 ± 0.000 | 0.021 ± 0.000 (+5/−0/5) | 0.272 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.549 (negligible) |

### llm_only, held-out: share of initially wrong lines fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.154 ± 0.018 | 0.515 ± 0.000 | 0.576 ± 0.000 | 0.748 ± 0.000 | 0.062 ± 0.000 (+5/−0/5) | 0.233 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.551 (negligible) |
| 10 | 0.240 ± 0.012 | 0.670 ± 0.000 | 0.688 ± 0.000 | 0.821 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.151 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.520 (negligible) |
| 20 | 0.335 ± 0.080 | 0.738 ± 0.000 | 0.766 ± 0.000 | 0.873 ± 0.000 | 0.028 ± 0.000 (+5/−0/5) | 0.135 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.520 (negligible) |
| 50 | 0.578 ± 0.033 | 0.842 ± 0.000 | 0.862 ± 0.000 | 0.930 ± 0.000 | 0.020 ± 0.000 (+5/−0/5) | 0.088 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.561 (small) |

### llm_only, held-out: share of initially wrong templates fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.155 ± 0.012 | 0.153 ± 0.000 | 0.162 ± 0.000 | 0.418 ± 0.000 | 0.009 ± 0.000 (+5/−0/5) | 0.265 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.561 (small) |
| 10 | 0.248 ± 0.006 | 0.261 ± 0.000 | 0.282 ± 0.000 | 0.536 ± 0.000 | 0.021 ± 0.000 (+5/−0/5) | 0.275 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.531 (negligible) |
| 20 | 0.330 ± 0.004 | 0.341 ± 0.000 | 0.339 ± 0.000 | 0.643 ± 0.000 | -0.002 ± 0.000 (+0/−5/5) | 0.302 ± 0.000 | oracle:1; frequent:2; dsize:2; random:2 | 0.531 (negligible) |
| 50 | 0.544 ± 0.010 | 0.542 ± 0.000 | 0.561 ± 0.000 | 0.791 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.249 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.541 (negligible) |

### llm_involved, all systems: share of initially wrong lines fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.098 ± 0.008 | 0.487 ± 0.000 | 0.552 ± 0.000 | 0.715 ± 0.000 | 0.065 ± 0.000 (+5/−0/5) | 0.229 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.568 (small) |
| 10 | 0.158 ± 0.017 | 0.645 ± 0.000 | 0.665 ± 0.000 | 0.802 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.157 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.512 (negligible) |
| 20 | 0.257 ± 0.080 | 0.715 ± 0.000 | 0.742 ± 0.000 | 0.852 ± 0.000 | 0.027 ± 0.000 (+5/−0/5) | 0.136 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.519 (negligible) |
| 50 | 0.472 ± 0.029 | 0.824 ± 0.000 | 0.841 ± 0.000 | 0.901 ± 0.000 | 0.017 ± 0.000 (+5/−0/5) | 0.076 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.549 (negligible) |

### llm_involved, all systems: share of initially wrong templates fixed

Eligible systems: 9/12; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.112 ± 0.012 | 0.105 ± 0.000 | 0.112 ± 0.000 | 0.355 ± 0.000 | 0.007 ± 0.000 (+5/−0/5) | 0.250 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.549 (negligible) |
| 10 | 0.203 ± 0.005 | 0.219 ± 0.000 | 0.240 ± 0.000 | 0.477 ± 0.000 | 0.020 ± 0.000 (+5/−0/5) | 0.258 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.519 (negligible) |
| 20 | 0.279 ± 0.009 | 0.296 ± 0.000 | 0.300 ± 0.000 | 0.622 ± 0.000 | 0.004 ± 0.000 (+5/−0/5) | 0.326 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.525 (negligible) |
| 50 | 0.478 ± 0.010 | 0.489 ± 0.000 | 0.510 ± 0.000 | 0.760 ± 0.000 | 0.021 ± 0.000 (+5/−0/5) | 0.271 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.549 (negligible) |

### llm_involved, held-out: share of initially wrong lines fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.117 ± 0.027 | 0.467 ± 0.000 | 0.529 ± 0.000 | 0.701 ± 0.000 | 0.062 ± 0.000 (+5/−0/5) | 0.234 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.551 (negligible) |
| 10 | 0.194 ± 0.011 | 0.623 ± 0.000 | 0.641 ± 0.000 | 0.773 ± 0.000 | 0.018 ± 0.000 (+5/−0/5) | 0.150 ± 0.000 | oracle:1; dsize:2; frequent:2; random:3 | 0.520 (negligible) |
| 20 | 0.289 ± 0.080 | 0.691 ± 0.000 | 0.719 ± 0.000 | 0.825 ± 0.000 | 0.028 ± 0.000 (+5/−0/5) | 0.134 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.520 (negligible) |
| 50 | 0.532 ± 0.033 | 0.797 ± 0.000 | 0.816 ± 0.000 | 0.883 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.086 ± 0.000 | oracle:1; dsize:1; frequent:1; random:2 | 0.561 (small) |

### llm_involved, held-out: share of initially wrong templates fixed

Eligible systems: 7/9; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.138 ± 0.011 | 0.124 ± 0.000 | 0.134 ± 0.000 | 0.418 ± 0.000 | 0.009 ± 0.000 (+5/−0/5) | 0.293 ± 0.000 | oracle:1; random:2; dsize:2; frequent:2 | 0.561 (small) |
| 10 | 0.248 ± 0.006 | 0.261 ± 0.000 | 0.282 ± 0.000 | 0.536 ± 0.000 | 0.021 ± 0.000 (+5/−0/5) | 0.275 ± 0.000 | oracle:1; dsize:2; frequent:2; random:2 | 0.531 (negligible) |
| 20 | 0.330 ± 0.004 | 0.341 ± 0.000 | 0.339 ± 0.000 | 0.643 ± 0.000 | -0.002 ± 0.000 (+0/−5/5) | 0.302 ± 0.000 | oracle:1; frequent:2; dsize:2; random:2 | 0.531 (negligible) |
| 50 | 0.544 ± 0.010 | 0.543 ± 0.000 | 0.561 ± 0.000 | 0.791 ± 0.000 | 0.019 ± 0.000 (+5/−0/5) | 0.248 ± 0.000 | oracle:1; dsize:2; random:2; frequent:2 | 0.541 (negligible) |

### Per-system D − frequency gains (mean gap, number of strictly positive seeds / 5)

| System | Subset | Metric | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|---|
| BGL | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.054 (5/5) | +0.000 (0/5) |
| BGL | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.143 (5/5) | +0.000 (0/5) |
| BGL | llm_only | lines | +0.000 (0/5) | +0.054 (5/5) | +0.054 (5/5) | +0.000 (0/5) |
| BGL | llm_only | templates | +0.000 (0/5) | +0.143 (5/5) | +0.143 (5/5) | +0.000 (0/5) |
| BGL | llm_involved | lines | +0.000 (0/5) | +0.054 (5/5) | +0.054 (5/5) | +0.000 (0/5) |
| BGL | llm_involved | templates | +0.000 (0/5) | +0.143 (5/5) | +0.143 (5/5) | +0.000 (0/5) |
| HDFS | all | lines | undefined | undefined | undefined | undefined |
| HDFS | all | templates | undefined | undefined | undefined | undefined |
| HPC | all | lines | undefined | undefined | undefined | undefined |
| HPC | all | templates | undefined | undefined | undefined | undefined |
| HPC | llm_only | lines | undefined | undefined | undefined | undefined |
| HPC | llm_only | templates | undefined | undefined | undefined | undefined |
| HPC | llm_involved | lines | undefined | undefined | undefined | undefined |
| HPC | llm_involved | templates | undefined | undefined | undefined | undefined |
| Hadoop | all | lines | +0.342 (5/5) | +0.000 (0/5) | +0.065 (5/5) | +0.069 (5/5) |
| Hadoop | all | templates | +0.023 (5/5) | +0.000 (0/5) | +0.023 (5/5) | +0.045 (5/5) |
| Hadoop | llm_only | lines | +0.342 (5/5) | +0.000 (0/5) | +0.050 (5/5) | +0.054 (5/5) |
| Hadoop | llm_only | templates | +0.023 (5/5) | +0.000 (0/5) | +0.023 (5/5) | +0.045 (5/5) |
| Hadoop | llm_involved | lines | +0.342 (5/5) | +0.000 (0/5) | +0.050 (5/5) | +0.054 (5/5) |
| Hadoop | llm_involved | templates | +0.023 (5/5) | +0.000 (0/5) | +0.023 (5/5) | +0.045 (5/5) |
| HealthApp | all | lines | +0.560 (5/5) | +0.000 (0/5) | -0.001 (0/5) | +0.000 (0/5) |
| HealthApp | all | templates | +0.042 (5/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HealthApp | llm_only | lines | +0.149 (5/5) | -0.001 (0/5) | -0.021 (0/5) | +0.000 (5/5) |
| HealthApp | llm_only | templates | +0.042 (5/5) | +0.000 (0/5) | -0.042 (0/5) | +0.042 (5/5) |
| HealthApp | llm_involved | lines | +0.149 (5/5) | -0.001 (0/5) | -0.021 (0/5) | +0.000 (5/5) |
| HealthApp | llm_involved | templates | +0.042 (5/5) | +0.000 (0/5) | -0.042 (0/5) | +0.042 (5/5) |
| Linux | all | lines | +0.000 (0/5) | -0.018 (0/5) | +0.003 (5/5) | +0.001 (5/5) |
| Linux | all | templates | +0.000 (0/5) | -0.007 (0/5) | +0.007 (5/5) | +0.000 (0/5) |
| Linux | llm_only | lines | -0.002 (0/5) | -0.034 (0/5) | +0.000 (0/5) | +0.001 (5/5) |
| Linux | llm_only | templates | +0.000 (0/5) | -0.014 (0/5) | +0.000 (0/5) | +0.007 (5/5) |
| Linux | llm_involved | lines | -0.002 (0/5) | -0.034 (0/5) | +0.000 (0/5) | +0.001 (5/5) |
| Linux | llm_involved | templates | +0.000 (0/5) | -0.014 (0/5) | +0.000 (0/5) | +0.007 (5/5) |
| Mac | all | lines | +0.000 (0/5) | +0.084 (5/5) | +0.102 (5/5) | +0.048 (5/5) |
| Mac | all | templates | +0.000 (0/5) | +0.012 (5/5) | +0.023 (5/5) | +0.023 (5/5) |
| Mac | llm_only | lines | +0.000 (0/5) | +0.087 (5/5) | +0.125 (5/5) | +0.073 (5/5) |
| Mac | llm_only | templates | +0.000 (0/5) | +0.012 (5/5) | +0.029 (5/5) | +0.029 (5/5) |
| Mac | llm_involved | lines | +0.000 (0/5) | +0.084 (5/5) | +0.121 (5/5) | +0.071 (5/5) |
| Mac | llm_involved | templates | +0.000 (0/5) | +0.012 (5/5) | +0.029 (5/5) | +0.029 (5/5) |
| OpenSSH | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.290 (5/5) | +0.000 (0/5) |
| OpenSSH | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.200 (5/5) | +0.000 (0/5) |
| OpenSSH | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenSSH | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| OpenStack | all | lines | undefined | undefined | undefined | undefined |
| OpenStack | all | templates | undefined | undefined | undefined | undefined |
| OpenStack | llm_only | lines | undefined | undefined | undefined | undefined |
| OpenStack | llm_only | templates | undefined | undefined | undefined | undefined |
| OpenStack | llm_involved | lines | undefined | undefined | undefined | undefined |
| OpenStack | llm_involved | templates | undefined | undefined | undefined | undefined |
| Spark | all | lines | +0.000 (0/5) | +0.000 (0/5) | -0.004 (0/5) | +0.000 (0/5) |
| Spark | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Spark | llm_only | lines | +0.158 (5/5) | +0.081 (5/5) | +0.047 (5/5) | +0.018 (5/5) |
| Spark | llm_only | templates | +0.000 (0/5) | +0.048 (5/5) | +0.048 (5/5) | +0.048 (5/5) |
| Spark | llm_involved | lines | +0.158 (5/5) | +0.081 (5/5) | +0.047 (5/5) | +0.018 (5/5) |
| Spark | llm_involved | templates | +0.000 (0/5) | +0.048 (5/5) | +0.048 (5/5) | +0.048 (5/5) |
| Thunderbird | all | lines | -0.060 (0/5) | +0.000 (0/5) | -0.005 (0/5) | -0.014 (0/5) |
| Thunderbird | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.005 (5/5) |
| Thunderbird | llm_only | lines | -0.060 (0/5) | -0.008 (0/5) | -0.005 (0/5) | +0.011 (5/5) |
| Thunderbird | llm_only | templates | +0.000 (0/5) | -0.005 (0/5) | +0.000 (0/5) | +0.015 (5/5) |
| Thunderbird | llm_involved | lines | -0.060 (0/5) | -0.008 (0/5) | -0.005 (0/5) | +0.011 (5/5) |
| Thunderbird | llm_involved | templates | +0.000 (0/5) | -0.005 (0/5) | +0.000 (0/5) | +0.015 (5/5) |
| Zookeeper | all | lines | +0.000 (0/5) | +0.447 (5/5) | +0.000 (0/5) | -0.004 (0/5) |
| Zookeeper | all | templates | +0.000 (0/5) | +0.167 (5/5) | +0.000 (0/5) | -0.167 (0/5) |
| Zookeeper | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | -0.004 (0/5) | +0.000 (0/5) |
| Zookeeper | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | -0.167 (0/5) | +0.000 (0/5) |
| Zookeeper | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | -0.004 (0/5) | +0.000 (0/5) |
| Zookeeper | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | -0.167 (0/5) | +0.000 (0/5) |

Headline: with the real parser, Most frequent achieves 72.1% of Oracle’s macro gain after 20 reviews: 0.614 versus 0.852 of wrong lines fixed (gap 0.238). D × size beats frequency on every seed in: all/all systems/lines/5, all/all systems/lines/10, all/all systems/lines/20, all/all systems/lines/50, all/all systems/templates/5, all/all systems/templates/10, all/all systems/templates/20, all/held-out/lines/5, all/held-out/lines/10, all/held-out/lines/20, all/held-out/lines/50, all/held-out/templates/5, all/held-out/templates/10, all/held-out/templates/20, llm_only/all systems/lines/5, llm_only/all systems/lines/10, llm_only/all systems/lines/20, llm_only/all systems/lines/50, llm_only/all systems/templates/5, llm_only/all systems/templates/10, llm_only/all systems/templates/20, llm_only/all systems/templates/50, llm_only/held-out/lines/5, llm_only/held-out/lines/10, llm_only/held-out/lines/20, llm_only/held-out/lines/50, llm_only/held-out/templates/5, llm_only/held-out/templates/10, llm_only/held-out/templates/50, llm_involved/all systems/lines/5, llm_involved/all systems/lines/10, llm_involved/all systems/lines/20, llm_involved/all systems/lines/50, llm_involved/all systems/templates/5, llm_involved/all systems/templates/10, llm_involved/all systems/templates/20, llm_involved/all systems/templates/50, llm_involved/held-out/lines/5, llm_involved/held-out/lines/10, llm_involved/held-out/lines/20, llm_involved/held-out/lines/50, llm_involved/held-out/templates/5, llm_involved/held-out/templates/10, llm_involved/held-out/templates/50, BGL/all/lines/20, BGL/all/templates/20, BGL/llm_only/lines/10, BGL/llm_only/lines/20, BGL/llm_only/templates/10, BGL/llm_only/templates/20, BGL/llm_involved/lines/10, BGL/llm_involved/lines/20, BGL/llm_involved/templates/10, BGL/llm_involved/templates/20, Hadoop/all/lines/5, Hadoop/all/lines/20, Hadoop/all/lines/50, Hadoop/all/templates/5, Hadoop/all/templates/20, Hadoop/all/templates/50, Hadoop/llm_only/lines/5, Hadoop/llm_only/lines/20, Hadoop/llm_only/lines/50, Hadoop/llm_only/templates/5, Hadoop/llm_only/templates/20, Hadoop/llm_only/templates/50, Hadoop/llm_involved/lines/5, Hadoop/llm_involved/lines/20, Hadoop/llm_involved/lines/50, Hadoop/llm_involved/templates/5, Hadoop/llm_involved/templates/20, Hadoop/llm_involved/templates/50, HealthApp/all/lines/5, HealthApp/all/templates/5, HealthApp/llm_only/lines/5, HealthApp/llm_only/lines/50, HealthApp/llm_only/templates/5, HealthApp/llm_only/templates/50, HealthApp/llm_involved/lines/5, HealthApp/llm_involved/lines/50, HealthApp/llm_involved/templates/5, HealthApp/llm_involved/templates/50, Linux/all/lines/20, Linux/all/lines/50, Linux/all/templates/20, Linux/llm_only/lines/50, Linux/llm_only/templates/50, Linux/llm_involved/lines/50, Linux/llm_involved/templates/50, Mac/all/lines/10, Mac/all/lines/20, Mac/all/lines/50, Mac/all/templates/10, Mac/all/templates/20, Mac/all/templates/50, Mac/llm_only/lines/10, Mac/llm_only/lines/20, Mac/llm_only/lines/50, Mac/llm_only/templates/10, Mac/llm_only/templates/20, Mac/llm_only/templates/50, Mac/llm_involved/lines/10, Mac/llm_involved/lines/20, Mac/llm_involved/lines/50, Mac/llm_involved/templates/10, Mac/llm_involved/templates/20, Mac/llm_involved/templates/50, OpenSSH/all/lines/20, OpenSSH/all/templates/20, Spark/llm_only/lines/5, Spark/llm_only/lines/10, Spark/llm_only/lines/20, Spark/llm_only/lines/50, Spark/llm_only/templates/10, Spark/llm_only/templates/20, Spark/llm_only/templates/50, Spark/llm_involved/lines/5, Spark/llm_involved/lines/10, Spark/llm_involved/lines/20, Spark/llm_involved/lines/50, Spark/llm_involved/templates/10, Spark/llm_involved/templates/20, Spark/llm_involved/templates/50, Thunderbird/all/templates/50, Thunderbird/llm_only/lines/50, Thunderbird/llm_only/templates/50, Thunderbird/llm_involved/lines/50, Thunderbird/llm_involved/templates/50, Zookeeper/all/lines/10, Zookeeper/all/templates/10.
