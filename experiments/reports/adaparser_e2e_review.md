# Real AdaParser: provenance, concentration and review outcomes

The official parser preloads 32 labelled Loghub-2.0 candidates per system; provenance shares count only templates with stream lines. Strict scoring; seeds 0–4; mean ± sample std of per-seed system macro means. Template failures use the first-line verdict, and a majority correction fixes one only when it equals that first-line GT. Line and template Oracles optimize their own metric. Parser cache seed 0; seeds 0–4 apply to review policies. Candidate/LLM/mixed/unknown provenance tracks final trie ancestry. LLM-only excludes mixed groups; LLM-involved includes them. Both restrictions exclude unknowns and refit within their own queue. Budgets are capped at the cohort’s available templates; actual review counts are saved per run.

## llama: partial: 3/13; provisional cached-state replay, fresh GPU check pending: Apache, HDFS, HPC

| System | Output templates | Candidate insertions | PA_0 | Wrong first-line | Wrong majority | Line share candidate / LLM / mixed / unknown | Template share candidate / LLM / mixed / unknown |
|---|---|---|---|---|---|---|---|
| Apache | 29 | 32 | 0.999 | 0.069 | 0.069 | 0.998 / 0.002 / 0.000 / 0.000 | 0.828 / 0.172 / 0.000 / 0.000 |
| HDFS | 13 | 32 | 1.000 | 0.000 | 0.000 | 1.000 / 0.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 / 0.000 |
| HPC | 22 | 32 | 0.965 | 0.136 | 0.136 | 0.856 / 0.144 / 0.000 / 0.000 | 0.727 / 0.273 / 0.000 / 0.000 |

| System | Gini | Top 1 / 5 / 10 / 20 line share | Top20 wrong first | Top20 wrong majority | ρ(size, first wrong) | ρ(size, majority wrong) |
|---|---|---|---|---|---|---|
| Apache | 0.815 | 0.386 / 0.843 / 0.981 / 0.997 | 0.000 | 0.000 | -0.32630621299906587 | -0.32630621299906587 |
| HDFS | 0.656 | 0.244 / 0.983 / 1.000 / 1.000 | 0.000 | 0.000 | None | None |
| HPC | 0.769 | 0.267 / 0.880 / 0.979 / 1.000 | 0.100 | 0.100 | -0.2924509453959643 | -0.2924509453959643 |

N5 comparison uses the same 2 systems for every parser, all stream lines; strong retains its 10 verified seed templates.

| Parser | Strict PA | PTA | RTA | FTA |
|---|---|---|---|---|
| AdaParser | 0.982 | 0.875 | 0.916 | 0.894 |
| Drain | 0.451 | 0.295 | 0.384 | 0.315 |
| base | 0.455 | 0.374 | 0.382 | 0.378 |
| strong | 0.803 | 0.730 | 0.747 | 0.738 |

### all, all systems: share of initially wrong lines fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.254 ± 0.237 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 1.000 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.500 (negligible) |
| 10 | 0.477 ± 0.095 | 0.499 ± 0.000 | 0.200 ± 0.273 | 1.000 ± 0.000 | -0.299 ± 0.273 (+0/−3/5) | 0.501 ± 0.000 | oracle:1; frequent:2; random:2; dsize:2 | 0.350 (medium) |
| 20 | 0.931 ± 0.062 | 0.500 ± 0.000 | 0.808 ± 0.172 | 1.000 ± 0.000 | 0.308 ± 0.172 (+4/−0/5) | 0.500 ± 0.000 | oracle:1; random:1; dsize:1; frequent:2 | 0.550 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### all, all systems: share of initially wrong templates fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.200 ± 0.209 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 1.000 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.500 (negligible) |
| 10 | 0.450 ± 0.173 | 0.167 ± 0.000 | 0.067 ± 0.091 | 1.000 ± 0.000 | -0.100 ± 0.091 (+0/−3/5) | 0.833 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.350 (medium) |
| 20 | 0.817 ± 0.109 | 0.333 ± 0.000 | 0.467 ± 0.112 | 1.000 ± 0.000 | 0.133 ± 0.112 (+4/−0/5) | 0.667 ± 0.000 | oracle:1; random:1; dsize:2; frequent:2 | 0.550 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### all, held-out: share of initially wrong lines fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.254 ± 0.237 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 1.000 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.500 (negligible) |
| 10 | 0.477 ± 0.095 | 0.499 ± 0.000 | 0.200 ± 0.273 | 1.000 ± 0.000 | -0.299 ± 0.273 (+0/−3/5) | 0.501 ± 0.000 | oracle:1; frequent:2; random:2; dsize:2 | 0.350 (medium) |
| 20 | 0.931 ± 0.062 | 0.500 ± 0.000 | 0.808 ± 0.172 | 1.000 ± 0.000 | 0.308 ± 0.172 (+4/−0/5) | 0.500 ± 0.000 | oracle:1; random:1; dsize:1; frequent:2 | 0.550 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### all, held-out: share of initially wrong templates fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.200 ± 0.209 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 1.000 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.500 (negligible) |
| 10 | 0.450 ± 0.173 | 0.167 ± 0.000 | 0.067 ± 0.091 | 1.000 ± 0.000 | -0.100 ± 0.091 (+0/−3/5) | 0.833 ± 0.000 | oracle:1; random:2; frequent:3; dsize:3 | 0.350 (medium) |
| 20 | 0.817 ± 0.109 | 0.333 ± 0.000 | 0.467 ± 0.112 | 1.000 ± 0.000 | 0.133 ± 0.112 (+4/−0/5) | 0.667 ± 0.000 | oracle:1; random:1; dsize:2; frequent:2 | 0.550 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_only, all systems: share of initially wrong lines fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.900 ± 0.223 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | oracle:1; frequent:1; dsize:1; random:1 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_only, all systems: share of initially wrong templates fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.867 ± 0.075 | 0.833 ± 0.000 | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.167 ± 0.000 | oracle:1; random:2; frequent:2; dsize:2 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_only, held-out: share of initially wrong lines fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.900 ± 0.223 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | oracle:1; frequent:1; dsize:1; random:1 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_only, held-out: share of initially wrong templates fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.867 ± 0.075 | 0.833 ± 0.000 | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.167 ± 0.000 | oracle:1; random:2; frequent:2; dsize:2 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_involved, all systems: share of initially wrong lines fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.900 ± 0.223 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | oracle:1; frequent:1; dsize:1; random:1 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_involved, all systems: share of initially wrong templates fixed

Eligible systems: 2/3; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.867 ± 0.075 | 0.833 ± 0.000 | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.167 ± 0.000 | oracle:1; random:2; frequent:2; dsize:2 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_involved, held-out: share of initially wrong lines fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.900 ± 0.223 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | oracle:1; frequent:1; dsize:1; random:1 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### llm_involved, held-out: share of initially wrong templates fixed

Eligible systems: 2/2; absent cohorts and zero initial-error denominators are excluded.

| Reviews | Random | Most frequent | D × size | Oracle | D − frequency (seed signs) | Oracle − frequency | SK-ESD | A12 D vs frequency |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.867 ± 0.075 | 0.833 ± 0.000 | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.167 ± 0.000 | oracle:1; random:2; frequent:2; dsize:2 | 0.500 (negligible) |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |
| 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 (+0/−0/5) | 0.000 ± 0.000 | random:1; frequent:1; dsize:1; oracle:1 | 0.500 (negligible) |

### Per-system D − frequency gains (mean gap, number of strictly positive seeds / 5)

| System | Subset | Metric | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|---|
| Apache | all | lines | +0.000 (0/5) | +0.000 (0/5) | +0.617 (4/5) | +0.000 (0/5) |
| Apache | all | templates | +0.000 (0/5) | +0.000 (0/5) | +0.400 (4/5) | +0.000 (0/5) |
| Apache | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| Apache | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HDFS | all | lines | undefined | undefined | undefined | undefined |
| HDFS | all | templates | undefined | undefined | undefined | undefined |
| HPC | all | lines | +0.000 (0/5) | -0.599 (0/5) | -0.000 (0/5) | +0.000 (0/5) |
| HPC | all | templates | +0.000 (0/5) | -0.200 (0/5) | -0.133 (0/5) | +0.000 (0/5) |
| HPC | llm_only | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_only | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_involved | lines | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |
| HPC | llm_involved | templates | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) | +0.000 (0/5) |

Headline: with the real parser, Most frequent achieves 50.0% of Oracle’s macro gain after 20 reviews: 0.500 versus 1.000 of wrong lines fixed (gap 0.500). D × size beats frequency on every seed nowhere among the tested budgets, subsets and systems.
