# N1-N3: updated after Llama cache restoration

Protocol: PROGRESS.md Decisions (N1-N3 and G2). Details: cpu_analyses_details.md; five-seed macros and SK-ESD/A12: cpu_analyses_stats.md. N1 concentration is deterministic. N2 is a simulation, not a measurement of a newly improved parser.

## N1: concentration and the frequency/Oracle gap

| Mean Gini | Top 1 / 5 / 10 / 20 line share |
|---|---|
| 0.863 | 0.323 / 0.718 / 0.800 / 0.855 |

| Cell | Systems | Wrong templates | Top20 wrong | Mean ρ(size, wrong) | Frequency @20 | Oracle @20 |
|---|---|---|---|---|---|---|
| llama/base | 13 | 0.587 | 0.623 | -0.075 | 0.861 | 0.878 |
| llama/strong | 13 | 0.416 | 0.498 | -0.081 | 0.791 | 0.821 |
| gemma/base | 12 | 0.537 | 0.568 | -0.068 | 0.833 | 0.863 |
| gemma/strong | 12 | 0.420 | 0.511 | -0.086 | 0.778 | 0.811 |

## N2: every tested flip level at the original headline budget, 20 reviews

| Cell | Flip rule / target | Frequency | D × size | Oracle | D − frequency | Positive seeds |
|---|---|---|---|---|---|---|
| llama/base | actual | 0.861 ± 0.000 | 0.851 ± 0.013 | 0.878 ± 0.000 | -0.010 ± 0.013 | 2/5 |
| llama/base | dlow/0.05 | 0.292 ± 0.032 | 0.449 ± 0.013 | 0.724 ± 0.008 | 0.157 ± 0.042 | 5/5 |
| llama/base | dlow/0.1 | 0.622 ± 0.057 | 0.716 ± 0.041 | 0.848 ± 0.025 | 0.095 ± 0.040 | 5/5 |
| llama/base | dlow/0.2 | 0.806 ± 0.035 | 0.833 ± 0.025 | 0.901 ± 0.017 | 0.027 ± 0.011 | 5/5 |
| llama/base | dlow/0.3 | 0.857 ± 0.005 | 0.873 ± 0.009 | 0.915 ± 0.005 | 0.016 ± 0.005 | 5/5 |
| llama/base | dlow/0.4 | 0.858 ± 0.006 | 0.868 ± 0.007 | 0.900 ± 0.004 | 0.010 ± 0.003 | 5/5 |
| llama/base | random/0.05 | 0.518 ± 0.044 | 0.518 ± 0.064 | 0.787 ± 0.017 | 0.000 ± 0.054 | 3/5 |
| llama/base | random/0.1 | 0.668 ± 0.076 | 0.644 ± 0.082 | 0.859 ± 0.032 | -0.024 ± 0.057 | 1/5 |
| llama/base | random/0.2 | 0.768 ± 0.058 | 0.751 ± 0.040 | 0.892 ± 0.027 | -0.017 ± 0.029 | 2/5 |
| llama/base | random/0.3 | 0.810 ± 0.023 | 0.785 ± 0.027 | 0.898 ± 0.013 | -0.026 ± 0.027 | 2/5 |
| llama/base | random/0.4 | 0.837 ± 0.024 | 0.814 ± 0.025 | 0.895 ± 0.016 | -0.023 ± 0.019 | 0/5 |
| llama/strong | actual | 0.791 ± 0.000 | 0.771 ± 0.004 | 0.821 ± 0.000 | -0.020 ± 0.004 | 0/5 |
| llama/strong | dlow/0.05 | 0.454 ± 0.010 | 0.492 ± 0.010 | 0.710 ± 0.012 | 0.038 ± 0.010 | 5/5 |
| llama/strong | dlow/0.1 | 0.651 ± 0.028 | 0.677 ± 0.039 | 0.820 ± 0.024 | 0.026 ± 0.021 | 5/5 |
| llama/strong | dlow/0.2 | 0.772 ± 0.033 | 0.773 ± 0.027 | 0.860 ± 0.015 | 0.002 ± 0.011 | 2/5 |
| llama/strong | dlow/0.3 | 0.808 ± 0.007 | 0.803 ± 0.010 | 0.865 ± 0.005 | -0.006 ± 0.006 | 1/5 |
| llama/strong | dlow/0.4 | 0.796 ± 0.005 | 0.786 ± 0.009 | 0.847 ± 0.004 | -0.010 ± 0.006 | 0/5 |
| llama/strong | random/0.05 | 0.527 ± 0.039 | 0.447 ± 0.052 | 0.741 ± 0.034 | -0.080 ± 0.039 | 0/5 |
| llama/strong | random/0.1 | 0.654 ± 0.044 | 0.601 ± 0.043 | 0.809 ± 0.017 | -0.052 ± 0.015 | 0/5 |
| llama/strong | random/0.2 | 0.741 ± 0.016 | 0.708 ± 0.018 | 0.842 ± 0.015 | -0.033 ± 0.013 | 0/5 |
| llama/strong | random/0.3 | 0.756 ± 0.007 | 0.727 ± 0.017 | 0.836 ± 0.009 | -0.029 ± 0.014 | 0/5 |
| llama/strong | random/0.4 | 0.777 ± 0.018 | 0.749 ± 0.017 | 0.837 ± 0.013 | -0.028 ± 0.008 | 0/5 |
| gemma/base | actual | 0.833 ± 0.000 | 0.814 ± 0.000 | 0.863 ± 0.000 | -0.019 ± 0.000 | 0/5 |
| gemma/base | dlow/0.05 | 0.551 ± 0.000 | 0.576 ± 0.000 | 0.771 ± 0.000 | 0.025 ± 0.000 | 5/5 |
| gemma/base | dlow/0.1 | 0.627 ± 0.000 | 0.723 ± 0.000 | 0.836 ± 0.000 | 0.096 ± 0.000 | 5/5 |
| gemma/base | dlow/0.2 | 0.731 ± 0.000 | 0.763 ± 0.000 | 0.834 ± 0.000 | 0.032 ± 0.000 | 5/5 |
| gemma/base | dlow/0.3 | 0.786 ± 0.000 | 0.804 ± 0.000 | 0.843 ± 0.000 | 0.018 ± 0.000 | 5/5 |
| gemma/base | dlow/0.4 | 0.782 ± 0.000 | 0.798 ± 0.000 | 0.829 ± 0.000 | 0.017 ± 0.000 | 5/5 |
| gemma/base | random/0.05 | 0.487 ± 0.053 | 0.496 ± 0.109 | 0.784 ± 0.015 | 0.009 ± 0.070 | 4/5 |
| gemma/base | random/0.1 | 0.639 ± 0.066 | 0.665 ± 0.072 | 0.848 ± 0.026 | 0.026 ± 0.049 | 4/5 |
| gemma/base | random/0.2 | 0.754 ± 0.037 | 0.748 ± 0.044 | 0.861 ± 0.036 | -0.006 ± 0.015 | 2/5 |
| gemma/base | random/0.3 | 0.790 ± 0.037 | 0.784 ± 0.033 | 0.859 ± 0.033 | -0.006 ± 0.012 | 2/5 |
| gemma/base | random/0.4 | 0.815 ± 0.023 | 0.792 ± 0.030 | 0.862 ± 0.022 | -0.022 ± 0.029 | 1/5 |
| gemma/strong | actual | 0.778 ± 0.000 | 0.772 ± 0.000 | 0.811 ± 0.000 | -0.007 ± 0.000 | 0/5 |
| gemma/strong | dlow/0.05 | 0.493 ± 0.000 | 0.519 ± 0.000 | 0.697 ± 0.000 | 0.026 ± 0.000 | 5/5 |
| gemma/strong | dlow/0.1 | 0.618 ± 0.000 | 0.653 ± 0.000 | 0.813 ± 0.000 | 0.034 ± 0.000 | 5/5 |
| gemma/strong | dlow/0.2 | 0.663 ± 0.000 | 0.689 ± 0.000 | 0.792 ± 0.000 | 0.026 ± 0.000 | 5/5 |
| gemma/strong | dlow/0.3 | 0.731 ± 0.000 | 0.747 ± 0.000 | 0.804 ± 0.000 | 0.015 ± 0.000 | 5/5 |
| gemma/strong | dlow/0.4 | 0.742 ± 0.000 | 0.752 ± 0.000 | 0.791 ± 0.000 | 0.010 ± 0.000 | 5/5 |
| gemma/strong | random/0.05 | 0.522 ± 0.011 | 0.538 ± 0.016 | 0.720 ± 0.020 | 0.016 ± 0.014 | 5/5 |
| gemma/strong | random/0.1 | 0.682 ± 0.033 | 0.679 ± 0.033 | 0.809 ± 0.029 | -0.003 ± 0.016 | 3/5 |
| gemma/strong | random/0.2 | 0.750 ± 0.037 | 0.740 ± 0.034 | 0.832 ± 0.029 | -0.011 ± 0.007 | 0/5 |
| gemma/strong | random/0.3 | 0.776 ± 0.016 | 0.768 ± 0.018 | 0.837 ± 0.018 | -0.007 ± 0.005 | 0/5 |
| gemma/strong | random/0.4 | 0.778 ± 0.008 | 0.772 ± 0.008 | 0.823 ± 0.011 | -0.006 ± 0.001 | 0/5 |

Random flips test accuracy improvements independent of D; D-low-risk flips deliberately favour D. Full budgets 5/10/20 and undefined no-error exclusions are in the details/statistics.

## N3: template-word anomaly detectors

| Model / detector | Templates / review order | F1, macro over HDFS and BGL |
|---|---|---|
| llama/IF | gt | 0.195 ± 0.025 |
| llama/IF | n0 | 0.209 ± 0.005 |
| llama/IF | frequent20 | 0.207 ± 0.001 |
| llama/IF | dsize20 | 0.207 ± 0.001 |
| llama/OCSVM | gt | 0.097 ± 0.000 |
| llama/OCSVM | n0 | 0.153 ± 0.000 |
| llama/OCSVM | frequent20 | 0.097 ± 0.000 |
| llama/OCSVM | dsize20 | 0.097 ± 0.000 |
| gemma/IF | gt | 0.195 ± 0.025 |
| gemma/IF | n0 | 0.149 ± 0.001 |
| gemma/IF | frequent20 | 0.207 ± 0.001 |
| gemma/IF | dsize20 | 0.207 ± 0.001 |
| gemma/OCSVM | gt | 0.097 ± 0.000 |
| gemma/OCSVM | n0 | 0.099 ± 0.000 |
| gemma/OCSVM | frequent20 | 0.041 ± 0.000 |
| gemma/OCSVM | dsize20 | 0.041 ± 0.000 |
