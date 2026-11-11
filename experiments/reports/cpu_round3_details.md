# CPU round 3 details

Protocol preregistered in PROGRESS.md Decisions. CPU only, no models/tokenizers downloaded, no GPU work or new LLM calls. Long jobs stopped at 04:17 UTC, about two minutes after conservative 04:15 cutoff; only owned process trees stopped. 05:00 UTC deadline is a conservative assumption (machine setup at 19:00 + 10 h), not a verified reservation end.

Official [AdaParser @232951a](https://github.com/wuyifan18/AdaParser/tree/232951a/benchmark/logparser/AdaParser) scoring-only code/regexes are vendored under MIT. L1: post_process_template(prediction), convert {variables} back to <*>, ignore regeneration flag, exact GT match. L2: L1 prediction then symmetric whitespace collapse and merging adjacent/whitespace-separated <*>. None remains failure. No API client imported.

Seed summaries: mean ± sample std of five seed macro means over systems. Held-out excludes HDFS/Spark/Linux. System exclusions and original strict split skips retained. Degenerate AUCs are undefined, never replaced by chance. U uses frozen settings and cached label-free features, rescored source/reference labels. No hyperparameter tuning.

## llama capped: label audit

| system | PASS strict / L1 / L2 | forgiven/broken samples L1; L2 | forgiven/broken inputs L1; L2 | outside 10–90% | reliable between share strict / L1 / L2 |
|---|---|---|---|---|---|
| loghub_apache_500 | 0.600 / 0.700 / 0.702 | 404/0; 407/0 | 50/0; 51/0 | none | 0.953 / 0.925 / 0.925 |
| loghub_bgl_500 | 0.244 / 0.302 / 0.306 | 276/3; 287/3 | 29/0; 31/0 | none | 0.966 / 0.970 / 0.970 |
| loghub_hdfs_500 | 0.282 / 0.474 / 0.476 | 869/0; 884/0 | 96/0; 97/0 | none | 0.960 / 0.951 / 0.962 |
| loghub_hpc_500 | 0.214 / 0.254 / 0.286 | 197/0; 346/0 | 20/0; 36/0 | none | 0.795 / 0.813 / 0.951 |
| loghub_hadoop_500 | 0.302 / 0.470 / 0.488 | 689/0; 765/0 | 84/0; 93/0 | none | 0.994 / 0.992 / 0.992 |
| loghub_healthapp_500 | 0.336 / 0.482 / 0.482 | 576/0; 597/0 | 73/0; 73/0 | none | 0.977 / 0.985 / 0.985 |
| loghub_linux_500 | 0.250 / 0.356 / 0.360 | 547/1; 556/1 | 53/0; 55/0 | none | 0.988 / 0.977 / 0.977 |
| loghub_mac_500 | 0.150 / 0.240 / 0.246 | 482/0; 512/0 | 45/0; 48/0 | none | 1.000 / 1.000 / 1.000 |
| loghub_openssh_500 | 0.406 / 0.504 / 0.504 | 359/0; 361/0 | 49/0; 49/0 | none | 0.857 / 0.825 / 0.826 |
| loghub_openstack_500 | 0.566 / 0.690 / 0.692 | 468/0; 489/0 | 62/0; 63/0 | none | 0.989 / 0.991 / 0.992 |
| loghub_spark_500 | 0.244 / 0.350 / 0.352 | 459/9; 479/9 | 54/1; 55/1 | none | 0.979 / 0.983 / 0.984 |
| loghub_thunderbird_500 | 0.220 / 0.282 / 0.290 | 368/11; 389/11 | 32/1; 36/1 | none | 1.000 / 1.000 / 1.000 |
| loghub_zookeeper_500 | 0.478 / 0.526 / 0.526 | 142/0; 143/0 | 24/0; 24/0 | none | 0.974 / 0.981 / 0.981 |
## llama natural: label audit

| system | PASS strict / L1 / L2 | forgiven/broken samples L1; L2 | forgiven/broken inputs L1; L2 | outside 10–90% | reliable between share strict / L1 / L2 |
|---|---|---|---|---|---|
| loghub_apache_nat500 | 0.818 / 0.950 / 0.950 | 677/0; 677/0 | 66/0; 66/0 | L1, L2 | n/a / n/a / n/a |
| loghub_bgl_nat500 | 0.166 / 0.236 / 0.248 | 347/0; 397/0 | 35/0; 41/0 | none | n/a / n/a / n/a |
| loghub_hadoop_nat500 | 0.228 / 0.538 / 0.552 | 1241/0; 1326/0 | 155/0; 162/0 | none | n/a / n/a / n/a |
| loghub_hdfs_nat500 | 0.470 / 0.774 / 0.774 | 1184/0; 1192/0 | 152/0; 152/0 | none | n/a / n/a / n/a |
| loghub_healthapp_nat500 | 0.328 / 0.714 / 0.728 | 1588/0; 1640/0 | 193/0; 200/0 | none | n/a / n/a / n/a |
| loghub_hpc_nat500 | 0.384 / 0.504 / 0.506 | 604/0; 611/0 | 60/0; 61/0 | none | n/a / n/a / n/a |
| loghub_linux_nat500 | 0.088 / 0.112 / 0.114 | 118/0; 131/0 | 12/0; 13/0 | strict | n/a / n/a / n/a |
| loghub_mac_nat500 | 0.190 / 0.244 / 0.258 | 308/5; 364/5 | 28/1; 35/1 | none | n/a / n/a / n/a |
| loghub_openssh_nat500 | 0.278 / 0.312 / 0.312 | 191/0; 194/0 | 17/0; 17/0 | none | n/a / n/a / n/a |
| loghub_openstack_nat500 | 0.206 / 0.248 / 0.248 | 153/0; 159/0 | 21/0; 21/0 | none | n/a / n/a / n/a |
| loghub_spark_nat500 | 0.400 / 0.588 / 0.610 | 701/0; 778/0 | 94/0; 105/0 | none | n/a / n/a / n/a |
| loghub_thunderbird_nat500 | 0.214 / 0.238 / 0.240 | 148/0; 157/0 | 12/0; 13/0 | none | n/a / n/a / n/a |
| loghub_zookeeper_nat500 | 0.788 / 0.792 / 0.792 | 29/0; 29/0 | 2/0; 2/0 | none | n/a / n/a / n/a |
## llama disjoint: AUC

13 systems with all five seeds; 65 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

### all completed/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.645 ± 0.022 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.656 ± 0.025 | 2 | 0.543 | 0.011 ± 0.033 | 3/5 |
| lookup_drain | 0.505 ± 0.005 | 3 | 0.166 | -0.140 ± 0.023 | 5/5 |
| out_entropy | 0.796 ± 0.014 | 1 | 0.896 | 0.151 ± 0.026 | 5/5 |
| random | 0.502 ± 0.013 | 3 | 0.141 | -0.143 ± 0.033 | 5/5 |
### all completed/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.657 ± 0.012 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.657 ± 0.023 | 2 | 0.517 | 0.001 ± 0.023 | 2/5 |
| lookup_drain | 0.508 ± 0.003 | 3 | 0.111 | -0.149 ± 0.013 | 5/5 |
| out_entropy | 0.796 ± 0.014 | 1 | 0.889 | 0.139 ± 0.017 | 5/5 |
| random | 0.511 ± 0.019 | 3 | 0.115 | -0.145 ± 0.019 | 5/5 |
### all completed/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.620 ± 0.019 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.602 ± 0.017 | 3 | 0.481 | -0.017 ± 0.007 | 5/5 |
| lookup_drain | 0.502 ± 0.004 | 4 | 0.123 | -0.118 ± 0.019 | 5/5 |
| out_entropy | 0.774 ± 0.011 | 1 | 0.875 | 0.155 ± 0.020 | 5/5 |
| random | 0.505 ± 0.014 | 4 | 0.144 | -0.114 ± 0.024 | 5/5 |
| U-best | 0.714 ± 0.023 | 2 | 0.759 | 0.095 ± 0.035 | 5/5 |
### all completed/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.619 ± 0.016 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.605 ± 0.018 | 2 | 0.469 | -0.014 ± 0.010 | 5/5 |
| lookup_drain | 0.506 ± 0.004 | 3 | 0.139 | -0.113 ± 0.018 | 5/5 |
| out_entropy | 0.774 ± 0.011 | 1 | 0.894 | 0.155 ± 0.017 | 5/5 |
| random | 0.512 ± 0.017 | 3 | 0.166 | -0.107 ± 0.011 | 5/5 |
### all completed/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.614 ± 0.019 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.605 ± 0.018 | 3 | 0.493 | -0.010 ± 0.011 | 4/5 |
| lookup_drain | 0.502 ± 0.005 | 4 | 0.139 | -0.112 ± 0.019 | 5/5 |
| out_entropy | 0.774 ± 0.012 | 1 | 0.887 | 0.160 ± 0.020 | 5/5 |
| random | 0.505 ± 0.014 | 4 | 0.163 | -0.109 ± 0.027 | 5/5 |
| U-best | 0.713 ± 0.021 | 2 | 0.759 | 0.099 ± 0.035 | 5/5 |
### all completed/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.614 ± 0.020 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.606 ± 0.018 | 2 | 0.485 | -0.008 ± 0.007 | 5/5 |
| lookup_drain | 0.506 ± 0.005 | 3 | 0.157 | -0.108 ± 0.020 | 5/5 |
| out_entropy | 0.774 ± 0.012 | 1 | 0.889 | 0.160 ± 0.024 | 5/5 |
| random | 0.513 ± 0.017 | 3 | 0.178 | -0.102 ± 0.021 | 5/5 |
### held-out/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.634 ± 0.022 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.637 ± 0.039 | 2 | 0.506 | 0.002 ± 0.041 | 3/5 |
| lookup_drain | 0.503 ± 0.005 | 3 | 0.215 | -0.131 ± 0.025 | 5/5 |
| out_entropy | 0.784 ± 0.016 | 1 | 0.867 | 0.150 ± 0.029 | 5/5 |
| random | 0.510 ± 0.012 | 3 | 0.194 | -0.124 ± 0.034 | 5/5 |
### held-out/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.646 ± 0.012 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.641 ± 0.034 | 2 | 0.500 | -0.005 ± 0.026 | 3/5 |
| lookup_drain | 0.506 ± 0.003 | 3 | 0.142 | -0.141 ± 0.011 | 5/5 |
| out_entropy | 0.784 ± 0.016 | 1 | 0.864 | 0.138 ± 0.013 | 5/5 |
| random | 0.508 ± 0.018 | 3 | 0.144 | -0.138 ± 0.019 | 5/5 |
### held-out/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.631 ± 0.020 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.591 ± 0.026 | 4 | 0.418 | -0.040 ± 0.013 | 5/5 |
| lookup_drain | 0.502 ± 0.003 | 5 | 0.138 | -0.129 ± 0.021 | 5/5 |
| out_entropy | 0.770 ± 0.013 | 1 | 0.836 | 0.139 ± 0.025 | 5/5 |
| random | 0.512 ± 0.012 | 5 | 0.150 | -0.119 ± 0.029 | 5/5 |
| U-best | 0.711 ± 0.031 | 2 | 0.700 | 0.080 ± 0.044 | 5/5 |
### held-out/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.630 ± 0.019 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.596 ± 0.024 | 3 | 0.403 | -0.035 ± 0.012 | 5/5 |
| lookup_drain | 0.503 ± 0.004 | 4 | 0.142 | -0.127 ± 0.019 | 5/5 |
| out_entropy | 0.770 ± 0.013 | 1 | 0.856 | 0.139 ± 0.022 | 5/5 |
| random | 0.509 ± 0.019 | 4 | 0.152 | -0.122 ± 0.012 | 5/5 |
### held-out/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.626 ± 0.023 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.595 ± 0.025 | 3 | 0.430 | -0.031 ± 0.017 | 5/5 |
| lookup_drain | 0.503 ± 0.004 | 4 | 0.158 | -0.123 ± 0.022 | 5/5 |
| out_entropy | 0.771 ± 0.013 | 1 | 0.850 | 0.145 ± 0.027 | 5/5 |
| random | 0.512 ± 0.014 | 4 | 0.172 | -0.114 ± 0.034 | 5/5 |
| U-best | 0.709 ± 0.029 | 2 | 0.700 | 0.083 ± 0.044 | 5/5 |
### held-out/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.623 ± 0.030 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.597 ± 0.024 | 2 | 0.430 | -0.026 ± 0.014 | 5/5 |
| lookup_drain | 0.503 ± 0.004 | 3 | 0.168 | -0.121 ± 0.028 | 5/5 |
| out_entropy | 0.771 ± 0.013 | 1 | 0.853 | 0.148 ± 0.033 | 5/5 |
| random | 0.509 ± 0.018 | 3 | 0.174 | -0.114 ± 0.029 | 5/5 |
## llama standard: AUC

8 systems with all five seeds; 40 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

### all completed/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.763 ± 0.012 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.717 ± 0.014 | 2 | 0.351 | -0.046 ± 0.007 | 5/5 |
| lookup_drain | 0.674 ± 0.008 | 3 | 0.258 | -0.089 ± 0.014 | 5/5 |
| out_entropy | 0.786 ± 0.002 | 1 | 0.587 | 0.023 ± 0.014 | 5/5 |
| random | 0.505 ± 0.013 | 4 | 0.001 | -0.258 ± 0.023 | 5/5 |
### all completed/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.807 ± 0.015 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.737 ± 0.008 | 2 | 0.263 | -0.071 ± 0.013 | 5/5 |
| lookup_drain | 0.742 ± 0.006 | 2 | 0.351 | -0.065 ± 0.014 | 5/5 |
| out_entropy | 0.791 ± 0.004 | 1 | 0.424 | -0.017 ± 0.019 | 4/5 |
| random | 0.504 ± 0.013 | 3 | 0.004 | -0.304 ± 0.013 | 5/5 |
### all completed/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.696 ± 0.015 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.626 ± 0.010 | 3 | 0.239 | -0.070 ± 0.010 | 5/5 |
| lookup_drain | 0.672 ± 0.006 | 2 | 0.421 | -0.024 ± 0.016 | 5/5 |
| out_entropy | 0.771 ± 0.003 | 1 | 0.819 | 0.075 ± 0.014 | 5/5 |
| random | 0.502 ± 0.014 | 4 | 0.024 | -0.194 ± 0.019 | 5/5 |
### all completed/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.750 ± 0.019 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.629 ± 0.014 | 2 | 0.092 | -0.121 ± 0.026 | 5/5 |
| lookup_drain | 0.735 ± 0.008 | 1 | 0.490 | -0.015 ± 0.019 | 4/5 |
| out_entropy | 0.767 ± 0.006 | 1 | 0.636 | 0.017 ± 0.017 | 4/5 |
| random | 0.497 ± 0.004 | 3 | 0.000 | -0.253 ± 0.020 | 5/5 |
### all completed/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.709 ± 0.010 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.625 ± 0.011 | 4 | 0.217 | -0.084 ± 0.016 | 5/5 |
| lookup_drain | 0.672 ± 0.009 | 3 | 0.408 | -0.037 ± 0.017 | 5/5 |
| out_entropy | 0.772 ± 0.002 | 1 | 0.781 | 0.063 ± 0.009 | 5/5 |
| random | 0.506 ± 0.012 | 5 | 0.000 | -0.204 ± 0.012 | 5/5 |
### all completed/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.746 ± 0.011 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.619 ± 0.013 | 3 | 0.079 | -0.127 ± 0.017 | 5/5 |
| lookup_drain | 0.731 ± 0.013 | 2 | 0.483 | -0.015 ± 0.009 | 5/5 |
| out_entropy | 0.770 ± 0.005 | 1 | 0.636 | 0.023 ± 0.011 | 5/5 |
| random | 0.504 ± 0.012 | 4 | 0.000 | -0.243 ± 0.011 | 5/5 |
### held-out/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.745 ± 0.014 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.691 ± 0.016 | 2 | 0.322 | -0.054 ± 0.009 | 5/5 |
| lookup_drain | 0.660 ± 0.008 | 2 | 0.277 | -0.085 ± 0.014 | 5/5 |
| out_entropy | 0.776 ± 0.003 | 1 | 0.611 | 0.031 ± 0.016 | 5/5 |
| random | 0.498 ± 0.014 | 3 | 0.000 | -0.247 ± 0.027 | 5/5 |
### held-out/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.791 ± 0.019 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.714 ± 0.008 | 2 | 0.276 | -0.077 ± 0.016 | 5/5 |
| lookup_drain | 0.720 ± 0.008 | 2 | 0.332 | -0.071 ± 0.017 | 5/5 |
| out_entropy | 0.778 ± 0.005 | 1 | 0.478 | -0.013 ± 0.023 | 4/5 |
| random | 0.509 ± 0.021 | 3 | 0.008 | -0.282 ± 0.016 | 5/5 |
### held-out/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.695 ± 0.023 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.606 ± 0.020 | 3 | 0.177 | -0.090 ± 0.014 | 5/5 |
| lookup_drain | 0.669 ± 0.005 | 2 | 0.417 | -0.027 ± 0.025 | 4/5 |
| out_entropy | 0.768 ± 0.003 | 1 | 0.782 | 0.072 ± 0.020 | 5/5 |
| random | 0.503 ± 0.015 | 4 | 0.032 | -0.193 ± 0.018 | 5/5 |
### held-out/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.749 ± 0.020 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.616 ± 0.011 | 2 | 0.090 | -0.133 ± 0.026 | 5/5 |
| lookup_drain | 0.724 ± 0.009 | 1 | 0.460 | -0.024 ± 0.019 | 4/5 |
| out_entropy | 0.765 ± 0.006 | 1 | 0.599 | 0.016 ± 0.019 | 4/5 |
| random | 0.499 ± 0.013 | 3 | 0.000 | -0.250 ± 0.029 | 5/5 |
### held-out/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.712 ± 0.012 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.606 ± 0.018 | 4 | 0.157 | -0.106 ± 0.020 | 5/5 |
| lookup_drain | 0.670 ± 0.008 | 3 | 0.394 | -0.042 ± 0.015 | 5/5 |
| out_entropy | 0.772 ± 0.002 | 1 | 0.736 | 0.060 ± 0.011 | 5/5 |
| random | 0.506 ± 0.011 | 5 | 0.000 | -0.205 ± 0.005 | 5/5 |
### held-out/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.746 ± 0.012 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.606 ± 0.012 | 2 | 0.071 | -0.141 ± 0.015 | 5/5 |
| lookup_drain | 0.719 ± 0.016 | 1 | 0.452 | -0.027 ± 0.009 | 5/5 |
| out_entropy | 0.769 ± 0.005 | 1 | 0.598 | 0.023 ± 0.013 | 5/5 |
| random | 0.506 ± 0.011 | 3 | 0.000 | -0.240 ± 0.011 | 5/5 |
## llama natural: AUC

0 systems with all five seeds; 0 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

## gemma capped: label audit

| system | PASS strict / L1 / L2 | forgiven/broken samples L1; L2 | forgiven/broken inputs L1; L2 | outside 10–90% | reliable between share strict / L1 / L2 |
|---|---|---|---|---|---|
| loghub_bgl_500 | 0.454 / 0.508 / 0.512 | 259/2; 287/2 | 27/0; 29/0 | none | 0.970 / 0.972 / 0.976 |
| loghub_hdfs_500 | 0.402 / 0.464 / 0.496 | 353/0; 458/0 | 31/0; 47/0 | none | 0.929 / 0.856 / 0.864 |
| loghub_hpc_500 | 0.540 / 0.594 / 0.630 | 211/0; 381/0 | 27/0; 45/0 | none | 0.798 / 0.783 / 0.897 |
| loghub_hadoop_500 | 0.434 / 0.586 / 0.602 | 713/0; 799/0 | 76/0; 84/0 | none | 0.988 / 0.993 / 0.995 |
| loghub_healthapp_500 | 0.524 / 0.624 / 0.668 | 474/0; 696/0 | 50/0; 72/0 | none | 0.971 / 0.967 / 0.965 |
| loghub_linux_500 | 0.404 / 0.574 / 0.586 | 815/0; 853/0 | 85/0; 91/0 | none | 0.992 / 0.990 / 0.986 |
| loghub_mac_500 | 0.290 / 0.356 / 0.360 | 342/0; 375/0 | 33/0; 35/0 | none | 1.000 / 1.000 / 1.000 |
| loghub_openssh_500 | 0.342 / 0.374 / 0.376 | 130/0; 139/0 | 16/0; 17/0 | none | 0.893 / 0.869 / 0.870 |
| loghub_openstack_500 | 0.704 / 0.788 / 0.796 | 389/0; 436/0 | 42/0; 46/0 | none | 0.983 / 0.991 / 0.992 |
| loghub_spark_500 | 0.364 / 0.464 / 0.470 | 455/0; 485/0 | 50/0; 53/0 | none | 0.975 / 0.974 / 0.975 |
| loghub_thunderbird_500 | 0.376 / 0.428 / 0.436 | 269/12; 290/12 | 27/1; 31/1 | none | 1.000 / 1.000 / 1.000 |
| loghub_zookeeper_500 | 0.608 / 0.656 / 0.656 | 222/0; 230/0 | 24/0; 24/0 | none | 0.991 / 0.986 / 0.986 |
## gemma natural: label audit

| system | PASS strict / L1 / L2 | forgiven/broken samples L1; L2 | forgiven/broken inputs L1; L2 | outside 10–90% | reliable between share strict / L1 / L2 |
|---|---|---|---|---|---|
| loghub_bgl_nat500 | 0.400 / 0.426 / 0.426 | 113/0; 117/0 | 13/0; 13/0 | none | n/a / n/a / n/a |
| loghub_hadoop_nat500 | 0.492 / 0.576 / 0.578 | 404/0; 435/0 | 42/0; 43/0 | none | n/a / n/a / n/a |
| loghub_hdfs_nat500 | 0.446 / 0.522 / 0.522 | 477/0; 480/0 | 38/0; 38/0 | none | n/a / n/a / n/a |
| loghub_healthapp_nat500 | 0.466 / 0.776 / 0.908 | 1417/0; 2073/0 | 155/0; 221/0 | L2 | n/a / n/a / n/a |
| loghub_hpc_nat500 | 0.636 / 0.660 / 0.678 | 188/0; 241/0 | 12/0; 21/0 | none | n/a / n/a / n/a |
| loghub_linux_nat500 | 0.332 / 0.346 / 0.368 | 76/0; 241/0 | 7/0; 18/0 | none | n/a / n/a / n/a |
| loghub_mac_nat500 | 0.266 / 0.298 / 0.310 | 193/8; 255/8 | 17/1; 23/1 | none | n/a / n/a / n/a |
| loghub_openssh_nat500 | 0.240 / 0.244 / 0.244 | 19/0; 19/0 | 2/0; 2/0 | none | n/a / n/a / n/a |
| loghub_openstack_nat500 | 0.282 / 0.310 / 0.310 | 133/0; 153/0 | 14/0; 14/0 | none | n/a / n/a / n/a |
| loghub_spark_nat500 | 0.726 / 0.752 / 0.752 | 143/0; 150/0 | 13/0; 13/0 | none | n/a / n/a / n/a |
| loghub_thunderbird_nat500 | 0.230 / 0.262 / 0.262 | 145/0; 155/0 | 16/0; 16/0 | none | n/a / n/a / n/a |
| loghub_zookeeper_nat500 | 0.386 / 0.462 / 0.466 | 390/0; 398/0 | 38/0; 40/0 | none | n/a / n/a / n/a |
## gemma disjoint: AUC

12 systems with all five seeds; 60 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

### all completed/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.659 ± 0.025 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.655 ± 0.017 | 2 | 0.476 | -0.004 ± 0.025 | 3/5 |
| lookup_drain | 0.508 ± 0.002 | 3 | 0.087 | -0.152 ± 0.026 | 5/5 |
| out_entropy | 0.824 ± 0.012 | 1 | 0.928 | 0.164 ± 0.033 | 5/5 |
| random | 0.507 ± 0.012 | 3 | 0.086 | -0.153 ± 0.033 | 5/5 |
### all completed/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.671 ± 0.019 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.661 ± 0.017 | 2 | 0.467 | -0.010 ± 0.022 | 3/5 |
| lookup_drain | 0.512 ± 0.001 | 3 | 0.024 | -0.159 ± 0.019 | 5/5 |
| out_entropy | 0.824 ± 0.012 | 1 | 0.946 | 0.152 ± 0.022 | 5/5 |
| random | 0.504 ± 0.009 | 3 | 0.029 | -0.167 ± 0.014 | 5/5 |
### all completed/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.647 ± 0.037 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.626 ± 0.023 | 3 | 0.409 | -0.021 ± 0.026 | 4/5 |
| lookup_drain | 0.506 ± 0.002 | 4 | 0.064 | -0.141 ± 0.038 | 5/5 |
| out_entropy | 0.796 ± 0.014 | 1 | 0.926 | 0.149 ± 0.043 | 5/5 |
| random | 0.508 ± 0.014 | 4 | 0.091 | -0.140 ± 0.035 | 5/5 |
| U-best | 0.744 ± 0.011 | 2 | 0.806 | 0.096 ± 0.046 | 5/5 |
### all completed/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.658 ± 0.017 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.634 ± 0.024 | 2 | 0.400 | -0.024 ± 0.027 | 4/5 |
| lookup_drain | 0.510 ± 0.002 | 3 | 0.068 | -0.148 ± 0.018 | 5/5 |
| out_entropy | 0.796 ± 0.014 | 1 | 0.909 | 0.138 ± 0.027 | 5/5 |
| random | 0.505 ± 0.011 | 3 | 0.071 | -0.153 ± 0.020 | 5/5 |
### all completed/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.646 ± 0.042 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.625 ± 0.029 | 3 | 0.403 | -0.021 ± 0.034 | 2/5 |
| lookup_drain | 0.506 ± 0.003 | 4 | 0.086 | -0.140 ± 0.044 | 5/5 |
| out_entropy | 0.792 ± 0.013 | 1 | 0.922 | 0.147 ± 0.053 | 5/5 |
| random | 0.509 ± 0.013 | 4 | 0.109 | -0.137 ± 0.040 | 5/5 |
| U-best | 0.749 ± 0.015 | 2 | 0.819 | 0.104 ± 0.051 | 5/5 |
### all completed/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.651 ± 0.017 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.634 ± 0.024 | 2 | 0.417 | -0.017 ± 0.016 | 4/5 |
| lookup_drain | 0.510 ± 0.002 | 3 | 0.103 | -0.141 ± 0.017 | 5/5 |
| out_entropy | 0.792 ± 0.013 | 1 | 0.912 | 0.141 ± 0.027 | 5/5 |
| random | 0.504 ± 0.010 | 3 | 0.096 | -0.147 ± 0.020 | 5/5 |
### held-out/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.658 ± 0.033 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.648 ± 0.026 | 2 | 0.464 | -0.010 ± 0.027 | 2/5 |
| lookup_drain | 0.509 ± 0.002 | 3 | 0.094 | -0.149 ± 0.034 | 5/5 |
| out_entropy | 0.823 ± 0.011 | 1 | 0.913 | 0.165 ± 0.039 | 5/5 |
| random | 0.508 ± 0.015 | 3 | 0.096 | -0.149 ± 0.039 | 5/5 |
### held-out/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.678 ± 0.024 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.659 ± 0.025 | 2 | 0.437 | -0.019 ± 0.025 | 4/5 |
| lookup_drain | 0.513 ± 0.001 | 3 | 0.026 | -0.166 ± 0.025 | 5/5 |
| out_entropy | 0.823 ± 0.011 | 1 | 0.931 | 0.145 ± 0.027 | 5/5 |
| random | 0.500 ± 0.013 | 3 | 0.026 | -0.178 ± 0.018 | 5/5 |
### held-out/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.652 ± 0.044 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.630 ± 0.034 | 3 | 0.414 | -0.022 ± 0.023 | 4/5 |
| lookup_drain | 0.507 ± 0.003 | 4 | 0.062 | -0.145 ± 0.047 | 5/5 |
| out_entropy | 0.800 ± 0.012 | 1 | 0.910 | 0.148 ± 0.048 | 5/5 |
| random | 0.509 ± 0.016 | 4 | 0.089 | -0.143 ± 0.033 | 5/5 |
| U-best | 0.742 ± 0.018 | 2 | 0.769 | 0.091 ± 0.058 | 5/5 |
### held-out/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.664 ± 0.016 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.633 ± 0.034 | 2 | 0.376 | -0.032 ± 0.024 | 4/5 |
| lookup_drain | 0.510 ± 0.002 | 3 | 0.067 | -0.154 ± 0.017 | 5/5 |
| out_entropy | 0.800 ± 0.012 | 1 | 0.889 | 0.135 ± 0.023 | 5/5 |
| random | 0.502 ± 0.010 | 3 | 0.068 | -0.162 ± 0.020 | 5/5 |
### held-out/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.651 ± 0.042 | 3 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.627 ± 0.037 | 3 | 0.401 | -0.025 ± 0.030 | 4/5 |
| lookup_drain | 0.506 ± 0.004 | 4 | 0.086 | -0.145 ± 0.044 | 5/5 |
| out_entropy | 0.796 ± 0.013 | 1 | 0.901 | 0.145 ± 0.054 | 5/5 |
| random | 0.510 ± 0.017 | 4 | 0.110 | -0.142 ± 0.031 | 5/5 |
| U-best | 0.753 ± 0.023 | 2 | 0.789 | 0.102 ± 0.057 | 5/5 |
### held-out/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.655 ± 0.026 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.633 ± 0.036 | 2 | 0.402 | -0.022 ± 0.021 | 4/5 |
| lookup_drain | 0.511 ± 0.002 | 3 | 0.116 | -0.145 ± 0.027 | 5/5 |
| out_entropy | 0.796 ± 0.013 | 1 | 0.888 | 0.141 ± 0.035 | 5/5 |
| random | 0.502 ± 0.009 | 3 | 0.104 | -0.153 ± 0.031 | 5/5 |
## gemma standard: AUC

7 systems with all five seeds; 35 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

### all completed/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.749 ± 0.006 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.706 ± 0.009 | 3 | 0.365 | -0.044 ± 0.010 | 5/5 |
| lookup_drain | 0.681 ± 0.006 | 3 | 0.331 | -0.068 ± 0.011 | 5/5 |
| out_entropy | 0.811 ± 0.003 | 1 | 0.739 | 0.062 ± 0.008 | 5/5 |
| random | 0.503 ± 0.009 | 4 | 0.000 | -0.247 ± 0.014 | 5/5 |
### all completed/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.806 ± 0.008 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.725 ± 0.004 | 2 | 0.244 | -0.081 ± 0.006 | 5/5 |
| lookup_drain | 0.750 ± 0.010 | 2 | 0.377 | -0.056 ± 0.012 | 5/5 |
| out_entropy | 0.810 ± 0.004 | 1 | 0.503 | 0.004 ± 0.007 | 3/5 |
| random | 0.503 ± 0.013 | 3 | 0.000 | -0.303 ± 0.018 | 5/5 |
### all completed/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.724 ± 0.011 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.650 ± 0.014 | 3 | 0.173 | -0.074 ± 0.017 | 5/5 |
| lookup_drain | 0.667 ± 0.009 | 3 | 0.335 | -0.057 ± 0.007 | 5/5 |
| out_entropy | 0.796 ± 0.003 | 1 | 0.850 | 0.072 ± 0.011 | 5/5 |
| random | 0.502 ± 0.009 | 4 | 0.000 | -0.222 ± 0.015 | 5/5 |
### all completed/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.764 ± 0.005 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.647 ± 0.011 | 3 | 0.070 | -0.118 ± 0.015 | 5/5 |
| lookup_drain | 0.737 ± 0.007 | 2 | 0.481 | -0.027 ± 0.008 | 5/5 |
| out_entropy | 0.799 ± 0.006 | 1 | 0.707 | 0.034 ± 0.005 | 5/5 |
| random | 0.500 ± 0.009 | 4 | 0.000 | -0.265 ± 0.012 | 5/5 |
### all completed/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.719 ± 0.006 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.634 ± 0.012 | 4 | 0.204 | -0.085 ± 0.012 | 5/5 |
| lookup_drain | 0.670 ± 0.005 | 3 | 0.349 | -0.049 ± 0.006 | 5/5 |
| out_entropy | 0.805 ± 0.004 | 1 | 0.841 | 0.086 ± 0.007 | 5/5 |
| random | 0.503 ± 0.002 | 5 | 0.000 | -0.217 ± 0.006 | 5/5 |
### all completed/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.768 ± 0.012 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.643 ± 0.011 | 4 | 0.095 | -0.125 ± 0.005 | 5/5 |
| lookup_drain | 0.735 ± 0.006 | 3 | 0.425 | -0.033 ± 0.013 | 5/5 |
| out_entropy | 0.805 ± 0.005 | 1 | 0.656 | 0.037 ± 0.014 | 5/5 |
| random | 0.505 ± 0.009 | 5 | 0.000 | -0.263 ± 0.021 | 5/5 |
### held-out/strict/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.730 ± 0.010 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.681 ± 0.015 | 3 | 0.323 | -0.049 ± 0.016 | 5/5 |
| lookup_drain | 0.668 ± 0.006 | 3 | 0.357 | -0.062 ± 0.015 | 5/5 |
| out_entropy | 0.821 ± 0.003 | 1 | 0.808 | 0.091 ± 0.012 | 5/5 |
| random | 0.505 ± 0.003 | 4 | 0.000 | -0.225 ± 0.013 | 5/5 |
### held-out/strict/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.791 ± 0.012 | 1 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.707 ± 0.004 | 2 | 0.245 | -0.084 ± 0.008 | 5/5 |
| lookup_drain | 0.745 ± 0.007 | 2 | 0.446 | -0.046 ± 0.017 | 5/5 |
| out_entropy | 0.825 ± 0.001 | 1 | 0.654 | 0.034 ± 0.012 | 5/5 |
| random | 0.502 ± 0.014 | 3 | 0.000 | -0.289 ± 0.019 | 5/5 |
### held-out/L1/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.721 ± 0.012 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.651 ± 0.009 | 3 | 0.214 | -0.071 ± 0.005 | 5/5 |
| lookup_drain | 0.661 ± 0.012 | 3 | 0.312 | -0.061 ± 0.008 | 5/5 |
| out_entropy | 0.807 ± 0.004 | 1 | 0.866 | 0.085 ± 0.015 | 5/5 |
| random | 0.502 ± 0.012 | 4 | 0.000 | -0.220 ± 0.019 | 5/5 |
### held-out/L1/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.766 ± 0.006 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.654 ± 0.007 | 3 | 0.099 | -0.112 ± 0.012 | 5/5 |
| lookup_drain | 0.733 ± 0.009 | 2 | 0.462 | -0.033 ± 0.010 | 5/5 |
| out_entropy | 0.812 ± 0.009 | 1 | 0.736 | 0.046 ± 0.008 | 5/5 |
| random | 0.504 ± 0.012 | 4 | 0.000 | -0.262 ± 0.016 | 5/5 |
### held-out/L2/@100

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.712 ± 0.008 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.633 ± 0.012 | 3 | 0.189 | -0.079 ± 0.013 | 5/5 |
| lookup_drain | 0.660 ± 0.006 | 3 | 0.352 | -0.053 ± 0.007 | 5/5 |
| out_entropy | 0.818 ± 0.004 | 1 | 0.915 | 0.105 ± 0.010 | 5/5 |
| random | 0.500 ± 0.003 | 4 | 0.000 | -0.213 ± 0.008 | 5/5 |
### held-out/L2/@200

| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |
|---|---|---:|---:|---|---:|
| clotho | 0.761 ± 0.015 | 2 | 0.500 | 0.000 ± 0.000 | 0/5 |
| MDSA | 0.647 ± 0.012 | 3 | 0.122 | -0.114 ± 0.010 | 5/5 |
| lookup_drain | 0.735 ± 0.008 | 2 | 0.445 | -0.026 ± 0.013 | 5/5 |
| out_entropy | 0.822 ± 0.004 | 1 | 0.755 | 0.061 ± 0.017 | 5/5 |
| random | 0.503 ± 0.005 | 4 | 0.000 | -0.258 ± 0.014 | 5/5 |
## gemma natural: AUC

0 systems with all five seeds; 0 completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.

## Streams and remaining work

Strict reproduction succeeded on the few completed stream cells, but no complete-model L1/L2 stream results finished. PA_0, N1/N5 and the 20-review frequency headline under lenient scoring are unresolved. Llama base remains 11 systems (HDFS/OpenSSH missing), strong Llama 13 and Gemma 12. Finish standard/natural suite jobs and every rescored D stream job before a full R1 conclusion. Stream script is unfinished and unvalidated beyond saved strict baselines.

## R2–R4

Not started; record detailed protocols before computation.
