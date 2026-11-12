# Correction opportunity and exact representation ceilings — investigation B

Executed87 parser/model/system conditions using stored assignments, GT and per-group corrections. Existing ARI/error distributions/review curves were reused; no severity relabelling or new review ranking was run. Counts apply only to saved prefixes (all available Linux rows); missing later parser coverage is excluded from observed error denominators and remains explicitly missing.

## Three distinct interventions

1. Change text while retaining operational group IDs: grouping remains exactly unchanged.
2. Let the downstream event ID be the corrected template string: different pure groups can coalesce when given the same correct string, repairing over-splitting, but a mixed group cannot be split.
3. Split/reassign members using GT: a separate diagnostic membership oracle, not a template review.

For contingency n(g,t), one output string per group has exact maximum correct lines C*=Σg max_t n(g,t), residual R*=N−C*. If free event relabelling/coalescing is allowed, R* is also the exact minimum members needing reassignment to obtain GT memberships. The number of extra pure fragments needed is Σg(|GT(g)|−1). None of these is an anomaly-detection F1 ceiling.

A GT type can be restored as an exact full line set by string correction/coalescing iff none of its lines lies in a mixed operational group. The simultaneously restorable type count is K−|GT types touching mixed groups|. Irreducible cross-GT pairs within existing groups = Σg[C(n_g,2)−Σt C(n_gt,2)]; they survive every operation that cannot split a group. The saved post-correction ARI is the result of **line-optimal majority correction**, not a proof that ARI itself is maximized.

**한국어 설명:** 그룹별 정답 빈도 중 최댓값을 더하면 단일 문자열 교정으로 동시에 맞출 수 있는 최대 줄 수가 된다. 그 차이 R*는 혼합 그룹의 소수 이벤트 때문에 불가피하게 남는 오차다. 순수 그룹의 분할은 같은 정답 문자열로 합칠 수 있지만, 원래 그룹 ID를 고정한 탐지기에서는 그 합침조차 일어나지 않는다. 탐지 F1의 상한이나 실제 사람 교정 성공률로 해석할 수 없다.

| System | Condition | Rows | Initially wrong text | Mixed groups | GT types in mixtures | Pure split GT types | Residual R* | Exact line ceiling | Fully restorable GT types |
|---|---|---|---|---|---|---|---|---|---|
| Apache | shared/Drain | 50000 | 13911 | 0 | 0 | 0 | 0 | 1.00000 | 29/29 |
| Apache | llama/real_AdaParser | 50000 | 35 | 0 | 0 | 0 | 0 | 1.00000 | 29/29 |
| BGL | shared/Drain | 50000 | 39906 | 4 | 8 | 1 | 17 | 0.99966 | 46/54 |
| BGL | llama/real_AdaParser | 50000 | 252 | 0 | 0 | 2 | 0 | 1.00000 | 54/54 |
| BGL | gemma/real_AdaParser | 50000 | 224 | 0 | 0 | 2 | 0 | 1.00000 | 54/54 |
| HDFS | shared/Drain | 50000 | 23567 | 0 | 0 | 1 | 0 | 1.00000 | 13/13 |
| HDFS | llama/real_AdaParser | 50000 | 0 | 0 | 0 | 0 | 0 | 1.00000 | 13/13 |
| HDFS | gemma/real_AdaParser | 50000 | 0 | 0 | 0 | 0 | 0 | 1.00000 | 13/13 |
| HPC | shared/Drain | 50000 | 40995 | 0 | 0 | 1 | 0 | 1.00000 | 20/20 |
| HPC | llama/real_AdaParser | 50000 | 1744 | 0 | 0 | 1 | 0 | 1.00000 | 20/20 |
| HPC | gemma/real_AdaParser | 50000 | 0 | 0 | 0 | 0 | 0 | 1.00000 | 20/20 |
| Hadoop | shared/Drain | 50000 | 23145 | 5 | 8 | 8 | 288 | 0.99424 | 206/214 |
| Hadoop | llama/real_AdaParser | 50000 | 3041 | 3 | 5 | 8 | 125 | 0.99750 | 209/214 |
| Hadoop | gemma/real_AdaParser | 50000 | 3886 | 1 | 2 | 9 | 17 | 0.99966 | 212/214 |
| HealthApp | shared/Drain | 50000 | 34811 | 2 | 4 | 4 | 194 | 0.99612 | 132/136 |
| HealthApp | llama/real_AdaParser | 50000 | 9671 | 1 | 2 | 12 | 17 | 0.99966 | 134/136 |
| HealthApp | gemma/real_AdaParser | 50000 | 10110 | 0 | 0 | 4 | 0 | 1.00000 | 136/136 |
| Linux | shared/Drain | 23921 | 21263 | 5 | 12 | 24 | 59 | 0.99753 | 326/338 |
| Linux | llama/real_AdaParser | 23921 | 2968 | 11 | 26 | 28 | 91 | 0.99620 | 312/338 |
| Linux | gemma/real_AdaParser | 23921 | 14840 | 8 | 19 | 13 | 167 | 0.99302 | 319/338 |
| Mac | shared/Drain | 50000 | 31184 | 2 | 4 | 73 | 14 | 0.99972 | 474/478 |
| Mac | gemma/real_AdaParser | 50000 | 14675 | 15 | 52 | 20 | 912 | 0.98176 | 426/478 |
| OpenSSH | shared/Drain | 50000 | 27985 | 1 | 2 | 0 | 3 | 0.99994 | 25/27 |
| OpenSSH | llama/real_AdaParser | 50000 | 23 | 1 | 2 | 1 | 11 | 0.99978 | 25/27 |
| OpenSSH | gemma/real_AdaParser | 50000 | 31 | 1 | 2 | 1 | 10 | 0.99980 | 25/27 |
| OpenStack | shared/Drain | 50000 | 48556 | 1993 | 17 | 6 | 4981 | 0.90038 | 28/45 |
| OpenStack | llama/real_AdaParser | 50000 | 532 | 0 | 0 | 0 | 0 | 1.00000 | 45/45 |
| OpenStack | gemma/real_AdaParser | 50000 | 0 | 0 | 0 | 0 | 0 | 1.00000 | 45/45 |
| Spark | shared/Drain | 50000 | 48556 | 4 | 8 | 3 | 232 | 0.99536 | 129/137 |
| Spark | llama/real_AdaParser | 50000 | 749 | 3 | 6 | 5 | 11 | 0.99978 | 131/137 |
| Spark | gemma/real_AdaParser | 50000 | 708 | 3 | 8 | 2 | 22 | 0.99956 | 129/137 |
| Thunderbird | shared/Drain | 50000 | 38242 | 21 | 14 | 19 | 242 | 0.99516 | 526/540 |
| Thunderbird | llama/real_AdaParser | 50000 | 17814 | 15 | 55 | 44 | 1359 | 0.97282 | 485/540 |
| Thunderbird | gemma/real_AdaParser | 50000 | 17723 | 16 | 49 | 17 | 2446 | 0.95108 | 491/540 |
| Zookeeper | shared/Drain | 50000 | 7522 | 2 | 14 | 0 | 203 | 0.99594 | 75/89 |
| Zookeeper | llama/real_AdaParser | 50000 | 2245 | 1 | 3 | 4 | 10 | 0.99980 | 86/89 |
| Zookeeper | gemma/real_AdaParser | 50000 | 2445 | 1 | 3 | 0 | 10 | 0.99980 | 86/89 |

Complete87-row results, including cached base/strong, are in correction_opportunity.{csv,json}; every record includes Korean calculation/interpretation/limitation fields. correction_mixed_groups.json.gz preserves each mixed group’s GT contingency. Base/strong/Drain share operational groups: their structural residual is identical even when text accuracy differs. Actual AdaParser uses its final assignments, not Drain groups.

## What the ceilings establish

HDFS: all parser families have R*=0; real AdaParser is already GT. BGL: Drain/base/strong R*=17 lines; both real AdaParsers R*=0, so all observed real-parser text/over-split discrepancies can theoretically be removed by unlimited group text correction plus string coalescing. Operational-ID-only corrections still cannot change partitioning.

**한국어 설명:** HDFS와 BGL real AdaParser의 R*=0은 그룹마다 정답 문자열을 줄 수 있는 진단적 조건에서 완전한 표현 복구가 가능하다는 뜻이다. 오류가 실제로 없다는 뜻은 아니다(BGL에는252/224줄의 문자열 오류가 있다). 예산 제한, 검토 순서, 전문가 정답 가용성 및 실제 탐지 이득은 별개다.

OpenStack Drain/cache R*=4981/50000=9.962%; line ceiling90.038%. Real AdaParser R*=0. On the known labelled VM-tag subset both real parsers are already exact GT, so this dataset cannot supply a real-AdaParser review recovery test in that population. Hadoop real-parser R*=125 Llama/17 Gemma; labels/provenance remain unresolved.

**한국어 설명:** OpenStack의4981줄은 저장된 Drain 운영 그룹을 유지하는 한 텍스트를 아무리 많이 고쳐도 남는 오차다. 같은 그룹을 사용하는 base/strong에도 동일하게 적용된다. real AdaParser의 다른 그룹 구조와 혼합해 비교하면 안 되며, 라벨 있는 VM 부분집합의 real-parser 오류0과도 구분해야 한다.

Thunderbird real-parser R*=1359/50000=2.718% (Llama),2446/50000=4.892% (Gemma); exact line ceilings97.282%/95.108%. Mixed groups touch55/540 and49/540 GT types respectively. These are genuine membership obstacles, unlike the near-perfect HDFS/BGL partitioning. The prefix has no alert labels, however, so this cannot establish overlap with anomalous evaluation units or their detectability.

**한국어 설명:** Thunderbird에서 그룹별 다수 정답을 모두 적용해도 Llama1359줄/Gemma2446줄이 남는다. 따라서 이 표현 오차를 없애려면 그룹 분할/재배정이 필요하다. 현재5만 줄에는 경보가 없으므로 이상 탐지 피해나 교정 후 회복률은 계산하지 않았다.

## Line categories and non-additivity

Saved fields distinguish pure-group wrong text, initially wrong lines inside mixed groups, gross newly correct lines, newly broken previously correct minority lines, net gain and unavoidable final residual. Changing a mixed group to its majority string can sacrifice previously correct minority members. R* is a final-state quantity, not always a subset of initially wrong lines. GT types touched by splitting and merging can overlap and are not summed. A correct word bag or exact template string is not independently verified semantic correctness.

If membership repair is unavailable, prioritizing reviews of mixed groups cannot exceed these ceilings. If it is available, its additional human cost and action definition must be measured separately; current template-review histories do not simulate that action. [Figure](../results_phase3/cpu_remaining/correction_opportunity.pdf) shows actual-AdaParser residuals and GT-type exposure, not detector recovery.
