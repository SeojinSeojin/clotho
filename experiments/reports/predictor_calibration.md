# Frozen failure-predictor calibration — investigation C

125 verified disjoint@100 jobs (13 Llama/12 Gemma systems×5 seeds), with15462 held-out evaluation observations across seeds. No base predictor was refitted, no hidden states/generations were regenerated and no severity proxies were reclassified. Primary label is strict FAIL iff at least3 of samples5–9 fail. We verify labels from those saved exact-match flags, original test/reference indices and the previous strict-score checks. Only identical eligible examples with all seven verified score methods enter comparisons; MDSA remains excluded after its earlier reconstruction failure.

## Score semantics before calibration

| Method | Stored quantity | Direct Brier/ECE? | Transformation |
|---|---|---|---|
| Clotho | GMM score_samples log-density; not cluster posterior | No | Only a separately fitted score→FAIL mapping |
| U / U+V5 | Classifier predict_proba(PASS) | Yes, uncalibrated probability forecast | P(FAIL)=1−P(PASS)=1+stored risk |
| D-zero | Source-trained classifier predict_proba(PASS) | Yes, uncalibrated probability forecast | Same complement |
| Output entropy | Entropy ranking score | No | Separate fitted map only |
| Token probability | Stored token log-score surrogate, can exceed unit range | No | Not P(correctness); separate fitted map |
| LLMLog | Reference-based preference/ranking score | No | Separate fitted map only |

U/D classifier probabilities are not automatically calibrated: class balancing, source weighting, early stopping and deployment prior shift can matter. No arbitrary score is clipped/min–max scaled into a claimed probability. Risk is stored as negative PASS score; probability complements use verified provenance, not the observed numeric range alone. Raw Clotho Brier/ECE is explicitly unavailable.

**한국어 설명:** Clotho의 값은 밀도의 로그이며 성공확률이 아니다. U/D는 분류기의 PASS 확률이므로1−PASS로 실패확률을 만들 수 있지만, 그 값이 실제 실패율과 맞는지는 별도로 검증해야 한다. 토큰 점수/엔트로피를0~1로 강제 변환해 확률이라고 부르지 않았다.

## Leakage and cost protocol

Within each saved job, SHA256(model,system,seed,GT template) parity partitions the original test templates into calibration/evaluation halves. There is no template overlap between these halves or input overlap with the original reference set. The original benchmark labels were already analyzed in earlier reports: “held out” here means excluded from fitting this new fixed calibration map, not a never-before-inspected dataset. Roles may differ across seeds, so repeated observations are correlated.

Map: StandardScaler + logistic sigmoid (C=1, no class balancing, seed fixed), fit only on calibration scores/labels; same map and hyperparameters for every method. All125 jobs satisfy the frozen minimum10 calibration rows/two examples each class. Calibration uses47–170 labels per job (median124), additional to original model training; labels shared across methods/seeds are not reacquired. Local D calibration therefore loses its zero-target-label deployment claim. No calibrated scores are substituted into existing review rankings.

92/905 sigmoid fits have negative slopes. The additional mapping can reverse a frozen score’s order when the small calibration set estimates a reversed relationship; this is recorded as instability, not a ranking improvement claim. Base scores remain untouched. Separate fitted coefficients and exact split manifests are saved.

**한국어 설명:** 같은 정답 템플릿은 보정용과 평가용에 동시에 들어가지 않도록 해시로 나눴다. 각 보정기는47~170개(중앙값124개)의 별도 라벨을 쓴다. 따라서 보정된 D를 무표적 라벨 방법이라고 할 수 없고, 원래 검토 정책의 순위를 바꾸지도 않았다. 작은 보정 집합에서 역방향 기울기가 나오면 이를 안정적 개선으로 해석하지 않는다.

## Brier, ECE and reliability

Brier=mean((p_failure−y_failure)^2); smaller is better. ECE uses10 equal-width bins, sum_b(n_b/N)|mean p_b−mean y_b|;5/15-bin sensitivities are saved. All-table statistics are mean±sample SD over five seed macros, with equal system weighting. ECE can reward a low-information prevalence forecast; it must be read alongside Brier/discrimination and bin counts.

| Model | Method | Probability variant | Systems | Brier | ECE10 |
|---|---|---|---|---|---|
| llama | U | raw | 13 | 0.1950 ± 0.0156 | 0.2091 ± 0.0130 |
| llama | U+V5 | raw | 13 | 0.1875 ± 0.0137 | 0.2004 ± 0.0124 |
| llama | D-zero | raw | 13 | 0.2044 ± 0.0272 | 0.1862 ± 0.0240 |
| llama | Clotho | local_sigmoid | 13 | 0.2236 ± 0.0256 | 0.1709 ± 0.0498 |
| llama | U | local_sigmoid | 13 | 0.1882 ± 0.0127 | 0.1708 ± 0.0295 |
| llama | U+V5 | local_sigmoid | 13 | 0.1843 ± 0.0111 | 0.1636 ± 0.0268 |
| llama | D-zero | local_sigmoid | 13 | 0.2076 ± 0.0241 | 0.1836 ± 0.0333 |
| llama | Output entropy | local_sigmoid | 13 | 0.1774 ± 0.0160 | 0.1371 ± 0.0307 |
| llama | Token probability | local_sigmoid | 13 | 0.2291 ± 0.0202 | 0.1383 ± 0.0415 |
| llama | LLMLog | local_sigmoid | 13 | 0.2043 ± 0.0143 | 0.1566 ± 0.0258 |
| llama | Calibration prevalence | local | 13 | 0.2265 ± 0.0190 | 0.1245 ± 0.0406 |
| gemma | U | raw | 12 | 0.1916 ± 0.0126 | 0.1613 ± 0.0209 |
| gemma | U+V5 | raw | 12 | 0.1877 ± 0.0073 | 0.1555 ± 0.0207 |
| gemma | D-zero | raw | 12 | 0.2305 ± 0.0089 | 0.2060 ± 0.0199 |
| gemma | Clotho | local_sigmoid | 12 | 0.2350 ± 0.0159 | 0.1697 ± 0.0071 |
| gemma | U | local_sigmoid | 12 | 0.1934 ± 0.0191 | 0.1551 ± 0.0147 |
| gemma | U+V5 | local_sigmoid | 12 | 0.1901 ± 0.0163 | 0.1502 ± 0.0085 |
| gemma | D-zero | local_sigmoid | 12 | 0.2222 ± 0.0181 | 0.1694 ± 0.0184 |
| gemma | Output entropy | local_sigmoid | 12 | 0.1673 ± 0.0092 | 0.1255 ± 0.0152 |
| gemma | Token probability | local_sigmoid | 12 | 0.2472 ± 0.0154 | 0.1446 ± 0.0188 |
| gemma | LLMLog | local_sigmoid | 12 | 0.2186 ± 0.0205 | 0.1695 ± 0.0130 |
| gemma | Calibration prevalence | local | 12 | 0.2472 ± 0.0165 | 0.1210 ± 0.0242 |

**한국어 설명:** 각 행의 Brier는 예측 실패확률과0/1 실패라벨 차이를 제곱해 평균한 값이다. ECE는10개 확률 구간의 예측값-관측 실패율 차이를 표본수로 가중한 값이다. 예를 들어 Llama raw U의 Brier0.1950/ECE0.2091은 순위가 유용해도 수치 확률은 어긋날 수 있음을 보여준다. 단순 유병률 예측이 낮은 ECE를 얻을 수 있어 ECE만으로 방법의 실용성을 판정할 수 없다.

| Model | Method | Cohort | Systems | ΔBrier local calibration−raw | System-cluster95% interval |
|---|---|---|---|---|---|
| llama | U | all | 13 | -0.0068 | [-0.0323,+0.0240] |
| llama | U+V5 | all | 13 | -0.0032 | [-0.0291,+0.0264] |
| llama | D-zero | all | 13 | +0.0031 | [-0.0153,+0.0249] |
| gemma | U | all | 12 | +0.0018 | [-0.0054,+0.0091] |
| gemma | U+V5 | all | 12 | +0.0024 | [-0.0024,+0.0075] |
| gemma | D-zero | all | 12 | -0.0083 | [-0.0153,-0.0010] |
| llama | U | held_out | 10 | +0.0029 | [-0.0314,+0.0392] |
| llama | U+V5 | held_out | 10 | +0.0065 | [-0.0249,+0.0428] |
| llama | D-zero | held_out | 10 | +0.0040 | [-0.0196,+0.0320] |
| gemma | U | held_out | 9 | -0.0008 | [-0.0082,+0.0055] |
| gemma | U+V5 | held_out | 9 | +0.0020 | [-0.0030,+0.0077] |
| gemma | D-zero | held_out | 9 | -0.0118 | [-0.0185,-0.0048] |

**한국어 설명:** 같은 시스템/시드의 보정 후 Brier에서 보정 전 Brier를 뺐다. 음수면 확률 오차 감소이며 신뢰구간은 시드 평균을 낸 시스템을4000번 재표집했다. Llama U의−0.0068 구간은0을 포함하고 Gemma U는+0.0018로 일관된 개선이 아니다. Gemma D의 평균 개선도 소수 시스템과 추가 보정 라벨에 조건부이며 다중 비교 보정 없는 기술적 구간이다. 탐지 효용이나 검토 비용 절감을 뜻하지 않는다.

## Calibration by system and capped-template frequency

| Model | System | Raw U Brier/ECE | Raw D Brier/ECE |
|---|---|---|---|
| llama | Apache | 0.2214/0.2064 | 0.4016/0.4605 |
| llama | BGL | 0.1805/0.1725 | 0.1657/0.0979 |
| llama | HDFS | 0.2027/0.2394 | 0.1799/0.1834 |
| llama | HPC | 0.1514/0.2509 | 0.0860/0.1138 |
| llama | Hadoop | 0.1880/0.2029 | 0.1557/0.1072 |
| llama | HealthApp | 0.1779/0.1729 | 0.1548/0.1337 |
| llama | Linux | 0.1777/0.1772 | 0.1488/0.0863 |
| llama | Mac | 0.1734/0.2584 | 0.1298/0.1182 |
| llama | OpenSSH | 0.2287/0.2032 | 0.2913/0.2956 |
| llama | OpenStack | 0.1605/0.1770 | 0.2577/0.2819 |
| llama | Spark | 0.2066/0.2118 | 0.1759/0.1143 |
| llama | Thunderbird | 0.1911/0.2086 | 0.1407/0.0905 |
| llama | Zookeeper | 0.2744/0.2368 | 0.3695/0.3367 |
| gemma | BGL | 0.2024/0.1406 | 0.2445/0.1682 |
| gemma | HDFS | 0.1574/0.1047 | 0.1715/0.1463 |
| gemma | HPC | 0.2159/0.1751 | 0.2392/0.1981 |
| gemma | Hadoop | 0.1683/0.0965 | 0.2126/0.1424 |
| gemma | HealthApp | 0.2206/0.2047 | 0.2675/0.2739 |
| gemma | Linux | 0.1601/0.1079 | 0.1816/0.1563 |
| gemma | Mac | 0.1623/0.0952 | 0.1986/0.1725 |
| gemma | OpenSSH | 0.3123/0.3550 | 0.2868/0.3342 |
| gemma | OpenStack | 0.1281/0.1904 | 0.2839/0.3192 |
| gemma | Spark | 0.1824/0.1466 | 0.2023/0.1396 |
| gemma | Thunderbird | 0.1709/0.0958 | 0.1982/0.1415 |
| gemma | Zookeeper | 0.2179/0.2228 | 0.2796/0.2796 |

**한국어 설명:** 시스템별 값은 해당 시스템의 다섯 시드 평균이다. 전체 평균이 좋더라도 시스템마다 확률 오차가 달라 하나의 임계값이나 보정기를 그대로 이전할 수 있다는 보장은 없다. 원시드/확률구간 표본수와 모든 방법은 calibration_per_system.csv 및 calibration_metrics.json에 있다.

| Model | Method | GT frequency in capped500 | Systems represented | Repeated-seed observations | Brier | ECE |
|---|---|---|---|---|---|---|
| llama | U | frequency1 | 12 | 2303 | 0.2111 ± 0.0347 | 0.2756 ± 0.0333 |
| llama | U | frequency2_5 | 11 | 1848 | 0.1541 ± 0.0441 | 0.2299 ± 0.0444 |
| llama | U | frequency6plus | 8 | 3959 | 0.2113 ± 0.0259 | 0.2473 ± 0.0196 |
| llama | D-zero | frequency1 | 12 | 2303 | 0.1578 ± 0.0263 | 0.1813 ± 0.0206 |
| llama | D-zero | frequency2_5 | 11 | 1848 | 0.1873 ± 0.0905 | 0.2235 ± 0.0983 |
| llama | D-zero | frequency6plus | 8 | 3959 | 0.2305 ± 0.0479 | 0.2546 ± 0.0454 |
| gemma | U | frequency1 | 11 | 2285 | 0.2300 ± 0.0467 | 0.2525 ± 0.0404 |
| gemma | U | frequency2_5 | 10 | 1745 | 0.2103 ± 0.0569 | 0.2549 ± 0.0529 |
| gemma | U | frequency6plus | 7 | 3322 | 0.1973 ± 0.0264 | 0.2259 ± 0.0244 |
| gemma | D-zero | frequency1 | 11 | 2285 | 0.2402 ± 0.0130 | 0.2717 ± 0.0269 |
| gemma | D-zero | frequency2_5 | 10 | 1745 | 0.2411 ± 0.0447 | 0.2795 ± 0.0444 |
| gemma | D-zero | frequency6plus | 7 | 3322 | 0.2427 ± 0.0158 | 0.2779 ± 0.0300 |

**한국어 설명:** 빈도는 원래 capped500 자료에서 같은 정답 템플릿이 등장한 횟수(1,2–5,6+)다. 이는 실제 운영 스트림 빈도가 아니며 예측기 특징으로 사용하지 않았다. 빈도 구간별 값은 조건부 보정 오차를 보여줄 뿐 드문 이벤트가 더 중요한 의미 오류라는 증거는 아니다. 시드 반복 관측 수를 독립 표본 수로 해석하면 안 된다.

## Cross-system support and limits

Raw D-zero results are genuine source-trained target probability evaluations in the existing frozen protocol; all-system tables include tuning systems, while paired held-out rows exclude HDFS/Spark/Linux. We also fit calibration maps on calibration-template rows of those three systems and transfer to held-out-system evaluation templates for **local-only Clotho and label-independent output/token baselines**. Source calibration templates are purged from target evaluation; all three transfer methods have identical retained examples.

| Model | Source-map method | Held-out systems | Brier | ECE | Purged repeated-seed observations |
|---|---|---|---|---|---|
| llama | Clotho | 10 | 0.2227 ± 0.0166 | 0.1545 ± 0.0111 | 13 |
| llama | Output entropy | 10 | 0.1726 ± 0.0186 | 0.1336 ± 0.0139 | 13 |
| llama | Token probability | 10 | 0.2315 ± 0.0182 | 0.1586 ± 0.0144 | 13 |
| gemma | Clotho | 9 | 0.2515 ± 0.0155 | 0.2126 ± 0.0406 | 20 |
| gemma | Output entropy | 9 | 0.1648 ± 0.0050 | 0.1236 ± 0.0214 | 20 |
| gemma | Token probability | 9 | 0.2596 ± 0.0153 | 0.1666 ± 0.0601 | 20 |

A clean transferred U/D calibrator is unavailable: the saved source-system U/D models can have trained on labels from the eventual held-out target. Treating those as clean source-calibration vectors would leak target-label dependency. No such comparison was fabricated. Target-reference template disjointness does not imply that every source-system template was unseen by U/D; inherited transductive label-free transforms and source overlap remain as in the original protocol.

**한국어 설명:** Clotho/출력 점수의 보정기를HDFS·Spark·Linux에서만 학습해 다른 시스템에 적용했고 겹치는 정답 템플릿은 평가에서 제외했다. U/D의 원본 보정용 점수를 만든 모델은 최종 대상 시스템 라벨을 이미 학습했을 수 있어, 동일 절차를 깨끗한 교차 시스템 보정이라고 주장하지 않았다. D의 보정 전 결과는 원래 source→target 예측의 확률 검증으로 해석한다.

## Implications

Frozen rankings cannot be used as risk probabilities without this semantic/probability check. Raw U/D have nontrivial calibration errors and local post-processing is not uniformly beneficial under template shift. Large target calibration sets and prior-label costs weaken any claim of a cost-free operational fix. Calibration is worth reporting as a limitation/diagnostic, not currently a reason for another GPU failure-predictor training grid. Capped500 labels predict stochastic five-sample strict failure, not final cached/AdaParser correctness, semantic harm or anomaly-detector damage. No natural/standard split or unsupported parser probability claim is made.

[Reliability and per-system figure](../results_phase3/cpu_remaining/calibration.pdf). Pooled reliability curves are descriptive with repeated seeds, without line-IID confidence bands; pooling can hide opposite system-specific calibration errors, so they need not equal the macro ECE table. All result records include Korean calculation, interpretation and limitation fields; fitted maps and probability vectors are persisted.
