# Downstream dataset selection — new CPU investigation A

Decision: no current dataset supports a strong confirmatory real-AdaParser review-to-anomaly-recovery claim. Thunderbird is the most defensible **future matched-coverage expansion candidate**, conditional on detector health and new predictions. A small matched OpenStack diagnostic exists but has zero real-AdaParser headroom and only four positive VMs from one injected case. No detector outcome was used to select a dataset.

## Criteria fixed before ranking

Protocol: configs/cpu_remaining_protocol.json. In order: (1) independently sourced anomaly labels with an explicit detection unit/quality audit; (2) complete, nonoverlapping train/validation/test units and evaluation positives; (3) original-ID aligned GT and actual parser outputs; (4) representation-visible errors and a correction mechanism able to repair them; (5) distribution shift, rarity, case independence and feasible inference cost. Passing these necessary conditions is not sufficient for anomaly recovery. Large error counts without labels do not make a suitable anomaly benchmark.

## Existing coverage and representation opportunities

| System | Available label evidence | Parser prefix lines | Prefix GT types | Effective GT types | Drain mixed groups | Ada mixed groups L/G | Ada residual lines L/G | Prefix-test OOV positions |
|---|---|---|---|---|---|---|---|---|
| Apache | not_provided_here | 50000 | 29 | 6.9 | 0 | 0/NA | 0/NA | 0.08% |
| BGL | confirmed_line_alerts | 50000 | 54 | 2.5 | 4 | 0/0 | 0/0 | 0.00% |
| HDFS | confirmed_block_labels | 50000 | 13 | 5.1 | 0 | 0/0 | 0/0 | 0.00% |
| HPC | not_provided_here | 50000 | 20 | 6.6 | 0 | 0/0 | 0/0 | 100.00% |
| Hadoop | job_labels_with_quality_warning | 50000 | 214 | 30.3 | 5 | 3/1 | 125/17 | 0.10% |
| HealthApp | not_provided_here | 50000 | 136 | 14.9 | 2 | 1/0 | 17/0 | 0.00% |
| Linux | not_provided_here | 23921 | 338 | 13.4 | 5 | 11/8 | 91/167 | 6.67% |
| Mac | not_provided_here | 50000 | 478 | 128.3 | 2 | NA/15 | NA/912 | 0.65% |
| OpenSSH | not_provided_here | 50000 | 27 | 5.2 | 1 | 1/1 | 11/10 | 0.16% |
| OpenStack | confirmed_four_injected_VM_IDs | 50000 | 45 | 12.9 | 1993 | 0/0 | 0/0 | 0.01% |
| Spark | not_provided_here | 50000 | 137 | 3.3 | 4 | 3/3 | 11/22 | 0.06% |
| Thunderbird | confirmed_line_alerts | 50000 | 540 | 85.3 | 21 | 15/16 | 1359/2446 | 0.27% |
| Zookeeper | not_provided_here | 50000 | 89 | 13.3 | 2 | 1/1 | 10/10 | 0.15% |

Effective types = exp(Shannon entropy of GT line frequencies), so dominance can make an apparently large vocabulary effectively small. OOV is measured under the saved prefix’s raw-line60/20/20 cuts, not a complete-session benchmark. All parser conditions shown have actual saved prefix outputs; Llama real AdaParser excludes Mac, Gemma excludes Apache. Full-horizon predictions remain missing for the labelled HDFS/BGL/Thunderbird settings; Linux covers all supplied rows but lacks anomaly labels. Full machine-readable rows, source hashes and Korean explanations are in dataset_selection.{json,csv}.

**한국어 설명:** 표의 혼합 그룹 수는 한 파서 그룹에 서로 다른 정답 이벤트가 들어간 횟수이고, 잔여 줄 수는 그룹마다 정답 다수 템플릿을 주어도 남는 오류다. 어휘/오류 수는 표현 변화를 뜻할 뿐 탐지 피해를 뜻하지 않는다. 라벨이 없는 시스템을 큰 오류 수 때문에 추천하지 않았다.

## Full-supplied-corpus event coverage

| Dataset | Supplied full rows | Full GT event types |
|---|---|---|
| BGL | 4631261 | 320 |
| HDFS | 11167740 | 46 |
| Hadoop | 179993 | 236 |
| OpenStack | 207632 | 48 |
| Thunderbird | 16601745 | 1241 |

**한국어 설명:** 전체 이벤트 수는 버전이 확인된 기존 감사 또는 해당Loghub2 ZIP의 템플릿 목록/빈도 합으로 계산했다. 전체 파일을 검사하지 않은 다른 시스템의 전체 이벤트 수는 추측하지 않았다. 제공 코퍼스와 원본 공개 코퍼스의 줄 수가 다를 수 있다.

## Thunderbird: full supplied-corpus audit

Scanned all 16,601,745 Loghub-2.0 rows and 1,241 annotated event types. This is the supplied16.6M corpus, **not** the original211M-row Thunderbird release. Raw alert lines=554; saved-prefix alerts=0. All554 alerts lie outside existing predictions. Every prefix GT row and every raw/structured row join was verified; timestamps have zero backwards steps, no missing epoch fields and no NUL truncations.

| Split | Normal windows | Anomalous windows | GT types | Unseen types vs full train | Unseen positions | Units missing predictions |
|---|---|---|---|---|---|---|
| train | 497945 | 107 | 1085 | 0 | 0.000% | 495552 |
| validation | 165869 | 147 | 824 | 138 | 3.720% | 166016 |
| test | 165850 | 168 | 597 | 52 | 2.712% | 166018 |

Windows contain20 nonoverlapping lines, any alert is positive, last short window retained; two boundary-crossing windows are recorded and purged. Test prevalence=168/166018=0.1012%, versus BGL6.4172%. Test unseen-event positions=2.7124% versus BGL78.9666% (existing BGL result reused). Saved Thunderbird windows are all normal; later error rates cannot be inferred from prefix errors.

Test positive windows form 167 consecutive-positive runs (longest=2 windows). These are temporal descriptors, not verified independent failures. Against **all normal training events**, all168 positive test windows and24385/165850 normal test windows contain an OOV event. This does not measure detector accuracy. Future evaluation needs AP, recall at low FPR, prevalence controls and time-block uncertainty; accuracy alone would be misleading.

**한국어 설명:** 전체 제공 파일을 한 번 순회해 첫 필드가 -가 아닌554줄을 확인했다.20줄 창으로 묶고 경계 창을 제거한 뒤 시험 양성은168개이며, 시험 이벤트 위치의2.7124%가 전체 훈련 어휘 밖이다. 기존 BGL보다 어휘 변화가 작다는 자료 특성이지, Thunderbird 탐지 성능이나 파싱 교정 효과가 더 좋다는 결과는 아니다. 이후 구간의 실제 파서 출력은 없다.

## OpenStack: verified labels, then zero real-parser headroom

The original archive contains three case files and an explicit label file listing four anomalous VM UUIDs. Exact raw-row hashes map every207632 supplied Loghub-2.0 row back to an original case file; no unmatched row is guessed. Original normal files contain188 extra rows omitted from Loghub-2.0, including four explicit VM-tag occurrences. Completeness below is relative to the supplied corpus, not reconstruction of all omitted original context.

Explicit [instance: UUID] tags identify 2069 VMs:1871 normal-source-only,4 labelled positive,194 other-case VMs left unknown. Only51549/207632 rows carry explicit VM tags; untagged context is not assigned to a VM by guesswork. Of these,335 normal and4 positive tag traces are completely inside saved parser coverage. All four positives share one abnormal-case file. Unknown VMs are not silently made negative.

| Diagnostic split | Normal VMs | Positive VMs | Tagged rows | GT types | Unseen positions |
|---|---|---|---|---|---|
| train | 208 | 2 | 5227 | 23 | 0.00% |
| validation | 70 | 1 | 1775 | 23 | 0.00% |
| test | 57 | 1 | 1448 | 23 | 0.00% |

A fixed UUID-hash normal60/20/20 split and positive2/1/1 allocation preserve VM/line disjointness. This is a stratified within-case diagnostic, not a chronological or independent-run generalization evaluation. SHA-based IDs and exclusions are saved in OpenStack_matched_VM_manifest.json.gz. One positive per validation/test partition makes F1 unstable; it is not a strong confirmatory benchmark.

| Parser | Label | VMs | VMs with wrong text | Wrong rows | Rows wrong after global text oracle |
|---|---|---|---|---|---|
| Drain | 0 | 335 | 335 | 8350 | 3011 |
| Drain | 1 | 4 | 4 | 100 | 40 |
| llama/base | 0 | 335 | 335 | 5011 | 3011 |
| llama/base | 1 | 4 | 4 | 64 | 40 |
| llama/strong | 0 | 335 | 335 | 3680 | 3011 |
| llama/strong | 1 | 4 | 4 | 48 | 40 |
| llama/AdaParser | 0 | 335 | 0 | 0 | 0 |
| llama/AdaParser | 1 | 4 | 0 | 0 | 0 |
| gemma/base | 0 | 335 | 335 | 5004 | 3011 |
| gemma/base | 1 | 4 | 4 | 64 | 40 |
| gemma/strong | 0 | 335 | 335 | 4016 | 3011 |
| gemma/strong | 1 | 4 | 4 | 52 | 40 |
| gemma/AdaParser | 0 | 335 | 0 | 0 | 0 |
| gemma/AdaParser | 1 | 4 | 0 | 0 | 0 |

**한국어 설명:** 원래 줄 ID로 정답과 파서 문자열을 비교하면, 선택된339개 VM 추적의8450줄에서 Llama/Gemma real AdaParser의 오류는 모두0이다. 따라서 이 입력 표현으로 GT와 real AdaParser를 비교하면 같은 시드의 결정적 탐지기는 같은 입력을 받는다. 전체 접두사의 다른 문맥 줄에 오류가 있다는 사실을 이 라벨 모집단의 교정 기회로 바꾸어 말할 수 없다. Drain 계열 오류는 존재하지만 단일 장애 실행의 작은 진단이라는 한계가 남는다.

## Hadoop and unlabelled systems

Original labels cover55 jobs (11 nominal normal/44 nominal anomalous), all55 directories and978 container logs. The repository’s attempted README label-file URL is absent; the label file was recovered from the original ZIP. [Official label-quality warning](https://github.com/logpai/loghub/issues/56) remains unresolved; proposed corrections were not applied as verified truth.

The provided Hadoop corpus has179993 rows versus394310 original archive rows; these versions are not silently equated. Exact original raw-row→job matching gives saved-prefix counts {'ambiguous_jobs': 267, 'unique_job': 49733}. Repeated lines with multiple candidate jobs are retained as ambiguous; no labels or session closure are invented. Hadoop therefore needs label validation and a complete provenance-preserving job join before anomaly evaluation. Only11 nominal normal jobs also constrain independent train/validation/test sampling.

**한국어 설명:** Hadoop의55개 작업 라벨을 세면 정상11/이상44이지만, 공개된 라벨 오류 경고가 있어 이를 검증된 정답으로 간주하지 않는다. 동일 문장이 여러 작업에 나타나면 줄의 작업 ID를 추측하지 않았다. 나머지 Spark/HPC/Zookeeper/Linux/Mac/HealthApp/Apache/OpenSSH에는 현재 제공 자료에서 독립 이상 라벨이 확인되지 않아 파서/검토 연구에는 쓸 수 있어도 이상 탐지 효용의 주 분석으로는 부적합하다.

Label provenance: [Loghub catalogue](https://github.com/logpai/loghub), [Thunderbird](https://github.com/logpai/loghub/tree/master/Thunderbird), [OpenStack](https://github.com/logpai/loghub/tree/master/OpenStack), [Hadoop](https://github.com/logpai/loghub/tree/master/Hadoop). Source texts/revision and archive hashes are saved under sources/. HDFS_v2/v3 are different variants without matched current outputs and are not substituted. No severity level, error keyword or parser correctness label is treated as an anomaly label.

## Defensible priority

Preserve the existing prepared HDFS/BGL GT controls. For a new GPU health experiment, use the separately prepared Thunderbird GT-only full-horizon configuration; its lower observed vocabulary drift and available positive windows justify checking detector health, not claiming utility. Real AdaParser on broader Thunderbird coverage is the most relevant subsequent parser condition **only after** a fixed coverage/provenance plan and evidence of errors on the labelled target population. OpenStack GT/Drain could be a cheap mechanistic sensitivity check, but real-AdaParser correction is mathematically uninformative on its selected VM subset. Prioritize consolidating existing findings rather than expensive review/detector grids.
