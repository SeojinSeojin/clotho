# Decision memo after CPU investigations A/B/C

1. **Viability:** the scientific question remains plausible, but current matched outputs do not support a confirmatory claim that real-AdaParser review improves anomaly detection. HDFS/BGL constraints remain; OpenStack’s labelled VM subset has exactly zero real-parser error; existing Thunderbird parser rows have no alerts. Dataset-wide grouping errors alone do not establish an evaluable labelled error population.

2. **Dataset/parser:** Thunderbird is the strongest conditional expansion candidate:1241 GT types, observed real-AdaParser merging in the saved prefix, positive windows in frozen full-horizon splits, and substantially less observed GT vocabulary shift than BGL. Start with GT-only detector health; later compare actual Llama/Gemma AdaParser under a newly frozen coverage/candidate/adaptation protocol, never a silently filled prefix. OpenStack GT/Drain is only a small within-case positive-control option, not an independent-run benchmark or real-AdaParser repair study.

3. **Key uncertainty:** do parser errors and a useful detector coexist on the same anomaly-labelled held-out units, and can the allowed human intervention repair the representation the detector consumes? Real Thunderbird merging may require membership reassignment, which existing template-only review cannot do. Additional membership review costs are unknown.

4. **Redundant work:** do not repeat the completed HDFS/BGL feasibility/representation/review-cost work, severity-proxy audit, original predictor AUC grids, template-only fixed-ID grouping comparisons, or deterministic GT-vs-real-AdaParser comparisons on the perfect HDFS/OpenStack-labelled subsets. Do not retrain a stronger detector simply to overcome missing predictions, absent positives or zero parsing headroom.

5. **Next GPU:** one small DeepLog-style GT-only health run on the new frozen Thunderbird windows,10k normal train, all validation/test at original prevalence; keep architecture/settings inherited from the earlier pilot. Compare UNK-only/always-anomaly controls, AP, recall and low FPR. Do not expand to review policies unless health and matched-coverage/error-overlap gates pass. The old BGL pilot/config is preserved; this is a separately recorded option, not a retrospective modification of that split.

Prepared CPU artifacts: data/cpu_remaining/{full_horizon,pilot}/Thunderbird; source windows in results_phase3/cpu_remaining/Thunderbird_units.jsonl.gz. Configuration: configs/remaining_gpu_health.json. With GPU-enabled environment and persisted inputs restored, first command:

```bash
.venv/bin/python experiments/log_parsing_remaining_gpu_health.py train --detector deeplog --seed 0 --device cuda
```

No GPU run occurred here. Exact official DeepLog/LogBERT reproduction is not claimed.

6. **Priority:** consolidate the existing empirical findings first, including these feasibility, correction-mechanism and probability-calibration limits. A bounded GT-health pilot is scientifically reasonable; expensive broader parser inference remains conditional. Improved failure prediction alone will not fix missing downstream labels, incomplete coverage or unrepairable groups. Independent annotation of the existing semantic worksheet remains more informative than another ungrounded GPU model search.

**한국어 요약:** 현재 자료로 real AdaParser 교정의 실제 이상 탐지 이득을 입증할 수는 없다. Thunderbird는 향후 후보지만 양성 창이 있는 구간의 파서 출력이 없고, 현재 관측된 혼합 오류는 텍스트 검토만으로 복구되지 않을 수 있다. 다음GPU는GT 탐지기가 의미 있는 기준선을 만드는지 확인하는 제한된 건강 점검에만 쓰는 것이 타당하다. 논문의 중심은 기존 실증 결과와 이 한계를 정리하는 데 두고, 긍정적 효용을 전제한 추가 모델 탐색은 보류한다.
