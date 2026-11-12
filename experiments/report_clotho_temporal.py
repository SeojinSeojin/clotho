"""Generate technical/Korean temporal reports from validated saved results."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
import pandas as pd,numpy as np
from cpu_research_common import load,save,sha
R=EXP/'results_phase3/clotho_temporal';REP=EXP/'reports'
def table(df):
 def cell(x):return 'undefined' if pd.isna(x) else (f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x))
 return '\n'.join(['|'+'|'.join(map(str,df.columns))+'|','|'+'|'.join(['---']*len(df.columns))+'|']+['|'+'|'.join(cell(x) for x in row)+'|' for row in df.itertuples(index=False,name=None)])
def main():
 v=load(R/'validation.json');assert v['status']=='passed';d=pd.read_csv(R/'metrics.csv');p=load(EXP/'configs/clotho_temporal_protocol.json')
 s=d.groupby(['model','method','cohort']).agg(AUROC_mean=('AUROC','mean'),AUROC_sd=('AUROC','std'),AUPRC_mean=('AUPRC','mean'),AUPRC_sd=('AUPRC','std'),n=('n','mean'),failures=('failures','mean'),prevalence=('prevalence','mean'),prevalence_sd=('prevalence','std')).reset_index();s.to_csv(R/'summary_v2.csv',index=False)
 c=pd.read_csv(R/'paired_contrasts.csv');f=pd.read_csv(R/'fit_metadata.csv');allmodels=[]
 for model in ('llama','gemma'):
  for method in p['methods']:
   x=s[(s.model==model)&(s.method==method)].set_index('cohort');allmodels.append(dict(model=model,method=method,past_AUROC=x.loc['earlier_holdout','AUROC_mean'],future_AUROC=x.loc['future','AUROC_mean'],future_minus_past=x.loc['future','AUROC_mean']-x.loc['earlier_holdout','AUROC_mean']))
 gap=pd.DataFrame(allmodels);gap.to_csv(R/'past_future_descriptive.csv',index=False)
 audit=[]
 for path in sorted((EXP/'data/stream').glob('*.json')):
  if path.stem.endswith('_same'):continue
  a=load(path)
  if not all(k in a for k in ('first','gt','n_lines')):continue
  earlier=[q for q in a['first'] if q<10000];future=[q for q in a['first'] if q>=10000];past_gt=set(a['gt'][:10000]);new=sum(a['gt'][q] not in past_gt for q in future);verified=path.stem in ('HDFS','BGL','Thunderbird')
  audit.append(dict(system=path.stem,source_rows=a['n_lines'],earlier_calls=len(earlier),future_calls=len(future),new_origin_calls=new,timestamps_verified=verified,eligible=verified and len(earlier)>=100 and len(future)>=20))
 pd.DataFrame(audit).to_csv(R/'feasibility_inventory.csv',index=False)
 feasibility='''# Clotho temporal feasibility and novelty audit

## Decision

A narrow authentic chronological evaluation **is executable** with saved Thunderbird artifacts. Execute it; do not invent timestamps or sort shuffled benchmark suites into a prospective claim. No exact temporal failure-prediction job or artifact was running/completed at launch (local processes plus private-Hub inventory checked). No density, PCA, covariance or acquisition search is reopened.

The eligible target is actual first fine-Drain cache-miss parser calls within original source rows1–50000. The first10000 rows contain531 candidate calls; the remaining40000 contain99 calls. Budget100 actual earlier inputs is feasible. Future calls include88 new-to-origin and11 previously observed-template calls, spanning95 distinct GT types (86 new-origin,9 seen-origin). All86 GT types first observed after the origin appear among these calls. This is one system, two LLMs and five reference seeds, not independent system replication.

## Sources read and previous research

AGENTS and latest PROGRESS first; paper_numbers, acquisition audit/pilot/Korean explanation; mechanism/component/covariance/PCA/Gaussian reports and explanations; original PCA/GMM/CV/acquisition implementation, phase3 seeded harness, transfer/paired-transfer, stream/triage/U/gate scripts and reports; saved stream/hidden/generation manifests and raw timestamp audits. The preceding acquisition study found no robust advantage over archived acquisition across both LLMs, so it recommended stopping policy development on that benchmark. That stop remains in effect.

|Existing evaluation|What it measures|Why it does not answer this question|
|---|---|---|
|Capped500 standard / GT-template-disjoint|Randomly sampled and shuffled input sets; held-out GT types|Template separation is not chronological emergence|
|Natural500|Uniform reservoir over the full supplied corpus|Sampling/row order is randomized, not an earlier-reference future-test protocol|
|Source/target transfer and paired transfer|Held-out systems, optional target labels|Target reference labels are not restricted to the actual past|
|TRIAGE/E review curves|Review of cached groups, sometimes final group sizes/majority GT|Retrospective outcomes/weights can use later group members; no same strictly earlier-label comparison|
|Creation-time gates, U/stream predictors|Sequential calls/group creation|Source-suite target references span the corpus; some PCA fits see all stream states|
|Full-horizon downstream feasibility|Authentic chronology, anomaly/event labels|Anomaly-alert labels are not LLM parsing correctness; parser outputs beyond the prefix are missing|

## Authentic chronology and artifact coverage

The stream builder reads the first50000 **contiguous original LineIds** from full structured files and stores first raw positions, content and GT per source row. Its first-call order comes from sequential Drain creation, not final cluster size. Original raw scans verified timestamp monotonicity and raw/structured row alignment. HDFS uses raw YYMMDD/HHMMSS; BGL/Thunderbird use the raw epoch field. This task reuses those validated scans; raw archives are not currently local and no fresh raw scan was performed. Exact calendar endpoints are absent from these compact caches: figures use authentic source positions, not fabricated elapsed clock time. Timestamp ties retain original source order.

|Verified raw source|Scanned rows|Missing timestamps / backward steps|Saved earlier/future first calls at row10000|
|---|---:|---|---|
|HDFS|11167740|0 /0|13 /1|
|BGL|4631261|0 /0|11 /40|
|Thunderbird (supplied Loghub2 corpus)|16601745|0 /0|531 /99|

Thunderbird is the supplied16.6M corpus, not the historical211M release. Its archive SHA256 is `ee76bfb6a75126dbf137dcac00a3ebe0c61f90963334354a67c09df1ca1bf831`; raw SHA256 `a276905e7517a7d2c1d658490a0f27919c666045c3454ba695a5e77f79e9cbfe`. The prior inventory records `original_prefix_GT_joins_verified=true`, zero missing epoch fields, zero backward steps, no NUL truncation. HDFS/BGL source audits are `results_phase3/cpu_research/feasibility_{HDFS,BGL}.json`; Thunderbird inventory and validation are in `results_phase3/cpu_remaining/`. The Thunderbird README provenance copy is preserved in its `sources/` directory. Other systems have saved row positions but no equivalent timestamp verification in the inspected artifacts; they are ineligible here.

'''+table(pd.DataFrame(audit))+'''

## Exact saved inputs and correctness labels

- `data/stream/Thunderbird.json`:50000 GT row entries,630 first positions and input contents, GT template strings. Final group sizes/families are not predictor features.
- `data/stream_hidden/llama/Thunderbird_layer16.npy`:630×4096; Gemma layer21:630×3584. Same input ordering, all finite. These are last-prompt-token hidden states under rules_template.
- `data/stream_gen/{llama,gemma}/Thunderbird.json`:630×10 uncorrected parsed samples, separately greedy output/token probabilities/AdaParser results. Only samples5–9 determine the unchanged strict majority label; unused outputs are not prediction features. `None` is a parsing failure, not missing generation. GT is the first input's template, not the final group's majority GT. Engine generation seed1 differs from some old suite caches; no old scores are substituted.
- Candidate labels are evaluator/oracle-private. The acquisition/model-selection API rejects access to any input not already acquired from the earlier pool. GT template IDs are used only for cohort definitions and validation after fitting.

## Frozen protocol, feasibility boundary and costs

`configs/clotho_temporal_protocol.json` was committed in a20f19e before new correctness outcomes were inspected. A metadata-only60% boundary had only16 future calls; the fixed10000-row boundary was selected for531/99 call feasibility, not predictive performance. There was no search over temporal cuts or model configurations. No alternative cutoff is evaluated after outcomes.

100 actual past references include10 seeded random warmup calls, then the original half-entropy/half-diversity Clotho acquisition restricted to earlier candidates. No synthetic references lack chronological provenance. The same ordered100 are passed to every method. Past-only Clotho PCA fits all531 earlier candidate covariates; LR/shared PCA fits only100 references. A separately labeled original all630 PCA scorer uses unlabeled future covariates but the same frozen reference order. Neither uses future labels. Full details and findings are in the generalization report.

Archived benchmark acquisition cannot be faithfully replayed here: its IDs/reference histories belong to different shuffled suites. Comparing that history against a newly constructed chronological policy would not preserve equal information or populations. Therefore no acquisition/diversity arm is run and future-diversity benefit remains unanswered. No method is selected using final future AUROC.

All states and outputs for the eligible target already exist: new GPU calls=0, generation=0. Every method requires531 candidate states for the common Clotho acquisition and99 future states. Saved single-layer extraction timings imply roughly22.0s/31.6s for the past Llama/Gemma candidates and4.1s/5.9s for future inputs on the original L40S; these are historical estimates, not current timing measurements. Equal input-label budget100 corresponds to500 repeated correctness checks if implementing this five-output oracle literally; it is not an experiment with100 single-generation outputs. Costs of GT/human template creation are not measured.

For a broader authentic test, full-horizon GT exists but aligned parsing outputs and hidden states beyond the saved50000-row prefix do not. A new chronological routing pass must first enumerate actual cache-miss inputs; their count and token lengths are not saved, so exact GPU hours cannot honestly be supplied. For each such call the exact missing unit is one state at each frozen model/layer and at least5 uncorrected parser outputs per model. If strict compatibility with existing stored10-output files is needed, use10. HDFS/BGL are short of100 earlier calls by87/89; adding real earlier raw inputs would change the cache-miss target and require new labels/states. No such substitution or workload is launched.

## Research gap and decision

Run the bounded Thunderbird pilot. It can falsify unqualified transfer of the shuffled-benchmark ranking, but cannot establish multi-system prospective reliability, real deployment cost, or causal effects of template novelty. Preserve all tuning stop decisions. Do not claim a new method merely from this evaluation.
'''
 (REP/'clotho_temporal_feasibility.md').write_text(feasibility)
 main=s[s.cohort=='future'][['model','method','n','failures','prevalence','AUROC_mean','AUROC_sd','AUPRC_mean','AUPRC_sd']]
 cohorts=s[s.cohort.isin(['new','seen'])][['model','method','cohort','n','failures','AUROC_mean','AUROC_sd','AUPRC_mean']]
 windows=s[s.cohort.str.startswith('lines')][['model','method','cohort','n','failures','AUROC_mean','AUPRC_mean']]
 technical='''# Prospective failure prediction on authentic Thunderbird chronology

## Decision and strongest observation

**The old simple-control advantage does not automatically transfer to actual future calls.** On the predefined Thunderbird prefix, past-only Clotho has higher mean future AUROC than frozen LR and shared Gaussian for both LLMs. Conditional paired intervals include zero. LR/shared score better on earlier held-out calls and deteriorate later; Clotho does not deteriorate on the aggregate future population. This challenges benchmark extrapolation, not the decision to stop density tuning. It does not establish a new winning predictor.

This is an executed one-system exploratory evaluation using authentic source chronology, not a proposed experiment or another random template-disjoint test. There was no exact job/result at launch;10 CPU jobs completed, plus one deterministic replay. No new GPU, outputs, hidden-state layers or feature search. [Feasibility audit](clotho_temporal_feasibility.md) documents provenance and exclusions.

## Frozen evaluation protocol and leakage boundary

Protocol commit a20f19e predates examination of new failure outcomes. Rows1–10000 form the available history; evaluate actual first-group parser calls in rows10001–50000. No refitting/update after the origin. Past candidates531, references100 actual earlier inputs, future99 identical across all methods/seeds. Five seeded uniform10-call warmups followed by original balanced Clotho acquisition. All methods share its acquired100 examples/order; reference sets differ across LLMs/seeds because acquired-label feedback differs. There are no synthetic warmups; this budget composition differs from old benchmark100=90 real+10 synthetic and is not silently compared as the same training population.

Let c_i be the number of strict GT template matches in uncorrected samples5–9. FAIL is y_i=1[c_i≤2]. The acquisition oracle exposes only already chosen earlier inputs. Core CV receives the continuous fraction c_i/5, as in the original harness; LR/shared receive its binary failure label. Future ground truth is used by evaluation only. GT labels define novelty after model fitting, never acquisition or predictor inputs. The history includes **all raw rows**, not only acquired templates: new-to-origin means GT absent in all first10000 rows. This is first observation in the supplied trace, not proof of a globally newly created software event.

- **clotho_past:**original seeded harness, adaptive PCA dimension,3-fold ordinary reference CV, PASS-only GMM, mixture adaptation, entropy ridge. Normalization remains original raw PCA. PCA fitting and entropy/acquisition covariates are restricted to531 earlier calls, and labels to acquired references. This is a necessary prospective adaptation, not literal original all-input preprocessing.
- **lr:**previously frozen reference-only raw PCA16, orthogonal basis, full-reference covariance whitening, balanced C1 logistic regression(tol1e−10,max3000). Final fit uses100 references only.
- **shared:**exactly the LR reference PCA16 coordinates; one mean per class; shared covariance 0.5(S_PASS+S_FAIL)+α Cov_reference. Score log p(x|FAIL)−log p(x|PASS); no empirical priors. Frozen αL=0.00967120956372625, αG=0.0038248306387871998, inherited per-model median from the earlier Gaussian study **excluding Thunderbird**, all other systems/splits/seeds. No new temporal outcomes tune α. This external calibration avoids importing the earlier target transductive covariance.
- **clotho_future_covariates:**same earlier ordered reference100, original all630 `PCA.fit_transform`, original downstream CV/GMM/ridge. Explicitly transductive: it sees unlabeled future states. It replays the past-only reference history, so it isolates future-preprocessing access conditional on that history, not the full original transductive acquisition algorithm. An exactly original end-to-end algorithm could change reference choices and would no longer hold training examples constant.

An initial unvalidated run used `.fit().transform()` for this fourth arm. Code review restored exact original `.fit_transform()` before examining its outcomes, then reran all10 jobs; initial files remain in `clotho_temporal_prevalidation`. No old validated artifacts were overwritten. No tolerance was relaxed.

## Main future results

Failure-positive AUROC; AUPRC is sklearn average precision. Values are mean±sample SD over5 reference seeds, not5 independent systems. The future inputs and their label prevalences do not vary with seed. All methods retain all10 model/seed jobs.

'''+table(main)+'''

High AUPRC must be read against prevalence0.8687/0.7576; an all-positive cohort has AP1 without any discrimination. Future Llama has only13 PASS examples, Gemma24. No empirical calibrated risk threshold is selected.

## New-template and seen-template cohorts

'''+table(cohorts)+'''

88 new-origin calls cover86 GT types;11 seen-origin calls cover9. Llama seen-origin has only1 PASS; Gemma has none, so seen AUROC is undefined for every method. AP1 for Gemma is a degenerate cohort, not perfect prediction. This is the strongest obstacle to attributing errors to novelty. New-origin Clotho AUROC is0.6566/0.7078; LR0.5469/0.6003; shared0.5577/0.5919. Failure information remains predictive in some unseen types, but uncertainty and limited PASS support prevent a broad reliable-transfer conclusion. The cohort is defined relative to the frozen origin; later calls on an already emerged novel type remain new-origin, not falsely called a fresh first-ever appearance at each call.

## Earlier holdout versus future: descriptive shift

Earlier holdout is the431 unacquired past calls per seed; its labels do not enter fitting or selection. Its covariates were available to Clotho acquisition/PCA, so it is not a completely untouched covariate population. Adaptive selection removes particular inputs from this earlier holdout; the resulting past/future change is conditional on that selection and cannot isolate temporal drift alone. Comparisons across time have different examples/GT/prevalences and are not paired-input causal effects.

'''+table(gap)+'''

LR drops0.1261 Llama and0.1160 Gemma; shared drops0.1371/0.1230. Clotho rises0.0465/0.0314. Earlier failure prevalence averages0.8037/0.6056, future0.8687/0.7576. These changes do not prove the simpler methods fail because of template novelty: template composition, input difficulty, class balance and other temporal drift change together.

## Paired method differences and uncertainty

Conditional GT-template cluster percentile bootstrap4000 draws(seed20261112), resampling the95 future GT types with all reference seeds/methods retained jointly. New/seen strata use86/9 types. No line/seed independence assumption and no system-population CI from one system. Temporal dependence between different templates is not modeled, so these intervals are exploratory conditional uncertainty, not formal cross-time significance. One-class bootstrap draws are invalid, counted, never replaced by invented scores. Only2652/4000 seen-Llama replicates have both classes; all seen-Gemma replicates are invalid.

'''+table(c[c.cohort.isin(['future','new'])])+'''

Future LR−Clotho differences are−0.0859[-0.1920,0.0183] and−0.0984[-0.2320,0.0374]. Shared−Clotho is−0.0948[-0.2094,0.0186] and−0.1071[-0.2437,0.0302]. All overlap zero; seed wins do not establish significance. Original future-covariate access changes mean AUROC by−0.0134/+0.0076, also inconclusive. It does not explain away this observed rank reversal.

## Temporal windows

Windows are original source positions supported by prior raw timestamp audits. Equal10000-row spans are not equal elapsed durations. Cells with only one class retain undefined AUROC; no boundary is moved after observing outcomes.

'''+table(windows)+'''

The last two Llama windows have8 FAIL and0 PASS each, so a monotonic deterioration claim is impossible. Gemma's last window has7 FAIL/1 PASS: Clotho0.9143 versus LR0.2571 is a conspicuous but extremely small-sample counterexample, not independent proof of nonlinear density superiority. Global AUROC cannot be read as an average of window AUROCs, because cross-window pairs also contribute.

## Computation and references

All candidates use saved last-prompt states; there is no output-derived feature. Common acquisition uses531 states, inference99;500 actual generation correctness checks per100-input budget if the five-output oracle is implemented literally. Existing generation stores10 outputs per630 input per model, shared across all simulated seeds. Labels are revealed only after selection. Diversity was not compared; archived shuffled-suite histories cannot be treated as past labels for this stream. No claim that diversity improves future prediction.

'''+table(f)+'''

Recorded Clotho times include adaptive acquisition/CV/scoring; the transductive control replays acquisition. LR/shared final fit times exclude common acquisition, PCA candidate feature extraction and label acquisition. Thus their tiny final-fit timings are not end-to-end deployment cost. Historical state extraction averages0.0414s/call Llama,0.0594s Gemma; CPU timing here is from current host and model inference timings are from the previous L40S run. No new runtime comparison is presented as like-for-like GPU throughput.

## Validation, examples and reproducibility

Validation passed:10 exact future populations; shared ordered references per seed;100 past-only oracle reveals; future labels never read by acquisition/CV; past-only PCA IDs checked; transductive IDs explicitly marked; finite scores and no missing generation rows; failure score direction checked; independent metric reconstruction; hashes of all used metadata/generations/states/protocol; full Llama seed0 rerun gives identical reference IDs and maximum score error0 for all4 arms (tolerance1e−10).

Code: `log_parsing_temporal.py`, `analyze_clotho_temporal.py`, `report_clotho_temporal.py`; pinned protocol/environment in configs. Results: `results_phase3/clotho_temporal/{metrics,summary_v2,paired_contrasts,fit_metadata,sampling_manifest,past_future_descriptive,feasibility_inventory}.csv`, jobs with all630 saved scores and labels, oracle histories/PCA fitting IDs, `validation.json`, deterministic replay, representative case JSON. Original `summary.csv` contains first-seed earlier-holdout count/prevalence fields; use **summary_v2.csv**, which averages these varying past populations correctly. Future counts/prevalence are identical in either. AUROC/AP aggregation is unchanged. Scripts reproduce summary_v2.

![AUROC cohorts](../results_phase3/clotho_temporal/figures/cohorts.png)

![Temporal windows](../results_phase3/clotho_temporal/figures/temporal_windows.png)

In seed0, a future Llama FAIL at original LineId40772 has LR risk percentile0.303 versus Clotho0.929; Gemma FAIL at26089 has0.010 versus0.970. Conversely, Gemma PASS at15905 has LR0.040 versus Clotho0.869, a counterexample to universal Clotho rank quality. Percentiles rank only the99 future inputs. The selected input and GT strings are retained privately. These are descriptive, not threshold-based mistakes. Representative cases in `representative_cases.json` were selected by a predefined descriptive rule (seed0 earliest PASS/FAIL, largest percentile disagreement within each label), never for training or tuning. They are risk ranks, not calibrated false-negative decisions. Cases are discussed in the Korean explanation.

## Threats to validity and research decision memo

1. **Does random template-disjoint predict real chronological performance?** It is not established. Here the earlier simple-control ordering reverses in mean, but different input population, warmup composition, generation seed and labels prevent attributing the reversal solely to chronology. This is an empirical counterexample to automatic extrapolation, not a benchmark correlation study.
2. **Does Clotho deteriorate as new templates emerge?** Not in aggregate here: future AUROC is higher than its past holdout. Seen/new discrimination is underidentified by zero/one seen PASS examples. No causal novelty effect.
3. **Do simple alternatives consistently improve future behavior?** No. Their predefined means are lower on both models, with inconclusive paired intervals. This does not prove density complexity is useful generally or reopen its tuning branch.
4. **Does reference diversity help at equal cost?** Unknown; not tested because an archived equal-information chronological replay is unavailable. No speculative new policy is introduced.
5. **What remains unresolved?** Predictive ranking under genuine drift with enough positive and negative outcomes across independent systems and meaningful future duration. The present50000-row prefix is a selected tiny portion of16.6M source rows. Cache-miss calls differ from all raw lines; unseen-template GT diagnostics do not encode known timestamps of actual template creation.
6. **Method or empirical contribution?** A careful empirical temporal robustness finding is currently strongest. No new method hypothesis is supported well enough. Stop density/PCA/covariance/acquisition tuning as requested. **One next experiment only:** independently collect/cache a preregistered longer chronological horizon on additional timestamp-verified systems, then rerun these exact frozen predictors and reference budget without tuning. First enumerate chronological cache-miss inputs to specify exact new extraction/generation cost. That collection requires separate resources and was not silently launched.
'''
 (REP/'clotho_temporal_generalization.md').write_text(technical)
 ko=r'''# 실제 시간 순서에서 Clotho가 파싱 실패를 예측할 수 있는가?

## 이번 연구에서 새로 확인한 것

무작위 템플릿 분리 실험에서 좋았던 간단한 모델이 실제 미래 로그에서도 우월하다고 바로 결론 내릴 수 없다. 이번 Thunderbird 탐색 실험에서는 **Clotho의 미래 AUROC 평균이 LR과 공유 공분산 Gaussian보다 두 LLM 모두 높았다.** 하지만 차이의 신뢰구간에0이 포함된다. 따라서 복잡한 밀도 모델을 다시 개발하자는 결과도, Clotho의 확실한 우월성을 입증한 결과도 아니다. 지금 필요한 것은 새로운 점수함수 탐색이 아니라 독립적인 시간 순서 검증이다.

## 1. 기존 자료로 진짜 시간 순서 평가가 가능한가?

기존 capped500은 템플릿별 표본 추출 후 섞었고, natural500도 전체 파일에서 무작위 추출했다. 그 배열 순서는 시간이 아니다. 반면 저장된 stream은 원본 파일 첫50000줄의 연속 LineId를 읽고, Drain이 새 그룹을 만들 때 첫 로그를 LLM에 보냈다. Thunderbird 원본16601745줄의 epoch 시간에 누락과 역행이 없다는 이전 검증 및 원본/GT 대응 검증을 재사용했다. 새로 원본 전체를 스캔한 것은 아니다. 압축된 캐시에 실제 달력 시각 범위가 없으므로 ‘10000줄 구간’을 몇 시간이라고 바꾸지 않았다.

HDFS/BGL도 시간 순서 근거는 있지만 첫10000줄의 실제 호출이13/11개뿐이다. 100개 과거 참조를 요구하는 동일 실험에는 부족하다. Thunderbird만 과거531개, 미래99개 호출이 있어 실행했다. 다른 시스템을 임의 행 순서로 추가하지 않았다. 처음60%를 과거로 두면 미래 호출16개뿐이어서, **정답을 보지 않은 메타데이터 단계**에서10000줄 경계를 정하고 고정했다. 결과가 좋은 경계를 찾지 않았다.

여기서 평가 단위는 전체 원본 로그가 아니라 캐시 미스에 따른 새 Drain 그룹의 첫 파싱 호출이다. 기존 그룹으로 들어간 이후 로그의 파싱 정답을 생성하지 않았고, 미래 그룹 다수결 GT를 개별 입력의 정답으로 대신하지 않았다.

## 2. 정확히 무엇을 바꾸고 무엇을 고정했는가?

Llama16층/Gemma21층의 저장된 입력 마지막 토큰 상태만 사용했다. 새 생성, GPU 추출, 출력 특징은 없다. 시드0–4마다 과거에서 실제10개 입력을 무작위로 선택해 정답을 공개하고, 기존 Clotho의 절반 불확실성/절반 다양성 방식으로100개까지 획득했다. 매 행동은 아직 모르는 정답을 보기 전에 결정한다. 각 비교 모델은 **동일한 순서의100개 과거 입력과 정답**을 사용하고 동일99개 미래 입력에 점수를 매긴다. 시드와 LLM 간에는 선택된 참조가 다를 수 있다.

정답은 저장된10개 비수정 생성 중5–9번째만 사용한다. 입력의 원래 GT와 정확히 일치한 수를 c라고 할 때

$$y_{FAIL}=\mathbb{1}[c\le2],\qquad c\in\{0,1,2,3,4,5\}.$$

AdaParser 수정 결과나 이후 그룹 구성원 다수결 정답을 쓰지 않는다. 이 정의는 이전 fair_leakfree와 같다. 다만 입력 모집단과 stream 생성 시드1은 이전 표본 실험과 다를 수 있다. 예전100개 예산은 실제90개+인공 초기10개였지만 이번100개는 모두 시간 순서가 확인되는 실제 과거 입력이다. 이 차이를 감춘 전후 성능 비교는 하지 않는다.

비교한 방법은 다음과 같다.

- **Clotho 과거 전용:** 원래 PASS 밀도 GMM, 내부 CV/차원/성분/정규화 절차를 유지하되 PCA와 획득 후보 공변량은 과거531개로 제한한다. CV는 이미 획득한 정답만 본다.
- **LR:** 참조100개에만 raw PCA16을 맞추고 참조 공분산으로 whitening한 뒤 balanced C1 로지스틱 회귀를 맞춘다. 이전 연구에서 고정한 설정이다.
- **공유 Gaussian:** LR과 똑같은 PCA16 좌표에서 PASS/FAIL 평균과 공유 공분산을 추정한다. 위험 점수는 log p(x|FAIL)−log p(x|PASS), 클래스 사전확률은 넣지 않는다. 정규화 α는 이전 연구의 Thunderbird를 제외한 시스템에서 미리 고정한0.0096712/0.0038248이며 이번 미래 정답을 보며 조절하지 않는다.
- **미래 공변량 Clotho:** 같은100개 과거 참조를 재생하되 원래처럼 미래를 포함한630개로 PCA를 fit_transform한다. 미래 정답은 보지 않지만 미래 입력 분포를 보는 transductive 설정이다. 실제 미래를 아직 모르는 배포와 구분한다. 획득 과정까지 원래 transductive 방식으로 다시 돌리면 참조 자체가 달라지므로 이번에는 하지 않았다.

원래 benchmark의 획득 기록은 다른 무작위 표본 ID에 대한 것이다. 이를 이 stream의 ‘과거에서 선택한 기록’으로 둔갑시키지 않았다. 따라서 다양성 정책의 미래 이득은 이번에 검증하지 않았다.

## 3. AUROC와 AUPRC는 무엇이며 어떤 수치가 나왔는가?

위험 점수가 높을수록 FAIL이다. AUROC는 임의 FAIL의 점수가 임의 PASS보다 높을 확률이며 동점은0.5로 센다. AUPRC는 sklearn average precision으로 계산한다. 평균과 표준편차는 같은 미래 자료에서 참조를 바꾼5개 시드에 대한 값이다. 독립된5개 시스템이라는 뜻은 아니다.

'''+table(main)+'''

미래 Llama는99개 중86개 실패(86.87%), Gemma는75개 실패(75.76%)다. 예를 들어 Llama LR의 AP0.9021은 높아 보이지만 실패 유병률 자체가0.8687이다. AP를 단독으로 보고 뛰어나다고 말할 수 없다. Clotho AUROC0.6642/0.6897, LR0.5784/0.5912, 공유 Gaussian0.5694/0.5826이 실제 미래 전체 결과다. 이들은 새로운 정답으로 고른 ‘우승 설정’이 아니라 사전에 고정한 방법이다.

## 4. 새 템플릿이 원인이라고 말할 수 있는가?

첫10000줄 전체에 GT가 한 번이라도 있었으면 seen-origin, 없었다면 new-origin으로 정했다. 참조100개에서만 보이지 않았다는 뜻과 다르다. 미래99호출 중88호출/86 GT유형이 new-origin,11호출/9유형이 seen-origin이다. 새로운 유형이 이후 여러 번 호출되어도 기준 시점에 없었다면 같은 new-origin 코호트다. ‘전 세계에서 새로 생긴 템플릿’이 아니라 ‘이 제공된 기록의 과거에서 처음 보이지 않았던 유형’이다.

'''+table(cohorts)+'''

Llama의 seen11개에는 PASS가 딱1개다. Gemma의 seen11개는 **전부 FAIL**이라 AUROC가 정의되지 않는다. 이때 AP1은 분류 성공을 뜻하지 않는다. 모든 FAIL을 어떻게 정렬해도1이다. 그러므로 새 템플릿 때문에 Clotho가 나빠진다고 seen/new 차이로 단정할 수 없다. 시점 변화에는 난도, 내용, 클래스 비율 등도 함께 바뀐다.

새 유형에서 LR이 전혀 정보를 못 읽는다는 뜻도 아니다. Gemma 새 유형 LR AUROC0.6003, Clotho0.7078이며 Llama에서는0.5469/0.6566이다. 제한된 미래 실패 신호가 있지만 안정적인 일반화를 입증하기에는 부족하다.

## 5. 과거 검증에서는 간단한 모델이 좋았는데 미래에는 왜 다른가?

획득하지 않은 과거431개를 별도로 평가했다. 이 정답은 학습/CV에 쓰지 않았다. 단 Clotho의 과거 PCA와 획득은 이들의 비라벨 입력 상태를 볼 수 있었으므로 완전히 보지 않은 입력 분포는 아니다.

'''+table(gap)+'''

Llama LR은0.7045→0.5784, Gemma는0.7072→0.5912로 내려갔다. 공유 Gaussian은0.7065→0.5694,0.7056→0.5826이다. 반면 Clotho는0.6177→0.6642,0.6582→0.6897이다. 이것이 가장 중요한 반례다. 과거 표본이나 무작위 분리에서 더 좋은 학습기가 미래에서도 더 좋다고 보장하지 않는다. 그러나 과거/미래는 같은 입력이 아니며 클래스 비율도 다르다. ‘LR 선형성이 원인’ 또는 ‘GMM 복잡성이 유용해서’라는 원인 설명은 아직 없다.

## 6. 차이는 얼마나 확실한가?

미래95개 GT유형을 묶음 단위로4000번 복원 추출했다. 한 묶음의 모든 입력과5개 참조 시드, 모든 모델 점수를 함께 유지한다. 로그 줄이나 시드를 독립 시스템으로 세지 않는다. 이는 **Thunderbird 내부에 조건부인 구간**이며 시스템 모집단에 대한 신뢰구간이 아니다. 다른 템플릿 사이의 시간 의존성까지 해결하지 않는다.

'''+table(c[c.cohort=='future'])+'''

LR−Clotho는 Llama−0.0859,95%구간[−0.1920,0.0183], Gemma−0.0984,[−0.2320,0.0374]이다. 모두0을 포함한다. Clotho가 평균에서 높아도 확실한 우월성이라고 쓰면 안 된다. 미래 입력 PCA를 허용한 Clotho와 과거 전용 Clotho의 차이도−0.0134/+0.0076으로 확실하지 않다. 이번 순위 역전이 단순히 미래 PCA 접근 때문이라는 근거는 없다.

seen Llama는 복원 표본4000개 중2652개만 양쪽 클래스가 있어 AUROC가 계산된다. Gemma seen은0개다. 정의되지 않는 경우를0.5로 채우거나 평가 경계를 옮기지 않았다.

## 7. 시간이 더 지나면 계속 나빠지는가?

원본10000줄 단위로 미래를4구간으로 나눴다. 호출 수는59,24,8,8이다. Llama의 마지막 두 구간은8개 전부 실패라 AUROC가 없다. Gemma 마지막 구간은7FAIL/1PASS이며 Clotho0.9143, LR0.2571이다. 눈에 띄는 차이지만 PASS가 하나뿐인 강한 작은 표본 반례다. 시간이 지날수록 일정하게 나빠진다고 결론 낼 근거는 없다. 전체 AUROC는 구간 간 PASS/FAIL 쌍도 포함하므로 구간 AUROC의 단순 평균과 다르다.

![시간 구간 결과](../results_phase3/clotho_temporal/figures/temporal_windows.png)

## 8. 비용과 검증에서 무엇을 확인했는가?

새 GPU/생성 비용은0이지만 저장된 상태를 공짜 배포 특징처럼 간주하지 않는다. 공통 획득에 과거531개 상태, 예측에 미래99개가 필요하다. 예전 L40S 기록으로 과거 상태 추출은 대략 Llama22.0초/Gemma31.6초, 미래4.1/5.9초다. 원래 생성10개 중5개로 판정하므로 실제 구현에서100입력의 이 반복 정답을 얻으려면500개 출력 확인이 필요하다. 사람의 GT작성 비용은 측정하지 않았다.

LR/공유 Gaussian 최종 학습이 빠르더라도 이 실험은 공통 Clotho 획득 비용을 부담한다. 이를 빼고 end-to-end 비용 우위라고 주장하지 않는다. 모든10개 작업의 참조/미래 ID 동일성, 과거100개 정답 공개, PCA fit인덱스, 입력 해시, 유한 점수, 지표 재계산을 확인했다. Llama 시드0 전체를 다시 실행해4방법 모두 점수 차이0, 참조ID 동일을 확인했다. 허용오차를 늘리지 않았다.

구체적으로 시드0의 Llama 실패40772번 줄은 LR위험 백분위0.303, Clotho0.929이다. Gemma 실패26089번 줄은0.010/0.970으로 LR이 상대적으로 안전하게 정렬했다. 반대로 Gemma 성공15905번 줄은 LR0.040, Clotho0.869로 Clotho의 보편적 우월성에 반례다. 이 백분위는99개 미래 입력 안의 순위이지 확률이나 임계값 기반 오류가 아니다. 대표 사례는 시드0의 가장 이른 PASS/FAIL, 그리고 같은 라벨 내 LR/Clotho 순위 차이가 가장 큰 입력이라는 고정 규칙으로 저장했다. 점수는 보정된 실패 확률이나 특정 임계값의 오탐/미탐이 아니다. 사례를 골라 방법을 조정하지 않았다.

## 9. 연구 판단

이 실험은 시간 순서 일반화가 기존 연구와 다른 미해결 문제임을 보여 준다. 다만 단일 시스템의 첫50000줄은 전체1660만 줄의 작은 일부이고, seen PASS가0/1개여서 템플릿 신규성 효과를 분리하지 못한다. LR의 기존 장점을 미래에서 확인했다고 말할 수 없고, Clotho의 복잡성을 정당화했다고도 말할 수 없다. 다양성 획득의 미래 효과는 알 수 없다.

지금은 **밀도/PCA/공분산/획득 튜닝을 재개하지 않는다.** 새 방법 가설도 제안하지 않는다. 가장 방어 가능한 기여는 평가 조건에 따라 방법 순위가 바뀔 수 있다는 신중한 실증 결과다. 다음 실험은 하나만 권한다. **추가 타임스탬프 검증 시스템의 더 긴 실제 시간 구간을 사전 등록하고, 이번 설정/예산을 그대로 재평가한다.** 이를 위해 먼저 실제 캐시 미스 호출을 열거하여 새 상태/최소5개 생성의 정확한 비용을 산정해야 한다. 저장되지 않은 미래 예측을 기존 benchmark 예측으로 대체하지 않았고 추가 GPU 작업을 시작하지 않았다.
'''
 (REP/'clotho_temporal_generalization_explained.md').write_text(ko)
 # Preserve content case records privately; a small report table can be reviewed without fitting.
 examples=load(R/'representative_cases.json');case_lines=[]
 for a in examples:case_lines.append(f"|{a['model']}|{a['selection']}|{a['LineId']}|{a['failure']}|{a['new_to_origin']}|{a['lr_percentile']:.3f}|{a['clotho_percentile']:.3f}|")
 case_text='\n\n## Representative case ranks\n\nThe full input and GT strings are in the private machine-readable cases. FAIL=1; percentiles increase with risk.\n\n|Model|Fixed selection rule|Source LineId|FAIL|New-origin|LR percentile|Clotho percentile|\n|---|---|---:|---:|---|---:|---:|\n'+'\n'.join(case_lines)+'\n'
 with (REP/'clotho_temporal_generalization.md').open('a') as q:q.write(case_text)
 with (REP/'clotho_temporal_generalization_explained.md').open('a') as q:q.write(case_text.replace('Representative case ranks','대표 사례의 위험 순위').replace('The full input and GT strings are in the private machine-readable cases. FAIL=1; percentiles increase with risk.','전체 입력/GT는 비공개 사례 JSON에 있다. FAIL=1이며 백분위가 높을수록 위험 점수가 높다.'))
 for fn in ('clotho_temporal_feasibility.md','clotho_temporal_generalization.md','clotho_temporal_generalization_explained.md'):assert (REP/fn).stat().st_size>5000
 save(R/'report_validation.json',dict(status='passed',summary_version=2,earlier_counts_prevalence='mean over changing431-input seed populations, not first seed',reports={f:sha(REP/f) for f in ('clotho_temporal_feasibility.md','clotho_temporal_generalization.md','clotho_temporal_generalization_explained.md')},future_mean_auc={f'{m}/{k}':float(s[(s.model==m)&(s.method==k)&(s.cohort=='future')].AUROC_mean.iloc[0]) for m in ('llama','gemma') for k in p['methods']}))
if __name__=='__main__':main()
