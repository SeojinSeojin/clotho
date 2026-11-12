"""Render technical and Korean explanation from validated component artifacts."""
from cpu_research_common import EXP, load, table
import itertools
import numpy as np
import pandas as pd
from log_parsing_component_ablation import ROOT, REPS, SELECTORS, SCORES

BASE = 'transductive/random/pass'


def main():
    validation = load(ROOT / 'validation.json')
    summary = load(ROOT / 'summary.json')
    paired = load(ROOT / 'paired.json')
    factors = load(ROOT / 'factor_contrasts.json')
    gaps = load(ROOT / 'full_test_gaps.json')
    shared = load(ROOT / 'shared_input_gap_reductions.json')
    cv = pd.read_csv(ROOT / 'cv_folds.csv')
    dimensions = pd.read_csv(ROOT / 'selected_dimensions.csv')
    fallback = load(ROOT / 'fallbacks.json')
    jobs = [load(p) for p in sorted((ROOT / 'jobs').glob('*/*/*/seed[0-4].json'))]
    def find(records, **kw):
        return next((r for r in records if all(r.get(k) == v for k, v in kw.items())), None)
    def sm(model, method, split='disjoint', cohort='original5', metric='AUROC'):
        return find(summary, model=model, method=method, split=split, cohort=cohort, metric=metric)
    def delta(model, method, split='disjoint', cohort='original5', metric='AUROC'):
        return find(paired, model=model, method=method, split=split, cohort=cohort, metric=metric)
    def fmt(r, uncertainty=True):
        if r is None:
            return 'ineligible'
        return f"{r['mean']:.4f} ± {r['seed_sd']:.4f}" if uncertainty else f"{r['mean']:.4f}"
    def ci(r):
        return 'ineligible' if r is None else f"{r['mean']:+.4f} [{r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f}] (N={r['n_systems']})"
    methods = [f'{rep}/{sel}/{score}' for rep, sel, score in itertools.product(REPS, SELECTORS, SCORES)]
    technical = ['# Which Clotho components limit unseen-template generalization?',
        '\n## 1. Research questions and identification scope\n',
        'We test scoring, representation fitting and final-reference PCA-dimension selection using stored Llama-3.1-8B/Gemma-2-9B input states. The question is why failure ranking deteriorates, not which test-selected configuration wins. Hypotheses and bounded controls were committed in configs/clotho_component_ablation_protocol.json before factorial outcomes. This is a controlled diagnostic with frozen original acquisition, not a new end-to-end Clotho training algorithm.',
        '\n## 2. Existing evidence and unresolved gaps\n',
        'The [pre-experiment audit](clotho_component_ablation_audit.md) distinguishes completed same-space probes from missing crossings. Earlier250-job recovery, same-space two-density/LR, geometry and stable10 results are reused. Earlier relative density added FAIL-specific capacity; reference-standardized PCA16+LR changed several factors. No grouped Clotho dimension-CV intervention existed. U/S1 uses other layers, source labels and real-only references; U-best adds output features. None provides a component-isolated decomposition.',
        '\n## 3. Exact original procedure and baseline validation\n',
        'Original PCA uses centered raw vectors, no z-scoring or whitening, fitted to all510 rows (500 suite+10 synthetic), including unlabeled outer tests. PASS labels mean at least3 strict matches among stored generations5–9. Final PASS-only full-covariance GMM uses seed-specific inherited k/reg_covar,30 initializations,500 iterations and tolerance0.001. Failure risk is −log p_PASS. Adaptive acquisition begins with10 synthetic references, adds10 balanced uncertainty/diversity rows at a time and selects PCA dimension during this path.',
        'Dimension CV is unstratified3-fold shuffled KFold(seed),10 GMM starts, and ranks PASS log density against the **continuous** five-generation success rate using Pearson correlation. With constant rates the original substitutes1−MAE. It considers current dimension and ±10, bounded5–50. Higher dimension wins on larger mean; lower dimension additionally requires its minimum fold score to exceed both the original minimum and0.1. This asymmetry is retained. k changes using mixture-weight perplexity; early entropy checks alter covariance regularization. We freeze these inherited density settings across the final-factor comparisons.',
        f"Validated{validation['jobs']}/250 original AUROCs at tolerance1e-12; baseline vectors are identical to the archived originals. All750 source JSON/NPZ/input files retain their original SHA256. Isolated new-computation reruns: {len(validation['deterministic_replays'])}, identical scores/CV/metrics. See baseline_validation.json and validation.json.",
        'Cross-hardware limitation: this Haswell VM recomputes float32 PCA with small numerical drift despite matching numpy/scipy/sklearn/torch versions and trying Python3.10/3.12. HDFS differed1.204e−5 in PCA coordinates and0.00399 in risk, while original AUROC stayed exact; fitting density on archived coordinates reduced risk drift to1.03e−11. We therefore preserve the archived selected coordinates and PASS/relative/LR vectors as the original controls. Other candidate dimensions come from a seeded replay; reference/test order, AUROC and the final selected dimension must match. This is artifact-anchored reproduction, **not bit-identical original re-execution**. No AUROC tolerance was relaxed; preliminary exceptions are retained with resolution markers.',
        '\n## 4. Factorial design and dependencies\n',
        table(['Factor', 'Levels', 'Held constant / limitation'], [
            ['Scoring', 'PASS GMM / FAIL−PASS log density / balanced LR C=1', 'Same selected coordinates and100 labels; LR reference-coordinate standardization fixed'],
            ['Representation', 'Original transductive raw PCA / reference-only raw PCA / reference-only standardized PCA', 'Same fair layer16/21 and inputs; no whitening; fixed16 cells isolate scope/scaling at identical dimension'],
            ['Selection', 'Exact original final KFold / oracle GT GroupKFold / frozen16', 'Same original local dimension candidates and PASS-based objective for every score'],
            ['Capacity controls', 'PASS2k; pooled−PASS; reg0.01/1', 'Original selected coordinates; approximately matched parameter budget; no test tuning']]),
        'Each original job has27 primary cells plus bounded capacity/regularization controls. Transformations are cached once per job/representation/dimension. Original acquisition and candidate neighborhood are frozen: replacing CV during earlier acquisition would change reference examples, violating the matched design. Thus the grouped intervention is final-stage and conditional on the original acquisition path. For the reference-only representations, original local dimension candidates are retained and the final decision is recomputed. This is not an independent full search through earlier dimension histories.',
        'Reference PCA/scaling fits all100 reference covariates once, including inner-validation covariates, but never outer-test covariates or labels. This mirrors the original transductive inner-preprocessing convention and is not strictly fold-nested preprocessing. Every final scoring model uses only reference labels. Layer, extraction position, LLM, prompt, generations, reference order, five seeds and failure definition remain fixed. Group IDs are reference GT strings only; synthetic rows receive unique pseudo-groups. No GT input feature is supplied to any predictor.',
        '\n## 5. Matched evaluation and statistical units\n',
        'AUROC and FAIL-positive average precision (AP/AUPRC) are computed within each system/seed; prevalence is reported alongside AP. Values are per-seed system macro means with sample SD over five seeds. Paired contrasts require the same system to have all five seeds in both methods. Confidence intervals resample systems4,000 times (seed20261112), keeping their five seeds together. Lines and seeds are never independent inference units. Interactions are paired differences-in-differences; uncertainty is descriptive across these13/12 systems, without multiple-testing-adjusted significance claims. Full-test split gaps and identical-input-intersection gap reductions are both saved.',
        '\n## 6. Main results\n',
        'Names: transductive=original raw PCA; reference_raw=raw reference-fit PCA; reference_scaled=reference z-score then PCA. random=original final KFold; grouped=oracle final GroupKFold; frozen16=predeclared dimension16. All27 configurations are shown; grouped rows may cover fewer systems. Use the matched factor tables, rather than subtracting different-population means. Original-coordinate/random scoring rows are **reproduced existing results**, not new observations.'
    ]
    mainrows = []
    for model in ('llama', 'gemma'):
        for method in methods:
            r, ap, pr, std, gap = sm(model, method), sm(model, method, metric='AP'), sm(model, method, metric='prevalence'), sm(model, method, 'standard'), find(gaps, model=model, method=method)
            mainrows.append([model, method, r['n_systems'] if r else 0, fmt(r), fmt(ap, False), fmt(pr, False), fmt(std), fmt(gap), 'reference' if method == BASE else ci(delta(model, method))])
    technical.append(table(['Model', 'Method', 'Disjoint N', 'Disjoint AUROC±seed SD', 'AP', 'FAIL prevalence', 'Standard AUROC±SD (own eligible N)', 'Standard−disjoint gap (matched N)', 'ΔClotho [system CI]'], mainrows))
    technical.extend(['\n## 7. Factor effects, capacity and interactions\n',
        'Primary explanatory contrasts below were specified independently of test performance. In particular, frozen16 compares PCA fitting scope while dimension and score remain fixed. Selection-based representation contrasts can also change the selected dimension and should not be interpreted as a pure fitting-scope effect. The fixed linear baseline is reference_scaled/frozen16/lr; paired_vs_fixed_linear.json reports every method against it. All interactions and per-system contrasts, including negative results, are in factor_contrasts.{json,csv}.'])
    selected_contrasts = [
        ('scoring', 'relative-pass|transductive|random'), ('scoring', 'lr-pass|transductive|random'),
        ('representation', 'reference_raw-transductive|frozen16|pass'), ('representation', 'reference_raw-transductive|frozen16|relative'),
        ('representation', 'reference_scaled-reference_raw|frozen16|lr'),
        ('selection', 'frozen16-random|transductive|pass'), ('selection', 'grouped-random|transductive|pass'),
        ('scoring_x_representation', 'relative|reference_raw-transductive|frozen16'),
        ('scoring_x_selection', 'relative|transductive'),
        ('capacity', 'relative-control/pass_2k'), ('capacity', 'relative-control/pooled_relative'),
        ('capacity', 'relative-transductive/random/lr')]
    rows = []
    for kind, name in selected_contrasts:
        rows.append([kind, name] + [ci(find(factors, model=m, split='disjoint', kind=kind, name=name)) for m in ('llama', 'gemma')])
    technical.append(table(['Factor', 'Matched disjoint contrast', 'Llama Δ[95%CI]', 'Gemma Δ[95%CI]'], rows))
    controls = ['control/pass_2k', 'control/pooled_relative', 'control/pass_reg0.01', 'control/relative_reg0.01', 'control/pass_reg1.0', 'control/relative_reg1.0']
    technical.append(table(['Control', 'Llama disjoint AUROC', 'Llama ΔClotho', 'Gemma disjoint AUROC', 'Gemma ΔClotho'], [[m, fmt(sm('llama', m)), ci(delta('llama', m)), fmt(sm('gemma', m)), ci(delta('gemma', m))] for m in controls]))
    technical.append('For q=d+d(d+1)/2 mean/covariance parameters per component, two k-component densities have2kq+2k−2 free parameters; one2k-component PASS density has2kq+2k−1. The budget differs by only one mixture-weight parameter, but training samples per parameter differ. PASS2k changes flexibility on the PASS class; pooled-background changes fitting population, with equal extra density capacity. Neither is a perfect isolation of a loss function. In an ideal exact-mixture model, log p_all−log p_PASS is monotonic in p_FAIL/p_PASS; empirical differences also reflect estimation and regularization. Gains must be read together with the reg0.01/1 sensitivity results; no regularizer is chosen on outer tests.')
    technical.append('\n### Shared-input generalization-gap reductions\n')
    gap_methods = [BASE, 'transductive/random/relative', 'transductive/random/lr', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr', 'transductive/grouped/pass']
    technical.append(table(['Method', 'Llama paired gap reduction', 'Gemma paired gap reduction'], [[m] + [ci(find(shared, model=mo, method=m)) for mo in ('llama', 'gemma')] for m in gap_methods]))
    technical.append('Positive means a smaller standard−disjoint gap on identical input intersections. It is not automatically better prediction: inspect absolute disjoint AUROC and whether standard performance fell. Reference pools necessarily differ between outer split types; these contrasts do not identify a causal familiarity-only effect.')
    technical.append('\n## 8. Oracle grouped-CV diagnostics\n')
    cvsum = []
    for (model, split, rep, sel), f in dimensions.groupby(['model', 'split', 'representation', 'selection']):
        if sel == 'frozen16':
            continue
        ff = cv[(cv.model == model) & (cv.split == split) & (cv.representation == rep) & (cv.selection == sel)]
        # Deduplicate fold diagnostics across the repeated candidate dimension rows.
        folds = ff.drop_duplicates(['system', 'seed', 'fold'])
        cvsum.append([model, split, rep, sel, f'{f.eligible.sum()}/{len(f)}', f'{f.dimension.mean():.2f}',
                      f'{folds.validation_seen_fraction.mean():.3f}', f'{folds.train_groups.mean():.1f}/{folds.validation_groups.mean():.1f}',
                      f'{folds.train_fail_prevalence.mean():.3f}/{folds.validation_fail_prevalence.mean():.3f}'])
    technical.append(table(['Model', 'Outer split', 'PCA', 'CV', 'Eligible jobs', 'Mean selected d', 'Validation seen-GT share', 'Mean train/validation groups', 'Train/validation FAIL share'], cvsum))
    technical.append('GroupKFold uses min(3,number of reference groups), never mixes a GT between train and validation, and is not label-stratified. We mark the cell ineligible if any fold lacks both validation classes, has fewer than2 validation rows, or has fewer than k passing training references. No random fallback is used for grouped CV. Original random CV retains its original insufficient-PASS fallback. Full indices, candidate validation scores, convergence status and class/group counts are in each job JSON and cv_folds.csv. These diagnostics describe actual final-step folds, unlike the earlier overlap-only audit.')
    technical.append(table(['Model', 'Split', 'System', 'Seed', 'Representation', 'Protocol reason'], [[r['model'], r['split'], r['system'], r['seed'], r['representation'], r.get('ineligible', r.get('fallback'))] for r in fallback]) if fallback else 'No grouped cells excluded and no original random-CV fallbacks.')
    technical.append('Grouped CV is **oracle-assisted development**, because GT reference templates construct its folds. GT test templates never select a model. Predictions after fitting require no GT template IDs; development availability and inference availability are different claims. A benefit here would motivate a separate approximate-group experiment, not establish a deployable grouped-development procedure.')
    technical.append('\n## 9. Reliable-label verification\n')
    technical.append('Existing40-generation files were found and restored for **all13 Llama and12 Gemma systems**, correcting the narrower availability described in the earlier mechanism report. No new outputs were generated. Each40-output file is checked against the500-row suite and GT order. Training/reference/CV labels remain the original five-output labels. The reliable evaluation target is FAIL if≤20/40 strict matches. Stable40 keeps rates≤0.1 or≥0.9 that agree with the original five-output target; stable10 requires all original10 correctness bits identical. Stability is a diagnostic subset, not proof of deterministic parsing.')
    reliable_methods = [BASE, 'transductive/random/relative', 'transductive/random/lr', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr']
    rr = []
    for model in ('llama', 'gemma'):
        for method in reliable_methods:
            for cohort in ('original5', 'reliable40', 'stable40', 'stable10'):
                r = sm(model, method, cohort=cohort)
                rr.append([model, method, cohort, r['n_systems'] if r else 0, fmt(r), 'reference' if method == BASE else ci(delta(model, method, cohort=cohort))])
    technical.append(table(['Model', 'Method', 'Evaluation label/cohort', 'Systems with all5 seeds', 'Disjoint AUROC', 'Δmatched Clotho'], rr))
    technical.append('Matched40 coverage is100% of each job’s original test inputs. Stable-cohort coverage and per-job counts are saved in metrics_long.csv. Label_sensitivity.json compares40 versus5 on identical full test inputs. Do not subtract AUCs from different stable-system populations to estimate a “noise penalty.” Method rankings under40 are descriptive for historically observed data; reliable test labels never enter training or model selection.')
    technical.append('\n## 10. Per-system findings\n')
    ps = pd.read_csv(ROOT / 'per_system.csv')
    ps = ps[(ps.split == 'disjoint') & (ps.cohort == 'original5') & (ps.metric == 'AUROC') & (ps.seeds == 5)]
    rows = []
    for (model, system), f in ps.groupby(['model', 'system']):
        dd = dict(zip(f.method, f.value))
        rows.append([model, system, f'{dd[BASE]:.4f}'] + [f'{dd[m]-dd[BASE]:+.4f}' if m in dd else 'ineligible' for m in ['transductive/random/relative', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr', 'transductive/grouped/pass']])
    technical.append(table(['Model', 'System', 'Clotho', 'Original relative Δ', 'Ref raw16 relative Δ', 'Ref scaled16 LR Δ', 'Grouped original PASS Δ'], rows))
    technical.extend(['\n## 11. Competing explanations\n',
        'Reference size100 means only90 real examples plus10 synthetic. Balanced acquisition changes class composition, and sparse class/template support affects covariance estimation and grouped-fold eligibility. Full-covariance k-component mixtures can have many more parameters than class examples; ridge regularization prevents singularity but does not ensure a good density. Labels are repeated-generation outcomes, with40-vote verification reducing but not eliminating uncertainty. Strong GT dependence and system distribution differences mean no line-level IID inference is justified. Capacity, fitting-population and regularization controls constrain explanations without perfectly equating estimation difficulty.',
        'Layer/source/output-feature differences prevent attributing the whole gap to published U/S1 to scoring. Published S1 is0.749/0.762 disjoint AUROC, U-best0.7764/0.7732; they are contextual, not controlled same-layer baselines. contextual U score vectors use identical outer test observations, and paired_vs_fixed_linear.json saves their paired difference from the predeclared fixed linear control. A same-space LR is the clean discriminator comparison.',
        '\n## 12. Threats to validity and omitted comparisons\n',
        'This is a retrospective diagnostic on capped500-message suites and two LLMs, not a prospective deployment test. Fair layers were previously chosen using three tuning systems; frozen16 was specified independently of this final test analysis but appears in earlier exploratory work. Candidate dimension neighborhoods and density regularizers inherit the original acquisition path. No full end-to-end grouped acquisition experiment, no whitening search and no classifier-specific tuning are performed. Reference transformations see inner-validation covariates. Cross-hardware alternative-dimension PCA reconstruction is recorded and original selected coordinates remain archived. System-cluster intervals with12/13 systems can be wide; no causal “template memorization” inference or universal dominant-component claim follows.',
        'Reference200 is secondary. The primary experiment reuses saved100-reference coordinates, order and densities; equivalent200-coordinate snapshots are not present in the mechanism artifacts. Extending exact recovery and validation to200 is omitted from this bounded main study. Consequently there is no claim that grouped-CV or representation effects survive200 references; this is an explicit remaining limitation, not a negative result.',
        '\n## 13. Conclusions and decision memo\n',
        '<!-- DECISION_MEMO: replace only after inspecting validated matched contrasts. -->',
        '\n## 14. Recommended next experiment\n',
        'The next experiment should distinguish the remaining explanations rather than search outer-test AUROC. Select its single primary contrast from the validated decision memo; freeze layers, references, population and metric before collecting any new result. No method novelty claim is made from improved AUROC alone.',
        '\n## Reproduction, figures and artifacts\n',
        'Entry points: log_parsing_component_ablation.py, log_parsing_component_analysis.py, validate_clotho_component_ablation.py, log_parsing_component_report.py. CPU environment: environment.json. Restore the original artifacts with log_parsing_mechanisms_restore.py and --outputs-only plus existing40-generation files. Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1; four processes; CUDA_VISIBLE_DEVICES empty. Jobs are atomic/resumable. The HDFS pilot took~15s/job; primary cost is recorded per job, with cached per-dimension transforms. Original predictions and prior reports are never overwritten.',
        'All machine-readable files live in results_phase3/clotho_component_ablation/: jobs/*/*/*/seedN.{json,npz}, metrics_long.csv, per_system.csv, per_seed_macro_available.csv (available-population description), summary.json (complete-five-seed populations), paired.json, paired_vs_fixed_linear.json, cv_folds.csv, selected_dimensions.csv, factor_contrasts.json, full_test_gaps.json, shared_input_jobs_*.csv, shared_input_gap_reductions.json, label_sensitivity.json, score_distributions.json, fallbacks.json, baseline_validation.json, validation.json, source_hashes.json and persistence_validation.json. Bootstrap draws and per-system paired estimates are retained in JSON. Figures use paper/common.py conventions and are PDF+PNG exports.',
    ])
    for name, caption in [('scoring_representation', 'Scoring×representation under original final random CV; see matched contrasts for inference.'), ('standard_disjoint', 'Absolute performance, with full-population split comparisons.'), ('grouped_cv', 'Oracle grouped−random contrasts on matched eligible systems with system-cluster intervals.'), ('factor_interactions', 'Nonparallel lines diagnose representation/scoring dependence; frozen16 isolates dimension.'), ('per_system_improvements', 'System-level paired deltas; black bars are cluster intervals of the mean.')]:
        technical.append(f'\n![{caption}](../results_phase3/clotho_component_ablation/figures/{name}.png)\n')
    (EXP / 'reports/clotho_component_ablation.md').write_text('\n\n'.join(technical) + '\n')
    # Korean explanation uses separate calculations and interpretation, not sentence translation.
    ko = ['# Clotho 구성요소 실험을 이해하고 방어하기',
        '## 1. 무엇을 원인이라고 부를 수 있는가?\n\n질문은 최고 AUROC 조합 찾기가 아니라, 같은 입력 정보와 같은 라벨에서 어느 계산을 바꾸었을 때 실패 예측이 달라지는가이다. 모델/층/프롬프트/생성/학습 라벨/참조 순서/테스트 순서/시드0–4는 고정했다. 원래100개 참조는 실제90개+합성10개다. FAIL은 원래 생성5–9의 엄격 일치가2개 이하인 경우다. 새 GPU 계산이나 생성은 없다.',
        '세 요인은 점수(PASS밀도/상대밀도/LR), 표현(전체raw PCA/참조raw PCA/참조표준화PCA), 최종 차원선택(random CV/GT그룹CV/고정16)이다.27조합을 같은 참조/테스트로 비교했다. 그러나 초기 CV가 바뀌면 uncertainty/diversity가 뽑는 참조도 달라진다. 그래서 초기 선택 이력은 그대로 두고 마지막100개 참조의 차원 선택에만 개입했다. 이것이 통제의 장점이며 전체 파이프라인 교체 효과를 주장하지 못하는 이유이기도 하다.',
        '## 2. 원래 모델을 정말 비교 기준으로 썼는가?\n\n기존250개 작업의 참조·테스트·AUROC를1e−12 허용오차로 검증했다. 원래 PCA는 정규화된 벡터가 아니라 평균만 뺀 raw 은닉 상태이며, 라벨 없는 테스트 벡터까지 포함한510개에 적합한다. fair_leakfree는 출력 표본 간 라벨 누출을 방지하는 규칙이지 PCA가 참조에만 적합된다는 뜻이 아니다.',
        '이번 Haswell 장비에서 원래 PCA를 다시 계산하면 HDFS 좌표가1.204e−5 정도 달랐고 위험점수 차이는0.00399였다. AUROC는 동일했지만 비트 단위 재현이라고 부를 수 없다. 저장된 원래 좌표에 밀도만 다시 적합하면 차이가1.03e−11로 줄었다. 그래서 검증된 원래 좌표/점수를 그대로 재사용하고 대안 계산을 붙였다. 원래 AUROC 허용오차는 늘리지 않았다. 새 실험 자체는 별도 임시 디렉터리 재실행으로 점수·CV·지표의 동일성을 확인했다. 수치 재현과 연구 가설의 타당성은 서로 다른 검증이다.',
        '## 3. 점수의 질문: 낯선가, 실패에 더 가까운가?\n\n원래 위험은 $r_P(x)=-\log p(x\mid PASS)$, 상대밀도 위험은 $r_R(x)=\log p(x\mid FAIL)-\log p(x\mid PASS)$다. 예를 들어 성공 밀도0.001/실패 밀도0.00001인 낯선 성공 A와 성공 밀도0.1/실패 밀도0.5인 실패 B를 생각하자. PASS만 보면 A의 위험6.91이 B의2.30보다 높다. 상대밀도는 A가−4.61, B가1.61이므로 순위가 뒤집힌다. 이 예는 가능한 메커니즘을 설명할 뿐 실제 데이터의 이득을 증명하지 않는다. 실제 이득은 같은 PCA 좌표와 같은100개 라벨로 계산한 아래 표다.',
    ]
    ko.append(table(['비교', 'Llama 비중복 AUROC', 'Gemma 비중복 AUROC', 'Llama Δ[시스템95%구간]', 'Gemma Δ[시스템95%구간]'], [[m, fmt(sm('llama', m)), fmt(sm('gemma', m)), ci(delta('llama', m)), ci(delta('gemma', m))] for m in ['transductive/random/relative', 'transductive/random/lr', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr']]))
    ko.extend([
        '상대밀도는 FAIL 전용 매개변수를 추가한다. 따라서 A1/A2 차이를 손실함수만의 효과라고 부르면 안 된다. 이를 보기 위해 PASS성분을2k로 늘린 모형과 전체 참조의 배경밀도를 추가한 모형을 비교했다. 공분산 정규화0.01/1도 미리 고정해 확인했다. 차원d일 때 평균/공분산의 성분당 자유도는 q=d+d(d+1)/2다. 두k-GMM은2kq+2k−2, 한2k-GMM은2kq+2k−1로 총 매개변수 수는 거의 같다. 그래도 PASS만 학습하는 표본 수와 FAIL을 따로 학습하는 난도는 같지 않다. 완벽한 인과 분리는 아니다.',
        table(['용량/목적 진단', 'Llama Δ[구간]', 'Gemma Δ[구간]'], [[name] + [ci(find(factors, model=m, split='disjoint', kind='capacity', name=name)) for m in ('llama', 'gemma')] for name in ['relative-control/pass_2k', 'relative-control/pooled_relative', 'relative-transductive/random/lr']]),
        '## 4. 표현의 질문: PCA 학습 범위인가, 표준화인가, 차원인가?\n\n참조raw PCA는 원래와 똑같이 raw 좌표를 쓰면서 PCA 적합 대상을100개 참조로 제한한다. 참조표준화PCA는 그 앞에 참조 평균/표준편차로 feature z-score를 추가한다. 모든 점수 모형이 같은 변환을 공유한다. 특히 **고정16**끼리 비교해야 차원 변화가 없는 PCA 학습범위 효과를 볼 수 있다. random CV끼리 비교하면 표현이 바뀌면서 선택 차원도 반응하므로 그것은 표현과 선택의 결합된 결과다.',
        table(['차원16에서 통제한 질문', 'Llama Δ[구간]', 'Gemma Δ[구간]'], [[name] + [ci(find(factors, model=m, split='disjoint', kind='representation', name=name)) for m in ('llama', 'gemma')] for name in ['reference_raw-transductive|frozen16|pass', 'reference_raw-transductive|frozen16|relative', 'reference_scaled-reference_raw|frozen16|lr']]),
        'PCA 축이 설명하는 전체 분산과 실패를 가르는 정보는 동일하지 않다. explained_variance는 각 작업에 저장했다. 이 실험은 특수한 실패 방향을 찾아낸 것도, 은닉 상태가 템플릿을 암기한다는 것도 증명하지 않는다. 참조PCA는 outer-test를 전혀 보지 않지만 inner-validation의 입력 공변량은 포함한다. 엄격한 fold별 nested PCA 실험은 별도 과제다.',
        '## 5. 내부 CV의 질문: 시험 문제와 다른 연습문제로 차원을 고르는가?\n\n원래CV는3-fold KFold이며 stratification이 없다. 검증GT가 학습에도 있는 경우가 있다. GT그룹CV는 같은 템플릿을 한 fold에만 놓아 이를0으로 만든다. 정답템플릿을 특징으로 넣거나 외부 테스트 정답으로 차원을 고르는 것은 아니다. 그러나 개발할 때 참조GT가 필요하므로 **oracle-assisted 진단**이다.',
        '예를 들어 A1,A2가 같은 템플릿이고 B1,B2가 다른 템플릿이면 random은 A1/B1 학습, A2/B2 검증이 가능하다. 그룹CV는 A가족 전체를 검증에 넣으므로 새 가족 일반화에 더 가까운 검증이 된다. 단, FAIL이 특정 가족에 몰려 있으면 어떤 fold가 한 클래스만 가져 AUROC나 적절한 실패 검증이 불가능해진다. 우리는 그룹 분리를 깨지 않고 해당 셀을 제외했다. 제외 이유/시스템/시드를 기술보고서와 fallbacks.json에 모두 기록했다.',
        '중요한 원래 선택목표는 binary AUROC가 아니라 생성5개 중 성공비율(0,0.2,…,1)과 PASS밀도 순위의 상관이다. 높은 차원 후보는 평균상관만 높으면 바뀌지만 낮은 차원 후보에는 최소fold점수 조건도 있다. 그룹만 바꾸고 이 비대칭 규칙은 유지했다. 따라서 그룹CV의 결과는 “그룹화가 항상 좋다”가 아니라 **이 원래 선택규칙을 유지했을 때의 최종단계 효과**다.',
        table(['그룹−random, 같은 적격 시스템', 'Llama Δ[구간]', 'Gemma Δ[구간]'], [[f'{r}/{s}'] + [ci(find(factors, model=m, split='disjoint', kind='selection', name=f'grouped-random|{r}|{s}')) for m in ('llama', 'gemma')] for r,s in [('transductive','pass'),('transductive','relative'),('reference_raw','pass'),('reference_scaled','lr')]]),
        '선택 차원/검증점수/학습·검증 실패비율/GT중복률은 selected_dimensions.csv, cv_folds.csv에 있다. 원래보다 차원을 많이 바꿨다는 것만으로 효과가 있는 것은 아니다. 같은 적격 시스템에서 외부 AUROC가 어떻게 변했는지를 함께 봐야 한다.200개 참조에서는 동일한 복원·검증을 완료하지 않아 효과의 지속성을 주장하지 않는다.',
        '## 6. 이득이 합쳐지는가? 그리고 격차를 정말 줄였는가?\n\n상호작용은 $[AUC(R_2,relative)-AUC(R_2,PASS)]-[AUC(R_1,relative)-AUC(R_1,PASS)]$처럼 계산한다.0이면 두 표현에서 점수 교체 효과가 같고,0이 아니면 단순 가산이 아니다. 모든 조합은 같은 시스템·시드를 함께 재표집한다. 상호작용 그림에서 선이 평행하지 않은 것은 시각적 단서이며, 확실성은 저장된 시스템구간과 함께 읽는다.',
        '일반화 격차는 $G=AUC_{standard}-AUC_{disjoint}$다. 격차 감소는 $G_{Clotho}-G_{new}$다. 예를 들어0.80→0.60인 모형을0.65→0.58로 바꾸면 격차는0.20→0.07로 줄지만 비중복 예측은 나빠진다. 그래서 **절대 비중복 AUROC와 격차 감소를 둘 다** 보고했다. 아래는 표준/비중복 테스트의 동일 입력 교집합으로 계산한 paired 결과다.',
        table(['동일입력 격차 감소', 'Llama Δ[구간]', 'Gemma Δ[구간]'], [[m]+[ci(find(shared, model=mo, method=m)) for mo in ('llama','gemma')] for m in gap_methods[1:]]),
        '## 7. 더 신뢰할 평가 라벨로 바꾸면 결론이 달라지는가?\n\n이미 저장된40회 출력이 Llama13/Gemma12 모두에 있었다. 새로 생성하지 않았다. 원래5회 라벨로 계속 학습하고 평가 정답만40회 다수결로 바꿨다. 예를 들어 원래2/5 성공이라 FAIL이었지만 새25/40 성공이면40회 평가에서는 PASS다. 예측점수와 학습은 고정되므로 지표 변화가 라벨정의에 얼마나 민감한지 볼 수 있다.40회도 진짜 확률을 완벽히 아는 것은 아니다.',
    ])
    ko.append(table(['모델', '방법', '원래5회 AUROC', '40회 AUROC', '40회에서 ΔClotho[구간]', 'stable40 ΔClotho[구간]'], [[mo,m,fmt(sm(mo,m)),fmt(sm(mo,m,cohort='reliable40')),ci(delta(mo,m,cohort='reliable40')),ci(delta(mo,m,cohort='stable40'))] for mo in ('llama','gemma') for m in reliable_methods[1:]]))
    ko.extend(['전체 테스트 입력의40회 매칭률은100%다. stable40은40회 성공률≤0.1 또는≥0.9이며 원래5회 라벨과 동의하는 입력만 남긴다. 부분집합에 클래스가 없거나 어떤 시드가 부적격이면 그 시스템 전체를 paired 평균에서 뺀다. 전체집합 AUC와 안정집합 AUC를 바로 빼서 라벨노이즈의 인과 기여라고 말하면 안 된다. 순위가 바뀌는지와 개선의 부호가 유지되는지를 matched 표로 확인한다.',
        '## 8. 숫자를 어떻게 집계했는가?\n\n한 작업 AUROC는 FAIL/PASS 쌍 중 FAIL 위험이 큰 비율(동점0.5)이다. AP는 precision을 recall 증가량으로 가중한 평균이며, 실패비율이 높으면 무작위 AP 기준도 높다. 각 시드에서 시스템들을 같은 가중치로 평균한 뒤5시드 평균±표본표준편차를 계산했다. CI는 시스템을4,000회 복원추출하며 해당 시스템의5시드를 함께 옮겼다. 로그 수나5개 시드를 독립 실험으로 세지 않는다. 모든시드에서 이겼다는 사실만으로 통계적 유의성을 주장하지 않는다.',
        '## 9. 연구자가 방어할 수 있는 결론과 못 하는 주장\n\n아래 결정 메모는 검증된 비교를 보고 작성한다. 기존 결과 재현, 새 통제 관찰, 메커니즘의 설명력, 아직 분리하지 못한 원인을 구분한다. U/S1의0.749/0.762와 U-best의0.7764/0.7732는 다른 층·소스 라벨·출력특징이 섞인 문맥 비교다. 이를 우리 점수 하나의 기여율로 분해하지 않는다. 참조GT를 사용하지 않는 고정16 모형은 추론 때도 GT가 필요 없지만, 그룹CV는 개발단계에 GT가 필요하다. 성능 향상만으로 새 방법의 독창성이 증명되지 않는다.',
        '<!-- KOREAN_DECISION: fill after validated results. -->',
        '기술보고서의27셀 표와 모든 제외 사유, per_system.csv의 시스템별 결과, paired.json/factor_contrasts.json의 구간, 다섯 PDF/PNG 그림을 함께 확인할 수 있다. 기존 산출물을 덮어쓰지 않았고 RESULTS_REPO 보존 검증은 persistence_validation.json에 있다.'
    ])
    (EXP / 'reports/clotho_component_ablation_explained.md').write_text('\n\n'.join(ko) + '\n')
    print('Rendered technical and Korean explanation; decision memo requires validated-result review.')


if __name__ == '__main__':
    main()
