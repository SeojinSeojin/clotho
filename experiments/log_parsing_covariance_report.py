"""Technical report and Korean researcher explanation from validated scaling results."""
from cpu_research_common import EXP,load,table
from log_parsing_covariance_scaling import ROOT,SCALES
import numpy as np
import pandas as pd


def main():
    S=load(ROOT/'summary.json');P=load(ROOT/'paired.json');L=load(ROOT/'paired_vs_lr.json')
    G=load(ROOT/'shared_input_gap_reductions.json');V=load(ROOT/'validation.json')
    checks=load(ROOT/'invariance_summary.json');A=load(ROOT/'aggregation_validation.json')
    M=pd.read_csv(ROOT/'metrics.csv');C=pd.read_csv(ROOT/'covariance_diagnostics.csv')
    def find(rows,**kw):return next(r for r in rows if all(r.get(k)==v for k,v in kw.items()))
    def sm(model,name,split='disjoint',metric='AUROC'):return find(S,model=model,split=split,method=name,metric=metric)
    def pair(model,name,split='disjoint',metric='AUROC'):
        score=name.split('/')[-1];base=f'refit/fixed/identity/{score}'if score!='lr'else'refit/lr_std/identity/lr'
        return find(P,model=model,split=split,method=name,reference=base,metric=metric)
    def fmt(r):return f"{r['mean']:.4f} ± {r['seed_sd']:.4f}"
    def ci(r):return f"{r['mean']:+.4f} [{r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f}]"
    models=('llama','gemma')
    results=[]
    for model in models:
        for family in ('fixed','transported','scale_aware'):
            for scale in SCALES:
                names=[f'refit/{family}/{scale}/{score}'for score in ('pass','relative')]
                results.append([model,family,scale,fmt(sm(model,names[0],'standard')),fmt(sm(model,names[0])),f"{sm(model,names[0],metric='AP')['mean']:.4f}",fmt(sm(model,names[1],'standard')),fmt(sm(model,names[1])),f"{sm(model,names[1],metric='AP')['mean']:.4f}"])
    main_table=table(['Model','Ridge','Scale','Standard PASS AUROC±SD','Disjoint PASS AUROC±SD','Disjoint PASS AP','Standard relative AUROC±SD','Disjoint relative AUROC±SD','Disjoint relative AP'],results)
    prevalence_table=table(['Model','Split','Systems','FAIL prevalence (system macro)'],
                           [[mo,sp,sm(mo,'anchor/archived/identity/pass',sp)['n_systems'],f"{sm(mo,'anchor/archived/identity/pass',sp,'prevalence')['mean']:.4f}"]for mo in models for sp in ('standard','disjoint')])
    names=['anchor/archived/identity/pass','refit/fixed/identity/relative','refit/lr_std/identity/lr',
           'refit/fixed/uniform0.1/relative','refit/fixed/diagonal/relative','refit/transported/diagonal/relative',
           'refit/fixed/unit_sd/relative','refit/scale_aware/identity/relative',
           'native/transported/diagonal/relative','native/transported/unit_sd/relative']
    both_table=table(['Model','Method','Standard AUROC±SD','Disjoint AUROC±SD','Disjoint Δidentity[system CI]'],
                     [[mo,name,fmt(sm(mo,name,'standard')),fmt(sm(mo,name)),ci(pair(mo,name))]for mo in models for name in names])
    effect_names=['refit/fixed/diagonal/pass','refit/fixed/diagonal/relative','refit/fixed/unit_sd/relative','refit/fixed/uniform0.1/relative','refit/scale_aware/identity/relative']
    effect_table=table(['Controlled disjoint contrast vs corresponding identity score','Llama ΔAUROC[95%CI]','Gemma ΔAUROC[95%CI]','Llama ΔAP','Gemma ΔAP'],
                      [[name]+[ci(pair(mo,name))for mo in models]+[f"{pair(mo,name,metric='AP')['mean']:+.4f}"for mo in models]for name in effect_names])
    inv_table=table(['Model','Split','Branch','Cells','Gate breaches','Maximum normalized score error','Maximum raw error','Maximum pair-inversion fraction'],
                   [[r['model'],r['split'],r['operation']+'/'+r['regularizer'],r['cells'],r['failures'],f"{r['maximum_normalized_error']:.3g}",f"{r['maximum_absolute_error']:.3g}",f"{r['maximum_pair_inversion']:.3g}"]for r in checks])
    native_rows=[]
    for mo in models:
        for scale in ('diagonal','unit_sd'):
            for score in ('pass','relative'):
                name=f'native/transported/{scale}/{score}'
                f=M[(M.model==mo)&(M.split=='disjoint')&(M.method==name)]
                native_rows.append([mo,scale,score,ci(pair(mo,name)),f"{pair(mo,name,metric='AP')['mean']:+.4f}",f"{f.strict_pair_inversions.mean():.2%}",f"{f.top_quintile_overlap.mean():.2%}"])
    native_table=table(['Model','Native KMeans scale','Score','Δdisjoint AUROC[CI]','ΔAP','Mean reversed test-pair share','Mean top20% overlap'],native_rows)
    cond_rows=[]
    for mo in models:
        for family in ('fixed','transported','scale_aware'):
            for scale in ('identity','diagonal','unit_sd'):
                f=C[(C.model==mo)&(C.split=='disjoint')&(C.operation=='refit')&(C.regularizer==family)&(C.scale==scale)]
                if f.empty:
                    f=C[(C.model==mo)&(C.split=='disjoint')&(C.operation=='refit')&(C.regularizer=='fixed')&(C.scale=='identity')]
                worst=f.groupby(['system','seed'])[['max_condition','backmapped_max_condition']].max()
                cond_rows.append([mo,family,scale,f"{worst.max_condition.median():.3g}",f"{worst.backmapped_max_condition.median():.3g}",f"{f.min_eigenvalue.min():.3g}",f"{f.max_precision_residual.max():.3g}"])
    cond_table=table(['Model','Ridge','Scale','Median worst κ, scaled','Median worst κ, original units','Minimum covariance eigenvalue','Maximum precision residual'],cond_rows)
    gap_table=table(['Method','Llama same-input gap reduction[CI]','Gemma same-input gap reduction[CI]'],
                    [[name]+[ci(find(G,model=mo,method=name))for mo in models]for name in ['refit/transported/diagonal/relative','refit/fixed/diagonal/relative','refit/fixed/unit_sd/relative','refit/fixed/uniform0.1/relative','refit/scale_aware/identity/relative','native/transported/diagonal/relative']])
    linear_table=table(['Method minus existing standardized LR','Llama Δdisjoint AUROC[CI]','Gemma Δdisjoint AUROC[CI]'],
                       [[name]+[ci(find(L,model=mo,split='disjoint',method=name,metric='AUROC'))for mo in models]for name in ['refit/fixed/identity/pass','refit/fixed/identity/relative','refit/fixed/uniform0.1/relative','refit/scale_aware/identity/relative']])
    per=pd.read_csv(ROOT/'per_system.csv');per=per[per.split=='disjoint']
    system_rows=[]
    for (mo,system),f in per.groupby(['model','system']):
        d=dict(zip(f.method,f.AUROC));base=d['refit/fixed/identity/relative']
        system_rows.append([mo,system,f'{base:.4f}']+[f'{d[name]-base:+.4f}'for name in ['refit/fixed/diagonal/relative','refit/fixed/unit_sd/relative','refit/scale_aware/identity/relative','native/transported/diagonal/relative']])
    per_table=table(['Model','System','Identity relative AUROC','Fixed diagonal Δ','Fixed unit-SD Δ','Scale-aware Δ','Native diagonal Δ'],system_rows)
    technical=[
    '# Does covariance regularization explain Clotho’s coordinate-scaling sensitivity?',
    '## Decision and execution status',
    '**It was not running and had not started.** Local processes/tmux, scripts, logs/results, private-Hub inventory, PROGRESS.md and the latest component report were audited. Only the prior completed study existed; this experiment was its stated next step. No duplicate was launched. The protocol was committed before outcomes (configs/clotho_covariance_scaling_protocol.json); this study then ran250 CPU jobs with four workers and no LLM generation, GPU extraction, new layer, parser or PCA fit.',
    '**Answer:** fixed isotropic ridge is a directly identified cause of scaling sensitivity **when initialization is held fixed**. Correct ridge transport removes all observed ranking changes. It is not the only cause for unconstrained refits: recomputing KMeans initialization in scaled coordinates changes rankings even with correct ridge. This does not establish that numerical sensitivity explains the original unseen-template penalty.',
    'The determinant-one diagonal scaling lowers relative-density disjoint AUROC0.6925→0.5782 (Llama) and0.6973→0.5920 (Gemma) under fixed ridge; transported ridge retains0.6925/0.6973. Native scaled KMeans instead reverses19.20%/21.28% of test-pair orderings on average for relative density. A stable scale-aware ridge can be less accurate:0.6492/0.6837. Mathematical invariance, estimator behavior and improved prediction are separate findings.',
    '## 1. Frozen data, controls and bounded design',
    'Use the250 archived fair_leakfree jobs:13 Llama systems/12 Gemma systems × standard/template-disjoint × seeds0–4. Keep exact ordered100 references (90 real+10 synthetic), labels, test inputs, original k/ridge and fair layers16/21. FAIL remains≤2 strict successes among generations5–9. Each job’s original selected PCA coordinates are fixed, with dimension varying by job; this is **not a new universal PCA16 representation**. No PCA directions, normalization before PCA, dimension selection or acquisition is rerun. GT template identities and test labels never determine scaling, ridge, initialization or fitting.',
    'The original cached PCA was transductive (fit on all510 unlabeled vectors, including test covariates). That inherited representation is held fixed. New scaling/ridge statistics use reference inputs only; this study does not turn the original PCA into a strictly inductive pipeline.',
    'Five positive diagonal transformations: identity; uniform×0.1; uniform×10; determinant-one geomspace(0.1,10,d); and unit reference SD (D_j=1/SD of PCA coordinate j over all100 references). The latter is scaling of existing PCA coordinates, not the earlier raw-hidden-state standardization followed by a new PCA. Every method has full13/12-system, five-seed coverage; no fallback or excluded job.',
    'All new refits use float64 arithmetic on the exact archived float32 coordinate values, full-covariance GMMs, original per-job component count and λ,30 restarts,500 iterations, tolerance0.001 and original seed. Original PASS/relative/LR score vectors remain as untouched anchors. Identity-refit AUROC and AP match the archived controls for every job; score values are not asserted bit-identical across arithmetic precisions. Logistic regression is the existing balanced C1 control with reference-only coordinate StandardScaler, not a different hidden representation.',
    '## 2. Mathematical invariance versus refitting',
    r'For row vectors, (z^prime=zD), with positive diagonal (D). A consistently transformed mixture has (mu_k^prime=mu_kD), (Sigma_k^prime=DSigma_kD), and unchanged weights. Then (log p_D(zD)=log p(z)-log|det D|). PASS risk changes only by the constant (+log|det D|); the FAIL/PASS log-density ratio cancels that constant exactly. AUROC, AP and rankings should be invariant. This is a change-of-variables identity, not evidence of better failure prediction.',
    'The analytical branch transports already-fitted parameters and precision-Cholesky matrices; it performs no fitting. It checks both the original archived PASS model and the two experimental identity densities. Matrix-ridge EM is separately refitted on transformed reference inputs: every weighted component covariance update adds a supplied positive matrix R. Its scalar special case matches stock sklearn exactly in four real-class checks. A separate SciPy Gaussian-mixture calculation independently checks the Jacobian identity.',
    table(['Ridge policy','Definition in scaled coordinates','Meaning in original units'],[
       ['Fixed',r'R_D=λI',r'R_back=λD^{-2}; rescaling changes the statistical regularizer'],
       ['Transported',r'R_D=λD²',r'R_back=λI; original prior retained'],
       ['Scale-aware',r'R_D=α diag Var_ref(ZD), α=λ/mean Var_ref(Z), frozen once',r'Diagonal-scale-equivariant; at identity trace matches original ridge, but shape changes']]),
    'For uniform multiplier s, fixed λ after scaling is equivalent to λ/s² before scaling. Thus×0.1 implies100×stronger regularization, while×10 implies100×weaker regularization. Eight direct real-data refits at the corresponding effective λ verify this identity, maximum normalized density error4.47e−12. α for the scale-aware rule is fixed before rescaling; recalibrating it separately after every D would test a different rule. This diagonal rule is not a claim of general affine or rotation invariance.',
    'Controlled initialization: compute each of30 seeded KMeans responsibility matrices once in the unscaled reference coordinates, separately for PASS/FAIL; reuse them across D and ridge policies. Iterate EM on the genuinely transformed inputs with the supplied matrix ridge. Correct ridge plus equal initial responsibilities makes each EM update equivariant, not just the final scoring formula. The native-initialization diagnostic recomputes KMeans on the diagonal/unit-SD inputs, preserving the seed and all other settings. Same seed does not imply same initial responsibilities under a changed distance metric.',
    '## 3. Numerical validation and conditioning',
    f"Validated{V['jobs']} jobs/{V['score_cells']:,} score cells;750 old source files unchanged. Two isolated job replays reproduce metadata, all scores and saved GMM parameters exactly. Original anchor AUROC tolerance remains1e−12. All{A['estimates_checked']:,} macro/paired/bootstrap estimates were independently reconstructed; maximum macro error{A['maximum_macro_error']:.3g}. No NaN/Inf score, nonconverged fit, nonpositive recorded covariance eigenvalue, fitting exception or invariance-gate breach occurred. Normalized gates were preregistered (analytic1e−8; controlled refit/LR1e−6), not relaxed.",
    inv_table,
    'Across all models/splits/scales, maximum normalized changes: analytical transport4.35e−13; transported controlled refit7.34e−11; scale-aware controlled refit2.27e−9; standardized LR3.57e−13. All have zero strict pair inversions and zero tie-status changes, hence identical AUROC/AP and top-quintile membership. Their small nonzero density errors are floating-point residuals, not prediction gains or material failures of invariance.',
    'Covariance condition numbers depend on coordinate units. The table reports descriptive medians of each job’s worst component/class condition number, plus minimum eigenvalue and maximum |Σ·precision−I| residual. Repeated seeds/components are not independent inferential units. Backmapping removes the arbitrary units. Global maximum scaled condition is1.69e7, minimum recorded eigenvalue4.59e−8 and maximum precision residual1.89e−10; this is not a catastrophic numerical breakdown.',
    cond_table,
    '## 4. Empirical prediction results: all scoring methods',
    'AUROC and AP (FAIL-positive average precision, not trapezoidal PR area) are calculated within each job. Means±sample SD are five per-seed system macro averages. Failure prevalence is identical across methods in a job and saved in metrics.csv/summary.json. System-cluster95% intervals resample systems4,000 times (seed20261112), retaining allfive seeds. Lines, test pairs, components and the14,250 score cells are not independent samples. No test-selected configuration is recommended and no multiple-comparison-adjusted significance claim is made.',
    prevalence_table,
    main_table,
    'The density policies have the same class references, k and covariance capacity. Fixed-versus-transported within a scaling changes only the covariance ridge with initial responsibilities held fixed. The following compares each score against its own identity fit; transported fits have the identical identity AUROC/AP, so these are also the fixed-minus-transported effects for the fixed-ridge rows.',
    effect_table,
    'Invertible scaling discards no hidden information. For the diagonal transform its Jacobian is1, so even the absolute density should be unchanged under analytical transport. The large fixed-ridge losses therefore identify a regularizer/units effect under this controlled fitting protocol, not information loss, a PCA-direction intervention or ordinary floating-point round-off. This reproduces across both LLMs.',
    'The stronger effective ridge from uniform×0.1 improves relative-density AUROC by+0.0246 for Llama but only+0.0082 with an interval crossing0 for Gemma. It can hurt PASS-only ranking. These are prespecified sensitivity results, not a new test-tuned model or universal regularization recommendation.',
    linear_table,
    'Transported ridge retains the original predictions; it does not improve them. The scale-aware diagonal rule is perfectly stable under the tested coordinate changes yet lowers relative-density AUROC by−0.0433 (Llama) /−0.0137 (Gemma; interval includes0). Scale invariance is a consistency property, not sufficient evidence of a statistically useful covariance prior. This specific rule should not be promoted as a better failure predictor.',
    '## 5. Initialization is a separate mechanism',
    native_table,
    'These refits use the **correct transported ridge** but native scaled-space KMeans. Diagonal relative-density pair inversions average19.20%/21.28% on disjoint inputs, despite macro AUROC changes only+0.0118/−0.0002. The inversion denominator is all unordered test pairs, including within-label pairs; many reorderings need not change AUROC, which compares FAIL against PASS only. Positive and negative system-level changes can also cancel. These are initialization/optimization effects, not a violation of the density change-of-variables theorem or evidence that a correct ridge alone guarantees estimator equivariance. The experiment does not prove global optimality of any multistart fit.',
    '## 6. Standard versus template-disjoint performance',
    both_table,
    'The table above shows full original test populations, which differ between split types. To investigate the extra unseen penalty, the following uses identical standard/disjoint test-index intersections, identical labels, and paired differences in gaps. Positive means a smaller gap relative to the corresponding identity score. Read absolute disjoint AUROC alongside gap changes.',
    gap_table,
    'Correctly transported refits preserve the gap exactly, because they preserve predictions. Stronger effective ridge from×0.1 narrows the relative-score gap by0.0330 [0.0031,0.0627] in Llama, but0.0111 [−0.0125,0.0346] in Gemma. Thus one bounded ridge intervention affects the gap in one model; it is not replicated evidence that numerical scaling sensitivity explains Clotho’s original generalization gap. Fixed diagonal and unit-SD scaling often worsen the gap. Earlier raw-versus-standardized PCA also changed directions, so its loss cannot be causally decomposed by subtracting these different-representation results.',
    '## 7. Per-system outcomes and alternative explanations',
    per_table,
    'Small class reference sets, high-dimensional full covariance, mixture initialization, covariance bias and template-level dependence remain relevant. This study isolates a coordinate-units/ridge mechanism at frozen directions, and separately exhibits initializer geometry. It does not identify a specific failed template type, remove dependence among logs, explain all U/S1 gains, or establish downstream/parser improvements. Natural-distribution deployment and reference200 are not evaluated. Test labels/GT never enter fitting or regularization selection.',
    'The diagnostic scaling range is prespecified, not an estimate of how often deployment would change units. A fixed isotropic ridge encodes a prior in the chosen coordinates; changing that prior is not necessarily a software bug. Transport is required when the intention is only to re-express the same model. The native-initialization branch was bounded to transported ridge at two nonuniform scalings; this design does not provide a full ridge×initialization interaction or an additive percentage attribution. Reference-variance rules include the10 synthetic references by design. The matrix-EM implementation uses pinned sklearn1.6.1 internals; another version must pass scalar-equivalence checks before its results are accepted.',
    '## 8. Research decision and one next step',
    '**Mathematical finding:** analytical transformation preserves density ratios and score ranks as expected. **Estimator finding:** covariance transport restores controlled-refit invariance; native initialization can independently change the fitted solution. **Empirical finding:** keeping an absolute isotropic ridge while changing units can lose≈0.10–0.11 disjoint AUROC in relative scoring; the scale-aware consistency fix is not an accuracy improvement. These claims are deliberately separate.',
    '**Next step:** repeat the earlier PCA-direction comparison at a fixed dimension, using common canonical initialization responsibilities and a full reference-covariance ridge with a scalar coefficient frozen in the canonical representation. Keep references, labels and same-space LR fixed. A covariance rule transforming consistently within a retained subspace removes unit artifacts; common responsibilities remove initializer-distance artifacts. Remaining differences would then test the retained PCA subspace rather than repeat this units experiment. No outer-test regularization search or new LLM inference is justified.',
    '## Reproduction and persistence',
    'New sources: log_parsing_covariance_scaling.py, log_parsing_covariance_analysis.py, validate_clotho_covariance_scaling.py, check_clotho_covariance_aggregation.py, log_parsing_covariance_report.py and log_parsing_covariance_restore.py. Protocol/environment are under configs/clotho_covariance_scaling_*. No core Clotho file changed. The runner uses an advisory lock, per-job completion records and execution.json (PID/status/timestamps) to prevent concurrent duplicate CLIs; completed jobs resume without refits. The restored PCA/label bundles suffice: no raw hidden extraction, tokenizer/model weight or torch dependency is needed.',
    '```bash\nset -a; source ~/.clotho_env; set +a\n.venv-mechanisms/bin/python experiments/log_parsing_covariance_restore.py\n.venv-mechanisms/bin/python experiments/log_parsing_covariance_scaling.py --workers 4\n.venv-mechanisms/bin/python experiments/log_parsing_covariance_analysis.py\n.venv-mechanisms/bin/python experiments/validate_clotho_covariance_scaling.py\n.venv-mechanisms/bin/python experiments/check_clotho_covariance_aggregation.py\n.venv-mechanisms/bin/python experiments/log_parsing_covariance_report.py\n```',
    'Results under results_phase3/clotho_covariance_scaling/: jobs/*/*/*/seedN.json (indices/labels, metrics, conditioning, convergence), seedN.npz (scores), seedN_models.npz (fitted weights/means/covariances/precision factors, scale diagonals and canonical starts); metrics.csv, covariance_diagnostics.csv, per_system.csv, per_seed.csv; summary/paired/contrasts/paired_vs_lr and shared_input_gap_reductions JSON/CSV with compressed bootstrap draws; shared_input_jobs_*.csv; source_hashes, launch_audit, execution, validation, aggregation_validation, invariance_summary/failures, identity_metric_agreement, computational_accounting, environment and persistence_validation metadata. Previous results are preserved. All new artifacts are uploaded and remotely hash-checked by the repository persistence workflow.',
    ]
    for name,caption in [('uniform_scaling','Fixed ridge varies with units; transported and scale-aware controlled refits are flat.'),('invariance_and_refitting','Analytical numerical residuals versus true fitting changes under native initialization.'),('ridge_effects','Paired fixed-minus-transported effects, same starts and test inputs; system-cluster95% intervals.'),('covariance_conditioning','Covariance conditioning after backmapping into original coordinate units.')]:
        technical.append(f'![{caption}](../results_phase3/clotho_covariance_scaling/figures/{name}.png)')
    (EXP/'reports/clotho_covariance_scaling.md').write_text('\n\n'.join(technical)+'\n')
    korean=[
    '# PCA 좌표의 단위만 바꾸었는데 Clotho 예측이 달라지는 이유',
    '**확인 결과 이 실험은 실행 중도 완료 상태도 아니었다.** 이전 연구의 다음 과제로만 적혀 있었고 로컬 프로세스/로그/결과와 원격 결과 저장소에 해당 작업이 없었다. 사전 프로토콜을 커밋한 뒤250개 CPU 작업을 실행했다. 새 LLM 출력, 층, 파서, PCA는 만들지 않았다.',
    '**핵심 답변:** 초기 책임확률을 고정하면, 고정된 isotropic 공분산 ridge가 좌표 스케일 민감성의 원인임을 직접 확인할 수 있다. 그러나 실제로 KMeans도 다시 계산하면 올바른 ridge를 사용해도 순위가 달라진다. 따라서 모든 민감성을 정규화 하나로 설명하거나, 기존 미관측 템플릿 격차가 수치오차 때문이었다고 결론내리면 안 된다.',
    '## 1. 무엇을 바꾸고 무엇을 고정했는가?',
    'Llama13개/Gemma12개 시스템, 표준/템플릿 비중복 분할, 시드0–4, 참조100개(실제90+합성10)의 기존 실험을 그대로 썼다. 원래 선택된 PCA 좌표를 파일에서 읽었다. 각 작업의 차원은 원래 값이며 모든 작업을16차원으로 다시 만든 실험이 아니다. 참조 순서, 학습 라벨, 테스트 인덱스, 층16/21, 성분 수k와 원래 λ는 고정했다. FAIL은 기존 생성5–9 중 엄격 일치가2개 이하인 경우다. 이 작업에서는40회 라벨로 바꾸지 않았다.',
    '원래 PCA는 라벨 없는 전체510개 벡터에 적합되어 테스트 입력의 공변량도 포함한다. 그 표현을 그대로 보존했다. 이번D와ridge는 참조에서만 계산하지만 전체 파이프라인이 새로 inductive PCA가 되었다고 주장하지 않는다.',
    '좌표에 양의 대각행렬D만 곱했다. 전체×0.1/×10, 대각 원소가0.1부터10까지인 부피 보존 변환, 참조에서 계산한 좌표별 SD의 역수를 사용한 변환이다. 모두 역행렬이 있으므로 입력 정보 자체는 잃지 않는다. 참조만으로 SD와 ridge를 정하며 테스트 정답이나 GT 템플릿은 전혀 사용하지 않는다. 이전 연구의 “은닉 특징을 표준화한 뒤 PCA 재적합”과는 다르다.',
    '## 2. 수학적으로 같아야 하는 것은 무엇인가?',
    r'좌표를 (z^prime=zD)로 바꿀 때, 평균은 (mu^prime=mu D), 공분산은 (Sigma^prime=DSigma D)로 함께 바꿔야 한다. 그러면 (log p_D(zD)=log p(z)-log|det D|)다. PASS 위험에는 모든 입력에 같은 상수만 더해지고, (log p_{FAIL}-log p_{PASS})에서는 상수가 상쇄된다. 그러므로 순위·AUROC·AP는 같아야 한다.',
    '예를 들어 두 좌표를0.1배/10배로 바꾸면 determinant=1이다. 모형도 일관되게 변환하면 밀도 값 자체도 그대로다. 전체를10배로 바꾸면 밀도 수치는 달라지지만 모든 입력에 공통인 보정이므로 순위는 유지된다. 이 등식은 정답 라벨을 전혀 쓰지 않는 수학적 성질이며 실패 예측이 좋아졌다는 실험 결과가 아니다.',
    '우리는 이미 적합된 모형을 변환하는 **분석적 변환**, 변환한 입력에 EM을 실제로 다시 실행하는 **통제 재적합**, 변환 공간에서 KMeans부터 다시 하는 **native 재적합**을 나눴다. 앞의 두 경우 초기 책임확률을 같게 두면 다른 요인을 분리할 수 있다. “동일 seed”만으로는 KMeans 초기 상태가 같다고 보장되지 않는다.',
    '## 3. 같은 λ가 왜 다른 정규화인가?',
    r'원래 EM 공분산 갱신이 (C_k+lambda I)라고 하자. 좌표를D배 한 뒤에도 같은 (lambda I)를 더하면 원래 단위에서는 (C_k+lambda D^{-2})가 된다. 전체s배는 원래 단위에서 (lambda/s^2)를 쓰는 것과 같다. 올바른 운반은 (R_D=lambda D^2)다.',
    '작은 예: 원래 분산0.02에 λ=0.1을 더하면0.12다. 좌표를10배 하면 분산은2가 된다. 여기에0.1을 더하고 원래 단위로 돌리면2.1/100=0.021로,0.12와 다른 모형이다. ridge도100배한10을 더해야(2+10)/100=0.12가 된다. 따라서 스케일에 따른 예측 변화가 반드시 부동소수점 오류인 것은 아니다. 실제 통계 모형의 공분산 가정이 바뀐 것이다.',
    r'세 번째 대조는 (R_D=alpha,diag(Var_{ref}(ZD)))다. (alpha=lambda/mean(Var_{ref}(Z)))는 원래 좌표에서 한 번 정하고 고정한다. 원점에서 ridge의 평균 크기는 원래 λ와 같지만 모양은 다르다. 분산이D²배 되므로 이 규칙은 대각 스케일에 일관된다. D마다α를 새로 정하거나 평균분산×I만 쓰면 이 규칙과 다른 실험이다. 모든 scale-aware 정규화의 우월성을 시험한 것도 아니다.',
    '## 4. 실제 검증 수치는 무엇인가?',
    '250개 작업,14,250개 점수 셀을 검증했다. 기존750개 소스 파일과 원래 AUROC는 보존했다. 원래 점수는 그대로 두고 새 재적합은 같은 좌표값을float64로 계산했다. identity 재적합의 AUROC/AP는 매작업 기존 대조와 같으며, 점수의 비트 동일성을 주장하지는 않는다. 두 작업은 별도 임시 경로에서 점수와 저장된 GMM 매개변수를 다시 만들어 동일함을 확인했다.',
    '분석적 변환의 최대 정규화 점수 오차는4.35e−13, 일관된 ridge의 통제 EM 재적합은7.34e−11, scale-aware 재적합은2.27e−9, 표준화 LR은3.57e−13이었다. 이들에서 순서가 반전된 쌍과 동점 상태 변화는0이며 AUROC/AP도 그대로다. 이 작은 잔차는 수학적 등식의 유한정밀도 구현 오차이지 성능 개선이 아니다. 분산/정밀도 곱 잔차의 최대는1.89e−10이고 모든 기록된 공분산이 양의 정부호였다.',
    '행렬 ridge의 λI 특수 경우는 실제 PASS/FAIL 데이터 네 번에서 stock sklearn과 오차0으로 일치했다. SciPy로 독립 계산한 가우시안 혼합 Jacobian 잔차는4.88e−15였다. 전체×0.1/×10을 원래 단위의 λ/s²로 직접 재적합한8개 대조도 최대 정규화 오차4.47e−12로 일치했다. λ를 여러 값 탐색해 좋은 수치를 고른 것이 아니라 같은 계산임을 확인한 것이다.',
    '## 5. 정답 예측 성능에는 얼마나 큰 차이를 만들었는가?',
    main_table,
    effect_table,
    '가장 분명한 통제 관찰은 부피 보존 대각 변환이다. 같은 원래 PCA 정보/학습 라벨/초기 책임확률인데 고정 ridge에서는 상대밀도 비중복 AUROC가 Llama0.6925→0.5782, Gemma0.6973→0.5920으로 떨어졌다. 차이는−0.1143 [−0.1635,−0.0642]/−0.1053 [−0.1511,−0.0591]이다. ridge도 운반하면 두 모델 모두 원래 값으로 유지된다. 이는 이 통제 실험에서 ridge의 좌표 단위 의존성이 예측 민감성을 만든다는 증거다. PCA 축을 바꾼 효과나 정보 손실로 설명할 수 없다.',
    '전체×0.1은 유효 ridge를100배 키운다. 상대밀도는 Llama+0.0246, Gemma+0.0082 좋아졌지만 Gemma 구간은0을 포함한다. PASS 점수에는 항상 유리하지 않다. 이 결과를 보고 최적λ를 골랐거나 공통 개선법을 만들었다고 해석하면 안 된다. 반대로 scale-aware는 스케일에는 안정적이지만 상대밀도0.6492/0.6837로 원래0.6925/0.6973보다 낮다. 수학적으로 안정적이라는 성질과 좋은 실패 경계는 다르다.',
    linear_table,
    '## 6. 올바른 ridge를 써도 KMeans를 다시 하면 왜 달라지는가?',
    native_table,
    '대각 변환에서 native KMeans를 쓰면 상대밀도의 테스트 쌍 순서가 평균19.20%/21.28% 반전된다. 반면 평균 AUROC 변화는+0.0118/−0.0002에 그친다. 쌍 반전은 PASS끼리/FAIL끼리의 순서도 포함하지만 AUROC는 FAIL−PASS 쌍만 본다. 또 어떤 시스템의 이득과 다른 시스템의 손해가 평균에서 상쇄될 수 있다. 따라서 AUROC 평균이 비슷하다는 이유로 점수/우선순위가 안정적이라고 말할 수 없다.',
    '이 경우도 ridge는 올바르게 운반했다. 바뀐 것은 유클리드 거리로 계산한 초기 군집과 이후 비볼록 EM 적합 경로다.30개 시작점을 써도 전역 최적해를 보장하지 않는다. 그러므로 **ridge 불변성만 고치면 실제 전체 추정기가 언제나 불변이다**라는 주장도 성립하지 않는다. 초기화 기준을 함께 고정해야 본 실험의 순위 불변성이 얻어진다.',
    '## 7. 이것이 미관측 템플릿 격차를 설명하는가?',
    gap_table,
    '위 격차는 표준/비중복의 테스트 입력 교집합으로 계산했다. 양수는 identity 점수 대비(표준AUROC−비중복AUROC)가 줄었다는 뜻이다. 일관된 ridge는 예측을 그대로 보존하므로 격차도 전혀 줄이지 않는다. ×0.1 상대밀도는 Llama에서0.0330 [0.0031,0.0627] 줄였지만 Gemma는0.0111 [−0.0125,0.0346]로 불확실하다. 두 모델에 재현된 미관측 격차 설명이라고 할 수 없다.',
    '좌표 민감성의 메커니즘은 확인했지만 기존 Clotho의 실제 좌표계가 얼마나 잘못된 λ를 쓰는지는 별개다. 이전 표준화PCA는 축까지 바꾸었다. 이번 손해를 그때의 손해에서 빼서 PCA 축과 ridge의 기여율을 계산하면 다른 실험 모집단/표현을 섞게 된다. 또한 이번 예측 차이는 주로 다른 공분산 사전/초기화가 만든 것이며 심각한 수치오류가 원래 일반화 격차를 만들었다는 증거가 아니다.',
    '## 8. 지표와 통계는 어떻게 계산했는가?',
    'AUROC는 임의 FAIL/PASS 쌍에서 FAIL 위험이 더 큰 비율(동점0.5)이다. AP는 recall 증가량으로 가중한 precision 평균이다. 실패율은 모델/시스템/분할별로 다르므로 함께 저장했다. 모든 모델은 같은 테스트/참조를 쓰며 제외된 작업은 없다. 시드별로 시스템을같은 가중치로 평균한 뒤5시드 평균±표본SD를 낸다. CI는 시스템을4,000회 재표집하고 그 안의5시드를 같이 보존했다.14,250개 셀이나 로그 쌍을 독립 실험으로 세지 않는다. 같은 부호가5시드에 나왔다는 이유만으로 유의성을 주장하지 않는다.',
    '## 9. 연구 판단과 다음 한 단계',
    '**수학적 사실:** 밀도와 ridge를 일관되게 운반하면 순위는 불변이다. **추정기 관찰:** 고정 λ와 바뀐 초기화가 각각 좌표 의존성을 만들 수 있다. **예측 관찰:** 이로 인해 같은 정보에서도 약0.10–0.11 AUROC 손해가 생길 수 있지만, 불변성을 가진 scale-aware 규칙 자체는 정확도를 개선하지 못했다. 세 주장을 구별해야 한다.',
    '**다음 단계:** 앞선 PCA 방향 비교를 고정 차원에서 반복하되, 원래 참조 공간에서 만든 초기 책임확률을 공통으로 쓰고 전체 참조 공분산에 비례하는 ridge를 사용한다. 계수는 canonical 표현에서 한 번 정해 고정한다. 참조·학습 라벨·같은 좌표의 LR도 고정한다. 이렇게 단위/ridge와 초기 거리 효과를 먼저 통제한 뒤 남는 차이가 PCA가 보존한 부분공간 때문인지 시험하는 것이 타당하다. 이번 결과만으로 바깥 테스트를 보고 λ를 고르거나 새 LLM 생성을 할 이유는 없다.',
    '## 재현과 보존',
    '프로토콜/환경: configs/clotho_covariance_scaling_*. 실행·통계·보고: log_parsing_covariance_{scaling,analysis,report}.py. 검증: validate_clotho_covariance_scaling.py, check_clotho_covariance_aggregation.py. 복원: log_parsing_covariance_restore.py(--outputs-only는 완료 산출물만). 작업 상태/PID는 execution.json, 전체 수치/분할/라벨/적합 매개변수/점수/조건수/부트스트랩은 results_phase3/clotho_covariance_scaling/에 있다. 네 PDF/PNG 그림도 저장했다. CPU만 사용하고 새 데이터나 이전 결과를 덮어쓰지 않았다. 원격 보존 해시는 persistence_validation.json에서 확인한다.'
    ]
    (EXP/'reports/clotho_covariance_scaling_explained.md').write_text('\n\n'.join(korean)+'\n')
    print('Rendered validated technical report and Korean explanation.')


if __name__=='__main__':main()
