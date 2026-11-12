"""Prior-work inventory, controlled pilot and Korean researcher explanation."""
from cpu_research_common import EXP,load,save,table
from log_parsing_reference_selection import ROOT,POLICIES
import pandas as pd,numpy as np
D=pd.read_csv(ROOT/'metrics.csv');E=load(ROOT/'estimates.json');PR=load(ROOT/'prior_inventory.json');V=load(ROOT/'validation.json')
LABEL=dict(random='Random',structural='Masked-text diversity',hidden='Hidden-state diversity',uncertainty='LR uncertainty',clotho_archived='Archived Clotho acquisition')
def get(model,name,kind='macro',metric='AUROC'):return next(e for e in E if e['model']==model and e['name']==name and e['kind']==kind and e['metric']==metric)
def contrast(model,left,right,b=100,metric='AUROC'):
 try:e=get(model,f'{left} - {right} @{b}','paired',metric)
 except StopIteration:
  e=get(model,f'{right} - {left} @{b}','paired',metric).copy();e['mean']=-e['mean'];e['ci_low'],e['ci_high']=-e['ci_high'],-e['ci_low']
 return e

def fmt(e):return f"{e['mean']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]"
def f(x):return f'{x:.4f}'
def macros():
 rows=[]
 for b in (50,100):
  for policy in POLICIES:
   vals=[]
   for model in ('llama','gemma'):
    e=get(model,f'{policy}/{b}');ap=get(model,f'{policy}/{b}',metric='AP');vals += [f"{e['mean']:.4f} ± {e['seed_sd']:.4f}",f(ap['mean'])]
   rows.append([b,LABEL[policy],*vals])
 return table(['Total budget','Policy','Llama AUROC','Llama AP','Gemma AUROC','Gemma AP'],rows)
def differences():
 rows=[]
 for b in (50,100):
  for policy,base in [('structural','random'),('hidden','random'),('uncertainty','random'),('clotho_archived','random'),('structural','clotho_archived'),('hidden','clotho_archived'),('uncertainty','clotho_archived')]:rows.append([b,LABEL[policy]+' − '+LABEL[base],fmt(contrast('llama',policy,base,b)),fmt(contrast('gemma',policy,base,b))])
 return table(['Budget','Paired AUROC contrast','Llama Δ [95% CI]','Gemma Δ [95% CI]'],rows)
def diagnostics():
 rows=[]
 for model in ('llama','gemma'):
  for policy in POLICIES:
   g=D[(D.model==model)&(D.policy==policy)&(D.budget==100)];rows.append([model,LABEL[policy],f(g.fail_n.mean()),f(g.structural_coverage.mean()),f(g.structural_redundancy.mean()),f(g.GT_coverage.mean()),f(g.hidden_cover_mean.mean())])
 return table(['Model','Policy@100','FAIL labels','Structural coverage','Structural redundancy','Candidate-GT coverage','Mean hidden covering distance'],rows)
def systems(model):
 d=D[(D.model==model)&(D.budget==100)].pivot(index=['system','seed'],columns='policy',values='AUROC').groupby('system').mean()
 return table(['System',*[LABEL[p]for p in POLICIES]],[[s,*[f(r[p])for p in POLICIES]]for s,r in d.iterrows()])
def cost_table():
 rows=[]
 for policy in POLICIES:
  g=D[(D.policy==policy)&(D.budget==100)];tim=g.selector_total_to100_s.mean();rows.append([LABEL[policy],90,10,f(g.deployment_candidate_state_count.mean()),'unavailable'if pd.isna(tim)else f(tim),'8 LR fits'if policy=='uncertainty'else'original adaptive GMM/CV, not retimed'if policy=='clotho_archived'else'none'])
 return table(['Policy@100','New real labels','Common initial labels','Preselection state count incl.initial','Selector CPU sec mean','Adaptive selector updates'],rows)
def quantity_table():
 return table(['Policy','Llama100−50 AUROC [CI]','Gemma100−50 AUROC [CI]'],[[LABEL[p],fmt(get('llama',p+' budget100 - 50','paired')),fmt(get('gemma',p+' budget100 - 50','paired'))]for p in POLICIES])

def main():
 jobs=[load(p)for p in (ROOT/'jobs').glob('*/*/seed[0-4].json')];pools={m:[len(a['pool'])for a in jobs if a['model']==m]for m in ('llama','gemma')}
 pooltable=table(['Model','Systems×seeds','Real candidate pool min/mean/max','Test n min/mean/max'],[[m,str(13 if m=='llama'else 12)+'×5','/'.join(f'{x:g}'for x in (min(pools[m]),np.mean(pools[m]),max(pools[m]))),'/'.join(f'{x:g}'for x in (min(500-v for v in pools[m]),np.mean([500-v for v in pools[m]]),max(500-v for v in pools[m])))]for m in ('llama','gemma')])
 inventory=table(['Existing experiment','Selection policy','Budget','Predictor','Evaluation population','Verified numerical result','What established','What remained unknown'],[
 ['Phase3 tuning','Uniform random vs original half-entropy/half-diversity','50/100/200 incl.10 initial','Adaptive original PASS GMM','Llama13 standard, old layer21/R10 labels; test complements differ','@1000.7296 random vs0.7639 balanced; held-out L16@1000.7402 vs0.7703','Original active policy can help in that setup','Not fair5 disjoint, fixed LR, or common standard tests'],
 ['Fairness Llama','Original references vs seeded uniform reference set','100/200 incl.10','MDSA, GMMb, lookups','13 systems×5; same disjoint tests and common standard T′','Cold100 MDSA0.6563 active vs0.6446 random; GMMb0.5641 vs0.5756 (see exact prior JSON)','Reference-policy effects already studied; predictor dependence exists','No LR/reference-only PCA16, no isolated diversity or LR uncertainty'],
 ['Fairness Gemma','Same policies','100/200 incl.10','Same baseline families','12×5; same matched T′ construction','Cold100 MDSA≈0.655 vs0.648; GMMb≈0.609 vs0.607','Modest, model-dependent effects','No demonstration that a new deployable policy improves the modern simple control'],
 ['Original quantity','Same adaptive policy extended','100→200','Original Clotho','Same cold tests; standard complement shrinks','Cold L0.644816→0.656625; G0.659430→0.671148','More references sometimes help; cold budget comparison exists','Class/coverage/geometry components and fixed-LR quantity effect'],
 ['Budget strategies / decomposition','Random, Clotho active, frequency-first Drain representatives','B50/100/200 real labels +10 initial; then B extra checks','Clotho/Drain/random rankings','500-input natural/capped, not template-disjoint future tests','L natural B100: found2047.2 Clotho vs1690 random; acquired FAIL927.8 vs847.6','Active selection helps failure discovery in observed population','Not future-template AUROC or equal total budgets with the100-including10 convention'],
 ['Drain strategy','Observable group-first / propagation','Same B+initial and2B checks','Drain labels+Clotho','Same500 population','L natural B100 Drain+Clotho1983.4 vsClotho2047.2','Template-like grouping is not automatically beneficial','Does not test GT-diverse training references for unseen-template prediction'],
 ['Source/target transfer','Source pools plus random target labels','Target0/10/25/50, extra source supervision','LR/MLP/refit/Clotho','Held-out systems; matched variants in paired report','L capped paired refit AUROC≈0.680→0.720→0.768→0.790','Target-label quantity/source adaptation useful','Not competing target acquisition policies or the same total-label budget'],
 ['U/S1, streams/gates/review','Original references fixed; risk/frequency review policies','Nominal100 or review budgets, source labels extra','Source-trained discriminators / parser-review queues','GT-disjoint suite or stream review, differing objectives','S1 cold0.7490/0.7619; U0.7764/0.7732','Source/input signals and review prioritization','Not a controlled reference-acquisition intervention'],
 ['Mechanism/component/PCA/Gaussian','Original acquired references held fixed','100; Gaussian bootstrap perturbs within same100','Densities/LR','13/12 systems, five seeds, matched cold tests','Reference PCA16 LR0.7188/0.7138; shared Gaussian0.7294/0.7179','Scoring/estimator/representation issues; conditional instability','No new pool acquisition; bootstrap is not fresh label selection']])
 audit=fr'''# Reference acquisition audit: existing evidence and the actual remaining gap

## Status and decision

At launch no exact acquisition task was running, completed locally or saved remotely (zero matching private-Hub files). AGENTS, latest PROGRESS, paper_numbers, mechanism/component/PCA/Gaussian reports and explanations, original sampling/CV code, tuning/fairness/strategy/transfer/U and budget reports were inspected. No GPU or old predictor rerun was needed for the audit.

**Reference acquisition is not an unstudied topic here.** Random versus original acquisition, label quantity, group-first selection and source-label adaptation already have evidence. The narrow missing comparison was their interaction with the fixed reference-only PCA16+whitened C1 LR recipe, under explicit label revelation and the exact old template-disjoint tests. Phases1–3 supported one bounded pilot, not a new acquisition algorithm. After the pilot, **stop acquisition-method development on this benchmark**: no tested policy establishes a robust improvement over the original reference history across both LLMs. Preserve conditional diversity-versus-random findings and the Llama budget50 exception; do not claim acquisition never matters.

## Existing inventory and numerical verification

{inventory}

Verification: `log_parsing_reference_acquisition_audit.py` restored250 existing fairness job outputs and independently reconstructed every report AUC cell at its printed precision; no models were fit. It also recomputed original Clotho100/200 macros from250 canonical jobs and checked saved strategy per-seed means. Tuning's saved four-decimal aggregate/report were verified; its old raw-model fitting was not replayed. Exact fairness numbers are in `prior_inventory.json`, not inferred from rounded table subtraction. `prior_fairness_metrics.csv` retains system/seed rows.

**Found narrative mismatch:** `log_parsing_strategy.md` puts455 versus419 acquired failures beside its B100 example. The saved detailed table identifies455.2 versus418.6 as the B50 Clotho-versus-Drain quantities; the same comparison at B100 is927.8 versus842.0. Separately, the machine decomposition gives B100 Clotho-versus-random acquired counts927.8 versus847.6 and total found2047.2 versus1690. Do not silently change the comparator when correcting the budget. The older strategy budget is B *real* examples plus10 initial references, unlike nominal reference100=90 real+10 here. We document this without editing the earlier artifacts.

## A–G gap assessment

|Factor|Already known|New controlled observation / unresolved boundary|
|---|---|---|
|A quantity|Original100/200 and target-label adaptation curves exist|Pilot50/100 compares identical cold tests with fixed learner; extra labels often help more than small policy point differences|
|B class balance|Actual PASS counts measured, learner class weights balanced|Original “balanced” means uncertainty/diversity, not PASS/FAIL balancing. No future-label oracle-balanced policy was treated as deployable; class effects remain noncausal diagnostics|
|C diversity|Original max-min component and Drain group-first strategies exist|Isolated masked-text/hidden max-min selection with modern LR was missing; not a novel selector|
|D difficulty|Original GMM component-assignment entropy and failure-discovery budgets exist|LR margin uncertainty now tested, without improvement over random/archived baseline. Margin uncertainty is not measured intrinsic label noise|
|E redundancy|Group concentration and coverage measured in review work|Pilot diversity reduces current-pool redundancy; this does not prove future label transfer|
|F distribution shift|GT-disjoint outer evaluation and cross-system transfer exist|No genuine prospective/chronological future-template acquisition experiment; random GT holdout cannot settle it|
|G predictor interaction|Recent estimator studies froze original acquisition|Narrow fixed-LR policy comparison is new; shared-Gaussian acquisition interaction remains untested; no formal cross-predictor interaction is estimated or used to restart the density branch|

## Feasibility and leakage audit

{pooltable}

Saved fair-layer raw vectors cover all500 suite inputs plus10 initial examples. Standard PCA-study metadata partitions all510 into reference/test with their original five-generation labels; joining them reconstructs a complete label truth store. A row called “test” in an older standard experiment may be a candidate in the new disjoint study: only the *current* disjoint test IDs are forbidden. The oracle hides every unselected current candidate label and permanently rejects current test queries. Old generation outputs need not be regenerated or rescored.

Frozen disjoint test IDs define candidate pool=500\test. GT IDs validate separation and compute diagnostics only; selectors never receive them. Candidate input text, masked patterns and fair hidden states are available before labels. Common10 initial labels plus queried real labels are the only supervised input to fitting/updates. No policy receives a fully labeled candidate array; the driver alone constructs a private `LabelOracle`. Selection actions are committed before it returns labels. Events record known-label counts, selected IDs and returned labels. Each policy reveals exactly40/90 real labels at nominal50/100, with no duplicates/test queries.

Original Clotho history is available as ordered100 references. Its monotone append history provides a legitimate50-prefix. P5 faithfully replays that archived outcome, **not** a new counterfactual algorithm execution. Original acquisition used component-assignment entropy, adaptive GMM/PCA, continuous acquired pass-rate supervision for inner selection, and transductive all510 PCA. New policies use majority-label feedback and candidate-only features. All feedback arises from the same stored five-generation supervision; the original baseline has richer feedback/transductive provenance and unavailable historical runtime. Do not call P5 a freshly profiled candidate-only deployable policy or silently strip these dependencies.

The existing standard test complement depends on original Clotho acquisition. New policies may select those IDs, so blindly retaining that test would leak training examples. This bounded pilot runs only existing disjoint populations; it neither constructs a new standard holdout nor reports an invalid standard-to-disjoint gap. Existing fairness T′ results remain historical evidence.

## Bounded pilot and final continue/stop assessment

Frozen protocol commit7a79681, `configs/clotho_reference_selection_protocol.json`, before new outcomes.125 jobs, five policies, budgets50/100,1,250 cells. Fixed learner recipe only: no C/layer/dimension/policy tuning. Selected references naturally change learned PCA/whitening; this is a total policy effect under a fixed pipeline, not label-composition-only causality. Details: [pilot report](clotho_reference_selection_pilot.md).

At100, hidden diversity−archived Clotho is+0.0075[−0.0010,0.0150] Llama/+0.0016[−0.0054,0.0088] Gemma; masked-text diversity+0.0045[−0.0166,0.0230]/+0.0061[−0.0082,0.0189]. None is clear in both models. Hidden diversity versus random is+0.0214[−0.0021,0.0451]/+0.0177[0.0068,0.0293]. Thus a benefit over uniform selection is plausible/conditional, already consistent with prior evidence, but not a new method advantage over the existing policy. Llama50 hidden−original+0.0170[0.0024,0.0331] is a real secondary finding without Gemma replication.

No oracle labels/GT features are needed by P1–P4, but hidden/uncertainty require all candidate states while random/text diversity need selected states only. Equal label counts do not imply equal compute/human minutes. No new actual extraction occurred; prospective state costs are reported as counts, not zero.

**Decision:**the narrow gap was worth a small controlled check; the result does not justify developing a novel acquisition method. Do not repeat these policies under new names, assume balanced labels/diversity are universally better, or restart PCA/density tuning. Remaining class/coverage/PCA mediation and real future shift are not causally isolated. If one confirmation were commissioned, freeze original versus one predetermined diversity policy on an untouched chronological unseen-template population, using a candidate-only reproduction of the original rule and matched supervision granularity/state-label cost accounting. That is the single falsifiable follow-up, not a recommendation for another search on these test systems.

## Ten-question research decision

1. **Already adequately studied?** Generic random/active, quantity and group-first questions have prior evidence; the strict fixed-LR acquisition comparison was missing.
2. **Still unknown?** Class/coverage/PCA causal mediation, other predictors and genuinely future chronological shift; not a claim that all policies are exhausted.
3. **Feasible/leakage-free?** Yes as a cached-label simulation, with candidate-only P1–P4 and immutable tests. P5 is a qualified archived transductive baseline.
4. **Equal-label improvement?** Gemma100 diversity improves over random; Llama50 hidden diversity improves over original. No tested budget gives a clear original-policy advantage across both LLMs.
5. **Across systems/LLMs?** Heterogeneous and not robust against the existing policy in both; HPC/OpenSSH are explicit counterexamples.
6. **Mechanism explained?** No: counts/coverage/redundancy associations are descriptive and PCA changes with selected references.
7. **Deployable?** P1–P4 need only available candidate inputs/acquired labels, with different feature costs. GT is an evaluation diagnostic; P5 retains historical transduction/richer feedback.
8. **New method justified?** No, stop method development on this benchmark rather than expand a sampling search.
9. **Distinct from Clotho?** The controlled learner interaction is new evidence; max-min/uncertainty policies are known and original Clotho already includes diversity. No algorithmic novelty established.
10. **Single decisive follow-up?** A preregistered original-versus-one-diversity comparison on an untouched chronological template-shift population, reproducing the original rule on candidates only, with equal feedback and explicit state/label costs; conditional confirmation only, not another tuning loop here.

Machine artifacts: prior inventory/source hashes, pilot manifests/predictions, metrics/paired bootstrap estimates, leakage/poison/replay records and figures in `results_phase3/clotho_reference_selection/`. Old results are unchanged. Reports and outputs are persisted/remote-hash checked before updating PROGRESS.
'''
 (EXP/'reports/clotho_reference_acquisition_audit.md').write_text(audit)
 pilot=fr'''# Reference selection pilot: fixed LR, equal labels, unseen templates

## Research question and frozen protocol

Does reference policy improve unseen-template prediction when the final learner recipe is held fixed? [Audit](clotho_reference_acquisition_audit.md) establishes the missing interaction, previous results and limitations. This is a bounded comparison of known selectors, not a new acquisition method. The predictor was selected from the earlier Gaussian study before acquisition outcomes: reference-only raw PCA16, full-reference whitening, balanced C1 LR(tol1e−10,max_iter3000), fair layers16/21. LR avoids inheriting the earlier Gaussian ridge coefficient's transductive calibration. No density tuning, new LLM/GPU/hidden extraction or output features.

125 original GT-disjoint jobs: Llama13/Gemma12 systems×seeds0–4, exact old test IDs/five-generation labels. No additional exclusion. No test input is selectable and no GT appears in a selector. Failure-positive AUROC/AP; FAIL is at most2 strict matches among samples5–9. The benchmark is repeatedly studied, so this remains exploratory, not a fresh confirmatory test.

{pooltable}

## Candidate construction, oracle and policies

Candidate pool is the complement of the immutable suite test IDs, verified GT-disjoint. The full saved labels reside only in an oracle, not feature/selection APIs. All start with the same10 labeled synthetic examples. New P1–P4 share10 uniformly random real warmup inputs (seed+42000). Selections then proceed in batches10. No label from a batch is revealed until its entire action is chosen. Nominal budget50/100 means40/90 real examples; evaluating50 uses only the observed50-prefix, even though the saved acquisition manifest continues to100.

- **Random:**seeded uniform order.
- **Masked-text structural diversity:**candidate-only char3–5 TFIDF(max512) on input text with numeric/hex tokens masked, then greedy max-min cosine distance from selected real anchors. This is an observable proxy, not a ground-truth parser/template ID.
- **Hidden diversity:**raw candidate+initial-only PCA16, greedy max-min Euclidean coverage of candidate states. No heldout covariance. The geometry is fixed before acquiring labels and is distinct from the final reference-fitted PCA.
- **LR uncertainty:**after20 total labels, fit the fixed recipe on acquired examples, select the10 smallest absolute decision margins, reveal labels and update. Eight updates to100. An undefined class/rank would invoke a logged random warmup fallback; **none occurred**. Model margin under balanced C1 is not a calibrated failure probability or empirical five-generation label uncertainty. High-risk sampling was not separately tested.
- **Archived Clotho:**exact original order and50-prefix; its original mix is five GMM component-entropy and five max-min samples per batch (diversity alone before sufficient PASS labels). It is not class-balanced selection. Historical transductive PCA/continuous-reference feedback and unavailable acquisition timing remain explicit. Its reference sets are evaluated with the *same LR*, not original PASS density.

Final PCA/whitening are refit only on each selected set. Fixed predictor means fixed algorithm/hyperparameters, **not the same fitted projection after reference inputs change**. Membership, class balance, order and learned projection can all mediate the total acquisition effect; no causal fraction is assigned to each.

## Main metrics: identical test populations

Mean±sample SD of the five seed macros; AP is average precision. Each method/budget retains13/12 systems, no line-weighted pooling. Test prevalence is unchanged by policy. System-cluster percentile bootstrap4000 draws(seed20261112), retaining five seeds and methods together; unadjusted intervals, no significance from seed wins alone.

{macros()}

{differences()}

At100, diversity point estimates beat uniform sampling, but Llama uncertainty includes zero. Neither diversity strategy clearly improves the archived original policy across both models. The50-budget hidden-diversity improvement is established for Llama only. LR uncertainty does not beat random despite matching the final learner's objective. No winning policy is selected using final test results.

AP comparisons are all machine-readable; at100 hidden AP0.8341/0.7434 versus original0.8310/0.7406, structural0.8292/0.7478, random0.8127/0.7294, uncertainty0.8166/0.7255 (Llama/Gemma). Higher AUROC point means do not guarantee AP gains: structural Llama AP is slightly below original. P3/P5 differ in initialization, selection geometry, feedback and transduction: this is not a pure entropy-removal ablation. No new standard results or generalization-gap reduction is claimed: the original standard evaluation population is acquisition-dependent.

## Reference quantity

{quantity_table()}

The archived policy100−50 improves+0.0288/+0.0252 with positive intervals in both models, whereas most policy−archived100 point differences are below0.01. This is not a universal effect-size ranking: random Llama quantity gain is uncertain and policies have different learning curves. Both budgets use exactly the same outer inputs. Increasing labels also changes fitted PCA/whitening under the frozen recipe, so this is a pipeline learning curve, not pure class-count causality.

## Composition, diversity, redundancy and shift

{diagnostics()}

Coverage and GT coverage concern **candidate** groups; heldout GT coverage is zero by construction for every policy. Structural redundancy=1−distinct masked patterns/real selected inputs. Hidden distances use the same candidate-fitted raw PCA per job and cannot be compared in absolute units across systems/LLMs. Diversity improves these objective-related metrics by design; that is an algorithm check, not independent evidence of future predictive information.

P2/P3 acquire fewer FAIL examples than archived Clotho while slightly improving AUROC points. P4 is closer to a50:50 label balance but has no gain and greater structural redundancy/poorer candidate coverage. Therefore neither “more FAIL” nor “balanced labels” is sufficient as a simple explanation. These are not randomized mediator interventions. Class composition, hidden geometry, coverage and PCA adaptation co-vary. `associations.json` contains only descriptive correlations of system-averaged paired effects; no line/seed-level mediation test. We did not independently measure intrinsic difficulty/noise or prove coverage causes transfer. Random GT holdout is not chronological future shift.

## Equal labels versus computation

{cost_table()}

State counts include10 initial examples; random/text diversity require90 new selected states, hidden/uncertainty the entire real candidate pool, and original acquisition saw all510 including future test covariates. Final test-state scoring is common and separate. In this CPU replay all states/labels are cached, so actual new extraction/generation count is zero; this does not make deployment feature acquisition free. Text selection also scans the candidate texts. Feature preparation time and final fit/prediction time are saved separately from selector time. Fixed masked-text/hidden distance preparation is needed by the corresponding diversity policy; for other policies it was computed only as a common diagnostic. LR uncertainty uses its own selected-reference projections, not the fixed hidden-diversity distance matrix. The selector column is total to100; a separate50 selector wall time was not measured. Historical original acquisition runtime is unavailable, not zero. Equal example counts are not measured equal human annotation minutes, and prior source/strategy studies use different label accounting.

## Per-system counterexamples

### Llama

{systems('llama')}

### Gemma

{systems('gemma')}

At100, structural diversity−original is+0.0821 on Llama HPC but−0.0395 on Gemma HPC. Llama OpenSSH loses0.0918, Gemma0.0180. Hidden diversity loses0.0315/0.0203 on OpenSSH. Better current-pool coverage is not uniformly better unseen-template prediction. These descriptive systems were not used to pick a deployable selector.

## Competing explanations and limits

|Hypothesis|Evidence|Counterevidence/unresolved|
|---|---|---|
|Density-oriented acquisition mismatches LR|A different fixed learner can change the value of selected sets|Matching acquisition uncertainty to LR does not help; inherited original histories remain competitive|
|Diversity improves transferable coverage|More input/GT coverage, lower redundancy, Gemma gains over random|No clear cross-model gain over original; HPC/OpenSSH counterexamples; proxy coverage is partly optimized by construction|
|Uncertainty finds useful hard cases|Valid adaptive selection uses only acquired labels|No clear AUROC gain; more redundant/narrow coverage. Not proof those labels are noisy; true uncertainty not measured|
|Class counts drive gains|Counts differ measurably|Near-balanced P4 is not best, and P2/P3 do not acquire more FAIL than P5. No class-matched causal ablation|
|There is little remaining policy headroom over original|Both budgets lack clear cross-model improvement over original; original already includes diversity|Llama50 conditional gain and finite-system uncertainty leave room; no optimality theorem or prospective guarantee|

Fixed13/12 systems, capped500 inputs, five noisy-generation labels, synthetic warmup and repeated research use limit external validity. The archived baseline has richer reference feedback and transductive geometry; it is not an otherwise identical candidate-only selector. The new policies use binary feedback and no future feature access. Reference histories are original outcomes, not live counterfactual reruns or fair historical timing benchmarks. Lower state demand for masked-text selection is potentially useful, but no prespecified noninferiority margin was tested, so equal performance at lower cost is not established. This pilot studies one final learner, not a full predictor×policy factorial or all shared-Gaussian interactions. No policy/class/representation hyperparameter search occurred.

## Validation, artifacts and decision

125/125 jobs,1,250 cells,440 cluster estimates. All queries unique, candidate/initial-only, exact40/90 real label budgets; all125 GT separations independently checked.14 poison/noninterference checks on predeclared Gemma BGL0/Llama HDFS0: flipping unacquired labels leaves next LR batches unchanged; flipping all labels leaves label-free orders unchanged; corrupting heldout states leaves the adaptive next batch unchanged; the oracle rejects test queries. Archived LR100 vectors are reused bitwise, raw reconstruction AUROC matches1e−12. Two complete job replays reproduce all26 arrays and non-timing metadata exactly; independent metrics/bootstrap reconstruction max error {V['aggregation_error']:.3e}. Old input hashes unchanged;25 Git suite hashes checked. No acquisition fallback/ineligible system or new GPU work.

Saved root `results_phase3/clotho_reference_selection/`:ordered pools/tests, labels/source hashes, per-step oracle histories, selected prefixes, update/fallback timing, risks, structural/hidden distance matrices, diagnostic coverage, per-system/per-seed metrics, all paired policy/budget estimates and bootstrap draws, prior inventory/verification, privacy and deterministic replay records,3 PDF+PNG figures. Label truth is stored for audit but never given to selectors as an array. `validation.json` records the isolation checks. Remote persistence hashes are checked before PROGRESS update.

**Decision:**reference policy can matter conditionally; this pilot does not justify a new acquisition method or broad search. Known diversity policies beat random most clearly for Gemma, and hidden50 helps Llama, but none robustly improves the original reference policy across both LLMs. Stop acquisition-method development on these test systems. If a single confirmation were commissioned, preregister original versus one fixed diversity policy on an untouched chronological unseen-template population, restricting the original rule to available candidate features and equal supervision granularity, with explicit state costs; it could falsify a real future-coverage benefit. Do not tune selectors here or revive density/PCA searches.

```bash
set -a; source ~/.clotho_env; set +a
# If needed, restore saved artifacts first:
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_restore.py  # --outputs-only for inspection
# CPU dependencies pinned in configs/clotho_reference_selection_environment.txt
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_analysis.py
.venv-mechanisms/bin/python experiments/validate_clotho_reference_selection.py
.venv-mechanisms/bin/python experiments/log_parsing_reference_acquisition_audit.py
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_report.py
```

![Budget versus policy](../results_phase3/clotho_reference_selection/figures/budget_policy.png)

![Matched effects versus random](../results_phase3/clotho_reference_selection/figures/paired_vs_random.png)

![System-level hidden diversity effects](../results_phase3/clotho_reference_selection/figures/system_hidden_vs_random.png)
'''
 (EXP/'reports/clotho_reference_selection_pilot.md').write_text(pilot)
 (EXP/'reports/clotho_reference_acquisition_explained.md').write_text(korean())
 save(ROOT/'decision.json',dict(prior_topic_already_studied=True,narrow_fixed_LR_gap_was_new=True,pilot_feasible_and_executed=True,primary_policies_tuned_on_test=False,clear_cross_model_gain_over_original=False,branch='stop acquisition-method development on current benchmark',conditional_findings=['Gemma100 diversity versus random','Llama50 hidden diversity versus archived original'],next_single_confirmation='untouched chronological template shift: candidate-only original rule versus one fixed diversity policy, equal feedback and explicit state/label cost accounting',new_method_novelty_claim=False))

def korean():
 return fr'''# 어떤 참조를 고르는가: 기존 연구와 고정 LR 선택 실험을 구분하기

**결론:** 참조 선택이 전혀 중요하지 않은 것은 아니다. 그러나 이번 자료에서 새 acquisition 방법을 개발할 근거는 부족하다. 원래 Clotho도 이미 uncertainty+diversity를 사용했다. 알려진 diversity 방법은 random보다 좋아지는 조건이 있지만, 원래 참조 이력보다 두 LLM에서 일관되게 나은 새 정책은 확인되지 않았다. 작은 양의 평균 차이를 새 방법의 독창성으로 포장하지 않는다.

## 1. 이미 무엇을 알고 있었고 왜 또 확인했는가?

Random 대 원래 ‘balanced’ 선택은 이미 비교되었다. 예전 Llama layer21 표준 분할에서100참조 AUROC는 random0.7296, balanced0.7639였다. 하지만 그 실험은10회 생성 라벨, adaptive PCA/GMM, 방법마다 달라지는 test complement를 썼다. 현재 five-generation GT비중복/고정LR의 답으로 그대로 가져올 수 없다.

이후 fairness 실험은 동일 GT비중복 test에서 MDSA 참조를 바꾸었다. Llama는 원래 참조 약0.656 대 random0.645, Gemma0.655 대0.648이었다. GMMb에서는 Llama가 오히려 random0.576 대 원래0.564였다. 즉 ‘참조 정책은 예측기와 상호작용한다’는 문제는 원래부터 있었다.250개 저장 fairness 작업으로 보고서 모든 AUC 셀을 다시 계산해 확인했고, 원래 Clotho100/200 quantity도250개 canonical 결과에서 재집계했다. 기존 예측기를 재학습하지 않았다.

실패 발견 budget 연구도 있었다. Llama natural에서 B100 실제 입력을 라벨링하고 나머지 중100개를 더 확인하면 Clotho는2047.2 실패, random은1690을 찾았다. 단, 이는 future-template AUROC가 아니다. 또 그 B100은실제100+초기10개라 이번 nominal100=실제90+초기10개와 다르다. 원래 짧은 strategy 설명의455 대419는 B50 수치가 B100 문장에 섞인 것이며, 같은 Clotho 대 Drain 비교의 B100 획득 FAIL 수는 상세표에서927.8 대842.0이다. 별도로 machine decomposition에서 Clotho 대 random B100은927.8 대847.6이고 총발견수는2047.2 대1690이다. Budget을바로잡으며비교대상까지몰래바꾸지않는다. 예전 파일을 덮어쓰지 않고 이 audit에 바로잡아 적었다.

기존 PCA/밀도/LR 연구는 참조를 고정했다. Gaussian 실험의 bootstrap도 같은100개 안의 perturbation이지 새 후보를 어떤 순서로 라벨링할지 비교한 것이 아니었다. 그래서 **고정 reference-PCA16 LR에서, 실제로 숨겨진 candidate 라벨을 하나씩 얻는 정책 비교**만 남은 좁은 질문이었다. 이 비교는 단순 이름 바꾸기가 아니라 predictor–acquisition interaction의 미측정 부분이다. 기존 연구가 없었다고 주장하지 않는다.

## 2. 저장된 모든 라벨이 있는데 어떻게 ‘모르는 라벨’을 흉내 내는가?

Oracle는 정답을 보관하지만 selector는 이를 보지 못한다. 비중복 outer test ID를 먼저 고정하고, 나머지 suite 입력만 candidate로 둔다. GT ID는 그 분할이 정말 비중복인지 검사할 때만 쓴다. 정책 함수는 텍스트/은닉 특징, 후보 ID, 지금까지 관측한 라벨만 받는다.

예를 들어 초기10개와 실제10개를 관측했다면 LR은20개의 라벨로만 학습한다. 다음10개 ID를 먼저 고른 다음 Oracle가 그10개 라벨을 돌려준다. 후보 전체 라벨로 LR을 학습한 뒤 ‘uncertain’을 고르는 것은 하지 않는다. Oracle는 test ID, 중복 조회와100개 초과 조회를 거절한다. Manifest에 known_before·선택 ID·반환 라벨을 남긴다.50 budget 결과는50-prefix의 정보만 학습에 사용한다.

모든510개의 원래five-generation 라벨은 예전 표준 실험의 ref/test 메타데이터를 합치면 복원된다. 예전 표준 실험에서 test였다는 이름 자체가 현재 학습 금지는 아니다. **현재의 disjoint test**만 영구 차단하고, 현재 후보 라벨은 선택 후 비용을 지불했을 때만 공개한다. 이는 이미 완전 라벨링된 자료의 acquisition simulation이지 새 사람 실험이나 미래 데이터 수집이 아니다.

원래 Clotho는100개 순서가 저장되어 있어50-prefix까지 재사용할 수 있다. 다만 원래 알고리즘은 acquired pass rate로 내부 차원을 선택하고 unlabeled test state까지 본 global PCA를 썼다. 따라서 P5는 충실한 저장 이력 대조이지 binary-only oracle로 원래 GMM 알고리즘을 새로 실행한 것이 아니다. 더 풍부한 참조 피드백과 transductive provenance, 과거 실행시간의 부재를 숨기지 않는다. 그래도 current test/future candidate 라벨을 미리 본 정책이라는 뜻은 아니다.

## 3. 무엇을 바꾸고 무엇을 고정했는가?

LLM/Llama13·Gemma12 시스템, layer16/21, seed0–4, strict samples5–9 majority 라벨, 기존 GT비중복 test를 고정했다. Predictor는 참조전용 raw PCA16+전체 참조 covariance whitening+balanced C1 LR로 미리 정했다.16차원이나C를 정책 결과에 맞춰 바꾸지 않았다. 원래 ‘balanced acquisition’은 PASS/FAIL을 반반 맞추는 것이 아니라 **GMM 성분 entropy와max-min diversity를 반반** 쓰는 것이다. 원래 entropy는 실패확률 불확실성과 같지 않다.

새 네 정책은 실제10개 random warmup이 같다. P1은계속 random, P2는숫자/hex를 마스킹한 텍스트의char TFIDF 거리로diversity, P3는candidate-only PCA16 은닉거리로diversity, P4는관측라벨 LR의결정 margin이0에가까운10개를 고른다. P5는저장Clotho순서다. 앞의 네 정책은 GT나 candidate 출력을 사용하지 않는다. 새 LLM/GPU/은닉 추출도 없다.

Nominal50/100은공통초기10+실제40/90개의라벨이다. 후보가달라지면최종PCA도다시적합되므로 **고정된 것은 학습 recipe이지 이미 학습된 projection이 아니다**. 참조 선택의 전체 파이프라인 효과를 측정하지만, 라벨 구성의 효과와 PCA 변화의 효과를 나누지는 못한다.

표준 split에서는원래Clotho참조의complement가test라서새정책이그test를선택할수있다. 그래서이번에는억지로표준test를유지하거나다른집단을같다고부르지않고, 기존disjoint test만평가했다. 새로운standard-to-disjoint gap 감소도 주장하지 않는다.

## 4. 실제 수치는 무엇이며 무엇을 뜻하는가?

AUROC는 FAIL 위험점수가 PASS보다큰확률(동점0.5)이며 AP는FAIL 양성 average precision이다. 시스템별지표를먼저구하고5seed macro의평균±표본SD를보고한다. 구간은4,000번시스템bootstrap하며그시스템의5seed를함께보존한다. 줄이나seed를독립시스템으로세지않았다.

{macros()}

{differences()}

100개에서hidden−random은Llama+0.0214[−0.0021,0.0451]/Gemma+0.0177[0.0068,0.0293]다. Structural−random도+0.0183[−0.0034,0.0400]/+0.0222[0.0003,0.0421]로Gemma에서더명확하다. 따라서‘다양성은아무영향없다’고할수없지만, 두모델에서확실히재현됐다고도할수없다.

더중요한original과의100개비교는hidden+0.0075[−0.0010,0.0150]/+0.0016[−0.0054,0.0088], structural+0.0045[−0.0166,0.0230]/+0.0061[−0.0082,0.0189]다. 모두0을포함한다. 원래정책도diversity를이미사용하므로random만이겨새기여가생겼다고말하면안된다.

50개에서는Llama hidden−original+0.0170[0.0024,0.0331]이실제양의근거다. 하지만Gemma는+0.0074[−0.0088,0.0271]로불확실하다. 이부조건을숨기지않되, 테스트를보고‘Llama50전용새정책’을만들거나모든seed승리만으로유의성을주장하지않는다. P4 LR uncertainty는100개에서random대비+0.0009/+0.0023이고구간은0을포함한다. 최종예측기와같은uncertainty를쓴다고자동으로좋아지지않는다.

{quantity_table()}

Original50→100은+0.0288/+0.0252로두모델구간이양수다. 기존정책보다새정책이더좋다는100개차이보다point효과가크다. 그러나Llama random의추가50라벨효과는불확실하므로‘더많으면항상개선’도아니다. 같은test에서학습budget이늘어난pipeline효과이며PCA와classcount도함께변한다.

## 5. FAIL을 더 많이 골라서 좋아졌는가? 다양성이 원인인가?

{diagnostics()}

Coverage는candidate의관측구조/GT종류중선택한비율이지미래test GT의coverage가아니다. Test GT coverage는모든방법0이다. Hidden distance는model별원래단위가달라Llama의0.1과Gemma의3을크기비교하지않는다. 같은model/pool안의정책차이만읽는다.

P2/P3는original보다FAIL라벨을더많이얻지않는데도AUROC point가약간높다. P4는거의반반에가까운class구성이지만성능우위가없고구조중복률은더크다. 따라서‘FAIL이많아서’, ‘반반이라서’라는단순설명은충분하지않다. 그렇다고classbalance의인과효과가없다고증명한것도아니다.

Diversity가자기가최적화한coverage를높이는것은예상되는algorithm check다. 그것이transfer가능한실패정보를더잘얻었다는독립증거는아니다. Class구성·coverage·PCA가동시에변하므로저장상관을mediation/원인으로읽지않는다. P4가참조를좁게골랐다는관찰은있지만, 그라벨이본질적으로noise가많다는추가40회검증은하지않았다.

가장중요한반례는HPC다. Structural−original이Llama에서는+0.0821인데Gemma에서는−0.0395다. Llama OpenSSH는−0.0918, Gemma는−0.0180이다. 현재후보구조를다양하게커버하는것과미지GT실패를잘예측하는것이같지않다.

## 6. 라벨 수가 같으면 비용도 같은가?

{cost_table()}

모든100개방법은실제90개+공통초기10개라벨을쓴다. 그러나random/텍스트diversity는선택된입력의state만필요하고, hidden/uncertainty는전체candidate state를라벨링전에필요로한다. 원래Clotho는510개전체와GMM/CV까지썼다. 지금CPU실험은저장state를읽어새추출0회지만배포에서도공짜라는뜻이아니다. 최종test입력의state는예측시별도로모두필요하다. 실제사람annotation시간도측정하지않았으므로equal count를equal human minutes라고부르지않는다. P5과거선택시간은NA이지0이아니다.

## 7. 무엇이 검증되었고 무엇이 남는가?

125작업×5정책×2budget=1,250셀이완료되었다. 추가제외/fallback은0이다. Test ID/라벨은원래와같고모든125개GT비중복을독립확인했다. Oracle는현재test/중복/초과조회를거절한다.14개poison검사에서아직공개하지않은라벨을바꿔도LR의다음선택은같고, label-free정책은전체라벨을바꿔도같다. Test state를바꿔도다음adaptive선택은같다. 원래LR100risk는bitwise재사용되고raw재계산AUC도1e−12에서일치한다. 두전체job재실행26개배열/비시간metadata와440개cluster추정치도독립검증했다.

하지만현재13/12개시스템을여러연구에재사용했고GT random holdout이지실제미래시간shift가아니다. 고정recipe에서PCA가참조선택에따라바뀌는mediator도남는다. P5는원래더풍부한feedback/global covariate를썼고, 다른정책은binary feedback/candidate-only라access가완전히같지않다. 새policy가배포가능한입력만사용한다는것과, 그효과가새시스템에확증됐다는것은다르다.

**판단:** 기존참조연구는있었지만현대simple predictor와의좁은상호작용은새롭게검사할가치가있었다. 결과는새acquisition방법개발을정당화하지못한다. 이benchmark에서다른이름의sampling/PCA/density탐색을계속하지말고, 조건부이득과반례를포함한통제된실증결과로정리한다. Original policy가 최적이라고 증명하거나 reference 선택이 무의미하다고도 하지 않는다. Structural policy의 적은 후보-state 비용은 유용할 수 있지만 noninferiority margin을 사전 정의해 검정하지 않았으므로 같은 성능을 보장하는 비용절감 방법이라고도 주장하지 않는다.

**만약확인실험하나를의뢰한다면:** original entropy/diversity 규칙을 candidate-only로 재현한 대조와 사전에 정한 diversity 정책 하나를 새로운 독립 chronological unseen-template population에서 동일 label 피드백과 명시적 state 비용으로 고정 비교한다. ‘현재coverage의작은이득이미래에도전달된다’는가설을가장직접반박/검증할수있다. 현재결과만으로그확장이나새방법개발을권고하는것은아니며, 여기서추가GPU실험은하지않았다.

[기존연구audit](clotho_reference_acquisition_audit.md), [상세pilot](clotho_reference_selection_pilot.md), `results_phase3/clotho_reference_selection/`의manifest·prior_inventory·metrics·paired bootstrap·validation·PDF/PNG를함께보면수치와한계를확인할수있다. 이전결과를덮어쓰지않았고원격hash보존후PROGRESS를갱신한다.
'''
if __name__=='__main__':main()
