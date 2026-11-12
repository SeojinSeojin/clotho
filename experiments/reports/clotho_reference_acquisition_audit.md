# Reference acquisition audit: existing evidence and the actual remaining gap

## Status and decision

At launch no exact acquisition task was running, completed locally or saved remotely (zero matching private-Hub files). AGENTS, latest PROGRESS, paper_numbers, mechanism/component/PCA/Gaussian reports and explanations, original sampling/CV code, tuning/fairness/strategy/transfer/U and budget reports were inspected. No GPU or old predictor rerun was needed for the audit.

**Reference acquisition is not an unstudied topic here.** Random versus original acquisition, label quantity, group-first selection and source-label adaptation already have evidence. The narrow missing comparison was their interaction with the fixed reference-only PCA16+whitened C1 LR recipe, under explicit label revelation and the exact old template-disjoint tests. Phases1–3 supported one bounded pilot, not a new acquisition algorithm. After the pilot, **stop acquisition-method development on this benchmark**: no tested policy establishes a robust improvement over the original reference history across both LLMs. Preserve conditional diversity-versus-random findings and the Llama budget50 exception; do not claim acquisition never matters.

## Existing inventory and numerical verification

| Existing experiment | Selection policy | Budget | Predictor | Evaluation population | Verified numerical result | What established | What remained unknown |
|---|---|---|---|---|---|---|---|
| Phase3 tuning | Uniform random vs original half-entropy/half-diversity | 50/100/200 incl.10 initial | Adaptive original PASS GMM | Llama13 standard, old layer21/R10 labels; test complements differ | @1000.7296 random vs0.7639 balanced; held-out L16@1000.7402 vs0.7703 | Original active policy can help in that setup | Not fair5 disjoint, fixed LR, or common standard tests |
| Fairness Llama | Original references vs seeded uniform reference set | 100/200 incl.10 | MDSA, GMMb, lookups | 13 systems×5; same disjoint tests and common standard T′ | Cold100 MDSA0.6563 active vs0.6446 random; GMMb0.5641 vs0.5756 (see exact prior JSON) | Reference-policy effects already studied; predictor dependence exists | No LR/reference-only PCA16, no isolated diversity or LR uncertainty |
| Fairness Gemma | Same policies | 100/200 incl.10 | Same baseline families | 12×5; same matched T′ construction | Cold100 MDSA≈0.655 vs0.648; GMMb≈0.609 vs0.607 | Modest, model-dependent effects | No demonstration that a new deployable policy improves the modern simple control |
| Original quantity | Same adaptive policy extended | 100→200 | Original Clotho | Same cold tests; standard complement shrinks | Cold L0.644816→0.656625; G0.659430→0.671148 | More references sometimes help; cold budget comparison exists | Class/coverage/geometry components and fixed-LR quantity effect |
| Budget strategies / decomposition | Random, Clotho active, frequency-first Drain representatives | B50/100/200 real labels +10 initial; then B extra checks | Clotho/Drain/random rankings | 500-input natural/capped, not template-disjoint future tests | L natural B100: found2047.2 Clotho vs1690 random; acquired FAIL927.8 vs847.6 | Active selection helps failure discovery in observed population | Not future-template AUROC or equal total budgets with the100-including10 convention |
| Drain strategy | Observable group-first / propagation | Same B+initial and2B checks | Drain labels+Clotho | Same500 population | L natural B100 Drain+Clotho1983.4 vsClotho2047.2 | Template-like grouping is not automatically beneficial | Does not test GT-diverse training references for unseen-template prediction |
| Source/target transfer | Source pools plus random target labels | Target0/10/25/50, extra source supervision | LR/MLP/refit/Clotho | Held-out systems; matched variants in paired report | L capped paired refit AUROC≈0.680→0.720→0.768→0.790 | Target-label quantity/source adaptation useful | Not competing target acquisition policies or the same total-label budget |
| U/S1, streams/gates/review | Original references fixed; risk/frequency review policies | Nominal100 or review budgets, source labels extra | Source-trained discriminators / parser-review queues | GT-disjoint suite or stream review, differing objectives | S1 cold0.7490/0.7619; U0.7764/0.7732 | Source/input signals and review prioritization | Not a controlled reference-acquisition intervention |
| Mechanism/component/PCA/Gaussian | Original acquired references held fixed | 100; Gaussian bootstrap perturbs within same100 | Densities/LR | 13/12 systems, five seeds, matched cold tests | Reference PCA16 LR0.7188/0.7138; shared Gaussian0.7294/0.7179 | Scoring/estimator/representation issues; conditional instability | No new pool acquisition; bootstrap is not fresh label selection |

Verification: `log_parsing_reference_acquisition_audit.py` restored250 existing fairness job outputs and independently reconstructed every report AUC cell at its printed precision; no models were fit. It also recomputed original Clotho100/200 macros from250 canonical jobs and checked saved strategy per-seed means. Tuning's saved four-decimal aggregate/report were verified; its old raw-model fitting was not replayed. Exact fairness numbers are in `prior_inventory.json`, not inferred from rounded table subtraction. `prior_fairness_metrics.csv` retains system/seed rows.

**Found narrative mismatch:** `log_parsing_strategy.md` puts455 versus419 acquired failures beside its B100 example. Saved machine decomposition identifies those as B50 quantities. For B100, the actual acquired counts are927.8 versus847.6; total found2047.2 versus1690. The older strategy budget is B *real* examples plus10 initial references, unlike nominal reference100=90 real+10 here. We document this without editing the earlier artifacts.

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

| Model | Systems×seeds | Real candidate pool min/mean/max | Test n min/mean/max |
|---|---|---|---|
| llama | 13×5 | 250/256.046/315 | 185/243.954/250 |
| gemma | 12×5 | 250/253.617/272 | 228/246.383/250 |

Saved fair-layer raw vectors cover all500 suite inputs plus10 initial examples. Standard PCA-study metadata partitions all510 into reference/test with their original five-generation labels; joining them reconstructs a complete label truth store. A row called “test” in an older standard experiment may be a candidate in the new disjoint study: only the *current* disjoint test IDs are forbidden. The oracle hides every unselected current candidate label and permanently rejects current test queries. Old generation outputs need not be regenerated or rescored.

Frozen disjoint test IDs define candidate pool=500\test. GT IDs validate separation and compute diagnostics only; selectors never receive them. Candidate input text, masked patterns and fair hidden states are available before labels. Common10 initial labels plus queried real labels are the only supervised input to fitting/updates. No policy receives a fully labeled candidate array; the driver alone constructs a private `LabelOracle`. Selection actions are committed before it returns labels. Events record known-label counts, selected IDs and returned labels. Each policy reveals exactly40/90 real labels at nominal50/100, with no duplicates/test queries.

Original Clotho history is available as ordered100 references. Its monotone append history provides a legitimate50-prefix. P5 faithfully replays that archived outcome, **not** a new counterfactual algorithm execution. Original acquisition used component-assignment entropy, adaptive GMM/PCA, continuous acquired pass-rate supervision for inner selection, and transductive all510 PCA. New policies use majority-label feedback and candidate-only features. All feedback arises from the same stored five-generation supervision; the original baseline has richer feedback/transductive provenance and unavailable historical runtime. Do not call P5 a freshly profiled candidate-only deployable policy or silently strip these dependencies.

The existing standard test complement depends on original Clotho acquisition. New policies may select those IDs, so blindly retaining that test would leak training examples. This bounded pilot runs only existing disjoint populations; it neither constructs a new standard holdout nor reports an invalid standard-to-disjoint gap. Existing fairness T′ results remain historical evidence.

## Bounded pilot and final continue/stop assessment

Frozen protocol commit7a79681, `configs/clotho_reference_selection_protocol.json`, before new outcomes.125 jobs, five policies, budgets50/100,1,250 cells. Fixed learner recipe only: no C/layer/dimension/policy tuning. Selected references naturally change learned PCA/whitening; this is a total policy effect under a fixed pipeline, not label-composition-only causality. Details: [pilot report](clotho_reference_selection_pilot.md).

At100, hidden diversity−archived Clotho is+0.0075[−0.0010,0.0150] Llama/+0.0016[−0.0054,0.0088] Gemma; masked-text diversity+0.0045[−0.0166,0.0230]/+0.0061[−0.0082,0.0189]. None is clear in both models. Hidden diversity versus random is+0.0214[−0.0021,0.0451]/+0.0177[0.0068,0.0293]. Thus a benefit over uniform selection is plausible/conditional, already consistent with prior evidence, but not a new method advantage over the existing policy. Llama50 hidden−original+0.0170[0.0024,0.0331] is a real secondary finding without Gemma replication.

No oracle labels/GT features are needed by P1–P4, but hidden/uncertainty require all candidate states while random/text diversity need selected states only. Equal label counts do not imply equal compute/human minutes. No new actual extraction occurred; prospective state costs are reported as counts, not zero.

**Decision:**the narrow gap was worth a small controlled check; the result does not justify developing a novel acquisition method. Do not repeat these policies under new names, assume balanced labels/diversity are universally better, or restart PCA/density tuning. Remaining class/coverage/PCA mediation and real future shift are not causally isolated. If one confirmation were commissioned, freeze original versus one predetermined diversity policy on an untouched chronological unseen-template population with matched supervision granularity and state/label cost. That is the single falsifiable follow-up, not a recommendation for another search on these test systems.

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
10. **Single decisive follow-up?** A preregistered original-versus-one-diversity comparison on an untouched chronological template-shift population, equal feedback and state/label costs; conditional confirmation only, not another tuning loop here.

Machine artifacts: prior inventory/source hashes, pilot manifests/predictions, metrics/paired bootstrap estimates, leakage/poison/replay records and figures in `results_phase3/clotho_reference_selection/`. Old results are unchanged. Reports and outputs are persisted/remote-hash checked before updating PROGRESS.
