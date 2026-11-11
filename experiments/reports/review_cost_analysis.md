# Review budget and label cost analysis

New analysis of16,320 stored outcomes: actual label counts, source acquisition/amortization and variable-cost frontiers. Existing equal-effort experiments were reused, not rerun. Base, strong, real AdaParser and residual-tail populations remain distinct. No rankings or hyperparameters changed; no curves were interpolated.

## Label-count audit

| Method | Real target labels | Source labels per target | Synthetic/examples and online feedback |
|---|---|---|---|
| Frequency / Random / consistency | 0 | 0 | Review actions only |
| Clotho reference100 | 90 | 0 | 10 synthetic tests; static stream rankings unavailable |
| U+V5 stream | 90 | 6000 Llama /5500 Gemma | 10 synthetic tests excluded from fitting; review feedback reused |
| D | 0 deployment warm-start | 6000 /5500 | Other systems:500 labelled inputs each; feedback refits |
| U-zero / U+V5-zero | 0 deployment warm-start | 6000 /5500 | Source-only U model plus review feedback |
| Strong parser (common) | 10 verified templates | 0 | Common to every strategy in that parser setting |
| Real AdaParser (common) | 32 labelled candidates | 0 | Common to every strategy; distinct from predictor labels |

Counts audited in source_data/source_ds/suite_ref and the stored reference100 protocol. Source labels are real capped-suite input labels, not nominal reference sizes. Source weights0.1/1 or repeated online weights do not create additional unique human acquisitions. The existing strong/base usize histories are the frozen U+V5 stream setting, not U-alone. U-alone and static Clotho review curves cannot be invented from capped-suite AUCs. Zero-target U is available for caches, not real AdaParser.

A “binary label” here is one hypothetical human correctness judgement per labelled input, although the experiment actually obtains PASS labels from stored generations/GT. Auditing five outputs might cost more than a single judgement; no measured human timings exist. We do not count machine-generated sample counts as five independent human labels. Synthetic tests, parser candidates, anomaly labels and review corrections are different resources. The32 AdaParser candidates are common existing template supervision; their original labelling effort is treated as32 review equivalents, and strong seeds as10. Sensitivity of this common cost shifts the budget axis without changing matched policy rankings.

## Cost definitions

Scenario A counts target labels plus online reviews; source acquisition is disclosed separately. Scenario B single deployment also charges all source labels (6000/5500). Scenario B amortized acquires a union of6500/6000 source-labelled inputs once for deployment to13/12 systems:500 per system; it does not charge each leave-one-out source pool again. This is an analytical portfolio scenario, not a new multi-system experiment. Labels can be reused across base/strong episodes of the same model; no cross-model or cross-parser correctness-label reuse is assumed. Further repeated-episode amortization is possible but not measured here.

For review/binary-label cost ratio r∈{1,5,10}, cost in review equivalents = common parser template reviews + online reviews + (target labels + scenario-specific source labels)/r. A reviewed item supplies its feedback label once; refitting or repeating its weight is not charged again. Tail analyses include the20 prior frequency reviews once before additional tail reviews. No dollars, GPU seconds or binary labels are added without a stated conversion.

Budgets20/50/100/150/200 select the largest actually stored affordable review depth, independently of outcomes. Stored full histories support5/10/20/50/100; zero-target histories and real-parser histories stop at50 or queue exhaustion. Thus some methods leave budget unused. These are measured attainable frontiers, not extrapolated optimal curves; especially a50-vs100-depth comparison cannot establish that a zero-target method would still lose if its missing100-review curve were run. Exhausted queues cost only actual reviews. Real AdaParser has no stored reviewed FTA, so it is unavailable rather than filled with another parser’s metric.

## Paired target-effort results

| Model | Budget | Outcome | Method − frequency | Systems | Mean delta | System-cluster95%CI | System win probability |
|---|---|---|---|---|---|---|---|
| llama | 20 | lines | D × size | 11 | -0.0113 | [-0.0352,+0.0034] | 0.59 |
| llama | 20 | lines | U-zero × size | 11 | +0.0002 | [-0.0058,+0.0049] | 0.64 |
| llama | 20 | groups | D × size | 11 | +0.0287 | [+0.0036,+0.0661] | 0.82 |
| llama | 20 | groups | U-zero × size | 11 | +0.0321 | [+0.0059,+0.0729] | 0.91 |
| llama | 100 | lines | D × size | 11 | -0.0005 | [-0.0037,+0.0016] | 0.64 |
| llama | 100 | lines | U+V5 × size | 11 | -0.1600 | [-0.2698,-0.0637] | 0.00 |
| llama | 100 | lines | U-zero × size | 11 | -0.0426 | [-0.0949,-0.0074] | 0.14 |
| llama | 100 | groups | D × size | 11 | +0.0207 | [+0.0042,+0.0397] | 0.82 |
| llama | 100 | groups | U+V5 × size | 11 | -0.3964 | [-0.5810,-0.2124] | 0.00 |
| llama | 100 | groups | U-zero × size | 11 | -0.0884 | [-0.1614,-0.0270] | 0.23 |
| llama | 150 | lines | D × size | 11 | -0.0005 | [-0.0037,+0.0016] | 0.64 |
| llama | 150 | lines | U+V5 × size | 11 | -0.0380 | [-0.0860,-0.0054] | 0.14 |
| llama | 150 | lines | U-zero × size | 11 | -0.0426 | [-0.0949,-0.0074] | 0.14 |
| llama | 150 | groups | D × size | 11 | +0.0207 | [+0.0042,+0.0397] | 0.82 |
| llama | 150 | groups | U+V5 × size | 11 | -0.0896 | [-0.1629,-0.0274] | 0.23 |
| llama | 150 | groups | U-zero × size | 11 | -0.0884 | [-0.1614,-0.0270] | 0.23 |
| gemma | 20 | lines | D × size | 12 | -0.0193 | [-0.0693,+0.0115] | 0.67 |
| gemma | 20 | lines | U-zero × size | 12 | +0.0002 | [-0.0153,+0.0110] | 0.79 |
| gemma | 20 | groups | D × size | 12 | +0.0282 | [+0.0081,+0.0516] | 0.83 |
| gemma | 20 | groups | U-zero × size | 12 | +0.0378 | [+0.0086,+0.0768] | 0.88 |
| gemma | 100 | lines | D × size | 12 | -0.0001 | [-0.0052,+0.0052] | 0.62 |
| gemma | 100 | lines | U+V5 × size | 12 | -0.1177 | [-0.2095,-0.0458] | 0.00 |
| gemma | 100 | lines | U-zero × size | 12 | -0.0299 | [-0.0612,-0.0065] | 0.12 |
| gemma | 100 | groups | D × size | 12 | +0.0236 | [+0.0057,+0.0512] | 0.79 |
| gemma | 100 | groups | U+V5 × size | 12 | -0.3787 | [-0.5321,-0.2226] | 0.00 |
| gemma | 100 | groups | U-zero × size | 12 | -0.0662 | [-0.1231,-0.0171] | 0.21 |
| gemma | 150 | lines | D × size | 12 | -0.0001 | [-0.0052,+0.0052] | 0.62 |
| gemma | 150 | lines | U+V5 × size | 12 | -0.0291 | [-0.0584,-0.0068] | 0.12 |
| gemma | 150 | lines | U-zero × size | 12 | -0.0299 | [-0.0612,-0.0065] | 0.12 |
| gemma | 150 | groups | D × size | 12 | +0.0236 | [+0.0057,+0.0512] | 0.79 |
| gemma | 150 | groups | U+V5 × size | 12 | -0.0695 | [-0.1197,-0.0263] | 0.21 |
| gemma | 150 | groups | U-zero × size | 12 | -0.0662 | [-0.1231,-0.0171] | 0.21 |

At100 target review equivalents and r=1, U+V5 can afford only10 stored reviews after90 real labels; frequency can afford100 (subject to exhaustion). Its wrong-line fraction deficit is0.1600 Llama and0.1177 Gemma. At150, U+V5 reaches its50-review point and the deficits shrink to0.0380/0.0291. This corrects the old nominal100-label/capped120-review comparison. No downstream detector improvement is inferred.

At20 target review equivalents, U-zero’s base-parser wrong-line gain is approximately zero, while wrong-group gains are+0.0321 Llama/+0.0378 Gemma. It avoids90 target labels and can help template coverage at small budgets, conditional on existing source supervision. This is a narrower operational advantage than generally improving corrected line volume.

## Total-label and variable-cost sensitivity

| Model | Method | Δ wrong-line fraction vs frequency | System-cluster95%CI |
|---|---|---|---|
| llama | D × size | -0.0370 | [-0.0809, -0.0062] |
| llama | U+V5 × size | -0.1155 | [-0.2115, -0.0316] |
| llama | U-zero × size | -0.0426 | [-0.0949, -0.0074] |
| gemma | D × size | -0.0333 | [-0.0645, -0.0087] |
| gemma | U+V5 × size | -0.0776 | [-0.1468, -0.0259] |
| gemma | U-zero × size | -0.0299 | [-0.0612, -0.0065] |

This table uses amortized source costs, r=10 and100 review-equivalent budget; source-dependent methods remain below frequency on base-parser line recovery in these supported curves. Single-deployment source methods cannot reach even their first measured point within20–200 review equivalents under r≤10 because5500/6000 source labels already cost550/600 reviews at r=10. “Zero target” does not mean zero total acquisition cost. With source labels treated as sunk, A is the relevant incremental deployment scenario.

## Statistics, Pareto and coverage

| Model | Scenario | Method | Dominated by (all included objectives) |
|---|---|---|---|
| llama | A_target | D × size | Nondominated |
| llama | A_target | Frequency | Nondominated |
| llama | A_target | Group consistency | D × size, LLMLog × size |
| llama | A_target | LLMLog × size | Nondominated |
| llama | A_target | Random | D × size, Frequency, Group consistency, LLMLog × size |
| llama | A_target | U+V5 × size | D × size, Frequency, LLMLog × size |
| llama | A_target | U+V5-zero × size | Nondominated |
| llama | A_target | U-zero × size | Nondominated |
| llama | B_single | Frequency | Nondominated |
| llama | B_single | Group consistency | LLMLog × size |
| llama | B_single | LLMLog × size | Nondominated |
| llama | B_single | Random | Frequency, Group consistency, LLMLog × size |
| llama | B_amortized | Frequency | Nondominated |
| llama | B_amortized | Group consistency | LLMLog × size |
| llama | B_amortized | LLMLog × size | Nondominated |
| llama | B_amortized | Random | Frequency, Group consistency, LLMLog × size |
| gemma | A_target | D × size | Nondominated |
| gemma | A_target | Frequency | Nondominated |
| gemma | A_target | Group consistency | LLMLog × size |
| gemma | A_target | LLMLog × size | Nondominated |
| gemma | A_target | Random | D × size, Frequency, Group consistency, LLMLog × size |
| gemma | A_target | U+V5 × size | D × size, Frequency, LLMLog × size |
| gemma | A_target | U+V5-zero × size | Nondominated |
| gemma | A_target | U-zero × size | Nondominated |
| gemma | B_single | Frequency | Nondominated |
| gemma | B_single | Group consistency | LLMLog × size |
| gemma | B_single | LLMLog × size | Nondominated |
| gemma | B_single | Random | Frequency, Group consistency, LLMLog × size |
| gemma | B_amortized | Frequency | Nondominated |
| gemma | B_amortized | Group consistency | LLMLog × size |
| gemma | B_amortized | LLMLog × size | Nondominated |
| gemma | B_amortized | Random | Frequency, Group consistency, LLMLog × size |

Five seeds are matched within system; outcomes are averaged within system before4000 system-cluster bootstrap draws (seed20261111). Paired seed macros, per-system deltas,95% intervals and paired system win probability (ties count½) are saved. Deterministic seed duplicates are not five independent populations. Held-out comparisons excluding HDFS/Spark/Linux are saved separately as cohort=held_out; all-system tables include tuning systems and are descriptive. “Zero target” refers to deployment fitting: the research tuning systems were previously labelled. Held-out systems provide the cleaner zero-target evidence. Scenario B source pools include the tuning systems for held-out deployments; no extra tuning labels are charged twice.

Pareto comparisons use exactly common systems at each model/parser/scenario/budget, minimizing actual human cost and maximizing all jointly available outcomes (wrong lines, wrong groups, FTA). A method is marked dominated only if another is no worse on every included objective and strictly better on at least one. Source/GPU costs are never silently combined. These are descriptive point frontiers, not uncertainty-adjusted dominance claims. A method can lose on lines and remain nondominated through group/FTA recovery; no universal winner is claimed. Full dominated_by/objective fields are in review_cost_pareto.json.

Coverage: Llama cached analyses11 systems (pre-restoration R2/R3 histories omit HDFS/OpenSSH), Gemma12; real AdaParser12 each, with Llama Mac excluded. No reconstitution of missing adaptive histories was attempted. Perfect-system wrong-line/group fractions are undefined and jointly excluded per metric, so metric cohorts can differ. Real-parser32-candidate costs are included: budget50 generally reaches10 stored online reviews, not20. Per-system seed outcomes are in review_cost_per_system.csv; full/tail remain separately named parser populations.

GPU/inference measurements from log_parsing_cost_quality.md describe capped-suite scoring and cannot reliably price D’s different layer/prompt or AdaParser’s correction calls. They are not converted into invented deployment GPU cost or dollar axes. Existing parser call/token/wall measurements remain in adaparser_e2e_details.md; no new GPU inference was measured.

[Paper-ready four-panel PDF](../results_phase3/cpu_research/review_cost.pdf): supported performance/effort curves, amortized label cost, descriptive Pareto points and explicitly hypothetical cost-ratio sensitivity. Connecting lines guide the eye only; comparisons use measured points. Frontiers/paired comparisons/coverage/Pareto records are in results_phase3/cpu_research/.
