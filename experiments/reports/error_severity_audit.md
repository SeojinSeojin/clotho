# Do failure predictors identify consequential parsing errors?

**Status:** completed CPU audit; semantic consequences remain unverified.

## Research questions and reference

We test whether frozen predictors and review queues identify formal extraction/representation discrepancies, detector-visible word changes, and measured detector improvements. These are distinct from verified semantic harm. No GPU inference or new generations were run.

The reference was absent locally; its author PDF was retrieved after a disk check. Sun et al., DOI [10.1016/j.jss.2026.113161](https://asi.cs.cityu.edu.hk/paper/JSS_j.jss.2026.113161.pdf), was available online 28 October 2026 (journal volume 2027). We read §5.1/Tables5–6, §5.2 and §7. Their taxonomy is expert-derived; corrections are an oracle-style intervention, not a deployable repair system. Their public/industrial datasets and five detector architectures differ from our two-system count/word diagnostic. Expert subjectivity and correction realism limit transfer of their findings. We do not import their labels, frequencies, or downstream effect sizes.

## Protocol and artifact reuse

A: stored sample0 and label samples5–9, capped and natural datasets reported separately. B primary: capped **template-disjoint @100**, original five splits; no standard/natural predictor claim is made. Reference100 includes10 synthetic tests; U fits90 real target-suite labels plus its frozen source pool. Predictor features/splits/training remain strict-label based. Clotho/MDSA/D/LLMLog CPU score recovery was needed because old jobs mostly stored scalar AUCs; U/U+V5 vectors reuse completed G5. Original Clotho/post-score AUCs and G5 indices are asserted; rejected vectors are omitted and logged. Post scores use samples0–4, token probability preserves its original sample0 definition; labels use5–9. No new target informed fitting, thresholding or selection.

C: 50,000-line prefixes (all available Linux lines), base/strong Drain groups and real AdaParser output-string groups kept separate. Strong retains verified seeds outside queues; real AdaParser retains32 labelled candidates. Replay existing full/tail histories, including online refits and top20 verdicts already used in tail histories. Missing static Clotho-risk/risk×size and some U/D-tail histories are reported as unavailable, not approximated by filtering adaptive sequences. U review means the frozen U+V5 configuration with nominal reference100 (90 real target labels), unlike D.

## Taxonomy, uncertainty and validation

Exclusive primary labels: exact; whitespace-only surface proxy; uniquely aligned variable-extraction proxy; nonmatching representation proxy; uncertain. Overlapping attributes: preserved variables, masked constants, slot changes, unsupported-word proxy, N3 word-bag changes and observed cross-line mask-role inconsistency. Attribute counts must not be summed. GT/predicted wildcard spans are matched to the actual line after whitespace collapse; lazy/greedy disagreement, GT nonmatch and3ms timeouts are uncertain. Majority structural labels require three definite positive or three definite negative votes among samples5–9; other cases are unknown and excluded jointly. Strict labels are unchanged. Word changes are differences in the exact lowercase[a-z]+ bag consumed by N3, not semantic labels. A one-token status change can matter; even whitespace-only changes may split string-based event IDs. No category is automatically called harmless or consequential.

Paper E1/E2 are measured from membership contingency, separately for fixed operational groups and predicted-string partitions. E3 is only an extraction proxy. E4 requires observed cross-line inconsistency (singletons are not evidence of consistency). Reference-disjoint novelty is a descriptor, not E5 handling failure; temporal adaptation evidence is absent. E6 semantic rewriting cannot be established from edit distance or unsupported tokens alone: **all semantic labels remain unverified**, not zero semantic errors.

Five synthetic regression cases test variable retention, constant masking, whitespace equivalence, invalid representation and exact matches. Real examples are sampled score-blind: first sorted readable sample0 per primary/model/system/suite, all three text fields<=160characters; skipped first hits are counted in `manual_sampling_key.csv`. `manual_review_blind.csv` contains blank expert fields and no scores/automatic tags. This supports inspection, but is not independent human validation.

## A. Error distributions

| Model | Audit systems | Prediction systems | Downstream systems |
|---|---|---|---|
| llama | 13 | 13 | 2 |
| gemma | 12 | 12 | 2 |

| Model | Suite | Majority label | Inputs | Positive fraction | Unknown fraction |
|---|---|---|---|---|---|
| llama | capped | strict | 6500 | 0.670 | 0.000 |
| llama | capped | structural_proxy | 6500 | 0.654 | 0.014 |
| llama | capped | word_input_change | 6500 | 0.436 | 0.000 |
| llama | natural | strict | 6500 | 0.649 | 0.000 |
| llama | natural | structural_proxy | 6500 | 0.639 | 0.009 |
| llama | natural | word_input_change | 6500 | 0.391 | 0.000 |
| gemma | capped | strict | 6000 | 0.546 | 0.000 |
| gemma | capped | structural_proxy | 6000 | 0.514 | 0.026 |
| gemma | capped | word_input_change | 6000 | 0.339 | 0.000 |
| gemma | natural | strict | 6000 | 0.592 | 0.000 |
| gemma | natural | structural_proxy | 6000 | 0.558 | 0.032 |
| gemma | natural | word_input_change | 6000 | 0.360 | 0.000 |

Counts and overlap by system/model/sample are in the machine-readable CSVs; descriptive natural inputs need not meet the original prediction benchmark class-balance eligibility.

## B. Predictor performance and paired comparisons

### llama: full-cohort predictors Clotho, D-zero, LLMLog, Output entropy, Token probability, U, U+V5

| Target | Unit | Predictor | AUROC systems | AUROC | AP systems | AP | Prevalence | P@20 | P@20/random | ΔAUROC−Clotho | Positive seeds/5 | System-cluster95%CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | line | Clotho | 13 | 0.645 ± 0.022 | 13 | 0.794 ± 0.015 | 0.673 ± 0.015 | 0.895 ± 0.019 | 1.394 ± 0.062 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict | line | D-zero | 13 | 0.727 ± 0.027 | 13 | 0.821 ± 0.026 | 0.673 ± 0.015 | 0.885 ± 0.035 | 1.348 ± 0.063 | 0.082 ± 0.040 | 5 | [0.034617582007539636, 0.1311567607456895] |
| strict | line | LLMLog | 13 | 0.720 ± 0.023 | 13 | 0.817 ± 0.020 | 0.673 ± 0.015 | 0.879 ± 0.028 | 1.366 ± 0.067 | 0.075 ± 0.035 | 5 | [0.03454335934980701, 0.1165528380573211] |
| strict | line | Output entropy | 13 | 0.796 ± 0.014 | 13 | 0.845 ± 0.014 | 0.673 ± 0.015 | 0.922 ± 0.019 | 1.422 ± 0.040 | 0.151 ± 0.026 | 5 | [0.1116406085239663, 0.19068786870315113] |
| strict | line | Token probability | 13 | 0.511 ± 0.022 | 13 | 0.689 ± 0.014 | 0.673 ± 0.015 | 0.689 ± 0.027 | 1.032 ± 0.043 | -0.134 ± 0.030 | 0 | [-0.18782622525086456, -0.07458102324796599] |
| strict | line | U | 13 | 0.776 ± 0.028 | 13 | 0.854 ± 0.027 | 0.673 ± 0.015 | 0.912 ± 0.025 | 1.404 ± 0.065 | 0.132 ± 0.046 | 5 | [0.09096782268824928, 0.17285715997408752] |
| strict | line | U+V5 | 13 | 0.783 ± 0.027 | 13 | 0.857 ± 0.025 | 0.673 ± 0.015 | 0.914 ± 0.017 | 1.409 ± 0.053 | 0.139 ± 0.043 | 5 | [0.09890292688925047, 0.17922924541133112] |
| strict | GT_template_retrospective | Clotho | 12 | 0.669 ± 0.021 | 12 | 0.838 ± 0.018 | 0.692 ± 0.014 | 0.778 ± 0.016 | 1.131 ± 0.017 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict | GT_template_retrospective | D-zero | 12 | 0.778 ± 0.018 | 12 | 0.882 ± 0.016 | 0.692 ± 0.014 | 0.821 ± 0.015 | 1.198 ± 0.025 | 0.108 ± 0.032 | 5 | [0.05854934003911411, 0.1596156102925551] |
| strict | GT_template_retrospective | LLMLog | 12 | 0.788 ± 0.013 | 12 | 0.895 ± 0.012 | 0.692 ± 0.014 | 0.822 ± 0.014 | 1.210 ± 0.017 | 0.118 ± 0.015 | 5 | [0.0702559643792366, 0.176862960515186] |
| strict | GT_template_retrospective | Output entropy | 12 | 0.876 ± 0.011 | 12 | 0.935 ± 0.009 | 0.692 ± 0.014 | 0.855 ± 0.021 | 1.259 ± 0.017 | 0.207 ± 0.025 | 5 | [0.15460143516651378, 0.26477374357707106] |
| strict | GT_template_retrospective | Token probability | 12 | 0.517 ± 0.022 | 12 | 0.745 ± 0.016 | 0.692 ± 0.014 | 0.714 ± 0.020 | 1.036 ± 0.022 | -0.153 ± 0.028 | 0 | [-0.22513765501654343, -0.08621643479400974] |
| strict | GT_template_retrospective | U | 12 | 0.814 ± 0.021 | 12 | 0.904 ± 0.013 | 0.692 ± 0.014 | 0.845 ± 0.012 | 1.240 ± 0.021 | 0.144 ± 0.037 | 5 | [0.08671172290206881, 0.2055883562978978] |
| strict | GT_template_retrospective | U+V5 | 12 | 0.820 ± 0.023 | 12 | 0.905 ± 0.014 | 0.692 ± 0.014 | 0.840 ± 0.017 | 1.233 ± 0.026 | 0.151 ± 0.038 | 5 | [0.09472520419875526, 0.2094959651103833] |
| strict_matched_structural | line | Clotho | 13 | 0.642 ± 0.021 | 13 | 0.788 ± 0.015 | 0.667 ± 0.015 | 0.874 ± 0.031 | 1.385 ± 0.065 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict_matched_structural | line | D-zero | 13 | 0.724 ± 0.027 | 13 | 0.816 ± 0.027 | 0.667 ± 0.015 | 0.882 ± 0.036 | 1.359 ± 0.059 | 0.082 ± 0.039 | 5 | [0.033868511368561297, 0.1334057040568753] |
| strict_matched_structural | line | LLMLog | 13 | 0.719 ± 0.022 | 13 | 0.813 ± 0.020 | 0.667 ± 0.015 | 0.876 ± 0.030 | 1.378 ± 0.062 | 0.077 ± 0.033 | 5 | [0.03463107611589733, 0.12325222117902851] |
| strict_matched_structural | line | Output entropy | 13 | 0.795 ± 0.014 | 13 | 0.842 ± 0.015 | 0.667 ± 0.015 | 0.932 ± 0.021 | 1.459 ± 0.048 | 0.153 ± 0.024 | 5 | [0.11268009367772631, 0.19619771132439062] |
| strict_matched_structural | line | Token probability | 13 | 0.514 ± 0.021 | 13 | 0.685 ± 0.015 | 0.667 ± 0.015 | 0.685 ± 0.029 | 1.031 ± 0.047 | -0.128 ± 0.028 | 0 | [-0.18288847767106844, -0.06825206783629185] |
| strict_matched_structural | line | U | 13 | 0.774 ± 0.029 | 13 | 0.851 ± 0.029 | 0.667 ± 0.015 | 0.906 ± 0.029 | 1.412 ± 0.067 | 0.132 ± 0.045 | 5 | [0.09450240939792666, 0.17460324726444088] |
| strict_matched_structural | line | U+V5 | 13 | 0.781 ± 0.029 | 13 | 0.854 ± 0.027 | 0.667 ± 0.015 | 0.907 ± 0.023 | 1.421 ± 0.045 | 0.139 ± 0.043 | 5 | [0.10094268457824092, 0.1814270137688721] |
| strict_matched_structural | GT_template_retrospective | Clotho | 12 | 0.665 ± 0.021 | 12 | 0.832 ± 0.019 | 0.689 ± 0.013 | 0.775 ± 0.014 | 1.131 ± 0.013 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict_matched_structural | GT_template_retrospective | D-zero | 12 | 0.776 ± 0.019 | 12 | 0.880 ± 0.017 | 0.689 ± 0.013 | 0.819 ± 0.015 | 1.202 ± 0.024 | 0.111 ± 0.030 | 5 | [0.058758830028423754, 0.16563051055338812] |
| strict_matched_structural | GT_template_retrospective | LLMLog | 12 | 0.786 ± 0.012 | 12 | 0.893 ± 0.012 | 0.689 ± 0.013 | 0.821 ± 0.014 | 1.214 ± 0.018 | 0.121 ± 0.014 | 5 | [0.07002608028974285, 0.1868984618455519] |
| strict_matched_structural | GT_template_retrospective | Output entropy | 12 | 0.876 ± 0.010 | 12 | 0.935 ± 0.008 | 0.689 ± 0.013 | 0.855 ± 0.022 | 1.267 ± 0.020 | 0.211 ± 0.023 | 5 | [0.15529385351085462, 0.27280822980458785] |
| strict_matched_structural | GT_template_retrospective | Token probability | 12 | 0.518 ± 0.023 | 12 | 0.743 ± 0.017 | 0.689 ± 0.013 | 0.711 ± 0.019 | 1.036 ± 0.020 | -0.147 ± 0.027 | 0 | [-0.21830916605439143, -0.07615034356770392] |
| strict_matched_structural | GT_template_retrospective | U | 12 | 0.812 ± 0.022 | 12 | 0.903 ± 0.014 | 0.689 ± 0.013 | 0.845 ± 0.012 | 1.247 ± 0.020 | 0.147 ± 0.036 | 5 | [0.08731809128094514, 0.21265250181019102] |
| strict_matched_structural | GT_template_retrospective | U+V5 | 12 | 0.819 ± 0.023 | 12 | 0.903 ± 0.015 | 0.689 ± 0.013 | 0.839 ± 0.017 | 1.238 ± 0.028 | 0.154 ± 0.038 | 5 | [0.09487620953795792, 0.21592749302595984] |
| structural_proxy | line | Clotho | 13 | 0.642 ± 0.020 | 13 | 0.787 ± 0.015 | 0.666 ± 0.015 | 0.874 ± 0.031 | 1.387 ± 0.065 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| structural_proxy | line | D-zero | 13 | 0.724 ± 0.027 | 13 | 0.815 ± 0.027 | 0.666 ± 0.015 | 0.880 ± 0.036 | 1.358 ± 0.060 | 0.081 ± 0.039 | 5 | [0.03311737257239421, 0.13232925347115163] |
| structural_proxy | line | LLMLog | 13 | 0.719 ± 0.022 | 13 | 0.813 ± 0.020 | 0.666 ± 0.015 | 0.876 ± 0.030 | 1.379 ± 0.062 | 0.076 ± 0.033 | 5 | [0.03415287452084747, 0.12223090099285495] |
| structural_proxy | line | Output entropy | 13 | 0.794 ± 0.014 | 13 | 0.841 ± 0.015 | 0.666 ± 0.015 | 0.931 ± 0.022 | 1.459 ± 0.048 | 0.152 ± 0.024 | 5 | [0.11240702282010868, 0.19504265306487378] |
| structural_proxy | line | Token probability | 13 | 0.514 ± 0.021 | 13 | 0.684 ± 0.015 | 0.666 ± 0.015 | 0.685 ± 0.029 | 1.032 ± 0.047 | -0.128 ± 0.028 | 0 | [-0.18265282037792782, -0.06812336538669565] |
| structural_proxy | line | U | 13 | 0.774 ± 0.029 | 13 | 0.850 ± 0.028 | 0.666 ± 0.015 | 0.905 ± 0.028 | 1.413 ± 0.066 | 0.131 ± 0.045 | 5 | [0.09440735608882012, 0.17397405276605912] |
| structural_proxy | line | U+V5 | 13 | 0.781 ± 0.028 | 13 | 0.853 ± 0.027 | 0.666 ± 0.015 | 0.906 ± 0.022 | 1.421 ± 0.045 | 0.138 ± 0.043 | 5 | [0.10022170843193966, 0.18059190769364095] |
| structural_proxy | GT_template_retrospective | Clotho | 12 | 0.665 ± 0.021 | 12 | 0.832 ± 0.019 | 0.689 ± 0.013 | 0.775 ± 0.014 | 1.131 ± 0.013 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| structural_proxy | GT_template_retrospective | D-zero | 12 | 0.776 ± 0.019 | 12 | 0.880 ± 0.017 | 0.689 ± 0.013 | 0.819 ± 0.015 | 1.202 ± 0.024 | 0.111 ± 0.030 | 5 | [0.05875242194142356, 0.16563051055338812] |
| structural_proxy | GT_template_retrospective | LLMLog | 12 | 0.786 ± 0.012 | 12 | 0.893 ± 0.012 | 0.689 ± 0.013 | 0.821 ± 0.014 | 1.214 ± 0.018 | 0.121 ± 0.014 | 5 | [0.06988988005925581, 0.1868984618455519] |
| structural_proxy | GT_template_retrospective | Output entropy | 12 | 0.876 ± 0.010 | 12 | 0.935 ± 0.008 | 0.689 ± 0.013 | 0.855 ± 0.022 | 1.267 ± 0.020 | 0.211 ± 0.024 | 5 | [0.1551816885648081, 0.2728055592106344] |
| structural_proxy | GT_template_retrospective | Token probability | 12 | 0.519 ± 0.023 | 12 | 0.743 ± 0.017 | 0.689 ± 0.013 | 0.711 ± 0.019 | 1.036 ± 0.020 | -0.147 ± 0.027 | 0 | [-0.21828476785236167, -0.07612710718481842] |
| structural_proxy | GT_template_retrospective | U | 12 | 0.812 ± 0.022 | 12 | 0.903 ± 0.014 | 0.689 ± 0.013 | 0.845 ± 0.012 | 1.247 ± 0.020 | 0.147 ± 0.036 | 5 | [0.08730455724109532, 0.21238858803311933] |
| structural_proxy | GT_template_retrospective | U+V5 | 12 | 0.819 ± 0.024 | 12 | 0.903 ± 0.015 | 0.689 ± 0.013 | 0.839 ± 0.017 | 1.238 ± 0.027 | 0.153 ± 0.038 | 5 | [0.09487620953795792, 0.21592749302595984] |
| word_input_change | line | Clotho | 13 | 0.622 ± 0.014 | 13 | 0.582 ± 0.031 | 0.441 ± 0.015 | 0.647 ± 0.067 | 1.555 ± 0.170 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| word_input_change | line | D-zero | 13 | 0.656 ± 0.031 | 13 | 0.592 ± 0.033 | 0.441 ± 0.015 | 0.638 ± 0.060 | 1.541 ± 0.210 | 0.033 ± 0.034 | 4 | [-0.013387776841786698, 0.08681094615809683] |
| word_input_change | line | LLMLog | 13 | 0.616 ± 0.013 | 13 | 0.543 ± 0.015 | 0.441 ± 0.015 | 0.542 ± 0.036 | 1.244 ± 0.135 | -0.006 ± 0.023 | 2 | [-0.021461874996740966, 0.010138624976723368] |
| word_input_change | line | Output entropy | 13 | 0.701 ± 0.011 | 13 | 0.598 ± 0.015 | 0.441 ± 0.015 | 0.678 ± 0.018 | 1.584 ± 0.054 | 0.079 ± 0.021 | 5 | [0.04022691014621554, 0.11372238559605041] |
| word_input_change | line | Token probability | 13 | 0.528 ± 0.018 | 13 | 0.480 ± 0.009 | 0.441 ± 0.015 | 0.502 ± 0.029 | 1.181 ± 0.176 | -0.095 ± 0.028 | 0 | [-0.15516748481715925, -0.04394464622223748] |
| word_input_change | line | U | 13 | 0.664 ± 0.025 | 13 | 0.607 ± 0.027 | 0.441 ± 0.015 | 0.669 ± 0.033 | 1.540 ± 0.113 | 0.041 ± 0.021 | 5 | [-0.004168692037450426, 0.09192700701496248] |
| word_input_change | line | U+V5 | 13 | 0.671 ± 0.021 | 13 | 0.609 ± 0.023 | 0.441 ± 0.015 | 0.672 ± 0.041 | 1.561 ± 0.145 | 0.049 ± 0.021 | 5 | [0.007216140126140443, 0.09762645986179036] |
| word_input_change | GT_template_retrospective | Clotho | 12 | 0.645 ± 0.013 | 12 | 0.675 ± 0.014 | 0.504 ± 0.016 | 0.636 ± 0.021 | 1.258 ± 0.036 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| word_input_change | GT_template_retrospective | D-zero | 12 | 0.686 ± 0.012 | 12 | 0.696 ± 0.016 | 0.504 ± 0.016 | 0.667 ± 0.015 | 1.328 ± 0.040 | 0.042 ± 0.019 | 5 | [0.0042191075357933695, 0.08747211887570656] |
| word_input_change | GT_template_retrospective | LLMLog | 12 | 0.673 ± 0.019 | 12 | 0.664 ± 0.024 | 0.504 ± 0.016 | 0.606 ± 0.030 | 1.216 ± 0.041 | 0.029 ± 0.031 | 4 | [0.0013819348376835046, 0.06364626144667242] |
| word_input_change | GT_template_retrospective | Output entropy | 12 | 0.760 ± 0.015 | 12 | 0.754 ± 0.014 | 0.504 ± 0.016 | 0.687 ± 0.033 | 1.393 ± 0.043 | 0.116 ± 0.025 | 5 | [0.060554001390609534, 0.18071848934669163] |
| word_input_change | GT_template_retrospective | Token probability | 12 | 0.549 ± 0.009 | 12 | 0.606 ± 0.009 | 0.504 ± 0.016 | 0.565 ± 0.021 | 1.126 ± 0.032 | -0.095 ± 0.021 | 0 | [-0.17193346082692287, -0.028532494381656504] |
| word_input_change | GT_template_retrospective | U | 12 | 0.700 ± 0.018 | 12 | 0.718 ± 0.017 | 0.504 ± 0.016 | 0.684 ± 0.016 | 1.375 ± 0.045 | 0.055 ± 0.024 | 5 | [-0.00791317602139621, 0.12666733924830378] |
| word_input_change | GT_template_retrospective | U+V5 | 12 | 0.709 ± 0.020 | 12 | 0.721 ± 0.020 | 0.504 ± 0.016 | 0.680 ± 0.030 | 1.371 ± 0.073 | 0.065 ± 0.028 | 5 | [0.00325800252124149, 0.13589679234569135] |

### gemma: full-cohort predictors Clotho, D-zero, LLMLog, Output entropy, Token probability, U, U+V5

| Target | Unit | Predictor | AUROC systems | AUROC | AP systems | AP | Prevalence | P@20 | P@20/random | ΔAUROC−Clotho | Positive seeds/5 | System-cluster95%CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | line | Clotho | 12 | 0.659 ± 0.025 | 12 | 0.707 ± 0.038 | 0.543 ± 0.018 | 0.810 ± 0.055 | 1.564 ± 0.152 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict | line | D-zero | 12 | 0.698 ± 0.015 | 12 | 0.710 ± 0.015 | 0.543 ± 0.018 | 0.752 ± 0.036 | 1.425 ± 0.117 | 0.038 ± 0.030 | 5 | [-0.007366996677518656, 0.08067876319413514] |
| strict | line | LLMLog | 12 | 0.680 ± 0.024 | 12 | 0.717 ± 0.013 | 0.543 ± 0.018 | 0.786 ± 0.024 | 1.502 ± 0.060 | 0.020 ± 0.040 | 3 | [-0.04175504951244586, 0.08340428405825635] |
| strict | line | Output entropy | 12 | 0.824 ± 0.012 | 12 | 0.813 ± 0.013 | 0.543 ± 0.018 | 0.927 ± 0.014 | 1.828 ± 0.083 | 0.164 ± 0.033 | 5 | [0.12147571487388588, 0.20361467621149235] |
| strict | line | Token probability | 12 | 0.493 ± 0.024 | 12 | 0.568 ± 0.018 | 0.543 ± 0.018 | 0.597 ± 0.026 | 1.098 ± 0.068 | -0.167 ± 0.040 | 0 | [-0.2483782521205957, -0.0939722003492322] |
| strict | line | U | 12 | 0.773 ± 0.015 | 12 | 0.788 ± 0.013 | 0.543 ± 0.018 | 0.864 ± 0.037 | 1.657 ± 0.183 | 0.114 ± 0.039 | 5 | [0.07143029476648709, 0.1534649121403935] |
| strict | line | U+V5 | 12 | 0.784 ± 0.010 | 12 | 0.794 ± 0.014 | 0.543 ± 0.018 | 0.868 ± 0.035 | 1.664 ± 0.183 | 0.125 ± 0.033 | 5 | [0.07975272731966566, 0.1655127065973731] |
| strict | GT_template_retrospective | Clotho | 12 | 0.668 ± 0.024 | 12 | 0.725 ± 0.026 | 0.550 ± 0.009 | 0.713 ± 0.022 | 1.302 ± 0.023 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict | GT_template_retrospective | D-zero | 12 | 0.720 ± 0.010 | 12 | 0.749 ± 0.007 | 0.550 ± 0.009 | 0.714 ± 0.023 | 1.297 ± 0.028 | 0.052 ± 0.027 | 5 | [-0.009950623894891587, 0.10982548154483664] |
| strict | GT_template_retrospective | LLMLog | 12 | 0.679 ± 0.020 | 12 | 0.741 ± 0.013 | 0.550 ± 0.009 | 0.724 ± 0.017 | 1.319 ± 0.027 | 0.011 ± 0.035 | 3 | [-0.06532291622727429, 0.08833217534646873] |
| strict | GT_template_retrospective | Output entropy | 12 | 0.863 ± 0.014 | 12 | 0.881 ± 0.007 | 0.550 ± 0.009 | 0.817 ± 0.015 | 1.524 ± 0.035 | 0.195 ± 0.036 | 5 | [0.14591862461007904, 0.24904221698016282] |
| strict | GT_template_retrospective | Token probability | 12 | 0.497 ± 0.021 | 12 | 0.586 ± 0.018 | 0.550 ± 0.009 | 0.562 ± 0.020 | 1.026 ± 0.031 | -0.171 ± 0.032 | 0 | [-0.24316899362179686, -0.08787985633956961] |
| strict | GT_template_retrospective | U | 12 | 0.785 ± 0.017 | 12 | 0.817 ± 0.010 | 0.550 ± 0.009 | 0.773 ± 0.008 | 1.418 ± 0.020 | 0.117 ± 0.037 | 5 | [0.0595169075952921, 0.17705048110698812] |
| strict | GT_template_retrospective | U+V5 | 12 | 0.801 ± 0.010 | 12 | 0.829 ± 0.008 | 0.550 ± 0.009 | 0.772 ± 0.016 | 1.414 ± 0.013 | 0.134 ± 0.028 | 5 | [0.07683635538274122, 0.19502968503619128] |
| strict_matched_structural | line | Clotho | 12 | 0.656 ± 0.022 | 12 | 0.691 ± 0.035 | 0.531 ± 0.018 | 0.783 ± 0.047 | 1.537 ± 0.126 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict_matched_structural | line | D-zero | 12 | 0.693 ± 0.016 | 12 | 0.698 ± 0.016 | 0.531 ± 0.018 | 0.741 ± 0.043 | 1.425 ± 0.124 | 0.038 ± 0.028 | 5 | [-0.009515618254335903, 0.0810773902198249] |
| strict_matched_structural | line | LLMLog | 12 | 0.684 ± 0.024 | 12 | 0.711 ± 0.013 | 0.531 ± 0.018 | 0.783 ± 0.023 | 1.525 ± 0.058 | 0.028 ± 0.038 | 3 | [-0.0327631752089406, 0.0875185880624233] |
| strict_matched_structural | line | Output entropy | 12 | 0.821 ± 0.012 | 12 | 0.805 ± 0.011 | 0.531 ± 0.018 | 0.923 ± 0.015 | 1.855 ± 0.090 | 0.166 ± 0.032 | 5 | [0.12331758867790969, 0.2059997109247921] |
| strict_matched_structural | line | Token probability | 12 | 0.495 ± 0.025 | 12 | 0.559 ± 0.021 | 0.531 ± 0.018 | 0.587 ± 0.028 | 1.105 ± 0.070 | -0.161 ± 0.036 | 0 | [-0.24577779743253653, -0.08622963566969012] |
| strict_matched_structural | line | U | 12 | 0.772 ± 0.014 | 12 | 0.779 ± 0.015 | 0.531 ± 0.018 | 0.858 ± 0.038 | 1.678 ± 0.185 | 0.116 ± 0.036 | 5 | [0.06839926555967722, 0.15774088206222855] |
| strict_matched_structural | line | U+V5 | 12 | 0.782 ± 0.010 | 12 | 0.785 ± 0.015 | 0.531 ± 0.018 | 0.860 ± 0.033 | 1.679 ± 0.181 | 0.126 ± 0.029 | 5 | [0.07948631869005367, 0.16790312357886966] |
| strict_matched_structural | GT_template_retrospective | Clotho | 12 | 0.662 ± 0.023 | 12 | 0.709 ± 0.022 | 0.541 ± 0.008 | 0.704 ± 0.020 | 1.307 ± 0.025 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| strict_matched_structural | GT_template_retrospective | D-zero | 12 | 0.714 ± 0.009 | 12 | 0.736 ± 0.004 | 0.541 ± 0.008 | 0.697 ± 0.017 | 1.287 ± 0.025 | 0.052 ± 0.027 | 5 | [-0.010624586035830493, 0.11081258589266678] |
| strict_matched_structural | GT_template_retrospective | LLMLog | 12 | 0.679 ± 0.022 | 12 | 0.733 ± 0.015 | 0.541 ± 0.008 | 0.717 ± 0.020 | 1.326 ± 0.038 | 0.017 ± 0.034 | 3 | [-0.053867054913794, 0.09126833957800107] |
| strict_matched_structural | GT_template_retrospective | Output entropy | 12 | 0.861 ± 0.014 | 12 | 0.875 ± 0.007 | 0.541 ± 0.008 | 0.809 ± 0.018 | 1.537 ± 0.037 | 0.199 ± 0.035 | 5 | [0.14709220684120378, 0.25233650359888915] |
| strict_matched_structural | GT_template_retrospective | Token probability | 12 | 0.499 ± 0.020 | 12 | 0.578 ± 0.017 | 0.541 ± 0.008 | 0.549 ± 0.017 | 1.021 ± 0.027 | -0.163 ± 0.029 | 0 | [-0.23108257936339718, -0.08267153535028572] |
| strict_matched_structural | GT_template_retrospective | U | 12 | 0.781 ± 0.016 | 12 | 0.803 ± 0.009 | 0.541 ± 0.008 | 0.764 ± 0.011 | 1.427 ± 0.022 | 0.119 ± 0.036 | 5 | [0.060639852346315866, 0.17970529124272172] |
| strict_matched_structural | GT_template_retrospective | U+V5 | 12 | 0.797 ± 0.010 | 12 | 0.818 ± 0.006 | 0.541 ± 0.008 | 0.761 ± 0.015 | 1.417 ± 0.018 | 0.135 ± 0.028 | 5 | [0.07700977978040018, 0.19708587928534735] |
| structural_proxy | line | Clotho | 12 | 0.656 ± 0.022 | 12 | 0.689 ± 0.033 | 0.527 ± 0.017 | 0.782 ± 0.047 | 1.550 ± 0.125 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| structural_proxy | line | D-zero | 12 | 0.696 ± 0.016 | 12 | 0.697 ± 0.016 | 0.527 ± 0.017 | 0.741 ± 0.043 | 1.438 ± 0.122 | 0.040 ± 0.028 | 5 | [-0.005456756188873902, 0.08128894936801659] |
| structural_proxy | line | LLMLog | 12 | 0.685 ± 0.023 | 12 | 0.709 ± 0.012 | 0.527 ± 0.017 | 0.782 ± 0.023 | 1.539 ± 0.053 | 0.029 ± 0.037 | 3 | [-0.03117610022879766, 0.08807539052960418] |
| structural_proxy | line | Output entropy | 12 | 0.819 ± 0.014 | 12 | 0.800 ± 0.011 | 0.527 ± 0.017 | 0.922 ± 0.016 | 1.869 ± 0.091 | 0.163 ± 0.032 | 5 | [0.12003689787579361, 0.2045425405854322] |
| structural_proxy | line | Token probability | 12 | 0.494 ± 0.025 | 12 | 0.552 ± 0.020 | 0.527 ± 0.017 | 0.578 ± 0.028 | 1.096 ± 0.069 | -0.162 ± 0.037 | 0 | [-0.24687682695229032, -0.08804761406744363] |
| structural_proxy | line | U | 12 | 0.771 ± 0.014 | 12 | 0.776 ± 0.015 | 0.527 ± 0.017 | 0.856 ± 0.040 | 1.687 ± 0.182 | 0.116 ± 0.036 | 5 | [0.06625622045834301, 0.15817860526443492] |
| structural_proxy | line | U+V5 | 12 | 0.781 ± 0.011 | 12 | 0.781 ± 0.015 | 0.527 ± 0.017 | 0.859 ± 0.033 | 1.692 ± 0.175 | 0.126 ± 0.030 | 5 | [0.0778951131479907, 0.1662811368040376] |
| structural_proxy | GT_template_retrospective | Clotho | 12 | 0.662 ± 0.022 | 12 | 0.707 ± 0.021 | 0.537 ± 0.007 | 0.702 ± 0.019 | 1.316 ± 0.026 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| structural_proxy | GT_template_retrospective | D-zero | 12 | 0.716 ± 0.010 | 12 | 0.735 ± 0.004 | 0.537 ± 0.007 | 0.696 ± 0.017 | 1.295 ± 0.026 | 0.054 ± 0.026 | 5 | [-0.006679992144293309, 0.11119110610729253] |
| structural_proxy | GT_template_retrospective | LLMLog | 12 | 0.680 ± 0.022 | 12 | 0.731 ± 0.016 | 0.537 ± 0.007 | 0.717 ± 0.020 | 1.340 ± 0.039 | 0.018 ± 0.035 | 4 | [-0.05248061962659463, 0.09069164366052898] |
| structural_proxy | GT_template_retrospective | Output entropy | 12 | 0.859 ± 0.014 | 12 | 0.872 ± 0.007 | 0.537 ± 0.007 | 0.804 ± 0.018 | 1.544 ± 0.035 | 0.197 ± 0.035 | 5 | [0.14496255004215147, 0.2495338626705013] |
| structural_proxy | GT_template_retrospective | Token probability | 12 | 0.497 ± 0.021 | 12 | 0.572 ± 0.019 | 0.537 ± 0.007 | 0.544 ± 0.017 | 1.018 ± 0.026 | -0.165 ± 0.031 | 0 | [-0.23362550396216505, -0.08445070996347752] |
| structural_proxy | GT_template_retrospective | U | 12 | 0.781 ± 0.016 | 12 | 0.800 ± 0.010 | 0.537 ± 0.007 | 0.760 ± 0.013 | 1.430 ± 0.024 | 0.119 ± 0.037 | 5 | [0.06083865480721357, 0.1800481664512128] |
| structural_proxy | GT_template_retrospective | U+V5 | 12 | 0.797 ± 0.011 | 12 | 0.815 ± 0.007 | 0.537 ± 0.007 | 0.758 ± 0.018 | 1.423 ± 0.021 | 0.135 ± 0.029 | 5 | [0.07714734450361037, 0.19698452740749822] |
| word_input_change | line | Clotho | 12 | 0.625 ± 0.022 | 12 | 0.492 ± 0.047 | 0.347 ± 0.018 | 0.571 ± 0.082 | 1.666 ± 0.258 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| word_input_change | line | D-zero | 12 | 0.633 ± 0.012 | 12 | 0.498 ± 0.023 | 0.347 ± 0.018 | 0.534 ± 0.057 | 1.574 ± 0.279 | 0.007 ± 0.028 | 2 | [-0.04467684938639124, 0.0538126509675588] |
| word_input_change | line | LLMLog | 12 | 0.622 ± 0.023 | 12 | 0.466 ± 0.021 | 0.347 ± 0.018 | 0.488 ± 0.030 | 1.396 ± 0.024 | -0.003 ± 0.039 | 2 | [-0.06627138323104495, 0.06333844474810475] |
| word_input_change | line | Output entropy | 12 | 0.736 ± 0.016 | 12 | 0.545 ± 0.020 | 0.347 ± 0.018 | 0.643 ± 0.035 | 1.937 ± 0.097 | 0.111 ± 0.036 | 5 | [0.06850179302271475, 0.14745468595391148] |
| word_input_change | line | Token probability | 12 | 0.488 ± 0.021 | 12 | 0.371 ± 0.014 | 0.347 ± 0.018 | 0.362 ± 0.020 | 1.059 ± 0.049 | -0.138 ± 0.040 | 0 | [-0.24874832954113915, -0.04977868262611348] |
| word_input_change | line | U | 12 | 0.701 ± 0.019 | 12 | 0.548 ± 0.022 | 0.347 ± 0.018 | 0.603 ± 0.032 | 1.796 ± 0.213 | 0.075 ± 0.031 | 5 | [0.040634696441086464, 0.11263133655239493] |
| word_input_change | line | U+V5 | 12 | 0.697 ± 0.020 | 12 | 0.541 ± 0.027 | 0.347 ± 0.018 | 0.587 ± 0.046 | 1.759 ± 0.199 | 0.072 ± 0.033 | 5 | [0.036226997937577866, 0.11072508070399817] |
| word_input_change | GT_template_retrospective | Clotho | 12 | 0.626 ± 0.024 | 12 | 0.513 ± 0.033 | 0.364 ± 0.009 | 0.484 ± 0.018 | 1.305 ± 0.035 | 0.000 ± 0.000 | 0 | [0.0, 0.0] |
| word_input_change | GT_template_retrospective | D-zero | 12 | 0.629 ± 0.013 | 12 | 0.527 ± 0.016 | 0.364 ± 0.009 | 0.491 ± 0.012 | 1.293 ± 0.018 | 0.003 ± 0.032 | 3 | [-0.049272215467745586, 0.05111306705608133] |
| word_input_change | GT_template_retrospective | LLMLog | 12 | 0.622 ± 0.029 | 12 | 0.509 ± 0.022 | 0.364 ± 0.009 | 0.458 ± 0.019 | 1.244 ± 0.057 | -0.004 ± 0.040 | 2 | [-0.06660374551112783, 0.06314797773146082] |
| word_input_change | GT_template_retrospective | Output entropy | 12 | 0.784 ± 0.021 | 12 | 0.652 ± 0.032 | 0.364 ± 0.009 | 0.573 ± 0.015 | 1.604 ± 0.051 | 0.158 ± 0.039 | 5 | [0.11879025361015685, 0.19784950158973252] |
| word_input_change | GT_template_retrospective | Token probability | 12 | 0.504 ± 0.031 | 12 | 0.417 ± 0.022 | 0.364 ± 0.009 | 0.369 ± 0.024 | 1.027 ± 0.058 | -0.122 ± 0.044 | 0 | [-0.21714690384903446, -0.030661885190898353] |
| word_input_change | GT_template_retrospective | U | 12 | 0.717 ± 0.018 | 12 | 0.615 ± 0.025 | 0.364 ± 0.009 | 0.511 ± 0.022 | 1.398 ± 0.023 | 0.090 ± 0.029 | 5 | [0.03714084589558468, 0.14580070323787336] |
| word_input_change | GT_template_retrospective | U+V5 | 12 | 0.720 ± 0.022 | 12 | 0.602 ± 0.027 | 0.364 ± 0.009 | 0.506 ± 0.035 | 1.381 ± 0.076 | 0.094 ± 0.032 | 5 | [0.034556273353256485, 0.15310478797004817] |

AUROC excludes one-class systems; AP excludes zero-positive systems, and AP=1 in all-positive populations is uninformative. Every aggregate requires all five eligible seeds per system; contributing counts are explicit. Raw CSVs include K5/10/20/50, recall and random lifts. GT-template units average risk and majority-label within GT event types solely for retrospective evaluation; this is not a deployable ranking. Strict-on-structural-known controls population changes when comparing failure definitions.

Mean±sample std describes five seed macros, not five independent datasets. Paired confidence intervals resample systems (1000 draws, seed20261111), averaging each system’s five seeds first; entire correlated template/line clusters stay inside systems. No line-IID or seed-IID p-values are used. Small system counts limit inference.

Scott–Knott ESD and A12 below use the existing `log_parsing_stats` implementation on matched system×seed observations, for comparability only. Repeated seeds are correlated: these ranks/effect sizes are descriptive, not independent-dataset significance tests. The system-cluster paired intervals above provide the inferential uncertainty.

| Model | Target | Predictor | Matched system×seeds | SK-ESD group | A12 vs Clotho | Magnitude |
|---|---|---|---|---|---|---|
| llama | strict | Clotho | 65 | 3 | 0.5 | negligible |
| llama | strict | D-zero | 65 | 2 | 0.7368047337278106 | large |
| llama | strict | LLMLog | 65 | 2 | 0.7192899408284024 | large |
| llama | strict | Output entropy | 65 | 1 | 0.8963313609467456 | large |
| llama | strict | Token probability | 65 | 4 | 0.1540828402366864 | large |
| llama | strict | U | 65 | 1 | 0.8549112426035503 | large |
| llama | strict | U+V5 | 65 | 1 | 0.8736094674556213 | large |
| llama | structural_proxy | Clotho | 65 | 3 | 0.5 | negligible |
| llama | structural_proxy | D-zero | 65 | 2 | 0.7349112426035503 | large |
| llama | structural_proxy | LLMLog | 65 | 2 | 0.7207100591715976 | large |
| llama | structural_proxy | Output entropy | 65 | 1 | 0.8951479289940828 | large |
| llama | structural_proxy | Token probability | 65 | 4 | 0.15597633136094674 | large |
| llama | structural_proxy | U | 65 | 1 | 0.8497041420118343 | large |
| llama | structural_proxy | U+V5 | 65 | 1 | 0.86698224852071 | large |
| llama | word_input_change | Clotho | 65 | 3 | 0.5 | negligible |
| llama | word_input_change | D-zero | 65 | 2 | 0.5926627218934911 | small |
| llama | word_input_change | LLMLog | 65 | 3 | 0.4727810650887574 | negligible |
| llama | word_input_change | Output entropy | 65 | 1 | 0.700828402366864 | medium |
| llama | word_input_change | Token probability | 65 | 4 | 0.23668639053254437 | large |
| llama | word_input_change | U | 65 | 2 | 0.6260355029585799 | small |
| llama | word_input_change | U+V5 | 65 | 2 | 0.6411834319526627 | medium |
| gemma | strict | Clotho | 60 | 3 | 0.5 | negligible |
| gemma | strict | D-zero | 60 | 3 | 0.6227777777777778 | small |
| gemma | strict | LLMLog | 60 | 3 | 0.585 | small |
| gemma | strict | Output entropy | 60 | 1 | 0.9277777777777778 | large |
| gemma | strict | Token probability | 60 | 4 | 0.11333333333333333 | large |
| gemma | strict | U | 60 | 2 | 0.8097222222222222 | large |
| gemma | strict | U+V5 | 60 | 2 | 0.8369444444444445 | large |
| gemma | structural_proxy | Clotho | 60 | 4 | 0.5 | negligible |
| gemma | structural_proxy | D-zero | 60 | 3 | 0.6261111111111111 | small |
| gemma | structural_proxy | LLMLog | 60 | 3 | 0.6052777777777778 | small |
| gemma | structural_proxy | Output entropy | 60 | 1 | 0.9188888888888889 | large |
| gemma | structural_proxy | Token probability | 60 | 5 | 0.12805555555555556 | large |
| gemma | structural_proxy | U | 60 | 2 | 0.8066666666666666 | large |
| gemma | structural_proxy | U+V5 | 60 | 2 | 0.8336111111111111 | large |
| gemma | word_input_change | Clotho | 60 | 3 | 0.5 | negligible |
| gemma | word_input_change | D-zero | 60 | 3 | 0.5972222222222222 | small |
| gemma | word_input_change | LLMLog | 60 | 3 | 0.5258333333333334 | negligible |
| gemma | word_input_change | Output entropy | 60 | 1 | 0.8497222222222223 | large |
| gemma | word_input_change | Token probability | 60 | 4 | 0.1736111111111111 | large |
| gemma | word_input_change | U | 60 | 2 | 0.7627777777777778 | large |
| gemma | word_input_change | U+V5 | 60 | 2 | 0.7452777777777778 | large |

## C. What review orders select

| Model | Parser | Scope | Ordering | Metric @20 | Systems | Count | Δfrequency | Positive seeds/5 |
|---|---|---|---|---|---|---|---|---|
| llama | base | full | drisk | net_lines_fixed | 11 | 6560.673 ± 1698.162 | -22703.873 ± 1698.162 | 0 |
| llama | base | full | drisk | structural_groups | 11 | 14.218 ± 0.746 | 1.491 ± 0.746 | 5 |
| llama | base | full | drisk | distinct_wrong_strings_fully_repaired | 11 | 12.800 ± 0.654 | 0.618 ± 0.654 | 3 |
| llama | base | full | dsize | net_lines_fixed | 13 | 26829.877 ± 525.749 | -377.200 ± 525.749 | 2 |
| llama | base | full | dsize | structural_groups | 13 | 13.046 ± 0.117 | 1.046 ± 0.117 | 5 |
| llama | base | full | dsize | distinct_wrong_strings_fully_repaired | 13 | 12.492 ± 0.088 | 0.954 ± 0.088 | 5 |
| llama | base | full | oracle_lines | net_lines_fixed | 13 | 27833.615 ± 0.000 | 626.538 ± 0.000 | 5 |
| llama | base | full | oracle_lines | structural_groups | 13 | 17.615 ± 0.000 | 5.615 ± 0.000 | 5 |
| llama | base | full | oracle_lines | distinct_wrong_strings_fully_repaired | 13 | 16.769 ± 0.000 | 5.231 ± 0.000 | 5 |
| llama | base | full | random | net_lines_fixed | 13 | 7372.769 ± 2399.352 | -19834.308 ± 2399.352 | 0 |
| llama | base | full | random | structural_groups | 13 | 12.308 ± 0.224 | 0.308 ± 0.224 | 4 |
| llama | base | full | random | distinct_wrong_strings_fully_repaired | 13 | 8.677 ± 0.408 | -2.862 ± 0.408 | 0 |
| llama | base | full | urisk | net_lines_fixed | 11 | 6129.800 ± 260.757 | -23134.745 ± 260.757 | 0 |
| llama | base | full | urisk | structural_groups | 11 | 16.109 ± 0.235 | 3.382 ± 0.235 | 5 |
| llama | base | full | urisk | distinct_wrong_strings_fully_repaired | 11 | 15.036 ± 0.270 | 2.855 ± 0.270 | 5 |
| llama | base | full | usize | net_lines_fixed | 13 | 26273.646 ± 29.360 | -933.431 ± 29.360 | 0 |
| llama | base | full | usize | structural_groups | 13 | 13.569 ± 0.042 | 1.569 ± 0.042 | 5 |
| llama | base | full | usize | distinct_wrong_strings_fully_repaired | 13 | 13.169 ± 0.034 | 1.631 ± 0.034 | 5 |
| llama | base | tail | drisk | net_lines_fixed | 11 | 303.164 ± 18.376 | -1139.836 ± 18.376 | 0 |
| llama | base | tail | drisk | structural_groups | 11 | 14.236 ± 0.434 | 2.509 ± 0.434 | 5 |
| llama | base | tail | drisk | distinct_wrong_strings_fully_repaired | 11 | 12.200 ± 0.491 | 2.564 ± 0.491 | 5 |
| llama | base | tail | dsize | net_lines_fixed | 11 | 1474.782 ± 28.284 | 31.782 ± 28.284 | 4 |
| llama | base | tail | dsize | structural_groups | 11 | 13.436 ± 0.175 | 1.709 ± 0.175 | 5 |
| llama | base | tail | dsize | distinct_wrong_strings_fully_repaired | 11 | 11.164 ± 0.175 | 1.527 ± 0.175 | 5 |
| llama | base | tail | oracle_lines | net_lines_fixed | 11 | 1904.636 ± 0.000 | 461.636 ± 0.000 | 5 |
| llama | base | tail | oracle_lines | structural_groups | 11 | 17.545 ± 0.000 | 5.818 ± 0.000 | 5 |
| llama | base | tail | oracle_lines | distinct_wrong_strings_fully_repaired | 11 | 15.091 ± 0.000 | 5.455 ± 0.000 | 5 |
| llama | base | tail | random | net_lines_fixed | 11 | 178.164 ± 53.541 | -1264.836 ± 53.541 | 0 |
| llama | base | tail | random | structural_groups | 11 | 12.327 ± 0.237 | 0.600 ± 0.237 | 5 |
| llama | base | tail | random | distinct_wrong_strings_fully_repaired | 11 | 7.891 ± 0.528 | -1.745 ± 0.528 | 0 |
| llama | base | tail | urisk | net_lines_fixed | 11 | 351.382 ± 25.865 | -1091.618 ± 25.865 | 0 |
| llama | base | tail | urisk | structural_groups | 11 | 15.309 ± 0.138 | 3.582 ± 0.138 | 5 |
| llama | base | tail | urisk | distinct_wrong_strings_fully_repaired | 11 | 13.982 ± 0.197 | 4.345 ± 0.197 | 5 |
| llama | base | tail | usize | net_lines_fixed | 11 | 1535.000 ± 19.093 | 92.000 ± 19.093 | 5 |
| llama | base | tail | usize | structural_groups | 11 | 14.018 ± 0.119 | 2.291 ± 0.119 | 5 |
| llama | base | tail | usize | distinct_wrong_strings_fully_repaired | 11 | 12.091 ± 0.064 | 2.455 ± 0.064 | 5 |
| llama | strong | full | drisk | net_lines_fixed | 13 | 2400.169 ± 421.725 | -9999.369 ± 421.725 | 0 |
| llama | strong | full | drisk | structural_groups | 13 | 11.769 ± 0.598 | 2.154 ± 0.598 | 5 |
| llama | strong | full | drisk | distinct_wrong_strings_fully_repaired | 13 | 8.231 ± 0.365 | -0.231 ± 0.365 | 1 |
| llama | strong | full | dsize | net_lines_fixed | 13 | 12143.000 ± 76.812 | -256.538 ± 76.812 | 0 |
| llama | strong | full | dsize | structural_groups | 13 | 10.846 ± 0.077 | 1.231 ± 0.077 | 5 |
| llama | strong | full | dsize | distinct_wrong_strings_fully_repaired | 13 | 9.031 ± 0.208 | 0.569 ± 0.208 | 5 |
| llama | strong | full | oracle_lines | net_lines_fixed | 13 | 12945.538 ± 0.000 | 546.000 ± 0.000 | 5 |
| llama | strong | full | oracle_lines | structural_groups | 13 | 15.308 ± 0.000 | 5.692 ± 0.000 | 5 |
| llama | strong | full | oracle_lines | distinct_wrong_strings_fully_repaired | 13 | 13.231 ± 0.000 | 4.769 ± 0.000 | 5 |
| llama | strong | full | random | net_lines_fixed | 13 | 1995.831 ± 877.360 | -10403.708 ± 877.360 | 0 |
| llama | strong | full | random | structural_groups | 13 | 10.323 ± 0.263 | 0.708 ± 0.263 | 5 |
| llama | strong | full | random | distinct_wrong_strings_fully_repaired | 13 | 6.262 ± 0.296 | -2.200 ± 0.296 | 0 |
| llama | strong | full | urisk | net_lines_fixed | 13 | 2838.923 ± 338.248 | -9560.615 ± 338.248 | 0 |
| llama | strong | full | urisk | structural_groups | 13 | 12.231 ± 0.163 | 2.615 ± 0.163 | 5 |
| llama | strong | full | urisk | distinct_wrong_strings_fully_repaired | 13 | 11.508 ± 0.208 | 3.046 ± 0.208 | 5 |
| llama | strong | full | usize | net_lines_fixed | 13 | 12389.154 ± 12.295 | -10.385 ± 12.295 | 1 |
| llama | strong | full | usize | structural_groups | 13 | 10.938 ± 0.100 | 1.323 ± 0.100 | 5 |
| llama | strong | full | usize | distinct_wrong_strings_fully_repaired | 13 | 9.862 ± 0.100 | 1.400 ± 0.100 | 5 |
| llama | strong | tail | drisk | net_lines_fixed | 13 | 175.446 ± 39.873 | -696.631 ± 39.873 | 0 |
| llama | strong | tail | drisk | structural_groups | 13 | 10.446 ± 0.213 | 1.062 ± 0.213 | 5 |
| llama | strong | tail | drisk | distinct_wrong_strings_fully_repaired | 13 | 8.138 ± 0.333 | 0.908 ± 0.333 | 5 |
| llama | strong | tail | dsize | net_lines_fixed | 13 | 914.846 ± 29.369 | 42.769 ± 29.369 | 5 |
| llama | strong | tail | dsize | structural_groups | 13 | 9.938 ± 0.064 | 0.554 ± 0.064 | 5 |
| llama | strong | tail | dsize | distinct_wrong_strings_fully_repaired | 13 | 7.908 ± 0.064 | 0.677 ± 0.064 | 5 |
| llama | strong | tail | oracle_lines | net_lines_fixed | 13 | 1214.231 ± 0.000 | 342.154 ± 0.000 | 5 |
| llama | strong | tail | oracle_lines | structural_groups | 13 | 13.000 ± 0.000 | 3.615 ± 0.000 | 5 |
| llama | strong | tail | oracle_lines | distinct_wrong_strings_fully_repaired | 13 | 11.154 ± 0.000 | 3.923 ± 0.000 | 5 |
| llama | strong | tail | random | net_lines_fixed | 13 | 108.662 ± 19.777 | -763.415 ± 19.777 | 0 |
| llama | strong | tail | random | structural_groups | 13 | 8.908 ± 0.240 | -0.477 ± 0.240 | 0 |
| llama | strong | tail | random | distinct_wrong_strings_fully_repaired | 13 | 4.831 ± 0.227 | -2.400 ± 0.227 | 0 |
| llama | strong | tail | urisk | net_lines_fixed | 13 | 202.769 ± 7.368 | -669.308 ± 7.368 | 0 |
| llama | strong | tail | urisk | structural_groups | 13 | 10.569 ± 0.193 | 1.185 ± 0.193 | 5 |
| llama | strong | tail | urisk | distinct_wrong_strings_fully_repaired | 13 | 9.877 ± 0.160 | 2.646 ± 0.160 | 5 |
| llama | strong | tail | usize | net_lines_fixed | 13 | 919.292 ± 11.088 | 47.215 ± 11.088 | 5 |
| llama | strong | tail | usize | structural_groups | 13 | 10.415 ± 0.042 | 1.031 ± 0.042 | 5 |
| llama | strong | tail | usize | distinct_wrong_strings_fully_repaired | 13 | 8.815 ± 0.117 | 1.585 ± 0.117 | 5 |
| llama | real_adaparser | full | dsize | net_lines_fixed | 12 | 1947.067 ± 26.247 | -44.767 ± 26.247 | 0 |
| llama | real_adaparser | full | dsize | structural_groups | 12 | 3.133 ± 0.126 | 0.050 ± 0.126 | 3 |
| llama | real_adaparser | full | dsize | distinct_wrong_strings_fully_repaired | 12 | 3.050 ± 0.126 | 0.133 ± 0.126 | 3 |
| llama | real_adaparser | full | oracle_lines | net_lines_fixed | 12 | 2529.583 ± 0.000 | 537.750 ± 0.000 | 5 |
| llama | real_adaparser | full | oracle_lines | structural_groups | 12 | 11.917 ± 0.000 | 8.833 ± 0.000 | 5 |
| llama | real_adaparser | full | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 12.167 ± 0.000 | 9.250 ± 0.000 | 5 |
| llama | real_adaparser | full | random | net_lines_fixed | 12 | 378.683 ± 214.804 | -1613.150 ± 214.804 | 0 |
| llama | real_adaparser | full | random | structural_groups | 12 | 4.133 ± 0.634 | 1.050 ± 0.634 | 5 |
| llama | real_adaparser | full | random | distinct_wrong_strings_fully_repaired | 12 | 4.183 ± 0.557 | 1.267 ± 0.557 | 5 |
| gemma | base | full | drisk | net_lines_fixed | 12 | 11165.750 ± 0.000 | -14933.083 ± 0.000 | 0 |
| gemma | base | full | drisk | structural_groups | 12 | 12.167 ± 0.000 | 1.083 ± 0.000 | 5 |
| gemma | base | full | drisk | distinct_wrong_strings_fully_repaired | 12 | 8.667 ± 0.000 | -1.583 ± 0.000 | 0 |
| gemma | base | full | dsize | net_lines_fixed | 12 | 25662.833 ± 0.000 | -436.000 ± 0.000 | 0 |
| gemma | base | full | dsize | structural_groups | 12 | 12.417 ± 0.000 | 1.333 ± 0.000 | 5 |
| gemma | base | full | dsize | distinct_wrong_strings_fully_repaired | 12 | 11.667 ± 0.000 | 1.417 ± 0.000 | 5 |
| gemma | base | full | oracle_lines | net_lines_fixed | 12 | 26970.333 ± 0.000 | 871.500 ± 0.000 | 5 |
| gemma | base | full | oracle_lines | structural_groups | 12 | 17.917 ± 0.000 | 6.833 ± 0.000 | 5 |
| gemma | base | full | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 16.750 ± 0.000 | 6.500 ± 0.000 | 5 |
| gemma | base | full | random | net_lines_fixed | 12 | 8119.233 ± 2182.919 | -17979.600 ± 2182.919 | 0 |
| gemma | base | full | random | structural_groups | 12 | 11.667 ± 0.283 | 0.583 ± 0.283 | 5 |
| gemma | base | full | random | distinct_wrong_strings_fully_repaired | 12 | 7.867 ± 0.402 | -2.383 ± 0.402 | 0 |
| gemma | base | full | urisk | net_lines_fixed | 12 | 11034.700 ± 1070.720 | -15064.133 ± 1070.720 | 0 |
| gemma | base | full | urisk | structural_groups | 12 | 12.783 ± 0.126 | 1.700 ± 0.126 | 5 |
| gemma | base | full | urisk | distinct_wrong_strings_fully_repaired | 12 | 12.017 ± 0.199 | 1.767 ± 0.199 | 5 |
| gemma | base | full | usize | net_lines_fixed | 12 | 26334.617 ± 41.079 | 235.783 ± 41.079 | 5 |
| gemma | base | full | usize | structural_groups | 12 | 13.100 ± 0.149 | 2.017 ± 0.149 | 5 |
| gemma | base | full | usize | distinct_wrong_strings_fully_repaired | 12 | 12.250 ± 0.156 | 2.000 ± 0.156 | 5 |
| gemma | base | tail | drisk | net_lines_fixed | 12 | 242.917 ± 0.000 | -779.833 ± 0.000 | 0 |
| gemma | base | tail | drisk | structural_groups | 12 | 10.083 ± 0.000 | 1.583 ± 0.000 | 5 |
| gemma | base | tail | drisk | distinct_wrong_strings_fully_repaired | 12 | 7.000 ± 0.000 | 0.333 ± 0.000 | 5 |
| gemma | base | tail | dsize | net_lines_fixed | 12 | 1116.333 ± 0.000 | 93.583 ± 0.000 | 5 |
| gemma | base | tail | dsize | structural_groups | 12 | 10.500 ± 0.000 | 2.000 ± 0.000 | 5 |
| gemma | base | tail | dsize | distinct_wrong_strings_fully_repaired | 12 | 8.667 ± 0.000 | 2.000 ± 0.000 | 5 |
| gemma | base | tail | oracle_lines | net_lines_fixed | 12 | 1520.750 ± 0.000 | 498.000 ± 0.000 | 5 |
| gemma | base | tail | oracle_lines | structural_groups | 12 | 14.917 ± 0.000 | 6.417 ± 0.000 | 5 |
| gemma | base | tail | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 12.750 ± 0.000 | 6.083 ± 0.000 | 5 |
| gemma | base | tail | random | net_lines_fixed | 12 | 139.267 ± 54.045 | -883.483 ± 54.045 | 0 |
| gemma | base | tail | random | structural_groups | 12 | 9.867 ± 0.267 | 1.367 ± 0.267 | 5 |
| gemma | base | tail | random | distinct_wrong_strings_fully_repaired | 12 | 5.967 ± 0.280 | -0.700 ± 0.280 | 0 |
| gemma | base | tail | urisk | net_lines_fixed | 12 | 201.333 ± 35.660 | -821.417 ± 35.660 | 0 |
| gemma | base | tail | urisk | structural_groups | 12 | 10.467 ± 0.292 | 1.967 ± 0.292 | 5 |
| gemma | base | tail | urisk | distinct_wrong_strings_fully_repaired | 12 | 8.983 ± 0.199 | 2.317 ± 0.199 | 5 |
| gemma | base | tail | usize | net_lines_fixed | 12 | 1256.933 ± 32.386 | 234.183 ± 32.386 | 5 |
| gemma | base | tail | usize | structural_groups | 12 | 11.167 ± 0.177 | 2.667 ± 0.177 | 5 |
| gemma | base | tail | usize | distinct_wrong_strings_fully_repaired | 12 | 9.483 ± 0.109 | 2.817 ± 0.109 | 5 |
| gemma | strong | full | drisk | net_lines_fixed | 12 | 925.500 ± 0.000 | -11728.750 ± 0.000 | 0 |
| gemma | strong | full | drisk | structural_groups | 12 | 11.333 ± 0.000 | 1.250 ± 0.000 | 5 |
| gemma | strong | full | drisk | distinct_wrong_strings_fully_repaired | 12 | 6.000 ± 0.000 | -2.667 ± 0.000 | 0 |
| gemma | strong | full | dsize | net_lines_fixed | 12 | 12554.417 ± 0.000 | -99.833 ± 0.000 | 0 |
| gemma | strong | full | dsize | structural_groups | 12 | 10.333 ± 0.000 | 0.250 ± 0.000 | 5 |
| gemma | strong | full | dsize | distinct_wrong_strings_fully_repaired | 12 | 9.083 ± 0.000 | 0.417 ± 0.000 | 5 |
| gemma | strong | full | oracle_lines | net_lines_fixed | 12 | 13253.083 ± 0.000 | 598.833 ± 0.000 | 5 |
| gemma | strong | full | oracle_lines | structural_groups | 12 | 16.417 ± 0.000 | 6.333 ± 0.000 | 5 |
| gemma | strong | full | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 13.917 ± 0.000 | 5.250 ± 0.000 | 5 |
| gemma | strong | full | random | net_lines_fixed | 12 | 1697.567 ± 828.773 | -10956.683 ± 828.773 | 0 |
| gemma | strong | full | random | structural_groups | 12 | 10.567 ± 0.406 | 0.483 ± 0.406 | 4 |
| gemma | strong | full | random | distinct_wrong_strings_fully_repaired | 12 | 5.900 ± 0.446 | -2.767 ± 0.446 | 0 |
| gemma | strong | full | urisk | net_lines_fixed | 12 | 2504.400 ± 206.817 | -10149.850 ± 206.817 | 0 |
| gemma | strong | full | urisk | structural_groups | 12 | 11.400 ± 0.384 | 1.317 ± 0.384 | 5 |
| gemma | strong | full | urisk | distinct_wrong_strings_fully_repaired | 12 | 9.617 ± 0.315 | 0.950 ± 0.315 | 5 |
| gemma | strong | full | usize | net_lines_fixed | 12 | 12674.000 ± 53.195 | 19.750 ± 53.195 | 4 |
| gemma | strong | full | usize | structural_groups | 12 | 11.167 ± 0.118 | 1.083 ± 0.118 | 5 |
| gemma | strong | full | usize | distinct_wrong_strings_fully_repaired | 12 | 9.700 ± 0.075 | 1.033 ± 0.075 | 5 |
| gemma | strong | tail | drisk | net_lines_fixed | 12 | 198.333 ± 0.000 | -486.083 ± 0.000 | 0 |
| gemma | strong | tail | drisk | structural_groups | 12 | 10.417 ± 0.000 | 2.167 ± 0.000 | 5 |
| gemma | strong | tail | drisk | distinct_wrong_strings_fully_repaired | 12 | 5.917 ± 0.000 | 0.333 ± 0.000 | 5 |
| gemma | strong | tail | dsize | net_lines_fixed | 12 | 752.417 ± 0.000 | 68.000 ± 0.000 | 5 |
| gemma | strong | tail | dsize | structural_groups | 12 | 9.000 ± 0.000 | 0.750 ± 0.000 | 5 |
| gemma | strong | tail | dsize | distinct_wrong_strings_fully_repaired | 12 | 6.167 ± 0.000 | 0.583 ± 0.000 | 5 |
| gemma | strong | tail | oracle_lines | net_lines_fixed | 12 | 1171.500 ± 0.000 | 487.083 ± 0.000 | 5 |
| gemma | strong | tail | oracle_lines | structural_groups | 12 | 14.167 ± 0.000 | 5.917 ± 0.000 | 5 |
| gemma | strong | tail | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 11.000 ± 0.000 | 5.417 ± 0.000 | 5 |
| gemma | strong | tail | random | net_lines_fixed | 12 | 104.167 ± 28.395 | -580.250 ± 28.395 | 0 |
| gemma | strong | tail | random | structural_groups | 12 | 9.033 ± 0.126 | 0.783 ± 0.126 | 5 |
| gemma | strong | tail | random | distinct_wrong_strings_fully_repaired | 12 | 4.367 ± 0.126 | -1.217 ± 0.126 | 0 |
| gemma | strong | tail | urisk | net_lines_fixed | 12 | 155.983 ± 7.942 | -528.433 ± 7.942 | 0 |
| gemma | strong | tail | urisk | structural_groups | 12 | 9.867 ± 0.484 | 1.617 ± 0.484 | 5 |
| gemma | strong | tail | urisk | distinct_wrong_strings_fully_repaired | 12 | 7.383 ± 0.139 | 1.800 ± 0.139 | 5 |
| gemma | strong | tail | usize | net_lines_fixed | 12 | 852.600 ± 37.158 | 168.183 ± 37.158 | 5 |
| gemma | strong | tail | usize | structural_groups | 12 | 10.217 ± 0.183 | 1.967 ± 0.183 | 5 |
| gemma | strong | tail | usize | distinct_wrong_strings_fully_repaired | 12 | 7.483 ± 0.149 | 1.900 ± 0.149 | 5 |
| gemma | real_adaparser | full | dsize | net_lines_fixed | 12 | 3389.250 ± 0.000 | 143.833 ± 0.000 | 5 |
| gemma | real_adaparser | full | dsize | structural_groups | 12 | 3.833 ± 0.000 | 0.583 ± 0.000 | 5 |
| gemma | real_adaparser | full | dsize | distinct_wrong_strings_fully_repaired | 12 | 3.500 ± 0.000 | 0.667 ± 0.000 | 5 |
| gemma | real_adaparser | full | oracle_lines | net_lines_fixed | 12 | 4119.833 ± 0.000 | 874.417 ± 0.000 | 5 |
| gemma | real_adaparser | full | oracle_lines | structural_groups | 12 | 11.333 ± 0.000 | 8.083 ± 0.000 | 5 |
| gemma | real_adaparser | full | oracle_lines | distinct_wrong_strings_fully_repaired | 12 | 11.083 ± 0.000 | 8.250 ± 0.000 | 5 |
| gemma | real_adaparser | full | random | net_lines_fixed | 12 | 314.267 ± 135.275 | -2931.150 ± 135.275 | 0 |
| gemma | real_adaparser | full | random | structural_groups | 12 | 3.400 ± 0.379 | 0.150 ± 0.379 | 3 |
| gemma | real_adaparser | full | random | distinct_wrong_strings_fully_repaired | 12 | 3.417 ± 0.475 | 0.583 ± 0.475 | 4 |

All budgets, selected group IDs, wrong groups, gross newly-correct lines, broken previously-correct lines, net gains, mixed GT groups, primary distributions and overlapping proxy attributes are in `review_metrics.csv`. “Fully repaired wrong string” requires every initially wrong eligible group sharing that string to be successfully corrected, not just touched. Validated v2 exports correct the eligible-queue denominator; original review files are preserved. Majority-template review does not repair fixed operational grouping. Structural groups use one real representative of each GT event present in the group; this is a proxy, not verified semantic value. Static Clotho risk and risk×size vectors were not stored; an active Clotho sequence is not substituted. Tail histories missing for regenerated Llama/base HDFS/OpenSSH are not invented. Real AdaParser has only Random/frequency/Dsize/Oracle here.

## D. Partition and downstream diagnostics

| Model | Parser | Systems | Mean ARI | Exact-set line share | Exact-set GT-template share | Mixed output groups (sum) | Split GT types (sum) |
|---|---|---|---|---|---|---|---|
| llama | base | 13 | 0.992025 | 0.856 | 0.879 | 76 | 138 |
| llama | strong | 13 | 0.992316 | 0.872 | 0.898 | 59 | 120 |
| llama | real_adaparser | 12 | 0.999486 | 0.933 | 0.928 | 35 | 123 |
| gemma | base | 12 | 0.995124 | 0.861 | 0.879 | 61 | 113 |
| gemma | strong | 12 | 0.995420 | 0.857 | 0.896 | 60 | 101 |
| gemma | real_adaparser | 12 | 0.997400 | 0.950 | 0.941 | 45 | 77 |

| Model | Parser | Partition IDs | First-line-wrong groups pooled | Wrong groups with exact GT line set | Fraction |
|---|---|---|---|---|---|
| llama | base | operational | 9257 | 1243 | 0.134 |
| llama | base | output_string | 9257 | 1249 | 0.135 |
| llama | strong | operational | 8737 | 1022 | 0.117 |
| llama | strong | output_string | 8737 | 1655 | 0.189 |
| llama | real_adaparser | operational | 627 | 309 | 0.493 |
| llama | real_adaparser | output_string | 627 | 309 | 0.493 |
| gemma | base | operational | 8860 | 974 | 0.110 |
| gemma | base | output_string | 8860 | 7562 | 0.853 |
| gemma | strong | operational | 8651 | 869 | 0.100 |
| gemma | strong | output_string | 8651 | 1299 | 0.150 |
| gemma | real_adaparser | operational | 627 | 429 | 0.684 |
| gemma | real_adaparser | output_string | 627 | 429 | 0.684 |

Conditional counts use actual GT membership sets: a predicted ID must contain exactly one GT type, and that GT type must occur in exactly that predicted ID. Output-string mode first coalesces equal strings. These are pooled groups, not system macros or claims that every line in a mixed group is wrong. Both directions are checked; purity alone is insufficient.

| System | GT test sessions | Empty TF-IDF sessions | Empty anomalous sessions | GT training vocabulary |
|---|---|---|---|---|
| HDFS | 2044 | 0 | 0 | 34 |
| BGL | 1250 | 1 | 0 | 74 |

The GT word representation is not empty for any anomalous test session in HDFS/BGL. Empty vectors therefore do not explain the weak thresholded baselines. Detector choice, thresholds, train/test distribution and prefix/session construction remain plausible causes; none is established causally by this audit.

| Model | System | Detector | Templates/review | Test sessions | Test positives | F1 | AP | ΔF1−N0 | ΔAP−N0 |
|---|---|---|---|---|---|---|---|---|---|
| llama | HDFS | IF | D_size | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.180 ± 0.085 |
| llama | HDFS | IF | GT | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.180 ± 0.085 |
| llama | HDFS | IF | U_V5_size | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.180 ± 0.085 |
| llama | HDFS | IF | frequency | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.180 ± 0.085 |
| llama | HDFS | IF | oracle_lines | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.180 ± 0.085 |
| llama | HDFS | IF | uncorrected | 2044 | 74 | 0.185 ± 0.000 | 0.187 ± 0.069 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| llama | HDFS | OCSVM | D_size | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.173 ± 0.000 |
| llama | HDFS | OCSVM | GT | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.173 ± 0.000 |
| llama | HDFS | OCSVM | U_V5_size | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.173 ± 0.000 |
| llama | HDFS | OCSVM | frequency | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.173 ± 0.000 |
| llama | HDFS | OCSVM | oracle_lines | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.173 ± 0.000 |
| llama | HDFS | OCSVM | uncorrected | 2044 | 74 | 0.186 ± 0.000 | 0.480 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| llama | BGL | IF | D_size | 1250 | 92 | 0.229 ± 0.003 | 0.634 ± 0.007 | -0.006 ± 0.010 | -0.026 ± 0.025 |
| llama | BGL | IF | GT | 1250 | 92 | 0.206 ± 0.051 | 0.630 ± 0.009 | -0.028 ± 0.056 | -0.030 ± 0.026 |
| llama | BGL | IF | U_V5_size | 1250 | 92 | 0.228 ± 0.003 | 0.630 ± 0.007 | -0.006 ± 0.010 | -0.030 ± 0.025 |
| llama | BGL | IF | frequency | 1250 | 92 | 0.229 ± 0.003 | 0.642 ± 0.008 | -0.006 ± 0.010 | -0.018 ± 0.025 |
| llama | BGL | IF | oracle_lines | 1250 | 92 | 0.228 ± 0.003 | 0.630 ± 0.007 | -0.006 ± 0.010 | -0.030 ± 0.025 |
| llama | BGL | IF | uncorrected | 1250 | 92 | 0.234 ± 0.011 | 0.659 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| llama | BGL | OCSVM | D_size | 1250 | 92 | 0.123 ± 0.000 | 0.075 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 |
| llama | BGL | OCSVM | GT | 1250 | 92 | 0.123 ± 0.000 | 0.075 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 |
| llama | BGL | OCSVM | U_V5_size | 1250 | 92 | 0.122 ± 0.000 | 0.074 ± 0.000 | 0.003 ± 0.000 | -0.000 ± 0.000 |
| llama | BGL | OCSVM | frequency | 1250 | 92 | 0.124 ± 0.000 | 0.075 ± 0.000 | 0.005 ± 0.000 | 0.001 ± 0.000 |
| llama | BGL | OCSVM | oracle_lines | 1250 | 92 | 0.122 ± 0.000 | 0.074 ± 0.000 | 0.003 ± 0.000 | -0.000 ± 0.000 |
| llama | BGL | OCSVM | uncorrected | 1250 | 92 | 0.119 ± 0.000 | 0.075 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| gemma | HDFS | IF | D_size | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.150 ± 0.157 |
| gemma | HDFS | IF | GT | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.150 ± 0.157 |
| gemma | HDFS | IF | U_V5_size | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.150 ± 0.157 |
| gemma | HDFS | IF | frequency | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.150 ± 0.157 |
| gemma | HDFS | IF | oracle_lines | 2044 | 74 | 0.185 ± 0.000 | 0.367 ± 0.097 | 0.000 ± 0.000 | 0.150 ± 0.157 |
| gemma | HDFS | IF | uncorrected | 2044 | 74 | 0.185 ± 0.000 | 0.217 ± 0.110 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| gemma | HDFS | OCSVM | D_size | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.172 ± 0.000 |
| gemma | HDFS | OCSVM | GT | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.172 ± 0.000 |
| gemma | HDFS | OCSVM | U_V5_size | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.172 ± 0.000 |
| gemma | HDFS | OCSVM | frequency | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.172 ± 0.000 |
| gemma | HDFS | OCSVM | oracle_lines | 2044 | 74 | 0.070 ± 0.000 | 0.308 ± 0.000 | -0.116 ± 0.000 | -0.172 ± 0.000 |
| gemma | HDFS | OCSVM | uncorrected | 2044 | 74 | 0.186 ± 0.000 | 0.480 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| gemma | BGL | IF | D_size | 1250 | 92 | 0.230 ± 0.003 | 0.628 ± 0.014 | 0.116 ± 0.003 | 0.041 ± 0.014 |
| gemma | BGL | IF | GT | 1250 | 92 | 0.206 ± 0.051 | 0.630 ± 0.009 | 0.092 ± 0.050 | 0.043 ± 0.009 |
| gemma | BGL | IF | U_V5_size | 1250 | 92 | 0.230 ± 0.002 | 0.628 ± 0.012 | 0.116 ± 0.003 | 0.042 ± 0.012 |
| gemma | BGL | IF | frequency | 1250 | 92 | 0.230 ± 0.002 | 0.640 ± 0.015 | 0.116 ± 0.003 | 0.053 ± 0.015 |
| gemma | BGL | IF | oracle_lines | 1250 | 92 | 0.230 ± 0.003 | 0.625 ± 0.012 | 0.116 ± 0.003 | 0.038 ± 0.013 |
| gemma | BGL | IF | uncorrected | 1250 | 92 | 0.114 ± 0.001 | 0.587 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| gemma | BGL | OCSVM | D_size | 1250 | 92 | 0.013 ± 0.000 | 0.073 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 |
| gemma | BGL | OCSVM | GT | 1250 | 92 | 0.123 ± 0.000 | 0.075 ± 0.000 | 0.110 ± 0.000 | 0.000 ± 0.000 |
| gemma | BGL | OCSVM | U_V5_size | 1250 | 92 | 0.013 ± 0.000 | 0.073 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 |
| gemma | BGL | OCSVM | frequency | 1250 | 92 | 0.013 ± 0.000 | 0.074 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 |
| gemma | BGL | OCSVM | oracle_lines | 1250 | 92 | 0.013 ± 0.000 | 0.073 ± 0.000 | 0.000 ± 0.000 | -0.002 ± 0.000 |
| gemma | BGL | OCSVM | uncorrected | 1250 | 92 | 0.013 ± 0.000 | 0.074 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

D refits vocabulary/IDF and detector consistently for every corrected representation on the chronological training half; anomaly labels never set review rankings or tune thresholds. IF100trees/auto contamination, seeds0–4; OCSVM rbf/gamma-scale/nu0.05. New decision-score/AP recovery asserts every available old N3 F1/precision/recall to1e-12. Only base-cache HDFS/BGL diagnostics are rerun; existing H count-based results and N3 analyses are preserved. Raw decision-score vectors are not used to train any failure predictor.

| System | Sessions | Anomalies | Test anomalies | GT count-PCA F1 | GT IM F1 |
|---|---|---|---|---|---|
| HDFS | 4088 | 123 | 74 | 0.0 | 0.18568665377176016 |
| BGL | 2500 | 233 | 92 | 0.056074766355140186 | 0.019417475728155338 |

High ARI can hide rare split/merge errors: exact-set template share is shown alongside line-weighted agreement. Fixed Drain group IDs do not imply fixed output-string partitions; equal cached strings can coalesce groups. Count models are invariant only under a bijective event relabelling, not under every correction. Word-bag differences establish detector-input sensitivity, not semantic harm; N3 ignores order and variable values. Weak GT count baselines and truncated prefix/block sessions limit the ability to detect useful downstream gains. Positive cases exist, so “no anomalies” cannot explain HDFS/BGL insensitivity.

## Representative real examples and independent review

Selection rule: first sorted readable sample0 example per primary/model/system/suite; the display takes the first such example per model/primary, independent of all predictor scores. Exact stored templates follow.

| ID | Automatic proxy | Log | Prediction | GT |
|---|---|---|---|---|
| llama/capped/Apache/3/s0 | variable_extraction_proxy | child process 29767 still did not exit, sending a SIGTERM | child <*> still did not exit, sending a <*> | child process <*> still did not exit, sending a SIGTERM |
| llama/capped/Apache/7/s0 | nonmatching_representation_proxy | jk2_init() Can't find child 4025 in scoreboard | jk2_*() Can't find child <*> in scoreboard | jk2_init() Can't find child <*> in scoreboard |
| llama/capped/Apache/33/s0 | uncertain | [client 220.33.28.14] Directory index forbidden by rule: /var/www/html/ | [client <*>] Directory index forbidden by rule: /<*>/<*>/ | [client <*>] Directory index forbidden by rule: <*> |
| llama/capped/Hadoop/417/s0 | surface_whitespace_proxy | DFS chooseDataNode: got # 1 IOException, will wait for 1857.517062515084 msec. | DFS chooseDataNode: got #  <*>  IOException, will wait for  <*>  msec. | DFS chooseDataNode: got # <*> IOException, will wait for <*> msec. |
| gemma/capped/BGL/0/s0 | variable_extraction_proxy | tlb error.........................1 | tlb  <*>.........................<*> | tlb error.........................<*> |
| gemma/capped/BGL/7/s0 | uncertain | Controlling BG/L rows [ 0 1 2 3 4 5 6 7 ] | Controlling BG/L rows [ <*><*><*><*><*><*><*><*> ] | Controlling BG/L rows [ <*> ] |
| gemma/capped/BGL/14/s0 | nonmatching_representation_proxy | regctl scancom interface.........................0 | regctl scancom <*> <*>.........................0 | regctl scancom interface.........................<*> |
| gemma/capped/BGL/134/s0 | surface_whitespace_proxy | L3 ecc control register: 00000000 | L3  ecc control register: <*> | L3 ecc control register: <*> |

## Missing comparisons and threats

- gemma/BGL/MDSA/seed0: strict replay failed: observed 0.5775201288244767, stored 0.5777133655394524
- gemma/BGL/MDSA/seed3: strict replay failed: observed 0.6039024390243903, stored 0.6039674796747967
- gemma/BGL/MDSA/seed4: strict replay failed: observed 0.5700666666666666, stored 0.5701333333333334
- gemma/HPC/MDSA/seed1: strict replay failed: observed 0.6605810234541577, stored 0.6605143923240938
- gemma/Hadoop/MDSA/seed0: strict replay failed: observed 0.636845165041018, stored 0.6367805697306375
- gemma/Hadoop/MDSA/seed1: strict replay failed: observed 0.6130983015954709, stored 0.6131626351003602
- gemma/Hadoop/MDSA/seed3: strict replay failed: observed 0.5982799369747899, stored 0.5984112394957983
- gemma/Hadoop/MDSA/seed4: strict replay failed: observed 0.6962179487179486, stored 0.6961538461538462
- gemma/HealthApp/MDSA/seed2: strict replay failed: observed 0.6473793103448277, stored 0.6473103448275862
- gemma/Linux/MDSA/seed4: strict replay failed: observed 0.7219427244582044, stored 0.7218782249742002
- gemma/Mac/MDSA/seed1: strict replay failed: observed 0.6523809523809524, stored 0.6521428571428571
- gemma/Mac/MDSA/seed2: strict replay failed: observed 0.6213467851709504, stored 0.6212667147089439
- gemma/Mac/MDSA/seed4: strict replay failed: observed 0.6980571428571428, stored 0.6981333333333333
- gemma/Thunderbird/MDSA/seed0: strict replay failed: observed 0.6387165775401069, stored 0.6387878787878788
- gemma/Thunderbird/MDSA/seed3: strict replay failed: observed 0.6115972222222222, stored 0.6114583333333333
- gemma/Thunderbird/MDSA/seed4: strict replay failed: observed 0.6451854882705946, stored 0.6451172940534643
- gemma/Zookeeper/MDSA/seed0: strict replay failed: observed 0.5932927831662008, stored 0.593128390596745
- gemma/Zookeeper/MDSA/seed1: strict replay failed: observed 0.4709922430510666, stored 0.47107304460245636
- llama/BGL/MDSA/seed0: strict replay failed: observed 0.5907598686844423, stored 0.5909200096084555
- llama/BGL/MDSA/seed1: strict replay failed: observed 0.6150519750519751, stored 0.6151351351351352
- llama/HDFS/MDSA/seed0: strict replay failed: observed 0.7480140774258421, stored 0.7479135243841126
- llama/HDFS/MDSA/seed4: strict replay failed: observed 0.7440794499618029, stored 0.7441558441558441
- llama/HPC/MDSA/seed3: strict replay failed: observed 0.4375853825136612, stored 0.4376707650273224
- llama/Hadoop/MDSA/seed4: strict replay failed: observed 0.673400187576654, stored 0.6735444773104393
- llama/HealthApp/MDSA/seed0: strict replay failed: observed 0.663668430335097, stored 0.6635802469135803
- llama/HealthApp/MDSA/seed2: strict replay failed: observed 0.6054964539007093, stored 0.6053600654664484
- llama/HealthApp/MDSA/seed3: strict replay failed: observed 0.7626777662157475, stored 0.7625390218522372
- llama/HealthApp/MDSA/seed4: strict replay failed: observed 0.6038617253192151, stored 0.6040174400498287
- llama/Linux/MDSA/seed0: strict replay failed: observed 0.773109243697479, stored 0.7729476405946994
- llama/Mac/MDSA/seed0: strict replay failed: observed 0.7744538232373386, stored 0.7743296921549156
- llama/Mac/MDSA/seed4: strict replay failed: observed 0.8393042575285565, stored 0.8390446521287642
- llama/Mac/real_adaparser: no successful real parser
- llama/OpenSSH/MDSA/seed1: strict replay failed: observed 0.5998563218390804, stored 0.5997765006385696
- llama/Spark/MDSA/seed0: strict replay failed: observed 0.7284169884169883, stored 0.7283397683397682
- llama/Spark/MDSA/seed1: strict replay failed: observed 0.7000892060660124, stored 0.6997323818019625
- llama/Spark/MDSA/seed2: strict replay failed: observed 0.7210878517501716, stored 0.7210020590253947
- llama/Thunderbird/MDSA/seed0: strict replay failed: observed 0.5764911634756995, stored 0.5767673048600883
- llama/Thunderbird/MDSA/seed1: strict replay failed: observed 0.5987782047492364, stored 0.5984826091240517
- llama/Thunderbird/MDSA/seed2: strict replay failed: observed 0.6399999999999999, stored 0.6399
- llama/Thunderbird/MDSA/seed3: strict replay failed: observed 0.5707614942528736, stored 0.5705818965517242
- llama/Thunderbird/MDSA/seed4: strict replay failed: observed 0.6555995532541374, stored 0.6557010864047113
- llama/Zookeeper/MDSA/seed0: strict replay failed: observed 0.5345856997233144, stored 0.5345128877238969
- llama/Zookeeper/MDSA/seed1: strict replay failed: observed 0.6510015408320493, stored 0.6509373394966614
- llama/Zookeeper/MDSA/seed2: strict replay failed: observed 0.663085434894493, stored 0.6631497683993823

No human-verified semantic labels, temporal E5 handling labels, severity scale or semantic AUROC is available. Grouping-error prediction AUROC is not computed: stored capped input scores and operational-group histories do not provide a preregistered shared group-risk vector/target mapping; we report actual partition agreement and reviewed mixed-GT groups instead. These are unknown, not evidence of absence. No standard/natural or individual-subtype predictor rerun was scheduled in this short CPU window. The broad structural union overlaps strict failure heavily; subtype-specific hard cases remain unresolved. Recovering exact old MDSA vectors/RNG history and static Clotho stream-risk vectors remains necessary for those comparisons. Label uncertainty, 3ms matching limits, GT convention errors, retrospective template aggregation and capped-distribution bias constrain interpretation. U contains output-validity checks similar to the proxy construction; superior proxy AUC can partly reflect this construct alignment rather than downstream importance. Grouping/semantics/downstream outcomes must not be conflated. D has no target-suite training labels; U has90 plus frozen sources, so their supervision budgets differ. No predictor/threshold/severity weight was tuned on this audit.

## Reproduction and artifact map

New scripts leave all earlier validated results intact. Detailed data are in private RESULTS_REPO, with the following paths relative to `experiments/`. Pull these narrow prefixes on a fresh machine.

| Artifact | Location |
|---|---|
| Suite outputs and GT | `data/results_<model>/log_parsing/inference_results/rules_template/` and original suite JSONL |
| Original hidden states/scores/splits | Original data/results paths and `results_phase3{,_gemma}/final/fair_leakfree/disjoint/`; frozen U vectors in `token_followups/suites/` |
| Per-input labels, exact templates and scores | `data/severity_audit/<model>/{capped,natural,joined_scores}_<System>.json.gz` |
| Group membership annotations / review histories | `data/severity_audit/<model>/stream_<parser>_<System>.json.gz` and `reviews_validated_<System>.json.gz` |
| Per-system A/B/D results | `results_phase3{,_gemma}/severity_audit/`, `prediction/`, `downstream/` |
| CSV metrics / blind labels / uncertainty / validation | `reports/error_severity_audit/` (raw CSVs stay private; aggregate report is in Git) |
| Paper artefacts | `paper/figures/error_{distribution,predictors,budget,review}.{pdf,tex}` and `paper/tables/error_severity*.tex` |

Run from the repo root with the existing CPU `.venv` and saved inputs; each model-specific producer uses `CLOTHO_MODEL=llama|gemma`, and existing per-system results are skipped. CPU jobs set OMP/BLAS threads to1.

```bash
# Source the existing secret environment without printing it.
# Only needed to regenerate missing new audit jobs, for each model:
CLOTHO_MODEL=llama .venv/bin/python experiments/log_parsing_error_severity.py --workers 6
CLOTHO_MODEL=llama .venv/bin/python experiments/log_parsing_severity_predict.py --workers 6
CLOTHO_MODEL=llama .venv/bin/python experiments/log_parsing_severity_downstream.py
# Repeat the three commands with CLOTHO_MODEL=gemma.
.venv/bin/python experiments/log_parsing_severity_validate.py
.venv/bin/python experiments/log_parsing_severity_partition.py
.venv/bin/python experiments/log_parsing_severity_report.py
.venv/bin/python experiments/paper/fig_error_distribution.py
.venv/bin/python experiments/paper/fig_error_predictors.py
.venv/bin/python experiments/paper/fig_error_budget.py
.venv/bin/python experiments/paper/fig_error_review.py
.venv/bin/python experiments/paper/table_error_severity.py
```

## Research interpretation and smallest follow-up

The broad structural proxy almost reproduces strict failure: majority positives are65.4% vs67.0% for Llama and51.4% vs54.6% for Gemma capped suites, with1.4%/2.6% unknown. Thus unchanged predictor rankings under this proxy provide limited evidence of a new severity signal. Whitespace-only differences do not explain most strict failures under this rule, but formal mismatch still does not establish practical harm.

| Model | Predictor | Strict AUROC | Structural proxy AUROC | Word-input-change AUROC |
|---|---|---|---|---|
| llama | Clotho | 0.645 ± 0.022 | 0.642 ± 0.020 | 0.622 ± 0.014 |
| llama | U | 0.776 ± 0.028 | 0.774 ± 0.029 | 0.664 ± 0.025 |
| llama | U+V5 | 0.783 ± 0.027 | 0.781 ± 0.028 | 0.671 ± 0.021 |
| llama | Output entropy | 0.796 ± 0.014 | 0.794 ± 0.014 | 0.701 ± 0.011 |
| gemma | Clotho | 0.659 ± 0.025 | 0.656 ± 0.022 | 0.625 ± 0.022 |
| gemma | U | 0.773 ± 0.015 | 0.771 ± 0.014 | 0.701 ± 0.019 |
| gemma | U+V5 | 0.784 ± 0.010 | 0.781 ± 0.011 | 0.697 ± 0.020 |
| gemma | Output entropy | 0.824 ± 0.012 | 0.819 ± 0.014 | 0.736 ± 0.016 |

U retains an advantage over Clotho for extraction/representation proxies; the word-input-change advantage is smaller. For U alone it is+0.041 in Llama (system-cluster interval crosses zero), +0.075 in Gemma (interval excludes zero). U+V5 is+0.049/+0.072 on that target. Output entropy has the highest mean AUROC among verified full-cohort methods for both new targets in both models; U has no established advantage over output entropy here. These are changed detector words, not semantic importance. Clotho remains above chance descriptively but is substantially weaker than the best output-based ranking.

At20 base/full reviews, U-risk selects+3.38/+1.70 structural-proxy groups per system relative to frequency (Llama11/Gemma12 matched systems), positive on5/5 seed macros, while fixing about23135/15064 fewer net lines. U×size selects+1.57/+2.02 proxy groups (13/12 systems) with−933/+236 net lines. D×size selects+1.05/+1.33 proxy groups with−377/−436 net lines. This answers the review question only at proxy level: more formally erroneous groups can be found without more lines fixed; greater consequence remains unverified. Frequency remains competitive for line utility. Paired system-cluster intervals in `review_summary.csv` must accompany inference; five positive seeds alone do not establish significance.

| Model | Parser | Group size | Groups pooled | Structural-proxy groups | Fraction |
|---|---|---|---|---|---|
| llama | base | rare_1_2 | 10686 | 8024 | 0.7508890136627363 |
| llama | base | frequent_3plus | 2302 | 1646 | 0.7150304083405734 |
| llama | strong | rare_1_2 | 10686 | 8021 | 0.7506082725060828 |
| llama | strong | frequent_3plus | 2302 | 1442 | 0.6264118158123371 |
| llama | real_adaparser | rare_1_2 | 418 | 233 | 0.5574162679425837 |
| llama | real_adaparser | frequent_3plus | 1379 | 373 | 0.2704858593183466 |
| gemma | base | rare_1_2 | 10684 | 9352 | 0.8753275926619244 |
| gemma | base | frequent_3plus | 2275 | 1413 | 0.6210989010989011 |
| gemma | strong | rare_1_2 | 10684 | 9288 | 0.8693373268438787 |
| gemma | strong | frequent_3plus | 2275 | 1283 | 0.5639560439560439 |
| gemma | real_adaparser | rare_1_2 | 371 | 126 | 0.33962264150943394 |
| gemma | real_adaparser | frequent_3plus | 1740 | 482 | 0.2770114942528736 |

Rare groups (size1–2) have a higher pooled structural-proxy fraction in all six model/parser cells; these are descriptive pooled counts, not a verified harm gradient or a system-adjusted causal effect. High output-partition ARI(0.992–0.999) coexists with exact-set line agreement0.856–0.950 andGT-template agreement0.879–0.941: it is incorrect to say every wrong template preserves the true partition. Corrections to fixed operational templates still cannot repair their membership errors.

Downstream results reject a simple “more correct templates ⇒ better detector” interpretation. Gemma/BGL IF improves fromF1≈0.114 to≈0.230 after20 reviews for frequency, D×size andU×size alike; frequency has the highest AP of these methods(≈0.640 vs≈0.628). Llama/BGL corrections slightly reduce IF F1/AP. HDFS corrections leave IF F1≈0.185 unchanged while AP rises, but reduce OCSVM F1≈0.186→0.070 andAP≈0.480→0.308. Hence thresholded F1 can hide ranking changes, and correction can hurt a detector. Gemma/BGL OCSVM stays≈0.013 after20 reviews despiteGT F1≈0.123, showing the chosen review budget does not recover its GT behavior. Two systems do not support a general downstream superiority claim for U or D.

Confirmed scope: we can test formal proxy detection, reviewed error composition and representation-specific detector effects with stored outputs. We cannot conclude that any method identifies expert-verified consequential or semantic errors best. Frequency remains the paired reference at each budget; proxy-group gains with fewer net corrected lines are reported explicitly in machine-readable contrasts and must not be renamed downstream gains. Negative/mixed F1 or AP changes should be retained even when parsing improves.

Highest-value next step: have two domain reviewers independently annotate the blinded stratified worksheet, resolve disagreements and report agreement before claiming semantic/severity performance. Then use an independent GT-template-disjoint evaluation set and frozen predictor rankings. For downstream validation, first establish a useful GT/oracle ceiling on complete sessions and a representation that observes the annotated errors. Reuse real AdaParser’s existing event partition or a fixed semantic/parameter-consuming detector if count features cannot observe the change; this changes the architecture question and must be stated. No new LLM generation is justified until that sensitivity/ceiling check succeeds.
