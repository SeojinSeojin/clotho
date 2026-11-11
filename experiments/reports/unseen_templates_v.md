# V: more zero-extra-call signals for unseen templates

Details: `unseen_details.md`. Disjoint split, 100 references, AUC, 5 seeds; paired differences (k/5 keeping the sign); tuned on HDFS, Spark, Linux. "Added" = extra feature group on U's best (Llama S1+S3, Gemma S1+S2+S3), classifier unchanged. Clotho 0.645 / 0.659; U's best 0.776 / 0.773; output entropy (5 extra calls) 0.796 / 0.824.

| signal | Llama alone | Llama added: minus U best | Gemma alone | Gemma added: minus U best |
|---|---|---|---|---|
| V1 multi-layer (concat / ensemble) | 0.711 / 0.720 | +0.005 (4/5) | 0.714 / 0.683 | -0.003 (3/5) |
| V2 decision-point confidence | needs tokenizer (GPU session) | | | |
| V3 reasoning text | 0.593 | -0.001 (4/5) | 0.645 | +0.005 (3/5) |
| V4 same-group consistency | 0.624 | +0.005 (3/5) | 0.651 | +0.008 (4/5) |
| V5 Drain disagreement | 0.683 | **+0.007 (5/5)** | 0.706 | **+0.011 (4/5)** |
| V6 soft lookup | 0.637 | -0.015 (5/5) | 0.626 | +0.009 (3/5) |
| V7 cross-system neighbours | 0.591 | -0.001 (4/5) | 0.598 | -0.002 (3/5) |
| V8a template-level targets | | -0.002 (5/5) | | -0.007 (4/5) |
| V8b template-grouped CV | | +0.001 (2/5) | | -0.005 (5/5) |

Best added signal: V5, U + V5 = 0.783 / 0.784 (+0.139 / +0.125 over Clotho, 5/5). Gap to output entropy: -0.012 ± 0.020 (3/5) Llama, -0.039 ± 0.007 (5/5) Gemma. Every added row except Llama U + V6 is in U's best Scott-Knott group (for Llama, output entropy's group too).

- Every V gain over U's best is at most +0.012: U's discriminative hidden-state model already carries most of the zero-call signal.
- V5 is the only signal that helps both models on most seeds. Alone (0.68 / 0.71) it beats Clotho.
- V6 (soft lookup) hurts on the disjoint split for Llama but is the strongest addition elsewhere: +0.034 / +0.029 over U's best on the standard split and +0.018 / +0.044 on natural suites (100 references), where neighbours share templates.
- V8 (template-level targets, grouped CV) brings nothing.
- V2: per-token log-probs are stored without token strings; aligning them needs the tokenizer (a model download), so it is deferred.
- Extra LLM calls: 0 for every V signal (V3 to V5 read sample 0; labels use samples 5 to 9).
