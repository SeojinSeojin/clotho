# N1-N3: CPU-only analyses (2026-11-10)

Protocol: PROGRESS.md Decisions (N1-N3). Details: `cpu_analyses_details.md`; figure `paper/figures/lorenz.pdf`. Share = share of mis-parsed lines fixed, macro mean over systems. Llama base: 11 systems (HDFS, OpenSSH cache never pushed).

## N1. Why Most frequent nearly matches the oracle

Group sizes are extremely concentrated: mean Gini 0.86; the top 1/5/10/20 groups hold 32/72/80/86% of lines. Being wrong is unrelated to size (mean Spearman -0.04 to -0.09 in the four model x parser cells). So Most frequent's first N groups hold about their line share of the errors (within 0.014 of the size-independent expectation). Among the 20 largest groups 50-65% are wrong. The oracle gains only by swapping the correct ones for the next-largest wrong groups, which are small. A proxy that uses only sizes and verdicts (gain = size if wrong) predicts the gap: r = 0.92, mean absolute error 0.003 at 20 reviews (48 system x cell points). Actual gap: 0.08 / 0.06 / 0.03 at 5 / 10 / 20 reviews.

## N2. When would risk prediction pay off? (simulation)

A simulation: wrong groups flipped to correct on the stored streams (5 seeds); D refit only on simulated verdicts.

- Random flips (errors stay a random subset): D x size never beats Most frequent at 20 reviews for Llama, base or strong, at any wrong-group share down to 5%. Gemma: from 10% (base, +0.026, 4/5 seeds) and 5% (strong, +0.016, 5/5).
- D-low-risk flips (errors left are the ones D rates as risky): D x size overtakes from 40% wrong (Llama strong: from 20%), by up to +0.12 at 5% wrong. This rule favours D by construction, so it is an upper bound.

A more accurate parser alone does not make risk prediction pay off; it pays off only if the parser's remaining errors are the ones D already flags as risky.

## N3. Does template text matter for a semantics-based detector?

Yes, unlike the count-based detectors in H. Template-word TF-IDF + Isolation Forest on BGL: ground truth 0.21; Gemma base cache 0.11, which 20 reviews raise to 0.23. The one-class SVM stays at 0.01 under Gemma's cache against 0.12 under ground truth. On HDFS, Gemma's wrong templates even raise one-class SVM F1 (0.19 vs 0.07 under ground truth). Most frequent and D x size give the same F1 (within 0.004). All detectors stay weak (F1 at most 0.23).
