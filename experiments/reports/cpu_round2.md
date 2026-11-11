# CPU round 2: unseen templates, same-group validation, review and gate

Details in the `*_details.md` reports. Both models, 5 seeds, tuned on HDFS, Spark, Linux; (k/5) = seeds keeping the sign. Llama base lacks HDFS and OpenSSH (cache never pushed).

## Which zero-extra-call signal helps most on unseen templates?

A discriminative model on the stored hidden states (S1): alone +0.104 / +0.102 AUC over Clotho on the template-disjoint split (Llama / Gemma, 5/5). Adding checks on the parser's own answer (S3) gives U's best: 0.776 / 0.773 vs Clotho 0.645 / 0.659. The best V addition, Drain disagreement (V5), adds only +0.007 / +0.011. Output entropy (5 extra calls) stays ahead at 0.796 / 0.824. Soft lookup (V6) helps only where templates repeat (standard split, natural suites).

## Does same-group validation catch wrong cached templates early?

Precisely, but rarely. At a group's 2nd line the check flags wrong groups with precision 0.83 to 0.89, but recall is 0.19 to 0.25 of all wrong groups (0.30 to 0.41 of those reaching that line). Flagged groups hold 28 to 43% of mis-parsed lines. Groups that never get a 2nd line (27 to 32%) hold only 2 to 14%. Reviewing flagged groups first loses to Most frequent by 0.09 to 0.36 share. As a gate deciding at the 2nd line (q = 20%), it fixes more than a random creation-time gate in all four model x parser cells (+0.023 to +0.090; 5/5 Llama, 3-4/5 Gemma) with a quarter of the reviews.

## Does any signal beat Most frequent in review, or random at the gate?

- Review: U + V5 as risk x size beats Most frequent at 5 and 10 reviews in three of four cells (+0.015 to +0.044, 5/5); level at 20 and 50; below for Llama base. It uses 100 labelled inputs of the same system. Earlier orderings beat Most frequent only for Llama strong at 5 reviews (output entropy x size +0.038, D x size +0.012) and never at 20: size is unrelated to wrongness and the 20 largest groups hold 86% of lines (N1).
- Gate (creation time, no size): every predictor has higher precision than random (+0.14 to +0.30). For share fixed, U + V5 beats random for Gemma at every q and Llama strong at q = 20. Output entropy (extra calls) does so in 4 of 12 cells on 5/5 seeds, LLMLog in 2, D in none. Any creation-time gate fixes only 10 to 38% of what retrospective Most frequent fixes with as many reviews (N4).

## Other N results

- N2 (simulation): with random flips a more accurate parser does not make D x size beat Most frequent at 20 reviews for Llama; it does only if the remaining errors are the ones D rates risky.
- N5: the LLM beats Drain on rare templates (1-2 lines) on 10 of 11-12 systems; on frequent groups the base LLM ties Drain and the strong parser wins mainly through its 10 verified seed templates.
