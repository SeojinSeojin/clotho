# Does a template-aware strategy find more failures per human check than Clotho alone?

Source: `log_parsing_strategy.py`; tables in `log_parsing_strategy_details.md` and `paper_numbers.md` section 12. A strategy labels B suite inputs, then a human checks the top B of its ranking of the rest (2B checks). Failures are summed over systems (Llama 13, Gemma 12), mean over 5 seeds; "every seed" = same sign on all 5 paired seeds.

**Natural suites: no.** Clotho alone finds the most failures at every budget, for both models, on every seed. Drain-propagate + Clotho finds 47 to 64 fewer (Llama, B = 50 / 100 / 200) and 26 to 50 fewer (Gemma). Llama at B = 100: Clotho 2047 failures (0.79 per check), Drain-propagate + Clotho 1983, Random 1690. One label per Drain group covers 95% of failures at B = 50, but Clotho's own reference choice puts more failures among the labelled inputs (455 vs 419).

**Capped suites: mostly a tie.** Drain-propagate + Clotho minus Clotho: Llama -2, -50 (every seed), +6; Gemma +10, +13, +32. Only Gemma's +32 at B = 200 holds on every seed (3029 vs 2997 failures).

**What Clotho adds to Drain.** Ranking unlabelled inputs by Clotho instead of at random adds 17 to 108 failures on every suite, model and budget, on every seed.

**Propagation noise.** An inherited group label is wrong for 8 to 10% of inputs (natural) and 11 to 16% (capped).

**Answer.** Among these four, a Drain-first strategy does not beat Clotho on natural logs and matches it on capped suites. Later variants (`log_parsing_strategy_explained.md`) show Drain-lookup ranking beats Clotho on natural logs at B = 200.
