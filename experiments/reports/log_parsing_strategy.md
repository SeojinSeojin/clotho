# Does a template-aware strategy find more failures per human check than Clotho alone?

Source: `log_parsing_strategy.py` (protocol in its docstring); full tables in `log_parsing_strategy_details.md` and `paper_numbers.md` section 12. Label from samples 5-9 (PASS if >= 3 of 5). A strategy labels B suite inputs, then a human checks the top B of its ranking of the rest (2B checks). Totals are summed over systems (Llama 13, Gemma 12), mean over 5 seeds; "every seed" = the paired difference has the same sign on all 5.

**Natural suites: no.** Clotho alone finds the most failures at every budget, for both models, on every seed. Drain-propagate + Clotho finds 47 to 64 fewer (Llama, B = 50 / 100 / 200) and 26 to 50 fewer (Gemma). At B = 100, Clotho finds 2047 failures in 200 checks per system (0.79 per check), against 1983 for Drain-propagate + Clotho and 1690 for Random (Llama). Labelling one input per Drain group already covers 95% of failures at B = 50, but Clotho's own reference choice (uncertainty plus diversity) puts more failures among the labelled inputs (455 vs 419 at B = 50).

**Capped suites: a tie, with a small edge only at B = 200.** For Llama the difference is -2, -50 (every seed) and +6 at B = 50 / 100 / 200. For Gemma it is +10, +13 and +32, but only the last is the same sign on every seed (3029 vs 2997 failures, 0.63 vs 0.62 per check).

**What Clotho adds to Drain.** On every suite, model and budget, ranking the unlabelled inputs by Clotho instead of at random adds 17 to 108 failures, on every seed.

**Propagation noise.** An inherited group label is wrong for 8 to 10% of inputs on the natural suites and 11 to 16% on the capped suites.

**Answer.** Per human check, a Drain-first strategy does not beat Clotho on naturally distributed logs. On template-capped suites it matches Clotho, and it beats it by about 1% only for Gemma at B = 200.
