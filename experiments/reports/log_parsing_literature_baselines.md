# Literature baselines: LLMLog selection score and AdaParser self-correction

Sources: `log_parsing_literature.py`, `log_parsing_llmlog.py`, `log_parsing_adaparser.py` (official code: LLMLog @648553d, AdaParser @232951a). Tables and all deviations: `log_parsing_literature_details.md`, `paper_numbers.md` section 11. Setting fair_leakfree, standard split, 5 seeds; pairs are @100 / @200; "every seed" = same sign on all 5 paired seeds.

**Scope note.** LLMLog's score picks logs for a human to annotate as in-context demonstrations. Here it ranks inputs by likely failure, which it was not built for. A low AUC is no claim that LLMLog fails at its own task.

**(1) LLMLog vs Clotho and Drain lookup.** LLMLog scores 0.675 / 0.675 (Llama, natural), 0.720 / 0.717 (Llama, capped), 0.620 / 0.611 (Gemma, natural) and 0.686 / 0.683 (Gemma, capped). Clotho is ahead on every seed in all four (+0.029 to +0.260). Lookup (Drain) beats it on the natural suites (+0.16 to +0.30) and on the capped suites except Llama @100 (-0.040). Confidence alone is within 0.002 of the full score; representativeness alone is at chance (0.47 to 0.51). In the top 50, LLMLog finds 38 to 43 failures, Clotho 43 to 50, Random 26 to 33.

**(2) Does AdaParser-style correction fix most failures?** No. On samples 5-9 it fixes 8.0% (Llama, natural), 15.6% (Llama, capped), 5.9% (Gemma, natural) and 9.0% (Gemma, capped) of failing inputs. It breaks 97, 122, 219 and 141 passing ones, mostly on OpenSSH, Linux and Thunderbird. There, its constant heuristic flags real variables such as process names; on Gemma natural Linux only 10 of 500 inputs still pass.

**Do the predictors still help afterwards?** Yes. Against the corrected label, Clotho reaches 0.843 / 0.850, 0.766 / 0.813, 0.901 / 0.875 (11 systems) and 0.801 / 0.839, at or above its original values. Clotho still beats Lookup (Drain) on every seed on the capped suites (+0.056 to +0.078); on the natural suites the lookup ties it (Llama) or leads (Gemma @200). Output entropy drops by 0.01 to 0.06. Of the failures in Clotho's top 10%, only 2.1% to 4.9% are fixed, against 6.0% to 15.3% of all test failures. So the flagged failures are the ones correction cannot fix.

**Cost.** The corrector makes 0.34 to 0.51 LLM calls per output and takes 0.009 to 0.018 s per output (batched vLLM). Clotho's forward pass takes 0.020 (Llama) and 0.035 (Gemma) s per input, and it runs before generation; the corrector needs the output first.
