# CPU round 3: robustness analyses

**R1 is partial; R2–R4 were not started.** The scorer audit is complete for both models and both suites. Previously eligible disjoint jobs completed; standard coverage is partial, natural AUC and stream robustness unfinished.

## R1. Lenient scoring

**Changes the interpretation of strict PASS rates; full robustness remains unresolved.** L1 forgives substantial numbers of failures but also breaks a few strict matches. Official AdaParser post-processing includes semantic abstraction of numbers, addresses and paths, beyond formatting. L2 adds only symmetric whitespace collapse and merging consecutive wildcards.

| model / suite | systems | strict PASS | L1 PASS | L2 PASS | L1 forgiven / broken inputs | L2 forgiven / broken inputs |
|---|---:|---:|---:|---:|---:|---:|
| llama / capped | 13 | 0.330 | 0.433 | 0.439 | 671 / 2 | 711 / 2 |
| llama / natural | 13 | 0.351 | 0.481 | 0.487 | 847 / 1 | 888 / 1 |
| gemma / capped | 12 | 0.454 | 0.535 | 0.549 | 488 / 1 | 574 / 1 |
| gemma / natural | 12 | 0.409 | 0.469 | 0.485 | 367 / 1 | 462 / 1 |

Per-system audit, reliable between-template variance and completed disjoint AUCs (including U-best @100, paired signs, SK-ESD and A12) are in [cpu_round3_details.md](cpu_round3_details.md). Standard coverage is partial; no natural AUC is reported.

**Stream headline:** lenient PA_0 and whether Most frequent remains close to Oracle with no predictor beating it at 20 reviews are unresolved. Existing strict results are unchanged.

## R2. Template metrics and the tail

**Not run; effect on the claim unknown.**

| status | reason |
|---|---|
| not started | R1 incomplete at time cutoff |

## R3. U without target labels

**Not run; effect on the claim unknown.**

| status | reason |
|---|---|
| not started | R1 incomplete at time cutoff |

## R4. Full-length concentration

**Not run; effect on the claim unknown.**

| status | reason |
|---|---|
| not started | R1 incomplete at time cutoff |
