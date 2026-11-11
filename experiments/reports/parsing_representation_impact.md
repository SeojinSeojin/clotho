# Parsing representation impact

Executed984 baseline/review rows, reusing stored assignments and rankings; no model generation or predictor refit. Grouping statistics use all50,000 matched lines/system. Sequence diagnostics use the old frozen complete units (39 anomalous HDFS blocks/116 positions;2,500 BGL windows/50,000 positions). Lexical results below are test-only (24 HDFS blocks;500 all-normal BGL windows). These are invalid anomaly-recovery populations but usable explicitly conditional representation diagnostics. No detector harm is inferred.

## Grouping, event sequences and lexical features

| System | Parser | String events | ARI | NMI | Pair P/R | GT-event split fraction | One-to-one changed positions | Test word cosine |
|---|---|---|---|---|---|---|---|---|
| HDFS | GT | 13 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 1.00000 |
| HDFS | Drain | 14 | 1.0000000 | 0.99999 | 1.0000000/1.0000000 | 0.077 | 0.00000 | 1.00000 |
| HDFS | llama/cache | 14 | 1.0000000 | 0.99999 | 1.0000000/1.0000000 | 0.077 | 0.00000 | 0.81094 |
| HDFS | llama/AdaParser | 13 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 1.00000 |
| HDFS | gemma/cache | 14 | 1.0000000 | 0.99999 | 1.0000000/1.0000000 | 0.077 | 0.00000 | 0.72959 |
| HDFS | gemma/AdaParser | 13 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 1.00000 |
| HDFS | llama/strong | 13 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 0.96790 |
| HDFS | gemma/strong | 13 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 1.00000 |
| BGL | GT | 54 | 1.0000000 | 1.00000 | 1.0000000/1.0000000 | 0.000 | 0.00000 | 1.00000 |
| BGL | Drain | 51 | 0.9999996 | 0.99968 | 0.9999997/1.0000000 | 0.019 | 0.00036 | 0.64835 |
| BGL | llama/cache | 50 | 0.9999996 | 0.99972 | 0.9999997/1.0000000 | 0.000 | 0.00034 | 0.64823 |
| BGL | llama/AdaParser | 66 | 0.9999848 | 0.99727 | 1.0000000/0.9999876 | 0.037 | 0.00140 | 0.99993 |
| BGL | gemma/cache | 50 | 0.9999996 | 0.99972 | 0.9999997/1.0000000 | 0.000 | 0.00034 | 0.64615 |
| BGL | gemma/AdaParser | 57 | 0.9999965 | 0.99942 | 1.0000000/0.9999971 | 0.037 | 0.00028 | 0.99993 |
| BGL | llama/strong | 50 | 0.9999996 | 0.99972 | 0.9999997/1.0000000 | 0.000 | 0.00034 | 1.00000 |
| BGL | gemma/strong | 50 | 0.9999996 | 0.99972 | 0.9999997/1.0000000 | 0.000 | 0.00034 | 1.00000 |

Partition metrics use contingency counts rather than quadratic pairs. Machine-readable outputs also include merged/split event counts, affected line frequencies, GT-event completeness and predicted-event purity (event macro means), plus line-weighted measures. None outputs are explicitly observed parsing failures; they form a diagnostic failure event, not missing predictions. Drain/base/strong operational IDs are kept separately from final-template-string event partitions. AdaParser uses actual final assignments, verified bijective to strings by the prior200,000 CSV joins.

IDs are never compared numerically. We report two diagnostic GT-informed mappings: majority-GT mapping (many-to-one, deliberately collapses splits) and maximum-weight one-to-one matching (Hungarian contingency assignment), retaining unmatched predicted events as distinct symbols. Unmatched/ambiguous/tied event counts and affected positions are saved. Both use GT across the matched prefix solely for representation comparison, never to construct a deployable detector. Sequence unigram/bigram/3-gram JS divergence is in bits; bigrams measure transition joint distributions. Edges never cross unit boundaries. Majority mapping alone would hide AdaParser over-splitting, so it is not the sole distortion measure.

HDFS AdaParser is identical to GT. BGL AdaParser has no mixed-GT final groups but splits2/54 GT types, affecting222 lines; Llama has66 final groups and Gemma57 vs54 GT. This is real partition change that event/context detectors can observe, although ARI is near1 and majority-mapped sequence changes are zero. Drain/cache have almost identical partitions despite much worse strict text accuracy. Neither high ARI nor a zero majority-mapped difference proves harmlessness.

Word and character3–5-gram TF-IDF share a single frozen vocabulary/IDF per system, fit on the union of unique training templates across conditions including GT (diagnostic only). Validation/test text is transformed without fitting. Token bags use lowercase[a-z]+ with wildcards removed. Saved measures include constant-token multiset loss/addition, token-frequency JS, vocabulary loss/introduction, word/character cosine, GT-event macro cosine, and rare(≤20 prefix lines)/frequent stratification. This is lexical preservation, not semantic correctness. Empty/out-of-vocabulary GT vectors are counted; a both-empty cosine convention of1 is explicit and must not be read as semantic agreement.

## Stored review replay

| System | Parser | Order | Reviews≤20 | Newly correct lines | Changed output groups | Test word cosine | Mapped sequence change |
|---|---|---|---|---|---|---|---|
| HDFS | llama/cache | frequent | 14.0 | 15895.0 | 9.0 | 1.00000 | 0.00000 |
| HDFS | llama/cache | dsize | 14.0 | 15895.0 | 9.0 | 1.00000 | 0.00000 |
| HDFS | llama/cache | oracle | 14.0 | 15895.0 | 9.0 | 1.00000 | 0.00000 |
| HDFS | llama/strong | frequent | 4.0 | 20.0 | 2.0 | 1.00000 | 0.00000 |
| HDFS | llama/strong | dsize | 4.0 | 20.0 | 2.0 | 1.00000 | 0.00000 |
| HDFS | llama/strong | usize | 4.0 | 20.0 | 2.0 | 1.00000 | 0.00000 |
| HDFS | llama/strong | U-zero x size | 4.0 | 20.0 | 2.0 | 1.00000 | 0.00000 |
| HDFS | llama/strong | U+V5-zero x size | 4.0 | 20.0 | 2.0 | 1.00000 | 0.00000 |
| HDFS | llama/AdaParser | frequent | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| HDFS | llama/AdaParser | dsize | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| HDFS | llama/AdaParser | oracle | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| HDFS | gemma/cache | frequent | 14.0 | 27652.0 | 10.0 | 1.00000 | 0.00000 |
| HDFS | gemma/cache | dsize | 14.0 | 27652.0 | 10.0 | 1.00000 | 0.00000 |
| HDFS | gemma/cache | usize | 14.0 | 27652.0 | 10.0 | 1.00000 | 0.00000 |
| HDFS | gemma/cache | U-zero x size | 14.0 | 27652.0 | 10.0 | 1.00000 | 0.00000 |
| HDFS | gemma/cache | U+V5-zero x size | 14.0 | 27652.0 | 10.0 | 1.00000 | 0.00000 |
| HDFS | gemma/strong | frequent | 4.0 | 1.0 | 1.0 | 1.00000 | 0.00000 |
| HDFS | gemma/strong | dsize | 4.0 | 1.0 | 1.0 | 1.00000 | 0.00000 |
| HDFS | gemma/strong | usize | 4.0 | 1.0 | 1.0 | 1.00000 | 0.00000 |
| HDFS | gemma/strong | U-zero x size | 4.0 | 1.0 | 1.0 | 1.00000 | 0.00000 |
| HDFS | gemma/strong | U+V5-zero x size | 4.0 | 1.0 | 1.0 | 1.00000 | 0.00000 |
| HDFS | gemma/AdaParser | frequent | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| HDFS | gemma/AdaParser | dsize | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| HDFS | gemma/AdaParser | oracle | 13.0 | 0.0 | 0.0 | 1.00000 | 0.00000 |
| BGL | llama/cache | frequent | 20.0 | 44315.0 | 10.0 | 0.99999 | 0.00034 |
| BGL | llama/cache | dsize | 20.0 | 44371.0 | 14.0 | 0.99999 | 0.00034 |
| BGL | llama/cache | usize | 20.0 | 44385.0 | 15.0 | 0.99999 | 0.00034 |
| BGL | llama/cache | U-zero x size | 20.0 | 44385.0 | 15.0 | 0.99999 | 0.00034 |
| BGL | llama/cache | U+V5-zero x size | 20.0 | 44385.0 | 15.0 | 0.99999 | 0.00034 |
| BGL | llama/strong | frequent | 20.0 | 495.0 | 4.0 | 1.00000 | 0.00034 |
| BGL | llama/strong | dsize | 20.0 | 513.0 | 6.0 | 1.00000 | 0.00034 |
| BGL | llama/strong | usize | 20.0 | 515.4 | 6.6 | 1.00000 | 0.00034 |
| BGL | llama/strong | U-zero x size | 20.0 | 495.0 | 4.0 | 1.00000 | 0.00034 |
| BGL | llama/strong | U+V5-zero x size | 20.0 | 509.0 | 5.0 | 1.00000 | 0.00034 |
| BGL | llama/AdaParser | frequent | 20.0 | 203.0 | 5.0 | 1.00000 | 0.00066 |
| BGL | llama/AdaParser | dsize | 20.0 | 189.0 | 4.0 | 1.00000 | 0.00066 |
| BGL | llama/AdaParser | oracle | 20.0 | 252.0 | 18.0 | 1.00000 | 0.00000 |
| BGL | gemma/cache | frequent | 20.0 | 44610.0 | 13.0 | 1.00000 | 0.00034 |
| BGL | gemma/cache | dsize | 20.0 | 44624.0 | 14.0 | 1.00000 | 0.00034 |
| BGL | gemma/cache | usize | 20.0 | 44657.6 | 16.4 | 1.00000 | 0.00034 |
| BGL | gemma/cache | U-zero x size | 20.0 | 44641.8 | 17.0 | 1.00000 | 0.00034 |
| BGL | gemma/cache | U+V5-zero x size | 20.0 | 44666.0 | 17.0 | 1.00000 | 0.00034 |
| BGL | gemma/strong | frequent | 20.0 | 579.0 | 9.0 | 1.00000 | 0.00034 |
| BGL | gemma/strong | dsize | 20.0 | 565.0 | 8.0 | 1.00000 | 0.00034 |
| BGL | gemma/strong | usize | 20.0 | 581.8 | 9.2 | 1.00000 | 0.00034 |
| BGL | gemma/strong | U-zero x size | 20.0 | 573.4 | 8.6 | 1.00000 | 0.00034 |
| BGL | gemma/strong | U+V5-zero x size | 20.0 | 579.0 | 9.0 | 1.00000 | 0.00034 |
| BGL | gemma/AdaParser | frequent | 20.0 | 208.0 | 2.0 | 1.00000 | 0.00028 |
| BGL | gemma/AdaParser | dsize | 20.0 | 220.0 | 3.0 | 1.00000 | 0.00004 |
| BGL | gemma/AdaParser | oracle | 20.0 | 224.0 | 7.0 | 1.00000 | 0.00000 |

Full budgets 5/10/20/50 and five seeds are in representation_{HDFS,BGL}.json. “Changed output groups” counts reviewed operational groups whose output string changes; it is not a claim that every member becomes correct. Oracle uses saved line-gain ordering, not a newly optimized lexical oracle. U/zero-target histories are included where stored; static Clotho rankings and real-AdaParser U are unavailable. Strong uses final C2 templates, including its verified seeds; no greedy substitute.

Newly wrong lines and net correct-line changes are saved separately; a majority-template intervention may sacrifice previously correct minority members. Operational membership grouping metrics are exactly invariant under every template-only review. Corrected text can coalesce string-derived event IDs, so the separately reported string partition/sequence can change; mixed groups cannot be split by this intervention. There are no stored grouping-aware human repairs. Calling the operational ARI improvement positive would be incorrect.

At BGL20 reviews, frequency and D restore the same test lexical cosine for both base caches and both AdaParsers, despite differing whole-stream wrong-line recovery. Both AdaParsers reach numerical cosine1 on this all-normal test. Frequency and D therefore do not demonstrate distinct recovered lexical information here. Effects on rarer templates/whole-stream group counts can differ; the cost analysis treats those as separate outcomes.

## Error-type integration and implications

The completed severity audit is reused: BGL Llama AdaParser primary proxies48 exact/12 nonmatching representation/5 variable extraction/1 uncertain; Gemma50/2/4/1. These are first-line automatic proxies, not independently verified semantic error labels. Variable-boundary and word proxies can change token bags without changing operational membership. High-frequency errors weight line-level lexical distortion more heavily by construction; event-macro and rare/frequent fields expose this weighting rather than assuming rare errors are less important. The stored217-example blinded semantic worksheet still lacks independent annotation; no new heuristic semantic facts are assigned.

For cached parsers, a semantic/text-sensitive detector has more representation headroom than fixed-group count models. For real BGL AdaParser, over-splitting provides sequence/context headroom while text distortion is small on the stored test. Broader valid matched outputs are absent, and strong detector health is unmeasured. Lexical changes, grouping changes and anomaly recovery remain three different claims.

[Four-panel PDF](../results_phase3/cpu_research/representation.pdf) shows ARI/NMI, one-to-one sequence changes, line-weighted lexical distributions and budget recovery. Scripts/results remain separate from validated earlier paper plots.
