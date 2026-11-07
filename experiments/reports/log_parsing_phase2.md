# log_parsing Phase 2: 14 Loghub-2.0 systems x 2 prompts (llama, vLLM generation, Clotho random/pca, layer 21)

Input PASS = more than 5 of 10 samples match the ground-truth template (strict_match). ROC-AUC is over inputs not yet in the reference set; reference size includes the 10 initial tests. `first fit` = reference size at which Clotho first fitted a GMM.


## rules_template

| system | output pass | parse fail | input PASS | first fit | AUC@50 | AUC@100 | AUC@200 | mean AUC | Clotho status |
|---|---|---|---|---|---|---|---|---|---|
| apache | 55.1% | 0.0% | 57.0% (285) | 20 | 0.723 | 0.726 | 0.749 | 0.749 | ok |
| bgl | 27.0% | 0.6% | 20.0% (100) | 20 | 0.560 | 0.613 | 0.710 | 0.705 | ok |
| hdfs | 30.4% | 0.0% | 27.2% (136) | 30 | 0.766 | 0.803 | 0.882 | 0.873 | ok |
| hpc | 26.8% | 0.0% | 18.4% (92) | 20 | 0.813 | 0.812 | 0.824 | 0.836 | ok |
| hadoop | 31.8% | 0.2% | 28.6% (143) | 20 | 0.666 | 0.665 | 0.742 | 0.738 | ok |
| healthapp | 34.5% | 0.0% | 29.6% (148) | 20 | 0.781 | 0.754 | 0.831 | 0.829 | ok |
| linux | 26.8% | 0.9% | 22.8% (114) | 20 | 0.672 | 0.690 | 0.770 | 0.758 | ok |
| mac | 17.1% | 0.4% | 12.0% (60) | 40 | 0.669 | 0.642 | 0.678 | 0.710 | ok |
| openssh | 41.0% | 0.0% | 35.4% (177) | 30 | 0.733 | 0.792 | 0.787 | 0.775 | ok |
| openstack | 51.9% | 0.0% | 54.2% (271) | 20 | 0.849 | 0.831 | 0.838 | 0.811 | ok |
| proxifier | 0.7% | 0.0% | 0.0% (0) | n/a | n/a | n/a | n/a | n/a | not run |
| spark | 26.9% | 0.2% | 22.0% (110) | 20 | 0.709 | 0.738 | 0.700 | 0.753 | ok |
| thunderbird | 24.7% | 0.4% | 20.4% (102) | 20 | 0.490 | 0.549 | 0.583 | 0.552 | ok |
| zookeeper | 44.8% | 0.3% | 44.6% (223) | 20 | 0.827 | 0.862 | 0.836 | 0.810 | ok |
| **macro avg (defined only, n systems)** | | | | | 0.712 (13) | 0.729 (13) | 0.764 (13) | 0.761 (13) | |
