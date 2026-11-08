# log_parsing Phase 2: 14 Loghub-2.0 systems x 2 prompts (llama, vLLM generation, Clotho random/pca, layer 21)

Input PASS = more than 5 of 10 samples match the ground-truth template (strict_match). ROC-AUC is over inputs not yet in the reference set; reference size includes the 10 initial tests. `first fit` = reference size at which Clotho first fitted a GMM.


## messages_template

| system | output pass | parse fail | input PASS | first fit | AUC@50 | AUC@100 | AUC@200 | mean AUC | Clotho status |
|---|---|---|---|---|---|---|---|---|---|
| apache | 43.1% | 0.0% | 43.8% (219) | 40 | 0.955 | 0.974 | 0.971 | 0.955 | ok |
| bgl | 35.1% | 0.9% | 33.0% (165) | 30 | 0.695 | 0.717 | 0.701 | 0.742 | ok |
| hdfs | 9.9% | 0.0% | 2.6% (13) | 200 | 0.866 | 0.866 | 0.866 | 0.615 | ok |
| hpc | 24.7% | 0.2% | 18.0% (90) | 50 | 0.585 | 0.919 | 0.891 | 0.883 | ok |
| hadoop | 24.0% | 0.3% | 21.2% (106) | 60 | 0.665 | 0.727 | 0.756 | 0.823 | ok |
| healthapp | 44.0% | 0.1% | 44.0% (220) | 30 | 0.702 | 0.845 | 0.860 | 0.873 | ok |
| linux | 29.3% | 0.4% | 27.8% (139) | 40 | 0.599 | 0.750 | 0.870 | 0.841 | ok |
| mac | 21.1% | 0.5% | 18.6% (93) | 50 | 0.667 | 0.731 | 0.799 | 0.801 | ok |
| openssh | 38.3% | 0.0% | 35.6% (178) | 30 | 0.881 | 0.832 | 0.860 | 0.866 | ok |
| openstack | 52.1% | 0.0% | 55.6% (278) | 30 | 0.785 | 0.881 | 0.900 | 0.841 | ok |
| proxifier | 0.2% | 0.0% | 0.0% (0) | n/a | n/a | n/a | n/a | n/a | not run |
| spark | 25.6% | 0.4% | 22.4% (112) | 60 | 0.667 | 0.796 | 0.800 | 0.845 | ok |
| thunderbird | 30.6% | 0.6% | 30.0% (150) | 40 | 0.549 | 0.597 | 0.584 | 0.637 | ok |
| zookeeper | 36.4% | 0.3% | 32.0% (160) | 30 | 0.740 | 0.857 | 0.787 | 0.773 | ok |
| **macro avg (defined only, n systems)** | | | | | 0.720 (13) | 0.807 (13) | 0.819 (13) | 0.807 (13) | |

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
