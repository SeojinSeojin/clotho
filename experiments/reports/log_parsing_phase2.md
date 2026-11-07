# log_parsing Phase 2: 14 Loghub-2.0 systems x 2 prompts (llama, vLLM generation, Clotho random/pca, layer 21)

Input PASS = more than 5 of 10 samples match the ground-truth template after normalization. ROC-AUC is over inputs not yet in the reference set; reference size includes the 10 initial tests. `first fit` = reference size at which Clotho first fitted a GMM.


## messages_template

| system | output pass | parse fail | input PASS | first fit | AUC@50 | AUC@100 | AUC@200 | mean AUC | Clotho status |
|---|---|---|---|---|---|---|---|---|---|
| apache | not generated | | | | | | | | | |
| bgl | not generated | | | | | | | | | |
| hdfs | 10.6% | 0.0% | 2.6% (13) | n/a | n/a | n/a | n/a | n/a | not run |
| hpc | not generated | | | | | | | | | |
| hadoop | not generated | | | | | | | | | |
| healthapp | not generated | | | | | | | | | |
| linux | not generated | | | | | | | | | |
| mac | not generated | | | | | | | | | |
| openssh | not generated | | | | | | | | | |
| openstack | not generated | | | | | | | | | |
| proxifier | not generated | | | | | | | | | |
| spark | not generated | | | | | | | | | |
| thunderbird | not generated | | | | | | | | | |
| zookeeper | not generated | | | | | | | | | |
| **macro avg (defined only, n systems)** | | | | | n/a | n/a | n/a | n/a | |

## rules_template

| system | output pass | parse fail | input PASS | first fit | AUC@50 | AUC@100 | AUC@200 | mean AUC | Clotho status |
|---|---|---|---|---|---|---|---|---|---|
| apache | not generated | | | | | | | | | |
| bgl | not generated | | | | | | | | | |
| hdfs | not generated | | | | | | | | | |
| hpc | not generated | | | | | | | | | |
| hadoop | not generated | | | | | | | | | |
| healthapp | not generated | | | | | | | | | |
| linux | not generated | | | | | | | | | |
| mac | not generated | | | | | | | | | |
| openssh | not generated | | | | | | | | | |
| openstack | not generated | | | | | | | | | |
| proxifier | not generated | | | | | | | | | |
| spark | not generated | | | | | | | | | |
| thunderbird | not generated | | | | | | | | | |
| zookeeper | not generated | | | | | | | | | |
| **macro avg (defined only, n systems)** | | | | | n/a | n/a | n/a | n/a | |
