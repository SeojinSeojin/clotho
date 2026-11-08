# Leakage in the original Clotho protocol (code reading only)

Question: in the original Clotho repo, are the pass labels and the post-generation metrics (LOHSvar, Sem-Ent, Tok-Prob, Tok-Ent) computed from the same generations?

**Answer: yes.** Each input has one generation run (10 samples, T=0.8). It writes one JSON file, and that file supplies both the label (fraction of the 10 samples that pass) and every post-generation metric. LOHSvar's output hidden states are recorded during that same run. Tok-Prob and Tok-Ent use sample 0, which is also one of the 10 samples behind the label. Sem-Ent and LOHSvar use all 10.

Source: upstream `github.com/coinse/clotho` at commit `e8aed3d` (`git show upstream/main:<path>`; the paths are the same in this fork). Line numbers refer to that commit.

| step | file:line | what it reads or writes |
|---|---|---|
| Generation | `experiments/data_extraction/2-1_generate_outputs_and_record_hidden_states.py:16-17` | `REPEAT = 10`, `TEMPERATURE = 0.8` |
| | same file `:170-171` | writes `inference_results/<tmpl>/<ds>_R10_T0.8.json` **and** `output_hidden_states/<tmpl>/<ds>_R10_T0.8/` |
| | same file `:195` | one `get_hidden_states_during_generation(..., repeat=REPEAT)` call yields both the 10 outputs and their hidden states |
| | same file `:207-219` | saves those hidden states (`token_<t>_layer_<L>/input_<i>.pt`) and the same 10 `inferences` into the JSON |
| **Label** | `RQ/analysis_utils.py:110-133` (`get_test_results`) | reads `<ds>_R{num_inferences}_T{temperature}.json` (defaults 10, 0.8) at `:123`; labels every sample at `:130` (`label_test_outputs`, `:47-61`); `test_score` = fraction passing over all 10 at `:132` |
| | `experiments/utils.py:95-111` | the same function for the experiment scripts (path `:100`, score `:110`); `run_clotho_iterations.py:42` and `run_baseline_OOD.py:33` call it |
| **LOHSvar** | `experiments/data_extraction/3-1_repeated_outputs_variance.py:95-100` | reads the same `_R10_T0.8.json` (for labels) and `:97` the same `output_hidden_states/..._R10_T0.8/` |
| | same file `:180-196` | variance over the 10 hidden states of the 10 labelled samples (`:195`); also stores `score = labels_aggregated[i]` from the same outputs (`:196`, `:104-105`) |
| | `RQ/analysis_utils.py:441-473` | `load_loh_variance_df`: reads `_R10_T0.8_output_state_variances.pkl` (`:459`) and attaches `test_score` from `get_test_results` (`:442`, `:472`) |
| **Tok-Prob / Tok-Ent** | `experiments/data_extraction/3-2_token_based_metrics.py:78-81` | reads `inference_results/<tmpl>/<ds>_R10_T0.8.json` |
| | same file `:96-116` | teacher-forced log-prob and entropy for every one of the 10 `inferences` |
| | `RQ/analysis_utils.py:476-496`, `:499-521` | keep `inference_index == 0` only (`:491`, `:515`) and attach `test_score` from the same JSON (`:479`/`:495`, `:502`/`:519`) |
| **Sem-Ent** | `experiments/data_extraction/3-3_semantic_entropy.py:70` | reads `inference_results/<tmpl>/<ds>_R10_T0.8.json` |
| | `clotho/metrics/semantic_entropy.py:124-181` (`pipeline`) | clusters all 10 `inferences` of each entry (`:149`, `:153-169`) |
| | `RQ/analysis_utils.py:397-414` | attaches `test_score` from `get_test_results` (`:399`, `:412`) |
| Evaluation | `RQ/RQ3-1_loh_metrics_evaluation.ipynb`, cell 1 (`compute_metrics_for_model`) | Spearman of each metric against that `test_score` |

Consequence: a post-generation metric that measures agreement among the 10 samples (Sem-Ent, LOHSvar, our output entropy) is partly a function of the label itself. If all 10 samples are identical, the label is 0 or 1 and the dispersion is minimal. Tok-Prob/Tok-Ent share one of the 10 labelled samples. Clotho and the pre-generation baselines (MDSA, GMM_base, SLL) use input hidden states only and cannot leak this way.

How we handle it: `fair_leakfree` (`experiments/log_parsing_phase3_final.py`) takes the label from samples 5-9 and the post-generation scores from samples 0-4. The measured leakage is in `log_parsing_phase3_final.md` (output entropy loses about 0.12 to 0.14 AUC net).
