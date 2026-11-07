# Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs

This repository contains the code and data for the paper "[Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs](https://coinse.github.io/publications/pdfs/Yoon2026cv.pdf)".

## Installation

We strongly recommend using the provided Dockerfile to set up the environment for running Clotho, as it ensures all dependencies are correctly installed and configured for GPU support.

### Docker Container Setup
**Step 1. Build Docker Image**
```bash
> cd docker # Dockerfile is in this directory
> docker build -f Dockerfile -t clotho-image .
```

**Step 2. Run Docker Container**

Run this command from the root of the repository (requires NVIDIA Docker runtime for GPU support):

```bash
docker run -dt --gpus all --entrypoint=/bin/bash -v .:/root/workspace --name clotho clotho-image:latest
```

To attach to the running container, use:

```bash
docker exec -it clotho /bin/bash
```

### Ephemeral GPU servers: `setup.sh` + sharded, resumable runs (no Docker)

For machines that are wiped after every session. `setup.sh` rebuilds everything in a few minutes. Long jobs run in shards (500 inputs by default), and each finished shard is uploaded to a private Hugging Face dataset repo (`RESULTS_REPO`). That way an interrupted session loses at most the shard that was running.

**One-time:** create an HF token with *write* access, accept the [Llama 3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and pick a repo name (it is created as private on first use).

**At the start of every session:**

```bash
export HF_TOKEN=hf_...                      # never commit this
export RESULTS_REPO=<hf-user>/clotho-results
git clone https://github.com/<you>/clotho && cd clotho && bash setup.sh   # ~1-5 min
source .venv/bin/activate
tmux new -s run                              # so an SSH drop doesn't kill the job

T="-t spell_check -d misspell_injected_wordnet"
python experiments/sharded/run_sharded.py --stage hidden   $T   # LIHS (1-1 script, HF forward pass)
python experiments/sharded/run_sharded.py --stage gen_vllm $T   # outputs for pass/fail labels (vLLM)
cd experiments && python run_clotho_iterations.py --target_task spell_check --target_llm llama --refset_extension_methods random --seeds 0
```

Re-run the same commands in the next session and finished shards are skipped. When every shard of a job is done, the runner merges them into the original files (`experiments/data/results_llama/...`) and uploads those under `data/` in `RESULTS_REPO`. `setup.sh` pulls `data/` back down in the next session.

| Stage | What runs | Output |
|---|---|---|
| `hidden` | `1-1_extract_input_hidden_states.py`, unmodified, once per shard | `input_hidden_states/.../layer_{21,23}/hidden_vectors.pt` |
| `gen_vllm` | `experiments/sharded/vllm_generate.py` in `.venv-vllm` | `inference_results/.../<dataset>_R10_T0.8.json` (no output hidden states) |
| `gen_hf` | `2-1_generate_outputs_and_record_hidden_states.py`, unmodified (slow) | same JSON + `output_hidden_states/` (kept in the repo as one tar per shard; merge extracts them locally) |

Useful flags: `--status` (what is done, locally or remote), `--limit_shards N` (smoke test; its merge goes to `experiments/sharded_work/merged_partial/`, never to `experiments/data`), `--shard_size N`, `--merge_only`, `--session_hours 10 --safety_margin_min 20` (won't start a shard that would overrun the session).
Progress, with time per shard and an ETA, is written to `experiments/sharded_work/<shard path>/progress.log` and uploaded with every shard.
`gen_vllm` first runs `experiments/sharded/check_prompt_parity.py`, which compares the real `clotho/models/llama.py` code path against the vLLM path (prompt strings, token ids, output post-processing) and refuses to generate on any mismatch. The vLLM path supports Llama only for now.
Environment knobs for `setup.sh`: `MODELS` (space-separated HF ids), `INSTALL_VLLM=0`, `VLLM_VERSION`.

### Installing Clotho Package
The `clotho` toolkit, provided as a Python package, includes code for dataset construction used in our study, hidden state extraction from Hugging Face–deployed open LLMs, and functions for computing both baseline and Clotho’s adequacy metrics based on the extracted hidden state vectors.

You can install the package using pip:

```bash
pip install -r requirements.txt
pip install -e . # install the clotho package with editable mode
```

The detailed package structure and descriptions are as follows:

```
.
|-- al
|   `-- sampling.py                       # Sampling strategies for reference set construction: random, exploration (diversity), exploitation (uncertainty), balanced
|-- dataset                               # Dataset of task-specific prompt templates and test inputs, and scripts to construct them
|-- inference.py                          # Inference pipeline for open LLMs
|-- metrics
|   |-- reference_based
|   |   `-- density_estimation.py         # Clotho's reference-based adequacy score model (based on GMM)
|   |-- logprobs.py                       # Token-log-probability–based baseline metrics implementation
|   |-- sa.py                             # MDSA, base GMM metrics implementation
|   `-- semantic_entropy.py               # Semantic entropy baseline metric implementation
|-- models
|   |-- gemma.py                          # Interface wrapper for Gemma model for hidden state extraction
|   |-- llama.py                          # Interface wrapper for LLaMA model for hidden state extraction
|   `-- mistral.py                        # Interface wrapper for Mistral model for hidden state extraction
`-- preprocessing
    `-- feature_reduction.py              # PCA dimensionality reduction for hidden state feature vectors
```

### Collecting LIHS and Inference Outputs from LLMs

To extract LIHS (**L**ast-token **I**nput **H**idden **S**tates) from LLMs and given task-specific test suite (dataset), run the following command from the root of the repository:

```bash
$ cd experiments/data_extraction
$ HF_TOKEN="your_huggingface_token"
$ python 1-1_extract_hidden_states.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name>
```

where `<task_name>` and `<dataset_name>` are defined in `clotho/exp_config.py` (e.g., `--task spell_check --dataset misspell_injected_wordnet`), and `<model_name>` is one of the supported models (`llama`, `gemma`, `mistral`).

You'll need to provide your own Hugging Face token as an environment variable. Additionally, you must first accept the terms of service for each model before accessing them via Hugging Face: [Llama](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Gemma](https://huggingface.co/google/gemma-2-9b-it), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

To collect the generated outputs from LLMs for the test suite:
```bash
$ cd experiments/data_extraction
$ python 2-1_generate_outputs_and_record_hidden_states.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name>
```

> Note: This process is computationally demanding (especially due to the decoding time for output token generation) and requires GPUs >=8GB VRAM. In our setup (NVIDIA RTX 4090), it took approximately 3-5 days for each model and task combination. We recommend to download the pre-generated outputs published in Zenodo (see [below](#scripts-for-exploratory-and-main-study-analyses)) instead of running the generation yourself.


### Running Clotho
To simulate iterations of Clotho's pre-generation adequacy modelling using the extracted LIHS, run the following command from the root of the repository:

```bash
$ cd experiments
$ python run_clotho_iterations.py --target_task <task_name> --target_llm <model_name> --refset_extension_methods <reference_sampling_method> --seeds <seed1> <seed2> ...
```

where `<reference_sampling_method>` is one of `['random', 'diversity_euclidean', 'uncertainty', 'balanced']`.

### [Optional] Compute Baseline Metrics
To compute the pre-generation baseline metrics (e.g., SLL, MDSA) based on the extracted LIHS, run the following commands:

```bash
$ cd experiments/data_extraction
$ python 1-2_extract_input_token_logprobs.py --model <model_name> # extracts token log-probabilities for all tasks for the specified model

$ cd ../ # go back to the `experiments/` directory
$ python run_baseline_OOD.py --target_llm <model_name> # computes MDSA and GMM_base scores for all tasks for the specified model
$ python run_baseline_SLL.py --target_llm <model_name> # computes SLL (sequence log likelihoods for input tokens) scores for all tasks for the specified model
```

To compute the post-generation baseline metrics (e.g., token entropy, semantic entropy, LOHS-Variance) based on the additional post-generation features, run the following command:

```bash
$ cd experiments/data_extraction
$ python 3-1_repeated_outputs_variance.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name> # Computes LOHS-Variance scores for the specified task and model
$ python 3-2_token_based_metrics.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name> # Computes token-based metrics for the specified task and model
$ python 3-3_semantic_entropy.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name> # Computes semantic entropy scores for the specified task and model
```

The calculated baseline metric scores will be saved in `experiments/data/results_{model_name}/{task_name}/precalculated_metrics`.

## Scripts for Exploratory and Main Study Analyses

The analysis scripts (Jupyter Notebooks) used in our study are available in the `RQ` directory.

<details>
  <summary>🧩 Click to download and locate required experiment result data for analyses</summary>

To actually run the notebooks for reproducing our analyses in the paper, please download the following two archives from our [Zenodo archive](https://zenodo.org/records/19693236) containing the generated LLM outputs and Clotho/baseline metric scores:

- data.tar.gz
  - Locate the compressed file in `experiments/` directory and extract it there: directories `experiments/data/results_llama`, `experiments/data/results_gemma`, `experiments/data/results_mistral`, and so on will be created.
- results_clotho_10seeds_fse2026.tar.gz
  - Locate the compressed file in `experiments/` directory and extract it there: directory `experiments/results_clotho` will be created.

</details>

- **`dataset_statistics.ipynb`**  
  Summarises dataset statistics (e.g., size, pass/fail rates per task and model).

- **`exploratory_study_1_task_specific.ipynb`**  
  Exploratory study (Sec. 2.1): task-specific separability of hidden states.

- **`exploratory_study_2_layer_selection.ipynb`**  
  Exploratory study (Sec. 2.2): identifying informative Transformer layers.

- **`RQ1-1_compare_before_gen_metrics_all_models.ipynb`**  
  RQ1-1: Compare pre-generation adequacy metrics across models (Tab. 3).

- **`RQ1-1_score_visualization.ipynb`**  
  RQ1-1: Visualise predicted scores vs. actual failure rates on a projected 2D plane (Fig. 4).

- **`RQ1-2_compare_sampling_methods.ipynb`**  
  RQ1-2: Compare sampling strategies (Fig. 5, Fig. 6).

- **`RQ2_ROC_AUC.ipynb`**  
  RQ2: Compute ROC-AUC for failure prediction and Mann-Whitney U Test (Tab. 4).

- **`RQ2_test_prioritization.ipynb`**  
  RQ2: Evaluate test prioritisation performance (Tab. 5).

- **`RQ3-1_loh_metrics_evaluation.ipynb`**  
  RQ3-1: Evaluate Clotho vs. post-generation metrics (Tab. 6).

- **`RQ3-2_loh_complementary_metrics.ipynb`**  
  RQ3-2: Analyse complementarity with post-generation uncertainty metrics (Fig. 7).

- **`RQ4_proprietary_model.ipynb`**  
  RQ4: Transfer scores from studied OLMs to GPT, Claude, and Gemini (Tab. 7, Tab. 8, Fig. 8).

- **`supplementary_RQ3_logistic_regression.ipynb`**  
  Supplementary: Additional analysis combining Clotho with a subset of post-generation metrics.


## Dataset of Task-specific LLM Prompts and Test Suites
The prompt templates, input data, and scripts used to synthesize (parts of) the datasets are available in the `clotho/datasets` directory.

For each task, `templates.py` defines the prompt templates with task-specific instructions and output format specifications, `output_parsers.py` implements the logic for parsing model outputs, and the input data files (e.g., `*.jsonl`) provide the test input instances used to generate LLM outputs.

```
.
|-- adding_odd_numbers
|   |-- integer_sequences_length_1_to_10_uniform.jsonl
|   |-- output_parsers.py
|   |-- templates.py
|   `-- test_generator.py    # Script to generate the random integer sequences
|-- github_typo_check
|   |-- github_typo_corpus_cleaned.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- json_repair
|   |-- synthetic_invalid_json    # Scripts to generate synthetic invalid JSON inputs
|   |-- invalid_json_dataset_2166.jsonl
|   |-- invalid_json_dataset_4397.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- model_name_extraction
|   |-- synthetic_abstract    # Scripts to generate synthetic abstracts
|   |-- ml_arxiv_papers_labelling    # Scripts to label ML arXiv papers with model names
|   |-- synthetic_abstracts_gpt4o_3600.jsonl
|   |-- ml_arxiv_papers_no_conflicting_labels.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- pos_detection
|   |-- cleaned_and_sampled_pos_tags.jsonl
|   |-- cleaned_and_sampled_pos_tags_trainset.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- spell_check
|   |-- misspell_injected_wordnet.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- syntactic_bug_detection
|   |-- syntactic_bug_injected.jsonl
|   |-- output_parsers.py
|   `-- templates.py
`-- topic_classification
    |-- ag_news_test.jsonl
    |-- output_parsers.py
    `-- templates.py
```

## 📖 Citation

If you use this artifact, please cite:

```latex
@misc{clotho2026artifact,
  title = {Artifact for "Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs"},
  doi   = {10.5281/zenodo.19693236},
}
```