# Frozen next GPU pilot

The smallest executable next step is a GT-only detector health pilot, not a matched parser recovery claim. Full-horizon matched parser outputs do not exist; do not train a recovery comparison on the old prefix. Decisions are frozen in configs/cpu_gpu_pilot.json and full_horizon split manifests. No GPU job ran in this CPU task.

## Data and conditions

Initial dataset BGL, full raw-line60/20/20 chronological cuts, nonoverlapping20-line any-alert windows; purge2 crossings and retain the final short unit with explicit within-unit left padding for its DeepLog context. All46312 validation/46313 test units are retained; train10000 normal units chosen by lowest SHA256(unit ID). HDFS prepared as a GT control with complete contained blocks and10000 normal training blocks; no shared raw lines across splits. Training vocabulary only; explicit UNK0 and contextual mask token at vocab+1. No test labels select vocabulary, units, rankings or hyperparameters. Both datasets’ exact IDs/input hashes are saved.

After GT health, obtain GT/Drain/base/strong/real-AdaParser assignments on identical frozen units. Predicted prepare requires a complete JSON map original LineId→actual event assignment; missing rows fail with the first missing ID. No GT fill. Template strings for semantic work need a matching event→text table; original IDs and final-vs-temporal assignment provenance must be retained. Parser seed candidates/overlap and adaptation need explicit new control, because current whole-prefix final outputs are transductive relative to detector splits.

## Detectors and frozen evaluation

First sequence: small DeepLog-style embedding32/LSTM64×2, context10, batch256, Adam0.001,10 epochs. This uses trainable event embeddings and mean next-event negative log probability per unit; it is a pilot variant, not exact official DeepLog reproduction. First contextual: masked Transformer128×2,4 heads, context20,15% mask, same batch/lr/epochs; evaluate mean masked-token loss over every position and aggregate within units. This lacks LogBERT’s hypersphere loss and is explicitly not an exact LogBERT replication. The official pinned LogBERT adapter remains a separate implementation task. First faithful semantic LogRobust awaits a pinned fixed word-vector source and attention/BiLSTM adapter; no substitute is silently labelled LogRobust.

Seeds0–4; initially seed0 GT only. Choose epoch by mean normal-validation loss and threshold by validation F1 (ties higher threshold); never test performance. Freeze precision/recall/F1/AP/AUROC/FPR, score direction and unit aggregation. Persist unit scores/labels/IDs, model checkpoint, config hash and validation history. AP/ROC require both test classes. Deterministic seeds and CUBLAS workspace are set. DeepLog short units use zero-left-padded within-unit context and one target, documented as a pilot convention, not an upstream protocol.

## Commands and execution order

CPU inputs are already prepared. After restoring data/cpu_research/ and results_phase3/cpu_research/ from RESULTS_REPO into experiments/, run:

```bash
.venv/bin/python experiments/log_parsing_gpu_pilot.py smoke
.venv/bin/python experiments/log_parsing_gpu_pilot.py train --system BGL --condition GT --detector deeplog --seed 0 --device cuda
```

The train command is the first GPU command. Use the normal GPU-enabled .venv next session; this session’s .venv-cpu has torch+cpu and cannot run CUDA. Install torch with a wheel matching the server driver plus numpy/scipy/scikit-learn if needed; no pretrained model/tokenizer is required. The command never downloads a model. No GPU memory figure was measured; models are small, but peak GPU allocation must be recorded during the first run instead of quoting an unverified GB estimate.

Then run BGL contextual seed0 and HDFS GT controls; inspect validation/data health without altering test splits. Continue seeds1–4 only if the fixed health gate passes. BGL severe temporal OOV drift must be reported by normal/anomaly strata, not used to choose a friendlier test cut. A template-semantic detector may be necessary to distinguish drift from anomalies, but that remains a hypothesis.

Before matched inference: verify complete assignment coverage and original-ID joins, candidate provenance, common unit/label hashes, no cross-split members, normal-only training, train-only vocabulary/IDF and input/target/mask dimensions. CPU smoke passed preprocessing and tiny randomly initialized CPU LSTM/Transformer forward shapes, with no training or smoke performance reported. GPU training/backpropagation and CUDA kernels remain untested.

Resource fallback: lower batch256→64 only for memory failure (record deviation), keep data/splits/architecture/epochs fixed; run seeds serially. Do not overlap GPU parser/model jobs. First GT health uses no LLM inference; defer expensive parser expansion until detector health is useful. Never expand reviews just because representation distances are nonzero.

Expansion requires valid matched outputs, both held-out classes, useful GT health under frozen metrics, and an interpretable GT–parser gap with paired time-block (BGL)/session (HDFS) uncertainty. Compare always-anomaly and UNK-only controls before interpreting sequence gains. Then evaluate stored-style review policies at0/5/10/20/50, keep template-only and membership interventions separate, and charge training/candidate/source labels. Retrained and frozen-detector correction experiments must remain distinct.

## Integration

Available HDFS real AdaParser is exactly GT; BGL real AdaParser changes partitions through two splitGT types and small lexical distortions. Existing frequency/D corrections recover the same stored BGL test lexical features by20 reviews, without a valid anomaly-recovery test. Label-aware U loses substantial line recovery at equal target effort when90 training labels are charged; zero-target U retains some small-budget group recovery but source acquisition is costly. These findings support a feasibility/cost contribution, not a downstream improvement claim. The highest-priority unknown is whether a useful GT detector and an actual parser-induced anomaly-performance gap coexist on the same valid fully covered units.
