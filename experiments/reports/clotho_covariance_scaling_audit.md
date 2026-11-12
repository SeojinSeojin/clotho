# Covariance/scaling experiment: launch audit

Read AGENTS.md, latest PROGRESS.md and the component report, including its decision memo. Inspected original GMM settings, saved PCA-coordinate/score bundles, and sklearn EM initialization/M-step implementation.

Status at2026-11-12 session start: **not started**. The only persistent Python process was the OS unattended-upgrade helper; tmux held Codex. No covariance/scaling job or script, local result/log, or matching private-results-repository artifact was found. The latest progress/report names this as the next experiment. Older component jobs are complete and are not rerun.

Reuse250 original-coordinate bundles and the corresponding component labels/reference/test indices. The old reference-standardized PCA experiment changed directions as well as scaling, so it does not answer the present question. Here every transformation is invertible diagonal rescaling of the same archived coordinates.

Freeze the protocol before outcomes. A matrix-ridge EM implementation permits direct evaluation on transformed inputs; its isotropic special case must reproduce stock sklearn. Controlled initial responsibilities isolate ridge effects; a small native-initialization diagnostic tests optimization separately. Analytical model transport is a separate no-refitting branch. Both original archived anchors and identity float64 refits are retained and clearly distinguished.
