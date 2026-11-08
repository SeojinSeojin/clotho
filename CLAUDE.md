# CLAUDE.md

Read `AGENTS.md` first (goal, constraints, hard rules, session start, repo map), then `PROGRESS.md` for current state.

Before anything else: follow the hard rules in `AGENTS.md`, especially these, learned the hard way:
- Wait on background jobs by PID, never with `pgrep -f` in a loop (it matches itself).
- Standalone vLLM scripts need `VLLM_USE_FLASHINFER_SAMPLER=0` (no `nvcc` on these servers).
- One GPU job at a time.
- Another agent session may share this checkout: `git pull --rebase` before pushing, and never kill processes you did not start.
