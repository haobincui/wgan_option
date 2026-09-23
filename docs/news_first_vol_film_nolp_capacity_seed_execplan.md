# FiLM Generator + NoLP Critic Capacity/Seed Experiment ExecPlan

This living plan implements a new exact-TTM experiment without modifying prior
formal roots.  The immutable training unit is `(stage, capacity, seed,
tolerance)`, with real LP text supplied only to the FiLM Generator and the
Critic text feature forced to zero without changing its parameter shape.

## Protocol

- Development: 6 capacities × 3 seeds × 2 tolerances = 36 jobs, train before
  2023-07-01 and select on the common 5m Q3 panel.
- Refit: all 36 cells restart from the same seeded initialization and replay
  the frozen per-epoch G/D LR traces through 2023Q3.
- Evaluation: all refit checkpoints use the same common 5m Q4 panel.  The 5m
  training lane is primary; 30m is secondary.  Q4 is retrospective and
  exploratory because the calendar period has historical prediction evidence.

## Progress

- [x] Exact model modes, six capacity profiles, parameter counts, seeds, data
  windows and inferential families frozen.
- [x] Independent orchestrator, analysis, report, CLI and tests implemented.
- [x] Mandatory verification attempted; the repository has no `make format`
  target, while Ruff, py_compile, diff-check and 107 relevant tests pass.
- [x] Dual-A30 36-worker one-epoch prelaunch smoke completed: peak 6,164 MiB
  per GPU, peak host RAM 55.68%, 36/36 jobs completed and every G/D updated.
- [ ] New formal root prepared and full pipeline launched under one background
  supervisor.
- [ ] Terminal results and lineage independently validated.

## Decisions

- No release tag exists.  The new CLI and persisted experiment schema are
  branch-local; no compatibility shim is added.  Existing checkpoint readers
  remain backward compatible.
- The experiment uses real_text only.  Capacity is the sole model factor.
- All six capacities are refit and evaluated on Q4; Q4 cannot alter the Q3
  frozen point leader or one-SE candidate.
- Benchmark concurrency is 18 workers/GPU only if memory is below 20 GiB/GPU,
  host RAM is below 85%, and all jobs update G/D without non-finite values.
  Otherwise formal concurrency is frozen at 12 workers/GPU.

## Recovery

Every action validates dataset, config, code, job, recipe and checkpoint hashes
before writing.  A supervisor lock prevents duplicate pipelines.  Interrupted
jobs are resumable only from a hash-consistent registry; Q4 remains inaccessible
until all 36 refit jobs and the two-checkpoint-per-job allowlist are frozen.
