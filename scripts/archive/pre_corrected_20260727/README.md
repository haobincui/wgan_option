# Pre-Corrected RQ Orchestration Archive

Archived on 2026-07-27 UTC.

These scripts reproduce earlier exploratory SVI/raw-vol, multi-seed, London
timezone, and RQ1/RQ2 launch workflows. They are retained for audit only.

They were superseded because they do not jointly enforce all corrected inputs:

- disjoint current/target half-open windows;
- CME Rule 19A TY option expiry;
- Black76 with frozen Treasury rates and exact prior TY futures;
- OTM positive-volume filtering and volume-weighted median IV;
- fold-train-only raw support masks;
- strict LP lineage and duplicate audit;
- common-random-number probabilistic evaluation;
- corrected RQ3 all-OOS conditional predictive-ability inference.

Use `scripts/rq123/start_corrected_pipeline_background.sh` for current runs.

The `rq3/` subdirectory contains the superseded all-surface and fast-window
news/quiet launchers. Their supporting Python modules remain importable under
`scripts/rq3/` only so historical outputs and regression tests can still be
audited; they are not active thesis entrypoints.
