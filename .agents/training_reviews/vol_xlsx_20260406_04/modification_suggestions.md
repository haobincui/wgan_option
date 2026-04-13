# Modification Suggestions

## 1. Tune the scheduler before changing the GAN architecture

The `04` run suggests that scheduler behavior is now a higher-priority lever than architecture redesign.

Reason:

- stability improved dramatically
- best and final validation accuracy both got worse relative to `03`
- this points more directly to a schedule tradeoff than to a broken generator or critic design

So the next move should be to refine the scheduler, not to redesign the GAN core.

## 2. Focus first on `reduce_lr_patience`, `reduce_lr_factor`, and `reduce_lr_min_lr`

The current schedule is:

- `factor = 0.5`
- `patience = 8`
- `min_lr = 1e-5`

That schedule starts reducing at epoch `22` and reaches the minimum learning rate by epoch `64`.

Recommended next trials:

- increase `reduce_lr_patience` to delay the first drop and later repeated drops
- test a gentler `reduce_lr_factor`, so each reduction is less aggressive
- test a higher `reduce_lr_min_lr`, so the run does not flatten into a low-learning-rate regime too early

The goal is to preserve part of the stability gain from `04` while recovering some of the validation sharpness seen in `03`.

## 3. Run paired `scheduler on` / `scheduler off` experiments

Do not infer scheduler benefit from `04` alone.

Recommended evaluation policy:

- keep the same seed when possible
- keep the same training infrastructure and checkpoint logic
- run one pair with scheduler off
- run one pair with scheduler on
- compare best-checkpoint metrics first

This is the cleanest way to separate:

- infrastructure effects
- scheduler effects
- random-run variation

## 4. For thesis-facing evaluation, prioritize best-checkpoint comparison over final-checkpoint comparison

This run makes that rule especially important.

- best epoch: `27`, `val_recon = 0.037823`
- final epoch: `100`, `val_recon = 0.039700`

If thesis conclusions are drawn from the final checkpoint alone, the run will look worse than it needs to.

The more defensible reporting rule is:

- use `generator_best.pt` as the primary evaluation artifact
- treat `generator.pt` as a diagnostic end-of-run artifact

## 5. Keep the best-checkpoint workflow, but fix `best_checkpoint.json` portability

The best-checkpoint workflow is still clearly worth keeping.

It captured epoch `27`, which is not a regular periodic checkpoint and would otherwise have been lost.

However, the metadata still has a relocation-safe problem:

- archived folder: `outputs/vol_xlsx_20260406-04`
- artifact path in metadata: `outputs/vol_xlsx/checkpoints/...`

Recommended fix:

- store filenames relative to `models_path`
- or store both artifact filenames and the run-time `models_path`

This would make archived experiment folders more self-contained for later thesis audit and reuse.

## 6. Practical interpretation to preserve

This run should not be labeled a failed run.

The more accurate description is:

- it is a healthy run
- it is materially more stable than `03`
- it is also more conservative and weaker on validation accuracy
- its main value is that it reveals a scheduler tradeoff worth tuning next

That is a useful experimental result, even if it is not the new best-performing run.
