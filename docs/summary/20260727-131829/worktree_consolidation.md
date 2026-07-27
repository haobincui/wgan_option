# Git Worktree Consolidation Record

## Operation

```text
operation_time_utc: 2026-07-27
surviving_worktree: /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone
surviving_branch: master
master_merge_head_before_this_record: fe7c72db577a4d0679ed32e57bbe74227ab68ae0
remote: https://github.com/haobincui/wgan_option.git
remote_push_performed: false
```

The previous RQ1, no-text continuation, RQ2 continuation, and London-time worktrees were
consolidated into local `master`. All tracked changes were committed before any worktree was
removed.

## Source Worktrees And Backup Commits

| Previous worktree | Branch | Preserved commit |
|---|---|---|
| `wgan_option` | `wgan_rq1` | `13d6aa4bb878411aaae49c24cbc8355fd71b03e3` |
| `wgan_option-rq1-no-text-continuation` | `wgan_rq1_no_text_continuation` | `26f3a23291eb78aa2019505ed61aa0ffcdd8d767` |
| `wgan_option-rq2-pair-continuation` | `wgan_rq2_pair_continuation` | `b0a6b174efefc3f9013a915ff84c736536a06e2e` |
| `wgan_option-rq123-london-timezone` | `wgan_rq123_london_timezone` | `ee61745018f33aed52b6249319c9e1096b8059e2` |

Backup commit subjects:

```text
13d6aa4 Add RQ1 results and scheduled-news evaluation workflows
26f3a23 Add controlled no-text continuation for RQ1
b0a6b17 Add RQ2 continuation representation workflow
ee61745 Add London-time RQ1-RQ3 recalculation workflow
```

## Merge Sequence

Local `master` was updated in this order:

1. Fast-forward from `82ba8b1` to the shared no-text continuation base `26f3a23`.
2. Merge the RQ1 worktree as `d0d3f09`.
3. Merge the RQ2 continuation worktree as `ad9f05b`.
4. Merge the London-time worktree as `fe7c72d`.

The RQ1 merge produced conflicts in:

```text
scripts/rq1_pair/rq1_pair_experiment.py
tests/test_scripts/test_rq1_pair_experiment.py
```

The London-time versions were used because they contain both the no-text continuation behavior
and the RQ1 results-pipeline additions. The final London merge used the London version for
overlapping conflict hunks while retaining files unique to the RQ1 and RQ2 branches.

Before deleting local topic branches, each source commit was verified as an ancestor of
`master`. The following local branches were then removed:

```text
wgan_rq1
wgan_rq1_no_text_continuation
wgan_rq2_pair_continuation
wgan_rq123_london_timezone
```

Remote branches were not deleted or modified.

## Experiment Artifact Preservation

Git-ignored outputs were not eligible for Git merge, so they were moved on the same filesystem
before the old worktrees were removed. No bulk copy was performed.

Archive root:

```text
outputs/archive/worktree_consolidation_20260727-131829/
```

It contains:

```text
wgan_option/
  outputs/
  logs/
  data_processed/
  main_tracked_data_processed_backup/

wgan_option-rq1-no-text-continuation/
  outputs/
  data_processed/
  data_raw/

wgan_option-rq2-pair-continuation/
  outputs/
  data_processed/
```

Aggregate archive measurements at consolidation time:

```text
apparent size: 720643472831 bytes
allocated size: 721945677824 bytes
```

Per-source values in `archive_manifest.csv` are measured independently and may double-count
hard-linked files. The aggregate measurement above counts hard links once within the archive
tree.

The canonical raw input was moved from the former primary worktree into:

```text
data/raw/
```

It is now a real directory rather than a symlink and measured `1290678471` apparent bytes at
consolidation time. Absolute symlinks inside archived continuation inputs were rewritten as
relative links. A final scan found no broken symlinks under the consolidation archive.

## Verification

The merged tree passed:

```text
git diff --check
conflict-marker scan
```

Targeted tests:

```bash
conda run -n py312 python -m unittest \
  tests.test_scripts.test_news_time \
  tests.test_scripts.test_rq1_pair_experiment \
  tests.test_scripts.test_rq2_pair_experiment \
  tests.test_scripts.test_rq3_scheduled_news
```

Result:

```text
Ran 30 tests in 11.047s
OK
```

Final worktree contract:

```text
git worktree count = 1
path               = /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone
branch             = master
old worktree paths = absent
```

At the time of consolidation, local `master` had not been pushed to `origin/master`.
