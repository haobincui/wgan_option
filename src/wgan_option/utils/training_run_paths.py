"""Backward-compatible shim for shared training/generate-result path helpers."""

from utils.training_paths import (
    GENERATE_RESOLVED_CONFIG_NAME,
    TRAINING_RESOLVED_CONFIG_NAME,
    TRAINING_RUN_LOG_NAME,
    checkpoint_filename,
    checkpoint_named_dir,
    generate_result_config_path,
    generate_result_dir,
    infer_run_dir_from_checkpoint,
    infer_training_output_root,
    prepare_timestamped_training_config,
    resolve_existing_run_dir,
    training_run_config_path,
    training_run_log_path,
    training_run_timestamp,
)

__all__ = [
    "TRAINING_RESOLVED_CONFIG_NAME",
    "GENERATE_RESOLVED_CONFIG_NAME",
    "TRAINING_RUN_LOG_NAME",
    "checkpoint_filename",
    "checkpoint_named_dir",
    "infer_training_output_root",
    "training_run_timestamp",
    "training_run_config_path",
    "training_run_log_path",
    "prepare_timestamped_training_config",
    "generate_result_dir",
    "generate_result_config_path",
    "infer_run_dir_from_checkpoint",
    "resolve_existing_run_dir",
]
