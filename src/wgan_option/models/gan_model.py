import math
import os
from dataclasses import asdict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from wgan_option.config import (
    FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
    Config,
    label_reliability_lineage,
    load_refit_recipe,
    normalize_refit_mode,
)
from wgan_option.models.common import (
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    critic_conditioning_fingerprint,
    critic_normalization_fingerprint,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GAUSSIAN_GENERATOR_NOISE_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    generator_current_input_fingerprint,
    generator_conditioning_fingerprint,
    generator_noise_fingerprint,
    generator_noise_tensor,
    normalize_generator_current_input_mode,
    normalize_generator_conditioning_mode,
    normalize_generator_noise_mode,
    normalize_critic_conditioning_mode,
    residual_output_fingerprint,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator
from wgan_option.utils.inference_helpers import infer_vol_surface_mc, load_vol_generator
from wgan_option.utils.news_first_experiment_core import (
    NO_FULL_TRAINING_STATE,
    RESUME_DYNAMIC_FULL_TRAINING_STATE,
    RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    SAVE_DYNAMIC_FULL_TRAINING_STATE,
    FullTrainingStateContract,
    load_full_training_state,
    load_full_training_state_contract,
    normalize_full_training_state_mode,
    save_full_training_state,
)
from wgan_option.utils.pure_cnn_film_graft import (
    apply_pure_cnn_graft_state,
    read_pure_cnn_graft_metadata,
)
from wgan_option.utils.reproducibility import seed_everything
from wgan_option.utils.training_artifacts import (
    write_best_checkpoint,
    write_metrics_csv,
    write_metrics_json,
)
from wgan_option.utils.vol_forecast_metrics import (
    mean_abs_error,
    resolve_monitor_metric,
    summarize_baseline_aware_metrics,
)
from wgan_option.utils.visualization import plot_training_curves
from wgan_option.utils.weighted_training import (
    apply_surface_mask,
    masked_mean_per_sample,
    reconstruction_training_weights,
    stable_noise_for_keys,
    stable_tensor_row_keys,
    training_weighted_mean,
    unpack_vol_training_batch,
    validated_label_reliability_weights,
    validated_sample_weights,
    validated_surface_mask,
    weighted_mean,
)


class WGAN_GP:
    """
    Conditional WGAN-GP for volatility-surface forecasting:
    (surface_t, text_t) -> surface_{t+h}
    """

    def __init__(
        self,
        config: Config,
        strike_grid: np.ndarray,
        maturity_grid_days: np.ndarray,
        embedding_dim: int,
    ):
        # Model initialization itself consumes RNG state, so the seed must be
        # established before constructing any module or optimizer.
        seed_everything(config.seed)
        self.config = config
        self.device = torch.device(
            "cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu"
        )
        self.strike_grid = torch.tensor(
            strike_grid, dtype=torch.float32, device=self.device
        ).clamp_min(1e-4)
        self.maturity_grid_days = torch.tensor(
            maturity_grid_days, dtype=torch.float32, device=self.device
        )
        self.tau_years = (self.maturity_grid_days / 365.0).clamp_min(1.0 / 365.0)
        self.embedding_dim = embedding_dim

        surface_height = int(len(maturity_grid_days))
        surface_width = int(len(strike_grid))
        self.G = Generator(
            channels=config.channels,
            embedding_dim=embedding_dim,
            noise_dim=config.noise_dim,
            surface_height=surface_height,
            surface_width=surface_width,
            base_channels=config.gen_base_channels,
            res_blocks=config.gen_res_blocks,
            text_hidden_dim=config.gen_text_hidden_dim,
            text_out_dim=config.gen_text_out_dim,
            hidden_dim=config.gen_hidden_dim,
            residual_output_mode=config.residual_output_mode,
            generator_noise_mode=config.generator_noise_mode,
            generator_current_input_mode=config.generator_current_input_mode,
            generator_conditioning_mode=config.generator_conditioning_mode,
            strike_grid=strike_grid,
            maturity_grid_days=maturity_grid_days,
            crossattn_heads=int(getattr(config, "gen_crossattn_heads", 4)),
            crossattn_text_tokens=int(getattr(config, "gen_crossattn_text_tokens", 4)),
            crossattn_dim=int(getattr(config, "gen_crossattn_dim", 128)),
            transformer_model_dim=int(getattr(config, "gen_transformer_model_dim", 96)),
            transformer_layers=int(getattr(config, "gen_transformer_layers", 4)),
            transformer_heads=int(getattr(config, "gen_transformer_heads", 8)),
            transformer_ffn_dim=int(getattr(config, "gen_transformer_ffn_dim", 384)),
            transformer_dropout=float(getattr(config, "gen_transformer_dropout", 0.1)),
            style_dim=int(getattr(config, "gen_style_dim", 128)),
            style_demodulate=bool(getattr(config, "gen_style_demodulate", True)),
        ).to(self.device)
        self.D = Discriminator(
            channels=config.channels,
            embedding_dim=embedding_dim,
            surface_height=surface_height,
            surface_width=surface_width,
            base_channels=config.disc_base_channels,
            res_blocks=config.disc_res_blocks,
            text_hidden_dim=config.disc_text_hidden_dim,
            hidden_dim=config.disc_hidden_dim,
            critic_normalization_mode=config.critic_normalization_mode,
            critic_conditioning_mode=config.critic_conditioning_mode,
        ).to(self.device)

        g_lr = (
            config.generator_learning_rate
            if config.generator_learning_rate > 0
            else config.learning_rate
        )
        d_lr = (
            config.discriminator_learning_rate
            if config.discriminator_learning_rate > 0
            else config.learning_rate
        )
        self.generator_optimizer_profile = str(
            getattr(config, "generator_optimizer_profile", "uniform_v1")
        )
        if self.generator_optimizer_profile == "uniform_v1":
            # Preserve the historical single-param-group Adam construction.
            self.g_optimizer = Adam(
                self.G.parameters(), lr=g_lr, betas=(config.beta_1, config.beta_2)
            )
        else:
            self.g_optimizer = Adam(
                self._split_generator_optimizer_groups(g_lr),
                betas=(config.beta_1, config.beta_2),
            )
        self.d_optimizer = Adam(
            self.D.parameters(), lr=d_lr, betas=(config.beta_1, config.beta_2)
        )
        self._initial_generator_learning_rate = self._optimizer_lr(self.g_optimizer)
        self._initial_generator_group_learning_rates = (
            self._generator_optimizer_group_learning_rates()
        )
        self._initial_discriminator_learning_rate = self._optimizer_lr(self.d_optimizer)

        self.critic_iter = config.discriminator_iter
        self.lambda_gp = config.lambda_gp
        self.lambda_recon = config.lambda_recon
        self.lambda_calendar = config.lambda_calendar
        self.lambda_butterfly = config.lambda_butterfly
        self.lambda_smooth = config.lambda_smooth
        self.lambda_delta_shrink = float(getattr(config, "lambda_delta_shrink", 0.0))
        self.use_calendar_constraint = config.use_calendar_constraint
        self.use_butterfly_constraint = config.use_butterfly_constraint
        self.use_smooth_constraint = config.use_smooth_constraint
        self.num_epochs = config.num_epochs
        self.best_checkpoint_metric = (
            str(getattr(config, "best_checkpoint_metric", "val_recon")).strip()
            or "val_recon"
        )
        self.baseline_penalty_weight = float(
            getattr(config, "baseline_penalty_weight", 2.0)
        )
        self.generator_noise_mode = normalize_generator_noise_mode(
            getattr(config, "generator_noise_mode", GAUSSIAN_GENERATOR_NOISE_MODE)
        )
        self.generator_current_input_mode = normalize_generator_current_input_mode(
            getattr(
                config,
                "generator_current_input_mode",
                FULL_CURRENT_GENERATOR_INPUT_MODE,
            )
        )
        self.generator_conditioning_mode = normalize_generator_conditioning_mode(
            getattr(
                config,
                "generator_conditioning_mode",
                BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
            )
        )
        self.critic_conditioning_mode = normalize_critic_conditioning_mode(
            getattr(
                config,
                "critic_conditioning_mode",
                LP_CONCAT_CRITIC_CONDITIONING_MODE,
            )
        )
        self.news_first_refit_mode = normalize_refit_mode(
            getattr(config, "news_first_refit_mode", "none")
        )
        self._refit_recipe: dict[str, object] | None = None
        if self.news_first_refit_mode == FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            self._refit_recipe = load_refit_recipe(
                str(config.news_first_refit_recipe_path),
                str(config.news_first_refit_recipe_sha256),
            )
            recipe_epochs = int(self._refit_recipe["num_epochs"])
            if int(config.num_epochs) != recipe_epochs:
                raise ValueError(
                    "Config num_epochs must equal the frozen refit recipe: "
                    f"{config.num_epochs} != {recipe_epochs}"
                )

        self.news_first_full_training_state_mode = normalize_full_training_state_mode(
            getattr(config, "news_first_full_training_state_mode", "none")
        )
        self._full_training_state_contract: FullTrainingStateContract | None = None
        self._restored_full_training_state: dict[str, object] | None = None
        self._saved_full_training_state_sha256 = ""
        if self.news_first_full_training_state_mode != NO_FULL_TRAINING_STATE:
            self._full_training_state_contract = load_full_training_state_contract(
                str(config.news_first_full_training_state_contract_path),
                str(config.news_first_full_training_state_contract_sha256),
                expected_mode=self.news_first_full_training_state_mode,
            )
        self.news_first_graft_state_path = str(
            getattr(config, "news_first_graft_state_path", "") or ""
        )
        self.news_first_graft_state_sha256 = str(
            getattr(config, "news_first_graft_state_sha256", "") or ""
        )
        self._loaded_graft_state: dict[str, object] | None = None

        self.model_path = config.models_path
        self.metrics_path = config.metrics_path

    def _set_seed(self, seed: int):
        """Backward-compatible seed hook for external callers."""

        seed_everything(seed)

    @staticmethod
    def _generator_parameter_group(name: str) -> str:
        if name.startswith("text_encoder."):
            return "text_encoder"
        if name.startswith(
            (
                "encoder_film_layers.",
                "bottleneck_film_layer.",
                "decoder_film_layers.",
            )
        ):
            return "film_projection"
        return "backbone"

    def _split_generator_optimizer_groups(
        self, backbone_learning_rate: float
    ) -> list[dict[str, object]]:
        if self.generator_optimizer_profile == "film_unet_split_lr_v1":
            group_names = ("backbone", "text_encoder", "film_projection")
            parameter_roles = {
                name: self._generator_parameter_group(name)
                for name, _ in self.G.named_parameters()
            }
            learning_rates = {
                "backbone": float(backbone_learning_rate),
                "text_encoder": float(self.config.generator_text_learning_rate),
                "film_projection": float(self.config.generator_film_learning_rate),
            }
        elif self.generator_optimizer_profile == "conditioning_split_lr_v2":
            group_names = ("backbone", "text_encoder", "conditioning_module")
            parameter_roles = self.G.parameter_roles()
            conditioning_learning_rate = float(
                getattr(self.config, "generator_conditioning_learning_rate", 0.0)
            )
            if conditioning_learning_rate <= 0.0:
                conditioning_learning_rate = float(
                    getattr(self.config, "generator_film_learning_rate", 0.0)
                )
            if conditioning_learning_rate <= 0.0:
                conditioning_learning_rate = float(backbone_learning_rate)
            learning_rates = {
                "backbone": float(backbone_learning_rate),
                "text_encoder": float(self.config.generator_text_learning_rate),
                "conditioning_module": conditioning_learning_rate,
            }
        else:
            raise ValueError(
                "Unsupported split Generator optimizer profile: "
                f"{self.generator_optimizer_profile!r}"
            )
        grouped: dict[str, list[torch.nn.Parameter]] = {
            name: [] for name in group_names
        }
        parameter_ids: list[int] = []
        for name, parameter in self.G.named_parameters():
            try:
                role = parameter_roles[name]
                grouped[role].append(parameter)
            except KeyError as exc:
                raise RuntimeError(
                    f"Generator parameter has no valid optimizer role: {name}"
                ) from exc
            parameter_ids.append(id(parameter))
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError(
                "Generator optimizer parameter groups contain duplicates"
            )
        expected_ids = {id(parameter) for parameter in self.G.parameters()}
        if set(parameter_ids) != expected_ids:
            raise RuntimeError("Generator optimizer parameter groups omit parameters")
        empty_groups = [name for name, parameters in grouped.items() if not parameters]
        if empty_groups:
            raise ValueError(
                f"{self.generator_optimizer_profile} requires non-empty Generator "
                f"groups: {empty_groups}"
            )
        return [
            {
                "params": grouped[name],
                "lr": learning_rates[name],
                "group_name": name,
            }
            for name in group_names
        ]

    def _generator_optimizer_group_learning_rates(self) -> dict[str, float]:
        if self.generator_optimizer_profile == "uniform_v1":
            return {}
        result: dict[str, float] = {}
        for group in self.g_optimizer.param_groups:
            name = str(group.get("group_name", ""))
            if not name or name in result:
                raise RuntimeError("Generator optimizer group names are invalid")
            result[name] = float(group["lr"])
        expected = (
            {"backbone", "text_encoder", "film_projection"}
            if self.generator_optimizer_profile == "film_unet_split_lr_v1"
            else {"backbone", "text_encoder", "conditioning_module"}
        )
        if set(result) != expected:
            raise RuntimeError(
                "Generator optimizer groups differ from the split-LR contract"
            )
        return result

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _generator_noise(
        self,
        reference: torch.Tensor,
        *,
        preserve_training_rng: bool,
    ) -> torch.Tensor:
        """Return an explicit latent batch for every WGAN training path."""

        return generator_noise_tensor(
            reference,
            batch_size=int(reference.shape[0]),
            noise_dim=int(self.config.noise_dim),
            mode=self.generator_noise_mode,
            preserve_gaussian_rng_progression=bool(preserve_training_rng),
        )

    def _generator_noise_lineage(self) -> dict[str, str]:
        mode = normalize_generator_noise_mode(self.generator_noise_mode)
        return {
            "generator_noise_mode": mode,
            "generator_noise_fingerprint": generator_noise_fingerprint(
                mode,
                int(self.config.noise_dim),
            ),
        }

    def _generator_current_input_lineage(self) -> dict[str, str]:
        mode = normalize_generator_current_input_mode(self.generator_current_input_mode)
        return {
            "generator_current_input_mode": mode,
            "generator_current_input_fingerprint": generator_current_input_fingerprint(
                mode
            ),
        }

    def _conditioning_lineage(self) -> dict[str, str]:
        generator_mode = normalize_generator_conditioning_mode(
            self.generator_conditioning_mode
        )
        critic_mode = normalize_critic_conditioning_mode(self.critic_conditioning_mode)
        return {
            "generator_conditioning_mode": generator_mode,
            "generator_conditioning_fingerprint": generator_conditioning_fingerprint(
                generator_mode
            ),
            "critic_conditioning_mode": critic_mode,
            "critic_conditioning_fingerprint": critic_conditioning_fingerprint(
                critic_mode
            ),
        }

    def _refit_lineage(self) -> dict[str, object]:
        mode = normalize_refit_mode(self.news_first_refit_mode)
        payload: dict[str, object] = {
            "news_first_refit_mode": mode,
            "news_first_refit_recipe_path": str(
                getattr(self.config, "news_first_refit_recipe_path", "") or ""
            ),
            "news_first_refit_recipe_sha256": str(
                getattr(self.config, "news_first_refit_recipe_sha256", "") or ""
            ),
        }
        if mode == FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            if self._refit_recipe is None:
                raise RuntimeError("Frozen refit recipe was not loaded")
            payload.update(
                {
                    "refit_recipe_schema_version": int(
                        self._refit_recipe["schema_version"]
                    ),
                    "refit_num_epochs": int(self._refit_recipe["num_epochs"]),
                    "refit_generator_lr_trace": [
                        {"epoch": int(row["epoch"]), "lr": float(row["g_lr"])}
                        for row in getattr(self, "_metrics_rows", [])
                        if int(row.get("epoch", 0)) >= 1 and "g_lr" in row
                    ],
                    "refit_discriminator_lr_trace": [
                        {"epoch": int(row["epoch"]), "lr": float(row["d_lr"])}
                        for row in getattr(self, "_metrics_rows", [])
                        if int(row.get("epoch", 0)) >= 1 and "d_lr" in row
                    ],
                }
            )
        return payload

    def _full_training_state_lineage(self) -> dict[str, object]:
        contract = self._full_training_state_contract
        payload: dict[str, object] = {
            "news_first_full_training_state_mode": (
                self.news_first_full_training_state_mode
            ),
            "news_first_full_training_state_contract_path": str(
                getattr(
                    self.config,
                    "news_first_full_training_state_contract_path",
                    "",
                )
                or ""
            ),
            "news_first_full_training_state_contract_sha256": str(
                getattr(
                    self.config,
                    "news_first_full_training_state_contract_sha256",
                    "",
                )
                or ""
            ),
            "full_training_state_output_sha256": (
                self._saved_full_training_state_sha256
            ),
        }
        if contract is not None:
            payload.update(
                {
                    "full_training_state_output_path": (
                        str(contract.output_path) if contract.output_path else ""
                    ),
                    "parent_full_training_state_path": (
                        str(contract.input_path) if contract.input_path else ""
                    ),
                    "parent_full_training_state_sha256": contract.input_sha256,
                }
            )
        return payload

    def _graft_state_lineage(self) -> dict[str, object]:
        return {
            "news_first_graft_state_path": self.news_first_graft_state_path,
            "news_first_graft_state_sha256": self.news_first_graft_state_sha256,
            "graft_state_loaded": self._loaded_graft_state is not None,
            "graft_target_generator_mode": (
                ""
                if self._loaded_graft_state is None
                else str(self._loaded_graft_state["target_generator_mode"])
            ),
            "graft_optimizer_state_restored": False,
            "graft_scheduler_state_restored": False,
        }

    def _restore_graft_state_if_configured(self, *, train_loader) -> None:
        if not self.news_first_graft_state_path:
            return
        if self.g_optimizer.state or self.d_optimizer.state:
            raise RuntimeError("Graft initialization requires fresh Adam state")
        artifact_path = str(self.news_first_graft_state_path)
        expected_sha256 = str(self.news_first_graft_state_sha256)
        metadata = read_pure_cnn_graft_metadata(
            artifact_path=artifact_path,
            expected_artifact_sha256=expected_sha256,
            map_location="cpu",
        )
        parent_lineage = metadata.get("parent_lineage")
        graft_lineage = metadata.get("graft_lineage")
        if not isinstance(parent_lineage, dict) or not isinstance(graft_lineage, dict):
            raise ValueError("Graft artifact lineage is incomplete")
        self._loaded_graft_state = apply_pure_cnn_graft_state(
            artifact_path=artifact_path,
            expected_artifact_sha256=expected_sha256,
            expected_parent_lineage=parent_lineage,
            expected_graft_lineage=graft_lineage,
            target_generator=self.G,
            target_discriminator=self.D,
            loader_generator=self._loader_generator(train_loader),
            restore_rng=True,
            map_location=self.device,
        )
        if self.g_optimizer.state or self.d_optimizer.state:
            raise RuntimeError("Graft loading unexpectedly mutated Adam state")

    @staticmethod
    def _loader_generator(train_loader) -> torch.Generator:
        loader_generator = getattr(train_loader, "generator", None)
        if not isinstance(loader_generator, torch.Generator):
            raise ValueError(
                "news_first_wgan_full_training_state_v1 requires the training "
                "DataLoader to expose its seeded torch.Generator"
            )
        return loader_generator

    def _restore_full_training_state_if_configured(
        self,
        *,
        train_loader,
        generator_scheduler: ReduceLROnPlateau | None,
        discriminator_scheduler: ReduceLROnPlateau | None,
    ) -> None:
        mode = self.news_first_full_training_state_mode
        if mode not in {
            RESUME_DYNAMIC_FULL_TRAINING_STATE,
            RESUME_FROZEN_LR_FULL_TRAINING_STATE,
        }:
            return
        contract = self._full_training_state_contract
        if (
            contract is None
            or contract.input_path is None
            or contract.expected_input_lineage is None
        ):
            raise RuntimeError("Full-state resume contract is incomplete")
        self._restored_full_training_state = load_full_training_state(
            contract.input_path,
            contract.input_sha256,
            generator=self.G,
            discriminator=self.D,
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            loader_generator=self._loader_generator(train_loader),
            expected_lineage=contract.expected_input_lineage,
            restore_schedulers=mode == RESUME_DYNAMIC_FULL_TRAINING_STATE,
            map_location=self.device,
        )
        self._initial_generator_learning_rate = self._optimizer_lr(self.g_optimizer)
        self._initial_generator_group_learning_rates = (
            self._generator_optimizer_group_learning_rates()
        )
        self._initial_discriminator_learning_rate = self._optimizer_lr(self.d_optimizer)

    def _save_selected_full_training_state(
        self,
        *,
        train_loader,
        completed_epoch: int,
        generator_scheduler: ReduceLROnPlateau | None,
        discriminator_scheduler: ReduceLROnPlateau | None,
    ) -> None:
        if self.news_first_full_training_state_mode not in {
            SAVE_DYNAMIC_FULL_TRAINING_STATE,
            RESUME_DYNAMIC_FULL_TRAINING_STATE,
        }:
            return
        contract = self._full_training_state_contract
        if contract is None or contract.output_path is None:
            raise RuntimeError("Dynamic full-state save contract has no output path")
        self._saved_full_training_state_sha256 = save_full_training_state(
            contract.output_path,
            generator=self.G,
            discriminator=self.D,
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            loader_generator=self._loader_generator(train_loader),
            completed_epoch=completed_epoch,
            lineage=contract.output_lineage,
            contract_sha256=contract.contract_sha256,
        )

    @staticmethod
    def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def _apply_refit_learning_rates(self, epoch: int) -> None:
        if self.news_first_refit_mode != FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            return
        if self._refit_recipe is None:
            raise RuntimeError("Frozen refit recipe was not loaded")
        index = int(epoch) - 1
        generator_trace = self._refit_recipe["generator_lr_trace"]
        discriminator_trace = self._refit_recipe["discriminator_lr_trace"]
        if not isinstance(generator_trace, list) or not isinstance(
            discriminator_trace, list
        ):
            raise RuntimeError("Frozen refit LR traces are malformed")
        self._set_generator_optimizer_lr(float(generator_trace[index]["lr"]))
        self._set_optimizer_lr(
            self.d_optimizer,
            float(discriminator_trace[index]["lr"]),
        )

    def _validate_completed_refit_trace(self) -> None:
        if self.news_first_refit_mode != FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            return
        if self._refit_recipe is None:
            raise RuntimeError("Frozen refit recipe was not loaded")
        actual_generator = [
            {"epoch": int(row["epoch"]), "lr": float(row["g_lr"])}
            for row in self._metrics_rows
            if int(row.get("epoch", 0)) >= 1
        ]
        actual_discriminator = [
            {"epoch": int(row["epoch"]), "lr": float(row["d_lr"])}
            for row in self._metrics_rows
            if int(row.get("epoch", 0)) >= 1
        ]
        if actual_generator != self._refit_recipe["generator_lr_trace"]:
            raise RuntimeError(
                "Generator LR replay drifted from the frozen refit recipe"
            )
        if actual_discriminator != self._refit_recipe["discriminator_lr_trace"]:
            raise RuntimeError(
                "Discriminator LR replay drifted from the frozen refit recipe"
            )

    def _grid_model_lineage(self) -> dict[str, object]:
        mode = str(self.config.critic_normalization_mode)
        return {
            "critic_normalization_mode": mode,
            "critic_normalization_fingerprint": critic_normalization_fingerprint(mode),
            "surface_shape": [
                int(self.maturity_grid_days.numel()),
                int(self.strike_grid.numel()),
            ],
            "strike_grid": [float(value) for value in self.strike_grid.tolist()],
            "maturity_days_grid": [
                int(round(float(value))) for value in self.maturity_grid_days.tolist()
            ],
            "surface_grid_profile": str(
                self.config.news_first_surface_grid_profile or ""
            ),
            "surface_grid_sha256": str(
                self.config.news_first_surface_grid_sha256 or ""
            ),
            "architecture_profile_sha256": str(
                self.config.news_first_architecture_profile_sha256 or ""
            ),
            "model_contract_sha256": str(
                self.config.news_first_model_contract_sha256 or ""
            ),
        }

    def _label_reliability_lineage(self) -> dict[str, object]:
        """Return the hash-bound train-label contract for persisted artifacts."""

        return label_reliability_lineage(self.config)

    def _forward_generator(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        noise: torch.Tensor,
        current_support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call the production Generator while retaining simple test doubles."""

        if (
            self.generator_current_input_mode
            == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        ):
            return self.G(
                current_surface,
                text_embedding,
                noise=noise,
                current_support_mask=current_support_mask,
            )
        if hasattr(self.G, "noise_dim"):
            return self.G(current_surface, text_embedding, noise=noise)
        return self.G(current_surface, text_embedding)

    @staticmethod
    def _optimizer_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(learning_rate)

    def _set_generator_optimizer_lr(self, backbone_learning_rate: float) -> None:
        if self.generator_optimizer_profile == "uniform_v1":
            self._set_optimizer_lr(self.g_optimizer, backbone_learning_rate)
            return
        initial = self._initial_generator_group_learning_rates
        backbone_initial = float(initial["backbone"])
        if backbone_initial <= 0.0:
            raise RuntimeError("Split Generator backbone LR must be positive")
        scale = float(backbone_learning_rate) / backbone_initial
        for group in self.g_optimizer.param_groups:
            name = str(group["group_name"])
            group["lr"] = float(initial[name]) * scale

    def _generator_learning_rate_metrics(self) -> dict[str, float]:
        return {
            f"g_lr_{name}": learning_rate
            for name, learning_rate in (
                self._generator_optimizer_group_learning_rates().items()
            )
        }

    def _lr_warmup_epochs(self) -> int:
        return max(0, int(getattr(self.config, "lr_warmup_epochs", 0)))

    def _lr_warmup_start_factor(self) -> float:
        return float(getattr(self.config, "lr_warmup_start_factor", 0.1))

    @staticmethod
    def _linear_warmup_learning_rate(
        *,
        target_learning_rate: float,
        start_factor: float,
        epoch: int,
        warmup_epochs: int,
    ) -> float:
        """Return an inclusive epoch-1-to-N linear warmup learning rate."""

        if warmup_epochs <= 0 or epoch > warmup_epochs:
            return float(target_learning_rate)
        if epoch <= 0:
            epoch = 1
        if warmup_epochs == 1:
            return float(target_learning_rate)
        progress = float(epoch - 1) / float(warmup_epochs - 1)
        factor = float(start_factor) + progress * (1.0 - float(start_factor))
        return float(target_learning_rate) * factor

    def _lr_warmup_active_for_epoch(self, epoch: int) -> bool:
        warmup_epochs = self._lr_warmup_epochs()
        return warmup_epochs > 0 and int(epoch) <= warmup_epochs

    def _apply_learning_rate_warmup(self, epoch: int) -> None:
        if not self._lr_warmup_active_for_epoch(epoch):
            return
        warmup_epochs = self._lr_warmup_epochs()
        start_factor = self._lr_warmup_start_factor()
        discriminator_lr = self._linear_warmup_learning_rate(
            target_learning_rate=self._initial_discriminator_learning_rate,
            start_factor=start_factor,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
        )
        if self.generator_optimizer_profile == "uniform_v1":
            generator_lr = self._linear_warmup_learning_rate(
                target_learning_rate=self._initial_generator_learning_rate,
                start_factor=start_factor,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
            )
            self._set_optimizer_lr(self.g_optimizer, generator_lr)
        else:
            for group in self.g_optimizer.param_groups:
                name = str(group["group_name"])
                group["lr"] = self._linear_warmup_learning_rate(
                    target_learning_rate=(
                        self._initial_generator_group_learning_rates[name]
                    ),
                    start_factor=start_factor,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                )
        self._set_optimizer_lr(self.d_optimizer, discriminator_lr)

    def _initialize_learning_rate_warmup(self, logger) -> None:
        warmup_epochs = self._lr_warmup_epochs()
        if warmup_epochs <= 0:
            return
        self._apply_learning_rate_warmup(1)
        logger.info(
            "Linear LR warmup active for epochs 1-%d: G %.6g -> %.6g, "
            "D %.6g -> %.6g. The post-warmup scheduler starts at epoch %d.",
            warmup_epochs,
            self._optimizer_lr(self.g_optimizer),
            self._initial_generator_learning_rate,
            self._optimizer_lr(self.d_optimizer),
            self._initial_discriminator_learning_rate,
            warmup_epochs + 1,
        )

    def _create_plateau_scheduler(self, optimizer) -> ReduceLROnPlateau:
        scheduler_min_lr = self._validated_plateau_min_lr(optimizer)
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.reduce_lr_factor),
            patience=int(self.config.reduce_lr_patience),
            min_lr=scheduler_min_lr,
        )

    def _validated_scheduler_min_lr(self, optimizer) -> float:
        """Return a scheduler floor that cannot increase an optimizer LR."""

        initial_lr = self._optimizer_lr(optimizer)
        scheduler_min_lr = float(self.config.reduce_lr_min_lr)
        if not math.isfinite(initial_lr) or initial_lr < 0.0:
            raise ValueError(
                f"Optimizer learning rate must be finite and non-negative, got {initial_lr}"
            )
        if not math.isfinite(scheduler_min_lr) or scheduler_min_lr < 0.0:
            raise ValueError(
                "reduce_lr_min_lr must be finite and non-negative, got "
                f"{scheduler_min_lr}"
            )
        if scheduler_min_lr > initial_lr:
            raise ValueError(
                "reduce_lr_min_lr cannot exceed the optimizer's initial learning "
                f"rate: {scheduler_min_lr} > {initial_lr}"
            )
        return scheduler_min_lr

    def _generator_scheduler_min_learning_rates(self) -> dict[str, float]:
        if self.generator_optimizer_profile == "uniform_v1":
            return {}
        fallback = float(self.config.reduce_lr_min_lr)
        if self.generator_optimizer_profile == "film_unet_split_lr_v1":
            configured = {
                "backbone": fallback,
                "text_encoder": float(self.config.generator_text_min_learning_rate)
                or fallback,
                "film_projection": float(self.config.generator_film_min_learning_rate)
                or fallback,
            }
        elif self.generator_optimizer_profile == "conditioning_split_lr_v2":
            conditioning_floor = float(
                getattr(
                    self.config,
                    "generator_conditioning_min_learning_rate",
                    0.0,
                )
            )
            if conditioning_floor <= 0.0:
                conditioning_floor = float(
                    getattr(self.config, "generator_film_min_learning_rate", 0.0)
                )
            if conditioning_floor <= 0.0:
                conditioning_floor = fallback
            configured = {
                "backbone": fallback,
                "text_encoder": float(self.config.generator_text_min_learning_rate)
                or fallback,
                "conditioning_module": conditioning_floor,
            }
        else:
            raise ValueError(
                "Unsupported split Generator optimizer profile: "
                f"{self.generator_optimizer_profile!r}"
            )
        initial = self._initial_generator_group_learning_rates
        for name, floor in configured.items():
            if not math.isfinite(floor) or floor < 0.0:
                raise ValueError(
                    f"Generator {name} scheduler floor is invalid: {floor}"
                )
            if floor > float(initial[name]):
                raise ValueError(
                    f"Generator {name} scheduler floor cannot exceed its initial "
                    f"learning rate: {floor} > {initial[name]}"
                )
        return configured

    def _validated_plateau_min_lr(self, optimizer) -> float | list[float]:
        if optimizer is self.g_optimizer and self.generator_optimizer_profile != (
            "uniform_v1"
        ):
            floors = self._generator_scheduler_min_learning_rates()
            return [
                floors[str(group["group_name"])] for group in optimizer.param_groups
            ]
        return self._validated_scheduler_min_lr(optimizer)

    def _learning_rate_contract(self) -> dict[str, object]:
        """Return checkpoint-safe generator/discriminator LR lineage."""

        g_initial = float(self._initial_generator_learning_rate)
        d_initial = float(self._initial_discriminator_learning_rate)
        trace = [
            {
                "epoch": int(row["epoch"]),
                "g_lr": float(row["g_lr"]),
                "d_lr": float(row["d_lr"]),
            }
            for row in getattr(self, "_metrics_rows", [])
            if "epoch" in row and "g_lr" in row and "d_lr" in row
        ]
        contract: dict[str, object] = {
            "lr_profile": self.config.news_first_lr_profile,
            "lr_profile_sha256": self.config.news_first_lr_profile_sha256,
            "initial_learning_rate": float(self.config.learning_rate),
            "generator_initial_learning_rate": g_initial,
            "discriminator_initial_learning_rate": d_initial,
            "lr_warmup_epochs": self._lr_warmup_epochs(),
            "lr_warmup_start_factor": self._lr_warmup_start_factor(),
            "generator_warmup_start_learning_rate": (
                self._linear_warmup_learning_rate(
                    target_learning_rate=g_initial,
                    start_factor=self._lr_warmup_start_factor(),
                    epoch=1,
                    warmup_epochs=self._lr_warmup_epochs(),
                )
            ),
            "discriminator_warmup_start_learning_rate": (
                self._linear_warmup_learning_rate(
                    target_learning_rate=d_initial,
                    start_factor=self._lr_warmup_start_factor(),
                    epoch=1,
                    warmup_epochs=self._lr_warmup_epochs(),
                )
            ),
            "scheduler_min_lr": float(self.config.reduce_lr_min_lr),
            "generator_scheduler_min_lr": float(self.config.reduce_lr_min_lr),
            "discriminator_scheduler_min_lr": float(self.config.reduce_lr_min_lr),
            "lr_trace": trace,
            "generator_lr_trace": [
                {"epoch": row["epoch"], "lr": row["g_lr"]} for row in trace
            ],
            "discriminator_lr_trace": [
                {"epoch": row["epoch"], "lr": row["d_lr"]} for row in trace
            ],
        }
        if self.generator_optimizer_profile != "uniform_v1":
            group_initial = dict(self._initial_generator_group_learning_rates)
            group_trace = [
                {
                    "epoch": int(row["epoch"]),
                    **{name: float(row[f"g_lr_{name}"]) for name in group_initial},
                }
                for row in getattr(self, "_metrics_rows", [])
                if "epoch" in row
                and all(f"g_lr_{name}" in row for name in group_initial)
            ]
            contract.update(
                {
                    "generator_optimizer_profile": self.generator_optimizer_profile,
                    "generator_group_initial_learning_rates": group_initial,
                    "generator_group_warmup_start_learning_rates": {
                        name: self._linear_warmup_learning_rate(
                            target_learning_rate=learning_rate,
                            start_factor=self._lr_warmup_start_factor(),
                            epoch=1,
                            warmup_epochs=self._lr_warmup_epochs(),
                        )
                        for name, learning_rate in group_initial.items()
                    },
                    "generator_group_scheduler_min_learning_rates": (
                        self._generator_scheduler_min_learning_rates()
                    ),
                    "generator_group_lr_trace": group_trace,
                }
            )
        return contract

    def _step_plateau_scheduler(
        self,
        *,
        scheduler: Optional[ReduceLROnPlateau],
        optimizer,
        metric_name: str,
        metric_value: float,
        logger,
        label: str,
    ) -> None:
        if scheduler is None:
            return
        old_lr = self._optimizer_lr(optimizer)
        scheduler.step(metric_value)
        new_lr = self._optimizer_lr(optimizer)
        if not math.isclose(old_lr, new_lr):
            logger.info(
                "%s ReduceLROnPlateau lowered LR from %.6g to %.6g using %s=%.6f",
                label,
                old_lr,
                new_lr,
                metric_name,
                metric_value,
            )

    def _normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _black_call_price(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        sigma: [B, H, W], strike dimension is W and maturity dimension is H.
        """
        sigma = sigma.clamp_min(1e-4)
        k = self.strike_grid.view(1, 1, -1)
        t = self.tau_years.view(1, -1, 1)
        sqrt_t = torch.sqrt(t)
        d1 = (torch.log(1.0 / k) + 0.5 * sigma.pow(2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return self._normal_cdf(d1) - k * self._normal_cdf(d2)

    @staticmethod
    def _reduce_per_sample(
        values: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
    ) -> torch.Tensor:
        if training_weights:
            return training_weighted_mean(values, sample_weight)
        return weighted_mean(values, sample_weight)

    def _calendar_penalty_per_sample(
        self,
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(1) < 2:
            return sigma.new_zeros(int(sigma.shape[0]))
        total_variance = sigma.pow(2) * self.tau_years.view(1, -1, 1)
        diff = total_variance[:, 1:, :] - total_variance[:, :-1, :]
        edge_mask = None
        if support_mask is not None:
            cells = support_mask.squeeze(1)
            edge_mask = cells[:, 1:, :] * cells[:, :-1, :]
        return masked_mean_per_sample(F.relu(-diff), edge_mask)

    def calendar_arbitrage_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._calendar_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    def _butterfly_penalty_per_sample(
        self,
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(2) < 3:
            return sigma.new_zeros(int(sigma.shape[0]))
        call_prices = self._black_call_price(sigma)
        second_diff = (
            call_prices[:, :, 2:]
            - 2.0 * call_prices[:, :, 1:-1]
            + call_prices[:, :, :-2]
        )
        triplet_mask = None
        if support_mask is not None:
            cells = support_mask.squeeze(1)
            triplet_mask = cells[:, :, 2:] * cells[:, :, 1:-1] * cells[:, :, :-2]
        return masked_mean_per_sample(F.relu(-second_diff), triplet_mask)

    def butterfly_arbitrage_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._butterfly_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    @staticmethod
    def _smoothness_penalty_per_sample(
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = generated_surface.squeeze(1)
        batch_size = int(sigma.shape[0])
        penalty = sigma.new_zeros(batch_size)
        cells = support_mask.squeeze(1) if support_mask is not None else None
        if sigma.size(1) > 1:
            maturity_mask = (
                cells[:, 1:, :] * cells[:, :-1, :] if cells is not None else None
            )
            penalty = penalty + masked_mean_per_sample(
                (sigma[:, 1:, :] - sigma[:, :-1, :]).pow(2),
                maturity_mask,
            )
        if sigma.size(2) > 1:
            strike_mask = (
                cells[:, :, 1:] * cells[:, :, :-1] if cells is not None else None
            )
            penalty = penalty + masked_mean_per_sample(
                (sigma[:, :, 1:] - sigma[:, :, :-1]).pow(2),
                strike_mask,
            )
        return penalty

    def smoothness_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._smoothness_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    def _constraint_warmup_epochs(self) -> int:
        return max(0, int(getattr(self.config, "constraint_warmup_epochs", 0)))

    def _constraint_switches_for_epoch(self, epoch: int) -> tuple[bool, bool, bool]:
        if int(epoch) <= self._constraint_warmup_epochs():
            return False, False, False
        return (
            bool(self.use_calendar_constraint),
            bool(self.use_butterfly_constraint),
            bool(self.use_smooth_constraint),
        )

    @staticmethod
    def _delta_shrink_per_sample(
        fake_future: torch.Tensor,
        current_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_mean_per_sample(
            torch.abs(fake_future - current_surface),
            support_mask,
        )

    def delta_shrink_penalty(
        self,
        fake_future: torch.Tensor,
        current_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._delta_shrink_per_sample(fake_future, current_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    @staticmethod
    def _allowed_monitor_metrics() -> set[str]:
        return {
            "val_recon",
            "val_current_recon",
            "val_baseline_gap",
            "val_hybrid_score",
            "val_calendar",
            "val_butterfly",
            "val_delta_shrink",
        }

    def calculate_gradient_penalty(
        self,
        real_surface: torch.Tensor,
        fake_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = real_surface.size(0)
        support_mask = validated_surface_mask(
            support_mask,
            reference_surface=real_surface,
        )
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_surface + (1.0 - alpha) * fake_surface
        interpolated.requires_grad_(True)

        interpolated_scores = self.D(
            apply_surface_mask(interpolated, support_mask),
            apply_surface_mask(current_surface, support_mask),
            text_embedding,
        )
        grad_outputs = torch.ones_like(interpolated_scores, device=self.device)
        gradients = autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        per_sample_penalty = (gradients.norm(2, dim=1) - 1.0) ** 2
        grad_penalty = (
            training_weighted_mean(per_sample_penalty, sample_weight) * self.lambda_gp
        )
        return grad_penalty

    def _generator_step(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        real_future: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        label_reliability_weight: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        current_support_mask: torch.Tensor | None = None,
        epoch: int = 1,
        loss_scale: float = 1.0,
        skip_optimizer_step: bool = False,
    ):
        if not skip_optimizer_step and loss_scale == 1.0:
            self.g_optimizer.zero_grad(set_to_none=True)
        noise = self._generator_noise(
            current_surface,
            preserve_training_rng=True,
        )
        fake_future = self._forward_generator(
            current_surface,
            text_embedding,
            noise,
            current_support_mask,
        )
        batch_size = int(fake_future.shape[0])
        support_mask = validated_surface_mask(
            support_mask,
            reference_surface=fake_future,
        )
        adv_per_sample = (
            -self.D(
                apply_surface_mask(fake_future, support_mask),
                apply_surface_mask(current_surface, support_mask),
                text_embedding,
            )
            .reshape(batch_size, -1)
            .mean(dim=1)
        )
        recon_per_sample = masked_mean_per_sample(
            torch.abs(fake_future - real_future),
            support_mask,
        )
        recon_weight = reconstruction_training_weights(
            sample_weight,
            label_reliability_weight,
            batch_size=batch_size,
            device=fake_future.device,
            dtype=fake_future.dtype,
        )
        adv_loss = training_weighted_mean(adv_per_sample, sample_weight)
        recon_loss = training_weighted_mean(recon_per_sample, recon_weight)
        cal_penalty = self.calendar_arbitrage_penalty(
            fake_future,
            sample_weight,
            training_weights=True,
            support_mask=support_mask,
        )
        bfly_penalty = self.butterfly_arbitrage_penalty(
            fake_future,
            sample_weight,
            training_weights=True,
            support_mask=support_mask,
        )
        smooth_penalty = self.smoothness_penalty(
            fake_future,
            sample_weight,
            training_weights=True,
            support_mask=support_mask,
        )
        delta_shrink = self.delta_shrink_penalty(
            fake_future,
            current_surface,
            sample_weight,
            training_weights=True,
            support_mask=support_mask,
        )
        use_calendar_constraint, use_butterfly_constraint, use_smooth_constraint = (
            self._constraint_switches_for_epoch(epoch)
        )

        g_loss = adv_loss + self.lambda_recon * recon_loss
        if use_calendar_constraint:
            g_loss = g_loss + self.lambda_calendar * cal_penalty
        if use_butterfly_constraint:
            g_loss = g_loss + self.lambda_butterfly * bfly_penalty
        if use_smooth_constraint:
            g_loss = g_loss + self.lambda_smooth * smooth_penalty
        if self.lambda_delta_shrink > 0.0:
            g_loss = g_loss + self.lambda_delta_shrink * delta_shrink
        (g_loss * loss_scale).backward()
        if not skip_optimizer_step:
            self.g_optimizer.step()

        return {
            "g_total": float(g_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_recon": float(recon_loss.detach().cpu()),
            "g_calendar": float(cal_penalty.detach().cpu()),
            "g_butterfly": float(bfly_penalty.detach().cpu()),
            "g_smooth": float(smooth_penalty.detach().cpu()),
            "g_delta_shrink": float(delta_shrink.detach().cpu()),
        }

    def _discriminator_step(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        real_future: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        current_support_mask: torch.Tensor | None = None,
    ):
        self.d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            noise = self._generator_noise(
                current_surface,
                preserve_training_rng=True,
            )
            fake_future = self._forward_generator(
                current_surface,
                text_embedding,
                noise,
                current_support_mask,
            )

        batch_size = int(real_future.shape[0])
        support_mask = validated_surface_mask(
            support_mask,
            reference_surface=real_future,
        )
        critic_current = apply_surface_mask(current_surface, support_mask)
        real_scores = (
            self.D(
                apply_surface_mask(real_future, support_mask),
                critic_current,
                text_embedding,
            )
            .reshape(batch_size, -1)
            .mean(dim=1)
        )
        fake_scores = (
            self.D(
                apply_surface_mask(fake_future, support_mask),
                critic_current,
                text_embedding,
            )
            .reshape(batch_size, -1)
            .mean(dim=1)
        )
        d_real = training_weighted_mean(real_scores, sample_weight)
        d_fake = training_weighted_mean(fake_scores, sample_weight)
        gp = self.calculate_gradient_penalty(
            real_future,
            fake_future,
            current_surface,
            text_embedding,
            sample_weight,
            support_mask,
        )
        d_loss = d_fake - d_real + gp
        d_loss.backward()
        self.d_optimizer.step()

        return {
            "d_total": float(d_loss.detach().cpu()),
            "d_real": float(d_real.detach().cpu()),
            "d_fake": float(d_fake.detach().cpu()),
            "gp": float(gp.detach().cpu()),
        }

    def _evaluate(self, val_loader) -> Dict[str, float]:
        if val_loader is None:
            return {}

        was_training = self.G.training
        self.G.eval()
        metric_sums = {
            "recon": 0.0,
            "current_recon": 0.0,
            "calendar": 0.0,
            "butterfly": 0.0,
            "delta_shrink": 0.0,
        }
        total_weight = 0.0
        mc_samples = max(1, int(getattr(self.config, "validation_mc_samples", 1)))
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = unpack_vol_training_batch(raw_batch)
                current_surface = self._to_device(batch.current_surface)
                text_embedding = self._to_device(batch.text_embedding)
                real_future = self._to_device(batch.target_surface)
                batch_size = int(current_surface.shape[0])
                weights = validated_sample_weights(
                    batch.sample_weight,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=current_surface.dtype,
                )
                validated_label_reliability_weights(
                    batch.label_reliability_weight,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=current_surface.dtype,
                    require_ones=True,
                )
                support_mask = validated_surface_mask(
                    batch.support_mask,
                    reference_surface=real_future,
                )
                current_support_mask = (
                    None
                    if batch.current_support_mask is None
                    else self._to_device(batch.current_support_mask)
                )
                stable_keys = batch.stable_noise_key
                if stable_keys is None:
                    stable_keys = stable_tensor_row_keys(
                        current_surface,
                        text_embedding,
                        real_future,
                    )

                if (
                    hasattr(self.G, "noise_dim")
                    and self.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE
                ):
                    noise = self._generator_noise(
                        current_surface,
                        preserve_training_rng=False,
                    )
                    fake_future = self._forward_generator(
                        current_surface,
                        text_embedding,
                        noise,
                        current_support_mask,
                    )
                elif hasattr(self.G, "noise_dim"):
                    fake_sum = torch.zeros_like(real_future)
                    for draw_index in range(mc_samples):
                        noise = stable_noise_for_keys(
                            stable_keys,
                            noise_dim=int(self.G.noise_dim),
                            base_seed=int(self.config.seed),
                            draw_index=draw_index,
                            device=self.device,
                            dtype=current_surface.dtype,
                        )
                        fake_sum = fake_sum + self._forward_generator(
                            current_surface,
                            text_embedding,
                            noise,
                            current_support_mask,
                        )
                    fake_future = fake_sum / float(mc_samples)
                else:
                    # Test doubles and deterministic generators may expose only
                    # the historical two-argument forward signature.
                    if (
                        self.generator_current_input_mode
                        == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                    ):
                        raise ValueError(
                            "current_support_masked evaluation requires a Generator "
                            "that accepts current_support_mask"
                        )
                    fake_future = self.G(current_surface, text_embedding)

                per_sample_metrics = {
                    "recon": masked_mean_per_sample(
                        torch.abs(fake_future - real_future), support_mask
                    ),
                    "current_recon": masked_mean_per_sample(
                        torch.abs(current_surface - real_future), support_mask
                    ),
                    "calendar": self._calendar_penalty_per_sample(
                        fake_future, support_mask
                    ),
                    "butterfly": self._butterfly_penalty_per_sample(
                        fake_future, support_mask
                    ),
                    "delta_shrink": self._delta_shrink_per_sample(
                        fake_future,
                        current_surface,
                        support_mask,
                    ),
                }
                for name, values in per_sample_metrics.items():
                    metric_sums[name] += float(torch.sum(values * weights).cpu())
                total_weight += float(torch.sum(weights).cpu())

        if was_training:
            self.G.train()
        if total_weight <= 0.0:
            return {}
        val_recon = metric_sums["recon"] / total_weight
        val_current_recon = metric_sums["current_recon"] / total_weight
        val_baseline_gap = val_recon - val_current_recon
        return {
            "val_recon": val_recon,
            "val_current_recon": val_current_recon,
            "val_baseline_gap": val_baseline_gap,
            "val_hybrid_score": val_recon
            + self.baseline_penalty_weight * max(0.0, val_baseline_gap),
            "val_calendar": metric_sums["calendar"] / total_weight,
            "val_butterfly": metric_sums["butterfly"] / total_weight,
            "val_delta_shrink": metric_sums["delta_shrink"] / total_weight,
        }

    def _init_metrics_file(self):
        self._metrics_file = os.path.join(self.metrics_path, "training_metrics.json")
        self._metrics_csv_file = os.path.join(self.metrics_path, "training_metrics.csv")
        self._best_checkpoint_file = os.path.join(
            self.metrics_path, "best_checkpoint.json"
        )
        self._initial_checkpoint_file = os.path.join(
            self.metrics_path,
            "initial_checkpoint.json",
        )
        self._best_learned_checkpoint_file = os.path.join(
            self.metrics_path,
            "best_learned_checkpoint.json",
        )
        self._fallback_calibration_file = os.path.join(
            self.metrics_path, "fallback_calibration.json"
        )
        self._metrics_rows = []
        write_metrics_json([], self._metrics_file)
        for stale_path in (
            self._metrics_csv_file,
            self._best_checkpoint_file,
            self._initial_checkpoint_file,
            self._best_learned_checkpoint_file,
            self._fallback_calibration_file,
            os.path.join(self.model_path, "generator_best.pt"),
            os.path.join(self.model_path, "discriminator_best.pt"),
            os.path.join(self.model_path, "generator_initial_epoch0.pt"),
            os.path.join(self.model_path, "discriminator_initial_epoch0.pt"),
            os.path.join(self.model_path, "generator_best_learned.pt"),
            os.path.join(self.model_path, "discriminator_best_learned.pt"),
        ):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    def _append_metrics_row(self, row):
        self._metrics_rows.append(row)
        self._write_metrics(self._metrics_rows)

    def _write_metrics(self, metrics_rows):
        write_metrics_json(metrics_rows, self._metrics_file)
        write_metrics_csv(metrics_rows, self._metrics_csv_file)

    def save_model(
        self, epoch: Optional[int] = None, *, label: Optional[str] = None
    ) -> Dict[str, str]:
        if epoch is not None and label is not None:
            raise ValueError(
                "Specify either epoch or label when saving a model, not both."
            )
        os.makedirs(self.model_path, exist_ok=True)

        suffix = ""
        if label:
            suffix = f"_{label}"
        elif epoch is not None:
            suffix = f"_epoch_{epoch:04d}"
        generator_path = os.path.join(self.model_path, f"generator{suffix}.pt")
        discriminator_path = os.path.join(self.model_path, f"discriminator{suffix}.pt")
        residual_output_mode = str(
            getattr(self.G, "residual_output_mode", self.config.residual_output_mode)
        )
        output_fingerprint = str(
            getattr(
                self.G,
                "residual_output_fingerprint",
                residual_output_fingerprint(residual_output_mode),
            )
        )

        torch.save(
            {
                "state_dict": self.G.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.embedding_dim,
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                "residual_output_mode": residual_output_mode,
                "residual_output_fingerprint": output_fingerprint,
                **self._generator_noise_lineage(),
                **self._generator_current_input_lineage(),
                **self._conditioning_lineage(),
                **self._grid_model_lineage(),
                **self._label_reliability_lineage(),
                **self._refit_lineage(),
                **self._full_training_state_lineage(),
                **self._graft_state_lineage(),
                **self._learning_rate_contract(),
            },
            generator_path,
        )
        torch.save(
            {
                "state_dict": self.D.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.embedding_dim,
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._generator_noise_lineage(),
                **self._generator_current_input_lineage(),
                **self._conditioning_lineage(),
                **self._grid_model_lineage(),
                **self._label_reliability_lineage(),
                **self._refit_lineage(),
                **self._full_training_state_lineage(),
                **self._graft_state_lineage(),
                **self._learning_rate_contract(),
            },
            discriminator_path,
        )
        return {
            "generator": generator_path,
            "discriminator": discriminator_path,
        }

    def _save_best_validation_checkpoint(
        self,
        *,
        epoch: int,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist one validation-selected checkpoint, including epoch zero."""

        artifact_paths = self.save_model(label="best")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "best_epoch": int(epoch),
                "best_metric": current_metric,
                "selection_scope": "baseline_inclusive",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._generator_noise_lineage(),
                **self._generator_current_input_lineage(),
                **self._conditioning_lineage(),
                **self._grid_model_lineage(),
                **self._label_reliability_lineage(),
                **self._refit_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._best_checkpoint_file,
        )

    def _save_best_learned_validation_checkpoint(
        self,
        *,
        epoch: int,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist the best trained checkpoint, excluding epoch zero."""

        if epoch < 1:
            raise ValueError("best learned checkpoint requires epoch >= 1")
        artifact_paths = self.save_model(label="best_learned")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "best_epoch": int(epoch),
                "best_learned_epoch_ge_1": int(epoch),
                "best_metric": current_metric,
                "selection_scope": "trained_epochs_only",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._generator_noise_lineage(),
                **self._generator_current_input_lineage(),
                **self._conditioning_lineage(),
                **self._grid_model_lineage(),
                **self._label_reliability_lineage(),
                **self._refit_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._best_learned_checkpoint_file,
        )

    def _save_initial_validation_checkpoint(
        self,
        *,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist the immutable pre-training validation checkpoint."""

        artifact_paths = self.save_model(label="initial_epoch0")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "initial_epoch": 0,
                "best_epoch": 0,
                "best_metric": current_metric,
                "selection_scope": "initial_epoch0",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._generator_noise_lineage(),
                **self._generator_current_input_lineage(),
                **self._conditioning_lineage(),
                **self._grid_model_lineage(),
                **self._label_reliability_lineage(),
                **self._refit_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._initial_checkpoint_file,
        )

    def _save_loss_curves(self, metrics_rows, logger) -> None:
        output_path = os.path.join(self.metrics_path, "loss_curves.png")
        plot_training_curves(
            metrics_rows,
            title="WGAN Vol Training Loss Curves",
            metric_groups=(
                (
                    "Primary losses",
                    ("g_recon", "val_recon", "g_total", "d_total", "gp"),
                ),
                (
                    "Constraint losses",
                    (
                        "g_calendar",
                        "g_butterfly",
                        "g_smooth",
                        "g_delta_shrink",
                        "val_calendar",
                        "val_butterfly",
                        "val_current_recon",
                        "val_hybrid_score",
                        "val_delta_shrink",
                    ),
                ),
            ),
            output_path=output_path,
        )
        logger.info("Loss curve plot saved to: %s", output_path)

    def _calibrate_fallback_threshold(self, val_samples: Sequence[object]) -> None:
        if self.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE:
            # A deterministic generator has identically zero MC dispersion;
            # emitting an uncertainty threshold would be misleading.
            return
        checkpoint_path = os.path.join(self.model_path, "generator_best.pt")
        if not val_samples or not os.path.exists(checkpoint_path):
            return

        model, train_config, _ = load_vol_generator(
            checkpoint_path, val_samples[0], self.device
        )
        uncertainty_scores: list[float] = []
        sample_metrics: list[dict[str, float]] = []

        for sample in val_samples:
            mean_surface, uncertainty_score, _ = infer_vol_surface_mc(
                model,
                sample,
                noise_dim=int(train_config.noise_dim),
                seed=int(self.config.seed),
                device=self.device,
                mc_samples=5,
            )
            current_surface = sample.current_surface[0]
            target_surface = sample.target_surface[0]
            support_mask = getattr(sample, "support_mask", None)
            uncertainty_scores.append(float(uncertainty_score))
            sample_metrics.append(
                {
                    "model_recon": mean_abs_error(
                        mean_surface, target_surface, support_mask
                    ),
                    "current_recon": mean_abs_error(
                        current_surface, target_surface, support_mask
                    ),
                    "uncertainty_score": float(uncertainty_score),
                }
            )

        if not sample_metrics:
            return

        score_array = np.asarray(uncertainty_scores, dtype=np.float32)
        candidate_percentiles = (50, 60, 70, 80, 90, 95)
        candidate_rows = []
        best_payload = None

        for percentile in candidate_percentiles:
            threshold = float(np.percentile(score_array, percentile))
            fallback_recon = []
            current_recon = []
            used_fallback_count = 0
            for item in sample_metrics:
                current_metric = float(item["current_recon"])
                model_metric = float(item["model_recon"])
                current_recon.append(current_metric)
                if float(item["uncertainty_score"]) > threshold:
                    fallback_recon.append(current_metric)
                    used_fallback_count += 1
                else:
                    fallback_recon.append(model_metric)
            summary = summarize_baseline_aware_metrics(
                fallback_recon,
                current_recon,
                baseline_penalty_weight=self.baseline_penalty_weight,
            )
            row = {
                "percentile": int(percentile),
                "uncertainty_threshold": threshold,
                "fallback_rate": float(used_fallback_count)
                / float(len(sample_metrics)),
                **summary,
            }
            candidate_rows.append(row)
            if best_payload is None or float(row["val_hybrid_score"]) < float(
                best_payload["val_hybrid_score"]
            ):
                best_payload = row

        assert best_payload is not None
        write_best_checkpoint(
            {
                "checkpoint_path": checkpoint_path,
                "mc_samples": 5,
                "baseline_penalty_weight": self.baseline_penalty_weight,
                "selected_percentile": int(best_payload["percentile"]),
                **self._label_reliability_lineage(),
                "uncertainty_threshold": float(best_payload["uncertainty_threshold"]),
                "selected_metrics": {
                    "val_recon": float(best_payload["val_recon"]),
                    "val_current_recon": float(best_payload["val_current_recon"]),
                    "val_baseline_gap": float(best_payload["val_baseline_gap"]),
                    "val_hybrid_score": float(best_payload["val_hybrid_score"]),
                    "fallback_rate": float(best_payload["fallback_rate"]),
                },
                "candidates": candidate_rows,
            },
            self._fallback_calibration_file,
        )

    def train(
        self,
        train_loader,
        val_loader=None,
        *,
        val_samples: Optional[Sequence[object]] = None,
    ):
        import logging

        logger = logging.getLogger("wgan_option.trainer")

        if self.news_first_refit_mode == FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            if val_loader is not None or bool(val_samples):
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 requires no validation loader or samples"
                )

        self._restore_graft_state_if_configured(train_loader=train_loader)
        self._initial_generator_learning_rate = self._optimizer_lr(self.g_optimizer)
        self._initial_generator_group_learning_rates = (
            self._generator_optimizer_group_learning_rates()
        )
        self._initial_discriminator_learning_rate = self._optimizer_lr(self.d_optimizer)
        self._init_metrics_file()

        metrics_rows = []
        num_batches = len(train_loader)
        monitor_metric = self.best_checkpoint_metric
        if monitor_metric not in self._allowed_monitor_metrics():
            raise ValueError(
                f"Unsupported best_checkpoint_metric='{monitor_metric}'. "
                f"Expected one of {sorted(self._allowed_monitor_metrics())}."
            )
        best_metric = None
        best_epoch = None
        best_learned_metric = None
        best_learned_epoch = None
        patience = max(1, int(self.config.early_stopping_patience))
        min_delta = float(self.config.early_stopping_min_delta)
        min_epochs = max(0, int(self.config.early_stopping_min_epochs))
        best_tracking_enabled = val_loader is not None
        early_stopping_enabled = bool(
            self.config.use_early_stopping and best_tracking_enabled
        )
        evaluate_initial_checkpoint = bool(
            getattr(self.config, "evaluate_initial_checkpoint", False)
        )
        g_scheduler = None
        d_scheduler = None
        epochs_without_improvement = 0

        scheduler_type = (
            str(getattr(self.config, "lr_scheduler_type", "none")).strip().lower()
        )
        if scheduler_type == "none" and self.config.use_reduce_lr_on_plateau:
            scheduler_type = "plateau"

        if scheduler_type == "plateau":
            if not best_tracking_enabled:
                logger.info(
                    "ReduceLROnPlateau requested but disabled because no validation split is available."
                )
                scheduler_type = "none"
            else:
                g_scheduler = self._create_plateau_scheduler(self.g_optimizer)
                d_scheduler = self._create_plateau_scheduler(self.d_optimizer)
                logger.info(
                    "ReduceLROnPlateau enabled for generator and discriminator using %s (factor=%.3f, patience=%d, min_lr=%.6g).",
                    monitor_metric,
                    float(self.config.reduce_lr_factor),
                    int(self.config.reduce_lr_patience),
                    float(self.config.reduce_lr_min_lr),
                )
        elif scheduler_type == "cosine":
            g_eta_min = self._validated_scheduler_min_lr(self.g_optimizer)
            d_eta_min = self._validated_scheduler_min_lr(self.d_optimizer)
            g_scheduler = CosineAnnealingLR(
                self.g_optimizer, T_max=self.num_epochs, eta_min=g_eta_min
            )
            d_scheduler = CosineAnnealingLR(
                self.d_optimizer, T_max=self.num_epochs, eta_min=d_eta_min
            )
            logger.info(
                "CosineAnnealingLR enabled for generator and discriminator "
                "(T_max=%d, g_eta_min=%.6g, d_eta_min=%.6g).",
                self.num_epochs,
                g_eta_min,
                d_eta_min,
            )
        else:
            logger.info("LR scheduler is disabled.")

        self._restore_full_training_state_if_configured(
            train_loader=train_loader,
            generator_scheduler=(
                g_scheduler if isinstance(g_scheduler, ReduceLROnPlateau) else None
            ),
            discriminator_scheduler=(
                d_scheduler if isinstance(d_scheduler, ReduceLROnPlateau) else None
            ),
        )
        self._initialize_learning_rate_warmup(logger)

        if not best_tracking_enabled:
            logger.info("Validation unavailable; best-checkpoint tracking disabled.")
            if self.config.use_early_stopping:
                logger.info(
                    "Early stopping requested but disabled because no validation split is available."
                )
            if evaluate_initial_checkpoint:
                logger.info(
                    "Initial-checkpoint evaluation requested but disabled because no validation split is available."
                )
        elif early_stopping_enabled:
            logger.info(
                "Best-checkpoint tracking enabled using %s. Early stopping active "
                "(patience=%d, min_delta=%.6f, min_epochs=%d).",
                monitor_metric,
                patience,
                min_delta,
                min_epochs,
            )
        else:
            logger.info(
                "Best-checkpoint tracking enabled using %s. Early stopping is disabled.",
                monitor_metric,
            )

        constraint_warmup_epochs = self._constraint_warmup_epochs()
        if constraint_warmup_epochs > 0:
            logger.info(
                "Constraint warmup active: enabled constraint losses will be skipped for the first %d epoch(s) and applied from epoch %d.",
                constraint_warmup_epochs,
                constraint_warmup_epochs + 1,
            )

        if best_tracking_enabled and evaluate_initial_checkpoint:
            initial_stats = {"epoch": 0}
            initial_stats.update(self._evaluate(val_loader))
            initial_stats["g_lr"] = self._optimizer_lr(self.g_optimizer)
            initial_stats["d_lr"] = self._optimizer_lr(self.d_optimizer)
            initial_stats.update(self._generator_learning_rate_metrics())
            metrics_rows.append(initial_stats)
            self._append_metrics_row(initial_stats)
            if monitor_metric in initial_stats:
                best_metric = resolve_monitor_metric(initial_stats, monitor_metric)
                best_epoch = 0
                epochs_without_improvement = 0
                self._save_initial_validation_checkpoint(
                    monitor_metric=monitor_metric,
                    current_metric=best_metric,
                    epoch_stats=initial_stats,
                )
                self._save_best_validation_checkpoint(
                    epoch=0,
                    monitor_metric=monitor_metric,
                    current_metric=best_metric,
                    epoch_stats=initial_stats,
                )
                logger.info(
                    "Initial validation checkpoint saved at epoch 0 with %s=%.6f",
                    monitor_metric,
                    best_metric,
                )
                if (
                    scheduler_type == "plateau"
                    and self._restored_full_training_state is None
                    and self._lr_warmup_epochs() == 0
                ):
                    self._step_plateau_scheduler(
                        scheduler=g_scheduler,
                        optimizer=self.g_optimizer,
                        metric_name=monitor_metric,
                        metric_value=best_metric,
                        logger=logger,
                        label="Generator",
                    )
                    self._step_plateau_scheduler(
                        scheduler=d_scheduler,
                        optimizer=self.d_optimizer,
                        metric_name=monitor_metric,
                        metric_value=best_metric,
                        logger=logger,
                        label="Discriminator",
                    )
            else:
                logger.warning(
                    "Initial-checkpoint evaluation produced no %s metric; epoch-0 best tracking was skipped.",
                    monitor_metric,
                )

        accum_steps = max(
            1, int(getattr(self.config, "gradient_accumulation_steps", 1))
        )
        if accum_steps > 1:
            logger.info(
                "Gradient accumulation enabled: %d steps (effective batch = %d).",
                accum_steps,
                self.config.batch_size * accum_steps,
            )

        for epoch in range(1, self.num_epochs + 1):
            self._apply_refit_learning_rates(epoch)
            self._apply_learning_rate_warmup(epoch)
            selected_as_best_learned = False
            selected_metric = 0.0
            running: Dict[str, list] = {}
            for batch_idx, raw_batch in enumerate(train_loader, 1):
                batch = unpack_vol_training_batch(raw_batch)
                current_surface = self._to_device(batch.current_surface)
                text_embedding = self._to_device(batch.text_embedding)
                real_future = self._to_device(batch.target_surface)
                sample_weight = self._to_device(batch.sample_weight)
                support_mask = validated_surface_mask(
                    batch.support_mask,
                    reference_surface=real_future,
                )
                current_support_mask = (
                    None
                    if batch.current_support_mask is None
                    else self._to_device(batch.current_support_mask)
                )

                d_stats = {}
                for _ in range(self.critic_iter):
                    d_stats = self._discriminator_step(
                        current_surface,
                        text_embedding,
                        real_future,
                        sample_weight,
                        support_mask,
                        current_support_mask,
                    )
                    for key, value in d_stats.items():
                        running.setdefault(key, []).append(value)

                is_accum_boundary = (batch_idx % accum_steps == 0) or (
                    batch_idx == num_batches
                )
                if accum_steps > 1 and (batch_idx - 1) % accum_steps == 0:
                    self.g_optimizer.zero_grad(set_to_none=True)

                g_stats = self._generator_step(
                    current_surface,
                    text_embedding,
                    real_future,
                    sample_weight=sample_weight,
                    label_reliability_weight=batch.label_reliability_weight,
                    support_mask=support_mask,
                    current_support_mask=current_support_mask,
                    epoch=epoch,
                    loss_scale=1.0 / accum_steps if accum_steps > 1 else 1.0,
                    skip_optimizer_step=not is_accum_boundary
                    if accum_steps > 1
                    else False,
                )
                for key, value in g_stats.items():
                    running.setdefault(key, []).append(value)

                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches:
                    logger.info(
                        "[Epoch %04d/%04d] Batch %d/%d  D=%.4f G=%.4f",
                        epoch,
                        self.num_epochs,
                        batch_idx,
                        num_batches,
                        d_stats.get("d_total", 0.0),
                        g_stats.get("g_total", 0.0),
                    )

            epoch_stats = {
                "epoch": epoch,
                **{
                    key: float(np.mean(values))
                    for key, values in running.items()
                    if values
                },
            }
            if val_loader is not None:
                epoch_stats.update(self._evaluate(val_loader))
            epoch_stats["g_lr"] = self._optimizer_lr(self.g_optimizer)
            epoch_stats["d_lr"] = self._optimizer_lr(self.d_optimizer)
            epoch_stats.update(self._generator_learning_rate_metrics())
            metrics_rows.append(epoch_stats)
            self._append_metrics_row(epoch_stats)

            val_info = ""
            if "val_recon" in epoch_stats:
                val_info = (
                    f" ValRecon={epoch_stats['val_recon']:.4f}"
                    f" Curr={epoch_stats.get('val_current_recon', 0.0):.4f}"
                    f" Hybrid={epoch_stats.get('val_hybrid_score', 0.0):.4f}"
                )

            logger.info(
                "[Epoch %04d/%04d] D=%.4f G=%.4f Recon=%.4f Cal=%.4f Bfly=%.4f%s",
                epoch,
                self.num_epochs,
                epoch_stats.get("d_total", 0.0),
                epoch_stats.get("g_total", 0.0),
                epoch_stats.get("g_recon", 0.0),
                epoch_stats.get("g_calendar", 0.0),
                epoch_stats.get("g_butterfly", 0.0),
                val_info,
            )

            if best_tracking_enabled and monitor_metric in epoch_stats:
                current_metric = resolve_monitor_metric(epoch_stats, monitor_metric)
                if best_learned_metric is None or current_metric < (
                    best_learned_metric - min_delta
                ):
                    best_learned_metric = current_metric
                    best_learned_epoch = epoch
                    selected_as_best_learned = True
                    selected_metric = current_metric
                if best_metric is None or current_metric < (best_metric - min_delta):
                    best_metric = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_best_validation_checkpoint(
                        epoch=epoch,
                        monitor_metric=monitor_metric,
                        current_metric=current_metric,
                        epoch_stats=epoch_stats,
                    )
                    logger.info(
                        "New best checkpoint saved at epoch %d with %s=%.6f",
                        epoch,
                        monitor_metric,
                        current_metric,
                    )
                elif early_stopping_enabled:
                    epochs_without_improvement += 1
                    logger.info(
                        "Early stopping patience %d/%d without %s improvement (current=%.6f, best=%.6f at epoch %d)",
                        epochs_without_improvement,
                        patience,
                        monitor_metric,
                        current_metric,
                        best_metric,
                        best_epoch,
                    )

                if scheduler_type == "plateau" and not self._lr_warmup_active_for_epoch(
                    epoch
                ):
                    self._step_plateau_scheduler(
                        scheduler=g_scheduler,
                        optimizer=self.g_optimizer,
                        metric_name=monitor_metric,
                        metric_value=current_metric,
                        logger=logger,
                        label="Generator",
                    )
                    self._step_plateau_scheduler(
                        scheduler=d_scheduler,
                        optimizer=self.d_optimizer,
                        metric_name=monitor_metric,
                        metric_value=current_metric,
                        logger=logger,
                        label="Discriminator",
                    )

            if scheduler_type == "cosine" and g_scheduler is not None:
                g_scheduler.step()
                d_scheduler.step()

            if selected_as_best_learned:
                self._save_selected_full_training_state(
                    train_loader=train_loader,
                    completed_epoch=epoch,
                    generator_scheduler=(
                        g_scheduler
                        if isinstance(g_scheduler, ReduceLROnPlateau)
                        else None
                    ),
                    discriminator_scheduler=(
                        d_scheduler
                        if isinstance(d_scheduler, ReduceLROnPlateau)
                        else None
                    ),
                )
                self._save_best_learned_validation_checkpoint(
                    epoch=epoch,
                    monitor_metric=monitor_metric,
                    current_metric=selected_metric,
                    epoch_stats=epoch_stats,
                )
                logger.info(
                    "New best learned checkpoint saved at epoch %d with %s=%.6f",
                    epoch,
                    monitor_metric,
                    selected_metric,
                )

            if epoch in tuple(
                getattr(self.config, "news_first_validation_snapshot_epochs", ())
            ):
                self.save_model(label=f"validation_epoch_{epoch:04d}")
                logger.info(
                    "Frozen validation-trajectory checkpoint saved at epoch %d",
                    epoch,
                )

            if epoch % self.config.save_every == 0:
                self.save_model(epoch)
                logger.info("Checkpoint saved at epoch %d", epoch)

            if (
                early_stopping_enabled
                and epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d. Best %s=%.6f at epoch %d; "
                    "best learned epoch=%s.",
                    epoch,
                    monitor_metric,
                    best_metric,
                    best_epoch,
                    best_learned_epoch,
                )
                break

        self._validate_completed_refit_trace()
        self.save_model()
        self._write_metrics(metrics_rows)
        if best_tracking_enabled and val_samples:
            try:
                self._calibrate_fallback_threshold(val_samples)
            except Exception as exc:
                logger.warning("Failed to calibrate fallback threshold: %s", exc)
        try:
            self._save_loss_curves(metrics_rows, logger)
        except Exception as exc:
            logger.warning("Failed to save loss curve plot: %s", exc)
