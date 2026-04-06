import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam

from wgan_option.config import Config
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator
from wgan_option.utils.visualization import plot_training_curves


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
        self.config = config
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")
        self.strike_grid = torch.tensor(strike_grid, dtype=torch.float32, device=self.device).clamp_min(1e-4)
        self.maturity_grid_days = torch.tensor(maturity_grid_days, dtype=torch.float32, device=self.device)
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
            hidden_dim=config.gen_hidden_dim,
        ).to(self.device)
        self.D = Discriminator(
            channels=config.channels,
            embedding_dim=embedding_dim,
            surface_height=surface_height,
            surface_width=surface_width,
            hidden_dim=config.disc_hidden_dim,
        ).to(self.device)

        self.g_optimizer = Adam(self.G.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
        self.d_optimizer = Adam(self.D.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))

        self.critic_iter = config.discriminator_iter
        self.lambda_gp = config.lambda_gp
        self.lambda_recon = config.lambda_recon
        self.lambda_calendar = config.lambda_calendar
        self.lambda_butterfly = config.lambda_butterfly
        self.lambda_smooth = config.lambda_smooth
        self.use_calendar_constraint = config.use_calendar_constraint
        self.use_butterfly_constraint = config.use_butterfly_constraint
        self.use_smooth_constraint = config.use_smooth_constraint
        self.num_epochs = config.num_epochs

        self.model_path = config.models_path
        self.metrics_path = config.metrics_path
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)

        self._set_seed(config.seed)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

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

    def calendar_arbitrage_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(1) < 2:
            return torch.zeros(1, device=self.device).squeeze()
        total_variance = sigma.pow(2) * self.tau_years.view(1, -1, 1)
        diff = total_variance[:, 1:, :] - total_variance[:, :-1, :]
        return F.relu(-diff).mean()

    def butterfly_arbitrage_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(2) < 3:
            return torch.zeros(1, device=self.device).squeeze()
        call_prices = self._black_call_price(sigma)
        second_diff = call_prices[:, :, 2:] - 2.0 * call_prices[:, :, 1:-1] + call_prices[:, :, :-2]
        return F.relu(-second_diff).mean()

    def smoothness_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        sigma = generated_surface.squeeze(1)
        penalty = torch.zeros(1, device=self.device).squeeze()
        if sigma.size(1) > 1:
            penalty = penalty + (sigma[:, 1:, :] - sigma[:, :-1, :]).pow(2).mean()
        if sigma.size(2) > 1:
            penalty = penalty + (sigma[:, :, 1:] - sigma[:, :, :-1]).pow(2).mean()
        return penalty

    def calculate_gradient_penalty(
        self,
        real_surface: torch.Tensor,
        fake_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_surface.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_surface + (1.0 - alpha) * fake_surface
        interpolated.requires_grad_(True)

        interpolated_scores = self.D(interpolated, current_surface, text_embedding)
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
        grad_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * self.lambda_gp
        return grad_penalty

    def _generator_step(self, current_surface: torch.Tensor, text_embedding: torch.Tensor, real_future: torch.Tensor):
        self.g_optimizer.zero_grad(set_to_none=True)
        fake_future = self.G(current_surface, text_embedding)
        adv_loss = -self.D(fake_future, current_surface, text_embedding).mean()
        recon_loss = F.l1_loss(fake_future, real_future)
        cal_penalty = self.calendar_arbitrage_penalty(fake_future)
        bfly_penalty = self.butterfly_arbitrage_penalty(fake_future)
        smooth_penalty = self.smoothness_penalty(fake_future)

        g_loss = adv_loss + self.lambda_recon * recon_loss
        if self.use_calendar_constraint:
            g_loss = g_loss + self.lambda_calendar * cal_penalty
        if self.use_butterfly_constraint:
            g_loss = g_loss + self.lambda_butterfly * bfly_penalty
        if self.use_smooth_constraint:
            g_loss = g_loss + self.lambda_smooth * smooth_penalty
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "g_total": float(g_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_recon": float(recon_loss.detach().cpu()),
            "g_calendar": float(cal_penalty.detach().cpu()),
            "g_butterfly": float(bfly_penalty.detach().cpu()),
            "g_smooth": float(smooth_penalty.detach().cpu()),
        }

    def _discriminator_step(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        real_future: torch.Tensor,
    ):
        self.d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_future = self.G(current_surface, text_embedding)

        d_real = self.D(real_future, current_surface, text_embedding).mean()
        d_fake = self.D(fake_future, current_surface, text_embedding).mean()
        gp = self.calculate_gradient_penalty(real_future, fake_future, current_surface, text_embedding)
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

        self.G.eval()
        recon, calendar, butterfly = [], [], []
        with torch.no_grad():
            for current_surface, text_embedding, real_future in val_loader:
                current_surface = self._to_device(current_surface)
                text_embedding = self._to_device(text_embedding)
                real_future = self._to_device(real_future)
                fake_future = self.G(current_surface, text_embedding)
                recon.append(float(F.l1_loss(fake_future, real_future).detach().cpu()))
                calendar.append(float(self.calendar_arbitrage_penalty(fake_future).detach().cpu()))
                butterfly.append(float(self.butterfly_arbitrage_penalty(fake_future).detach().cpu()))
        self.G.train()
        return {
            "val_recon": float(np.mean(recon)) if recon else 0.0,
            "val_calendar": float(np.mean(calendar)) if calendar else 0.0,
            "val_butterfly": float(np.mean(butterfly)) if butterfly else 0.0,
        }

    def _init_metrics_file(self):
        self._metrics_file = os.path.join(self.metrics_path, "training_metrics.json")
        self._metrics_rows = []
        with open(self._metrics_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    def _append_metrics_row(self, row):
        self._metrics_rows.append(row)
        with open(self._metrics_file, "w", encoding="utf-8") as f:
            json.dump(self._metrics_rows, f, indent=2, ensure_ascii=False)

    def _write_metrics(self, metrics_rows):
        output_file = os.path.join(self.metrics_path, "training_metrics.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, indent=2, ensure_ascii=False)

    def save_model(self, epoch: Optional[int] = None):
        suffix = f"_epoch_{epoch:04d}" if epoch is not None else ""
        generator_path = os.path.join(self.model_path, f"generator{suffix}.pt")
        discriminator_path = os.path.join(self.model_path, f"discriminator{suffix}.pt")

        torch.save(
            {
                "state_dict": self.G.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.embedding_dim,
            },
            generator_path,
        )
        torch.save(
            {
                "state_dict": self.D.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.embedding_dim,
            },
            discriminator_path,
        )

    def _save_loss_curves(self, metrics_rows, logger) -> None:
        output_path = os.path.join(self.metrics_path, "loss_curves.png")
        plot_training_curves(
            metrics_rows,
            title="WGAN Vol Training Loss Curves",
            metric_groups=(
                ("Primary losses", ("g_recon", "val_recon", "g_total", "d_total", "gp")),
                (
                    "Constraint losses",
                    ("g_calendar", "g_butterfly", "g_smooth", "val_calendar", "val_butterfly"),
                ),
            ),
            output_path=output_path,
        )
        logger.info("Loss curve plot saved to: %s", output_path)

    def train(self, train_loader, val_loader=None):
        import logging
        logger = logging.getLogger("wgan_option.trainer")

        self._init_metrics_file()

        metrics_rows = []
        num_batches = len(train_loader)
        for epoch in range(1, self.num_epochs + 1):
            running: Dict[str, list] = {}
            for batch_idx, (current_surface, text_embedding, real_future) in enumerate(train_loader, 1):
                current_surface = self._to_device(current_surface)
                text_embedding = self._to_device(text_embedding)
                real_future = self._to_device(real_future)

                d_stats = {}
                for _ in range(self.critic_iter):
                    d_stats = self._discriminator_step(current_surface, text_embedding, real_future)
                    for key, value in d_stats.items():
                        running.setdefault(key, []).append(value)

                g_stats = self._generator_step(current_surface, text_embedding, real_future)
                for key, value in g_stats.items():
                    running.setdefault(key, []).append(value)

                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches:
                    logger.info(
                        "[Epoch %04d/%04d] Batch %d/%d  D=%.4f G=%.4f",
                        epoch, self.num_epochs, batch_idx, num_batches,
                        d_stats.get("d_total", 0.0), g_stats.get("g_total", 0.0),
                    )

            epoch_stats = {
                "epoch": epoch,
                **{key: float(np.mean(values)) for key, values in running.items() if values},
            }
            epoch_stats.update(self._evaluate(val_loader))
            metrics_rows.append(epoch_stats)
            self._append_metrics_row(epoch_stats)

            val_info = ""
            if "val_recon" in epoch_stats:
                val_info = f" ValRecon={epoch_stats['val_recon']:.4f}"

            logger.info(
                "[Epoch %04d/%04d] D=%.4f G=%.4f Recon=%.4f Cal=%.4f Bfly=%.4f%s",
                epoch, self.num_epochs,
                epoch_stats.get("d_total", 0.0),
                epoch_stats.get("g_total", 0.0),
                epoch_stats.get("g_recon", 0.0),
                epoch_stats.get("g_calendar", 0.0),
                epoch_stats.get("g_butterfly", 0.0),
                val_info,
            )

            if epoch % self.config.save_every == 0:
                self.save_model(epoch)
                logger.info("Checkpoint saved at epoch %d", epoch)

        self.save_model()
        self._write_metrics(metrics_rows)
        try:
            self._save_loss_curves(metrics_rows, logger)
        except Exception as exc:
            logger.warning("Failed to save loss curve plot: %s", exc)
