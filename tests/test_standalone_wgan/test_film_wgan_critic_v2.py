import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import FilmWGANTrainConfig, _validate_train_fields  # noqa: E402
from film_wgan.losses import gradient_penalty  # noqa: E402
from film_wgan.models import FilmWGANCritic, FilmWGANGenerator  # noqa: E402


def _critic() -> FilmWGANCritic:
    return FilmWGANCritic(
        surface_height=4,
        surface_width=4,
        embedding_dim=3,
        base_channels=4,
        res_blocks=1,
        text_hidden_dim=8,
        text_out_dim=4,
        fusion_hidden_dim=16,
        conditioning_mode="residual_film",
        critic_conditioning_mode="transition_matching",
        text_dropout=0.0,
        current_surface_channels=2,
        matching_logit_scale=4.0,
    )


class TestTransitionMatchingCritic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.critic = _critic().train()
        self.current = torch.randn(3, 2, 4, 4)
        self.future = torch.randn(3, 1, 4, 4)
        self.transition = torch.randn(3, 1, 4, 4)
        self.text = torch.randn(3, 3)
        self.support = torch.tensor(
            [
                [[[1.0, 1.0, 0.0, 0.0]] * 4],
                [[[1.0, 0.0, 1.0, 0.0]] * 4],
                [[[1.0, 1.0, 1.0, 1.0]] * 4],
            ]
        ).reshape(3, 1, 4, 4)

    def test_train_mode_scores_are_bitwise_deterministic(self):
        first_adv = self.critic(
            self.future * self.support,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        first_match = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        second_adv = self.critic(
            self.future * self.support,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        second_match = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        self.assertTrue(torch.equal(first_adv, second_adv))
        self.assertTrue(torch.equal(first_match, second_match))

    def test_wasserstein_head_is_text_independent(self):
        first = self.critic(
            self.future * self.support,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        second = self.critic(
            self.future * self.support,
            self.current,
            self.text.flip(0) * 100.0,
            has_text=torch.zeros(3),
            support_mask=self.support,
        )
        self.assertTrue(torch.equal(first, second))

    def test_wasserstein_head_ignores_unsupported_future_values_and_gradients(self):
        future = self.future.detach().clone().requires_grad_(True)
        baseline = self.critic(
            future,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        gradients = torch.autograd.grad(baseline.sum(), future)[0]

        perturbed = future.detach().clone()
        perturbed[self.support == 0.0] = 1.0e6
        changed = self.critic(
            perturbed,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )

        self.assertTrue(torch.equal(baseline.detach(), changed))
        self.assertEqual(
            float(torch.max(torch.abs(gradients[self.support == 0.0]))),
            0.0,
        )
        self.assertGreater(
            float(torch.max(torch.abs(gradients[self.support > 0.0]))),
            0.0,
        )

    def test_matching_logits_are_bounded_and_masked_by_has_text(self):
        logits = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        masked = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.zeros(3),
        )
        self.assertLessEqual(float(torch.max(torch.abs(logits))), 4.0 + 1e-6)
        self.assertTrue(torch.equal(masked, torch.zeros_like(masked)))

    def test_matching_ignores_unsupported_transition_values(self):
        perturbed = self.transition.clone()
        perturbed[self.support == 0.0] = 1.0e6
        first = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        second = self.critic.matching_logits(
            perturbed,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        self.assertTrue(torch.equal(first, second))

    def test_matching_future_gradient_is_zero_outside_support(self):
        transition = self.transition.detach().clone().requires_grad_(True)
        logits = self.critic.matching_logits(
            transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        gradients = torch.autograd.grad(logits.sum(), transition)[0]
        self.assertEqual(float(torch.max(torch.abs(gradients[self.support == 0.0]))), 0.0)
        self.assertGreater(float(torch.max(torch.abs(gradients[self.support > 0.0]))), 0.0)

    def test_gp_does_not_depend_on_matching_head_parameters(self):
        real = self.future.detach().clone()
        fake = real.clone()
        baseline = gradient_penalty(
            critic=self.critic,
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=self.current,
            text_embedding=self.text,
            lambda_gp=2.0,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        with torch.no_grad():
            for parameter in self.critic.transition_encoder.parameters():
                parameter.add_(torch.randn_like(parameter) * 100.0)
            for parameter in self.critic.transition_projection.parameters():
                parameter.add_(torch.randn_like(parameter) * 100.0)
        changed = gradient_penalty(
            critic=self.critic,
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=self.current,
            text_embedding=self.text,
            lambda_gp=2.0,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        self.assertTrue(torch.equal(baseline, changed))

    def test_adversarial_and_matching_parameter_groups_are_isolated(self):
        adv_before = self.critic(
            self.future,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        with torch.no_grad():
            for module in (
                self.critic.transition_encoder,
                self.critic.transition_projection,
                self.critic.text_encoder,
            ):
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter) * 10.0)
        adv_after = self.critic(
            self.future,
            self.current,
            self.text,
            has_text=torch.ones(3),
            support_mask=self.support,
        )
        self.assertTrue(torch.equal(adv_before, adv_after))

        match_before = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        with torch.no_grad():
            for module in (
                self.critic.conv1,
                self.critic.conv2,
                self.critic.conv3,
                self.critic.classifier,
            ):
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter) * 10.0)
        match_after = self.critic.matching_logits(
            self.transition,
            self.text,
            support_mask=self.support,
            has_text=torch.ones(3),
        )
        self.assertTrue(torch.equal(match_before, match_after))


class TestV2Configuration(unittest.TestCase):
    @staticmethod
    def _config() -> FilmWGANTrainConfig:
        return FilmWGANTrainConfig(
            conditioning_mode="residual_film",
            critic_conditioning_mode="transition_matching",
            normalize_current_surface=True,
            normalize_target_delta=True,
            critic_text_dropout=0.0,
            gradient_penalty_mode="support_masked",
            lambda_mismatch=0.0,
            lambda_critic_matching=0.1,
            lambda_generator_matching=0.01,
        )

    def test_v2_config_accepts_independent_generator_dropout(self):
        config = replace(self._config(), text_dropout=0.3)
        _validate_train_fields(config)
        self.assertEqual(config.text_dropout, 0.3)
        self.assertEqual(config.critic_text_dropout, 0.0)

    def test_v2_rejects_stochastic_critic(self):
        with self.assertRaisesRegex(ValueError, "critic_text_dropout=0"):
            _validate_train_fields(replace(self._config(), critic_text_dropout=0.1))

    def test_v2_requires_support_masked_gp(self):
        with self.assertRaisesRegex(ValueError, "gradient_penalty_mode=support_masked"):
            _validate_train_fields(
                replace(self._config(), gradient_penalty_mode="legacy_full_grid")
            )

    def test_generator_matching_requires_a_trained_matching_head(self):
        with self.assertRaisesRegex(
            ValueError,
            "lambda_generator_matching requires a positive",
        ):
            _validate_train_fields(
                replace(self._config(), lambda_critic_matching=0.0)
            )

    def test_deterministic_forecast_rejects_transition_matching_critic(self):
        with self.assertRaisesRegex(ValueError, "does not construct a critic"):
            _validate_train_fields(
                replace(
                    self._config(),
                    forecast_mode="deterministic",
                    lambda_adv=0.0,
                )
            )

    def test_generator_can_retain_dropout_when_critic_has_none(self):
        generator = FilmWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=3,
            noise_dim=2,
            base_channels=4,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=16,
            conditioning_mode="residual_film",
            text_dropout=0.3,
        )
        generator_dropouts = [
            module for module in generator.modules() if isinstance(module, torch.nn.Dropout)
        ]
        critic_dropouts = [
            module for module in _critic().modules() if isinstance(module, torch.nn.Dropout)
        ]
        self.assertTrue(any(module.p == 0.3 for module in generator_dropouts))
        self.assertTrue(critic_dropouts)
        self.assertTrue(all(module.p == 0.0 for module in critic_dropouts))


if __name__ == "__main__":
    unittest.main()
