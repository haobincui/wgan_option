import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.losses import (  # noqa: E402
    critic_transition_matching_loss,
    generator_transition_matching_loss,
    gradient_penalty,
)


class _LinearFutureCritic(torch.nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(
        self,
        future_surface,
        current_surface,
        text_embedding,
        has_text=None,
        support_mask=None,
    ):
        del current_surface, text_embedding, has_text, support_mask
        return (future_surface * self.weights).flatten(start_dim=1).sum(dim=1, keepdim=True)


class _QuadraticFutureCritic(torch.nn.Module):
    def forward(
        self,
        future_surface,
        current_surface,
        text_embedding,
        has_text=None,
        support_mask=None,
    ):
        del current_surface, text_embedding, has_text, support_mask
        return future_surface.square().flatten(start_dim=1).sum(dim=1, keepdim=True)


class TestSupportAwareGradientPenalty(unittest.TestCase):
    @staticmethod
    def _inputs():
        real = torch.tensor([[[[0.1, -0.2], [0.3, -0.4]]]], dtype=torch.float32)
        fake = real.clone()
        current = torch.zeros(1, 1, 2, 2)
        text = torch.zeros(1, 2)
        return real, fake, current, text

    def test_gp_ignores_unsupported_critic_weights(self):
        real, fake, current, text = self._inputs()
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        first = gradient_penalty(
            critic=_LinearFutureCritic(torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])),
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=1.0,
            support_mask=support,
        )
        second = gradient_penalty(
            critic=_LinearFutureCritic(
                torch.tensor([[[[1.0, 1.0e6], [-1.0e6, 1.0e6]]]])
            ),
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=1.0,
            support_mask=support,
        )
        self.assertEqual(float(first), 0.0)
        self.assertEqual(float(second), 0.0)

    def test_gp_is_invariant_to_unsupported_real_and_fake_values(self):
        real, fake, current, text = self._inputs()
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        # A quadratic critic makes the input gradient value-dependent.  This
        # assertion would fail if unsupported interpolated values reached the
        # critic; the former linear fixture made that regression invisible.
        critic = _QuadraticFutureCritic()
        torch.manual_seed(123)
        baseline = gradient_penalty(
            critic=critic,
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=2.0,
            support_mask=support,
        )
        perturbed_real = real.clone()
        perturbed_fake = fake.clone()
        perturbed_real[0, 0, 0, 1:] = 1.0e6
        perturbed_real[0, 0, 1, :] = -1.0e6
        perturbed_fake[0, 0, 0, 1:] = -2.0e6
        perturbed_fake[0, 0, 1, :] = 2.0e6
        torch.manual_seed(123)
        perturbed = gradient_penalty(
            critic=critic,
            real_future_surface=perturbed_real,
            fake_future_surface=perturbed_fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=2.0,
            support_mask=support,
        )
        self.assertTrue(torch.equal(baseline, perturbed))

    def test_all_ones_support_matches_legacy_gp(self):
        real, fake, current, text = self._inputs()
        critic = _LinearFutureCritic(torch.full((1, 1, 2, 2), 0.5))
        legacy = gradient_penalty(
            critic=critic,
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=3.0,
        )
        supported = gradient_penalty(
            critic=critic,
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=current,
            text_embedding=text,
            lambda_gp=3.0,
            support_mask=torch.ones_like(real),
        )
        self.assertTrue(torch.equal(legacy, supported))

    def test_empty_support_is_rejected(self):
        real, fake, current, text = self._inputs()
        with self.assertRaisesRegex(ValueError, "at least one supported"):
            gradient_penalty(
                critic=_LinearFutureCritic(torch.ones(1, 1, 2, 2)),
                real_future_surface=real,
                fake_future_surface=fake,
                current_surface=current,
                text_embedding=text,
                lambda_gp=1.0,
                support_mask=torch.zeros_like(real),
            )


class TestBoundedMatchingLosses(unittest.TestCase):
    def test_zero_logits_have_log_two_loss(self):
        positive = torch.zeros(2)
        negative = torch.zeros(2, 2)
        self.assertAlmostEqual(
            float(critic_transition_matching_loss(positive, negative)),
            math.log(2.0),
            places=6,
        )
        self.assertAlmostEqual(
            float(generator_transition_matching_loss(positive, negative)),
            math.log(2.0),
            places=6,
        )

    def test_relative_generator_loss_rewards_positive_margin(self):
        good = generator_transition_matching_loss(
            torch.tensor([2.0]),
            torch.tensor([[-2.0, -1.0]]),
        )
        bad = generator_transition_matching_loss(
            torch.tensor([-2.0]),
            torch.tensor([[2.0, 1.0]]),
        )
        self.assertLess(float(good), float(bad))

    def test_critic_loss_matches_asymmetric_balanced_softplus_formula(self):
        positive = torch.tensor([2.0, -0.5])
        negative = torch.tensor([[-1.0, 0.25], [1.5, -2.0]])
        expected = 0.5 * (
            torch.nn.functional.softplus(-positive).mean()
            + torch.nn.functional.softplus(negative).mean()
        )
        actual = critic_transition_matching_loss(positive, negative)
        self.assertAlmostEqual(float(actual), float(expected), places=7)


if __name__ == "__main__":
    unittest.main()
