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
    GradientPenaltyTerms,
    gradient_penalty,
    gradient_penalty_terms,
)
from film_wgan.models import (  # noqa: E402
    VOL_CEIL,
    VOL_FLOOR,
    reconstruct_future_surface,
    reconstruct_future_surface_terms,
)


class _LinearCritic(torch.nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, future, current, text, has_text=None, support_mask=None):
        del current, text, has_text, support_mask
        return (future * self.weights).flatten(start_dim=1).sum(dim=1, keepdim=True)


class TestGradientPenaltyTermsV3(unittest.TestCase):
    def test_terms_expose_raw_norms_and_zero_unsupported_gradient(self):
        real = torch.zeros(2, 1, 2, 2)
        fake = torch.ones_like(real)
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]).expand_as(real)
        weights = torch.tensor(
            [
                [[[1.0, 99.0], [99.0, 99.0]]],
                [[[2.0, -99.0], [99.0, -99.0]]],
            ]
        )
        terms = gradient_penalty_terms(
            critic=_LinearCritic(weights),
            real_future_surface=real,
            fake_future_surface=fake,
            current_surface=torch.zeros_like(real),
            text_embedding=torch.zeros(2, 3),
            lambda_gp=3.0,
            support_mask=support,
        )
        self.assertIsInstance(terms, GradientPenaltyTerms)
        torch.testing.assert_close(terms.raw_norms, torch.tensor([1.0, 2.0]))
        self.assertEqual(float(terms.unsupported_max_abs_gradient), 0.0)
        self.assertAlmostEqual(float(terms.unscaled_penalty), 0.5)
        self.assertAlmostEqual(float(terms.penalty), 1.5)

    def test_legacy_wrapper_returns_identical_scalar(self):
        inputs = dict(
            critic=_LinearCritic(torch.ones(1, 1, 2, 2)),
            real_future_surface=torch.zeros(1, 1, 2, 2),
            fake_future_surface=torch.ones(1, 1, 2, 2),
            current_surface=torch.zeros(1, 1, 2, 2),
            text_embedding=torch.zeros(1, 2),
            lambda_gp=2.0,
        )
        torch.manual_seed(7)
        structured = gradient_penalty_terms(**inputs)
        torch.manual_seed(7)
        scalar = gradient_penalty(**inputs)
        self.assertTrue(torch.equal(structured.penalty, scalar))


class TestReconstructionTermsV3(unittest.TestCase):
    def test_structured_reconstruction_preserves_exact_clamped_log(self):
        current = torch.ones(1, 3, requires_grad=True)
        delta = torch.tensor([[-20.0, 0.25, 20.0]], requires_grad=True)
        terms = reconstruct_future_surface_terms(current, delta)
        torch.testing.assert_close(
            terms.clamped_log_surface,
            torch.tensor([[math.log(VOL_FLOOR), 0.25, math.log(VOL_CEIL)]]),
        )
        self.assertEqual(terms.clipped_mask.tolist(), [[True, False, True]])
        torch.testing.assert_close(
            terms.surface,
            torch.tensor([[VOL_FLOOR, math.exp(0.25), VOL_CEIL]]),
        )
        torch.testing.assert_close(reconstruct_future_surface(current, delta), terms.surface)


if __name__ == "__main__":
    unittest.main()
