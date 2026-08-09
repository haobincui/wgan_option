import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.models import FilmWGANCritic  # noqa: E402


class TestMaskOnlyMatcherControlV3(unittest.TestCase):
    def test_zero_transition_has_zero_logits_while_retaining_real_support(self):
        torch.manual_seed(9)
        critic = FilmWGANCritic(
            surface_height=4,
            surface_width=4,
            embedding_dim=3,
            base_channels=4,
            res_blocks=0,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=8,
            conditioning_mode="residual_film",
            critic_conditioning_mode="transition_matching",
            text_dropout=0.0,
        )
        support = torch.tensor(
            [
                [[[1.0, 1.0, 0.0, 0.0]] * 4],
                [[[0.0, 0.0, 1.0, 1.0]] * 4],
            ]
        ).reshape(2, 1, 4, 4)
        logits = critic.matching_logits(
            torch.zeros_like(support),
            torch.randn(2, 3),
            support_mask=support,
            has_text=torch.ones(2),
        )
        torch.testing.assert_close(logits, torch.zeros_like(logits), atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
