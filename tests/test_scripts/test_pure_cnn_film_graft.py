"""Contracts for the branch-local Pure-CNN to FiLM graft artifact."""

from __future__ import annotations

import inspect
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wgan_option.models.common import (
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator
from wgan_option.utils import pure_cnn_film_graft as graft_module
from wgan_option.utils.pure_cnn_film_graft import (
    GRAFT_STATE_KIND,
    PureCnnFilmGraftError,
    apply_pure_cnn_graft_state,
    graft_pure_cnn_fresh_restart,
    graft_pure_cnn_to_film,
    load_pure_cnn_to_film_graft_state,
    read_pure_cnn_graft_metadata,
    save_pure_cnn_to_film_graft_state,
    sha256_file,
    verify_graft_text_invariance,
)


class PureCnnFilmGraftTests(unittest.TestCase):
    embedding_dim = 5
    noise_dim = 3

    @staticmethod
    def _generator(mode: str) -> Generator:
        return Generator(
            channels=1,
            embedding_dim=PureCnnFilmGraftTests.embedding_dim,
            noise_dim=PureCnnFilmGraftTests.noise_dim,
            surface_height=8,
            surface_width=8,
            base_channels=2,
            res_blocks=0,
            text_hidden_dim=4,
            text_out_dim=3,
            hidden_dim=7,
            residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
            generator_current_input_mode=(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
            generator_conditioning_mode=mode,
            strike_grid=np.linspace(0.97, 1.03, 8, dtype=np.float32),
            maturity_grid_days=np.asarray(
                [1, 2, 3, 6, 10, 17, 26, 38], dtype=np.float32
            ),
        )

    @staticmethod
    def _critic() -> Discriminator:
        return Discriminator(
            channels=1,
            embedding_dim=PureCnnFilmGraftTests.embedding_dim,
            surface_height=8,
            surface_width=8,
            base_channels=2,
            res_blocks=0,
            text_hidden_dim=3,
            hidden_dim=7,
            critic_conditioning_mode=(LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE),
        )

    @classmethod
    def _models(
        cls, target_mode: str = FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
    ) -> tuple[Generator, Generator, Discriminator, Discriminator]:
        torch.manual_seed(11)
        source_generator = cls._generator(
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        )
        source_critic = cls._critic()
        with torch.no_grad():
            for parameter in source_generator.parameters():
                parameter.uniform_(-0.08, 0.08)
            for parameter in source_critic.parameters():
                parameter.uniform_(-0.08, 0.08)
        torch.manual_seed(29)
        target_generator = cls._generator(target_mode)
        target_critic = cls._critic()
        return source_generator, target_generator, source_critic, target_critic

    @classmethod
    def _probe(cls) -> dict[str, object]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(101)
        current = torch.rand((2, 1, 8, 8), generator=generator) + 0.1
        mask = torch.ones_like(current)
        mask[:, :, 0, 0] = 0.0
        noise = torch.randn((2, cls.noise_dim), generator=generator)
        matched = torch.randn((2, cls.embedding_dim), generator=generator)
        shuffle = matched.flip(0).clone()
        return {
            "current_surface": current,
            "current_support_mask": mask,
            "noise": noise,
            "text_embeddings": {
                "zero": torch.zeros_like(matched),
                "matched": matched,
                "shuffle": shuffle,
            },
        }

    @staticmethod
    def _lineages() -> tuple[dict[str, object], dict[str, object]]:
        parent = {
            "fold": "f1_2023q1",
            "seed": 42,
            "parent_job_id": "parent-f1-seed42",
            "parent_model_contract_sha256": "a" * 64,
        }
        graft = {
            "fold": "f1_2023q1",
            "seed": 42,
            "arm": "lp_matched",
            "graft_model_contract_sha256": "b" * 64,
        }
        return parent, graft

    def _save_fixture(self, root: Path) -> tuple[dict[str, object], tuple[object, ...]]:
        source_g, target_g, source_d, target_d = self._models()
        parent = root / "parent.pt"
        torch.save(
            {
                "generator_state_dict": source_g.state_dict(),
                "discriminator_state_dict": source_d.state_dict(),
            },
            parent,
        )
        generator_optimizer = Adam(target_g.parameters(), lr=5.0e-7)
        discriminator_optimizer = Adam(target_d.parameters(), lr=5.0e-7)
        generator_scheduler = ReduceLROnPlateau(generator_optimizer)
        discriminator_scheduler = ReduceLROnPlateau(discriminator_optimizer)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(2026)
        parent_lineage, graft_lineage = self._lineages()
        probe = self._probe()
        result = save_pure_cnn_to_film_graft_state(
            artifact_path=root / "graft.pt",
            manifest_path=root / "graft.json",
            parent_checkpoint_path=parent,
            expected_parent_checkpoint_sha256=sha256_file(parent),
            pure_cnn_generator=source_g,
            target_generator=target_g,
            pure_cnn_critic=source_d,
            target_discriminator=target_d,
            generator_optimizer=generator_optimizer,
            critic_optimizer=discriminator_optimizer,
            generator_scheduler=generator_scheduler,
            critic_scheduler=discriminator_scheduler,
            loader_generator=loader_generator,
            parent_lineage=parent_lineage,
            graft_lineage=graft_lineage,
            **probe,
        )
        context = (
            source_g,
            target_g,
            source_d,
            target_d,
            loader_generator,
            parent_lineage,
            graft_lineage,
            probe,
        )
        return result, context

    def test_strict_graft_copies_only_shared_generator_and_complete_critic(
        self,
    ) -> None:
        source_g, target_g, source_d, target_d = self._models()
        plan = graft_pure_cnn_to_film(
            pure_cnn_generator=source_g,
            film_generator=target_g,
            pure_cnn_critic=source_d,
            film_critic=target_d,
        )
        self.assertEqual(
            plan["target_generator_mode"],
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        )
        self.assertTrue(plan["copied_generator_keys"])
        self.assertTrue(plan["new_generator_keys"])
        self.assertTrue(plan["film_projection_keys"])
        source_state = source_g.state_dict()
        target_state = target_g.state_dict()
        for name in plan["copied_generator_keys"]:
            torch.testing.assert_close(target_state[name], source_state[name])
        for name, value in target_d.state_dict().items():
            torch.testing.assert_close(value, source_d.state_dict()[name])

        verification = verify_graft_text_invariance(
            pure_cnn_generator=source_g,
            film_generator=target_g,
            **self._probe(),
        )
        self.assertTrue(verification["passed"])
        self.assertLessEqual(verification["maximum_absolute_error"], 1.0e-7)

    def test_nonzero_film_projection_and_shape_drift_are_rejected(self) -> None:
        source_g, target_g, source_d, target_d = self._models()
        with torch.no_grad():
            target_g.encoder_film_layers[0].projection.weight[0, 0] = 1.0
        with self.assertRaisesRegex(PureCnnFilmGraftError, "exactly zero"):
            graft_pure_cnn_to_film(
                pure_cnn_generator=source_g,
                film_generator=target_g,
                pure_cnn_critic=source_d,
                film_critic=target_d,
            )

        mismatched = Generator(
            channels=1,
            embedding_dim=self.embedding_dim,
            noise_dim=self.noise_dim,
            surface_height=8,
            surface_width=8,
            base_channels=4,
            res_blocks=0,
            text_hidden_dim=4,
            text_out_dim=3,
            hidden_dim=7,
            residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
            generator_current_input_mode=(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
            generator_conditioning_mode=(
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ),
            strike_grid=np.linspace(0.97, 1.03, 8, dtype=np.float32),
            maturity_grid_days=np.asarray(
                [1, 2, 3, 6, 10, 17, 26, 38], dtype=np.float32
            ),
        )
        with self.assertRaisesRegex(PureCnnFilmGraftError, "shape mismatch"):
            graft_pure_cnn_to_film(
                pure_cnn_generator=source_g,
                film_generator=mismatched,
                pure_cnn_critic=source_d,
                film_critic=self._critic(),
            )

    def test_identity_pure_cnn_restart_has_no_new_keys(self) -> None:
        source_g, target_g, source_d, target_d = self._models(
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        )
        plan = graft_pure_cnn_fresh_restart(
            pure_cnn_generator=source_g,
            target_generator=target_g,
            pure_cnn_critic=source_d,
            target_critic=target_d,
        )
        self.assertEqual(
            plan["target_generator_mode"],
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        )
        self.assertEqual(plan["new_generator_keys"], ())
        self.assertEqual(plan["film_projection_keys"], ())
        verification = verify_graft_text_invariance(
            pure_cnn_generator=source_g,
            film_generator=target_g,
            **self._probe(),
        )
        self.assertLessEqual(verification["maximum_absolute_error"], 1.0e-7)

    def test_cuda_rng_state_is_projected_to_worker_visible_topology(self) -> None:
        loader = torch.Generator(device="cpu").manual_seed(17)
        first = torch.tensor([1, 2, 3], dtype=torch.uint8)
        second = torch.tensor([4, 5, 6], dtype=torch.uint8)
        captured = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [first, second],
            "loader_generator": loader.get_state(),
        }
        with mock.patch.object(
            graft_module, "capture_rng_state", return_value=captured
        ):
            state, topology = graft_module._capture_graft_rng_state(
                loader,
                expected_cuda_device_count=1,
                cuda_source_device_index=1,
            )
            self.assertEqual(len(state["torch_cuda"]), 1)
            torch.testing.assert_close(state["torch_cuda"][0], second)
            self.assertEqual(
                topology,
                {
                    "source_cuda_device_count": 2,
                    "source_cuda_device_index": 1,
                    "worker_cuda_device_count": 1,
                },
            )
            with self.assertRaisesRegex(PureCnnFilmGraftError, "unavailable"):
                graft_module._capture_graft_rng_state(
                    loader,
                    expected_cuda_device_count=1,
                    cuda_source_device_index=2,
                )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            self.assertRaisesRegex(ValueError, "device count differs"),
        ):
            graft_module._restore_graft_rng_state(
                state, loader_generator=torch.Generator().manual_seed(3)
            )

    def test_artifact_is_hash_bound_and_apply_never_accepts_optimizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, context = self._save_fixture(root)
            (
                source_g,
                _,
                source_d,
                _,
                saved_loader,
                parent_lineage,
                graft_lineage,
                probe,
            ) = context
            payload = torch.load(
                root / "graft.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["kind"], GRAFT_STATE_KIND)
            self.assertEqual(
                set(payload["rng_state"]),
                {
                    "python",
                    "numpy",
                    "torch_cpu",
                    "torch_cuda",
                    "dataloader_generator",
                },
            )
            self.assertEqual(payload["generator_optimizer_state_dict"]["state"], {})
            self.assertEqual(payload["discriminator_optimizer_state_dict"]["state"], {})
            self.assertTrue(payload["epoch0_equivalence"]["passed"])
            self.assertFalse(
                payload["optimizer_reset_proof"][
                    "apply_loads_optimizer_or_scheduler_state"
                ]
            )
            metadata = read_pure_cnn_graft_metadata(
                artifact_path=result["artifact_path"],
                expected_artifact_sha256=result["artifact_sha256"],
            )
            self.assertEqual(metadata["parent_lineage"], parent_lineage)
            self.assertEqual(metadata["graft_lineage"], graft_lineage)
            self.assertEqual(
                metadata["target_generator_mode"],
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            )
            signature = inspect.signature(load_pure_cnn_to_film_graft_state)
            self.assertNotIn("generator_optimizer", signature.parameters)
            self.assertNotIn("critic_optimizer", signature.parameters)
            self.assertNotIn("generator_scheduler", signature.parameters)
            self.assertNotIn("critic_scheduler", signature.parameters)

            _, loaded_g, _, loaded_d = self._models()
            with torch.no_grad():
                for parameter in loaded_g.parameters():
                    parameter.fill_(0.25)
                for parameter in loaded_d.parameters():
                    parameter.fill_(0.25)
            expected_loader_state = saved_loader.get_state().clone()
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored_loader = torch.Generator(device="cpu")
            restored_loader.manual_seed(999)
            loaded = load_pure_cnn_to_film_graft_state(
                artifact_path=result["artifact_path"],
                manifest_path=result["manifest_path"],
                expected_manifest_sha256=result["manifest_sha256"],
                expected_parent_lineage=parent_lineage,
                expected_graft_lineage=graft_lineage,
                target_generator=loaded_g,
                target_discriminator=loaded_d,
                loader_generator=restored_loader,
                restore_rng=True,
            )
            self.assertTrue(loaded["rng_restored"])
            torch.testing.assert_close(
                restored_loader.get_state(), expected_loader_state
            )
            verify_graft_text_invariance(
                pure_cnn_generator=source_g,
                film_generator=loaded_g,
                **probe,
            )
            for name, value in loaded_d.state_dict().items():
                torch.testing.assert_close(value, source_d.state_dict()[name])

            _, direct_g, _, direct_d = self._models()
            direct = apply_pure_cnn_graft_state(
                artifact_path=result["artifact_path"],
                expected_artifact_sha256=result["artifact_sha256"],
                expected_parent_lineage=parent_lineage,
                expected_graft_lineage=graft_lineage,
                target_generator=direct_g,
                target_discriminator=direct_d,
            )
            self.assertEqual(direct["manifest_path"], "")
            self.assertFalse(direct["rng_restored"])
            verify_graft_text_invariance(
                pure_cnn_generator=source_g,
                film_generator=direct_g,
                **probe,
            )

            with self.assertRaisesRegex(PureCnnFilmGraftError, "lineage"):
                load_pure_cnn_to_film_graft_state(
                    artifact_path=result["artifact_path"],
                    manifest_path=result["manifest_path"],
                    expected_manifest_sha256=result["manifest_sha256"],
                    expected_parent_lineage=parent_lineage,
                    expected_graft_lineage={**graft_lineage, "arm": "wrong"},
                    target_generator=self._models()[1],
                    target_discriminator=self._models()[3],
                )

    def test_artifact_tamper_and_nonfresh_optimizer_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _ = self._save_fixture(root)
            manifest_bytes = (root / "graft.json").read_bytes()
            with (root / "graft.json").open("ab") as stream:
                stream.write(b"tamper")
            _, target_g, _, target_d = self._models()
            parent_lineage, graft_lineage = self._lineages()
            with self.assertRaisesRegex(PureCnnFilmGraftError, "manifest SHA256"):
                load_pure_cnn_to_film_graft_state(
                    artifact_path=result["artifact_path"],
                    manifest_path=result["manifest_path"],
                    expected_manifest_sha256=result["manifest_sha256"],
                    expected_parent_lineage=parent_lineage,
                    expected_graft_lineage=graft_lineage,
                    target_generator=target_g,
                    target_discriminator=target_d,
                )
            (root / "graft.json").write_bytes(manifest_bytes)
            with (root / "graft.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(PureCnnFilmGraftError, "size/SHA256"):
                load_pure_cnn_to_film_graft_state(
                    artifact_path=result["artifact_path"],
                    manifest_path=result["manifest_path"],
                    expected_manifest_sha256=result["manifest_sha256"],
                    expected_parent_lineage=parent_lineage,
                    expected_graft_lineage=graft_lineage,
                    target_generator=target_g,
                    target_discriminator=target_d,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_g, target_g, source_d, target_d = self._models()
            parent = root / "parent.pt"
            torch.save(
                {
                    "generator_state_dict": source_g.state_dict(),
                    "discriminator_state_dict": source_d.state_dict(),
                },
                parent,
            )
            generator_optimizer = Adam(target_g.parameters(), lr=5.0e-7)
            discriminator_optimizer = Adam(target_d.parameters(), lr=5.0e-7)
            generator_optimizer.state[next(iter(target_g.parameters()))]["step"] = (
                torch.tensor(1.0)
            )
            parent_lineage, graft_lineage = self._lineages()
            with self.assertRaisesRegex(PureCnnFilmGraftError, "must be fresh"):
                save_pure_cnn_to_film_graft_state(
                    artifact_path=root / "graft.pt",
                    manifest_path=root / "graft.json",
                    parent_checkpoint_path=parent,
                    expected_parent_checkpoint_sha256=sha256_file(parent),
                    pure_cnn_generator=source_g,
                    target_generator=target_g,
                    pure_cnn_critic=source_d,
                    target_discriminator=target_d,
                    generator_optimizer=generator_optimizer,
                    critic_optimizer=discriminator_optimizer,
                    generator_scheduler=ReduceLROnPlateau(generator_optimizer),
                    critic_scheduler=ReduceLROnPlateau(discriminator_optimizer),
                    loader_generator=torch.Generator().manual_seed(2),
                    parent_lineage=parent_lineage,
                    graft_lineage=graft_lineage,
                    **self._probe(),
                )


if __name__ == "__main__":
    unittest.main()
