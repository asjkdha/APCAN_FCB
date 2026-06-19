import importlib
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class FCBLayerTests(unittest.TestCase):
    def test_deepsparse_preserves_shape_and_rejects_wrong_size(self):
        from models.fcb import DeepSparse

        layer = DeepSparse(channels=2, num_rows=8, num_cols=10)
        x = torch.randn(3, 2, 8, 10)

        y = layer(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))

        with self.assertRaisesRegex(
            ValueError, r"FCB expects spatial size \(8, 10\), but got \(4, 10\)"
        ):
            layer(torch.randn(1, 2, 4, 10))

    def test_fcblayer_is_residual_when_branch_is_zero(self):
        from models.fcb import FCBLayer

        layer = FCBLayer(n_feat=2, num_rows=8, num_cols=10, act=nn.Identity())
        with torch.no_grad():
            layer.fourier.weights_real.zero_()
            layer.fourier.weights_imag.zero_()
            layer.pointwise.weight.zero_()
            layer.pointwise.bias.zero_()

        x = torch.randn(1, 2, 8, 10)

        self.assertTrue(torch.allclose(layer(x), x))


class APCANIntegrationTests(unittest.TestCase):
    def test_apcan_uses_fcb_and_keeps_scale_two_output_shape(self):
        from models.APCAN_1 import APCAN

        opt = Namespace(
            scale=2,
            nch_out=1,
            use_fcb=True,
            fcb_rows=16,
            fcb_cols=16,
            n_resgroups=1,
            n_resblocks=1,
            n_feats=4,
            reduction=2,
        )
        model = APCAN(opt)

        y = model(torch.randn(1, 9, 16, 16))

        self.assertEqual(tuple(y.shape), (1, 1, 32, 32))

        with self.assertRaisesRegex(
            ValueError, r"FCB expects spatial size \(16, 16\), but got \(8, 8\)"
        ):
            model(torch.randn(1, 9, 8, 8))

    def test_apcan_passes_fcb_options_to_layer(self):
        from models.APCAN_1 import APCAN
        from models.fcb import FCBLayer

        opt = Namespace(
            scale=2,
            nch_out=1,
            use_fcb=True,
            fcb_rows=16,
            fcb_cols=16,
            fcb_init="glorot",
            fcb_alpha=0.9,
            fcb_gamma_init=0.002,
            fcb_rho_min=0.35,
            fcb_tau=0.08,
            fcb_sigma_theta=0.2,
            fcb_directions="0,45,90",
            fcb_residual_scale=0.004,
            n_resgroups=1,
            n_resblocks=1,
            n_feats=4,
            reduction=2,
        )

        model = APCAN(opt)
        fcb_layer = next(module for module in model.modules() if isinstance(module, FCBLayer))
        fourier = fcb_layer.global_branch.fourier
        prior = fourier.freq_prior

        self.assertAlmostEqual(fourier.alpha, 0.9)
        self.assertAlmostEqual(
            float(torch.nn.functional.softplus(fourier.raw_gamma_rad).detach()), 0.002
        )
        self.assertAlmostEqual(
            float(torch.nn.functional.softplus(fourier.raw_gamma_sim).detach()), 0.002
        )
        self.assertAlmostEqual(float(fcb_layer.residual_scale.detach()), 0.004)
        self.assertAlmostEqual(prior.rho_min, 0.35)
        self.assertAlmostEqual(prior.tau, 0.08)
        self.assertAlmostEqual(prior.sigma_theta, 0.2)
        self.assertTrue(np.allclose(prior.directions, [0.0, np.pi / 4.0, np.pi / 2.0]))


class DataPreprocessTests(unittest.TestCase):
    def test_shared_percentile_normalize_uses_raw_range_and_adds_gt_channel(self):
        from data.sim_dataset import shared_percentile_normalize

        frames = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
        wide_field = frames.mean(axis=0)
        gt = np.full((4, 4), 1000, dtype=np.float32)
        vmin = np.percentile(frames, 0.1)
        vmax = np.percentile(frames, 99.9)
        denom = max(vmax - vmin, 1e-8)

        norm_frames, norm_wf, norm_gt = shared_percentile_normalize(
            frames, wide_field, gt
        )

        self.assertEqual(norm_frames.dtype, np.float32)
        self.assertEqual(norm_wf.dtype, np.float32)
        self.assertEqual(norm_gt.dtype, np.float32)
        self.assertEqual(norm_gt.shape, (1, 4, 4))
        self.assertTrue(
            np.allclose(norm_frames, np.clip((frames - vmin) / denom, 0, 1))
        )
        self.assertTrue(
            np.allclose(norm_wf, np.clip((wide_field - vmin) / denom, 0, 1))
        )
        self.assertTrue(np.all(norm_gt == 1))


class GTMappingTests(unittest.TestCase):
    def _make_opt(self, root, **overrides):
        values = dict(
            root=root,
            scale=2,
            task="simin_simout",
            nch_in=9,
            nch_out=1,
            data_norm="minmax",
            out="checkpoint",
            model="apcan_1_actin",
            ntest=1,
            gt_mapping_mode="grouped",
            gt_group_size=12,
            filename_digits=8,
            raw_index_start=1,
            gt_index_start=1,
        )
        values.update(overrides)
        return Namespace(**values)

    def _touch_gt(self, root, split, name):
        gt_dir = os.path.join(root, split)
        os.makedirs(gt_dir, exist_ok=True)
        path = os.path.join(gt_dir, name)
        with open(path, "wb"):
            pass
        return path

    def test_grouped_gt_mapping_uses_shared_gt_for_raw_groups(self):
        from data.sim_dataset import SIMDataset

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "training", "00000001"))
            os.makedirs(os.path.join(root, "training", "00000012"))
            os.makedirs(os.path.join(root, "training", "00000013"))
            gt1 = self._touch_gt(root, "training_gt", "00000001.tif")
            gt2 = self._touch_gt(root, "training_gt", "00000002.tif")

            dataset = SIMDataset(self._make_opt(root), "train")

            self.assertEqual(dataset.get_gt_path(os.path.join(root, "training", "00000001")), gt1)
            self.assertEqual(dataset.get_gt_path(os.path.join(root, "training", "00000012")), gt1)
            self.assertEqual(dataset.get_gt_path(os.path.join(root, "training", "00000013")), gt2)

    def test_one_to_one_gt_mapping_keeps_raw_index(self):
        from data.sim_dataset import SIMDataset

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "training", "00000013"))
            gt13 = self._touch_gt(root, "training_gt", "00000013.tif")

            dataset = SIMDataset(self._make_opt(root, gt_mapping_mode="one_to_one"), "train")

            self.assertEqual(dataset.get_gt_path(os.path.join(root, "training", "00000013")), gt13)

    def test_validate_uses_validate_gt_and_missing_gt_has_context(self):
        from data.sim_dataset import SIMDataset

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "validate", "00000013"))
            gt2 = self._touch_gt(root, "validate_gt", "00000002.tif")
            dataset = SIMDataset(self._make_opt(root), "valid")

            self.assertEqual(dataset.get_gt_path(os.path.join(root, "validate", "00000013")), gt2)

            with self.assertRaisesRegex(
                FileNotFoundError, r"GT file not found: .*gt_mapping_mode=grouped.*gt_group_size=12"
            ):
                dataset.get_gt_path(os.path.join(root, "validate", "00000025"))


class PresetTests(unittest.TestCase):
    def test_parser_exposes_fcb_control_options(self):
        from option.options import parser

        action_dests = {action.dest for action in parser._actions}

        for key in [
            "fcb_init",
            "fcb_alpha",
            "fcb_gamma_init",
            "fcb_rho_min",
            "fcb_tau",
            "fcb_sigma_theta",
            "fcb_directions",
            "fcb_residual_scale",
        ]:
            self.assertIn(key, action_dests)

    def test_apply_preset_sets_defaults_but_preserves_cli_overrides(self):
        from option.options import apply_preset, parser

        opt = parser.parse_args(["--preset", "fcb_small", "--n_feats", "64"])
        apply_preset(opt, {"preset", "n_feats"})

        self.assertTrue(opt.use_fcb)
        self.assertEqual(opt.root, "./dataset/F-actin")
        self.assertEqual(opt.n_resgroups, 2)
        self.assertEqual(opt.n_resblocks, 2)
        self.assertEqual(opt.n_feats, 64)
        self.assertEqual(opt.gt_mapping_mode, "grouped")

    def test_apcan_base_preset_disables_fcb(self):
        from option.options import apply_preset, parser

        opt = parser.parse_args(["--preset", "apcan_base"])
        apply_preset(opt, {"preset"})

        self.assertFalse(opt.use_fcb)
        self.assertEqual(opt.n_resgroups, 4)
        self.assertEqual(opt.n_resblocks, 4)
        self.assertEqual(opt.n_feats, 64)

    def test_train_options_applies_preset_and_cli_override(self):
        with patch.object(sys, "argv", ["train.py", "--preset", "fcb_debug", "--n_feats", "64"]):
            train_script = importlib.import_module("train")
            opt = train_script.options()

        self.assertTrue(opt.use_fcb)
        self.assertEqual(opt.batchSize, 1)
        self.assertEqual(opt.nepoch, 20)
        self.assertEqual(opt.n_resgroups, 1)
        self.assertEqual(opt.n_feats, 64)

    def test_train_options_restores_log_values_without_overriding_cli_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, "apcan_1_actin.txt")
            with open(logfile, "w", encoding="utf-8") as fid:
                fid.write(
                    "Namespace(model='apcan_1_actin', task='simin_simout', "
                    "nch_in=9, nch_out=1, n_resgroups=3, n_resblocks=3, "
                    "n_feats=48, reduction=8, use_fcb=True, fcb_rows=502, "
                    "fcb_cols=502, imageSize=502, scale=2, batchSize=2, "
                    "batchSize_test=1, gt_mapping_mode='one_to_one', "
                    "gt_group_size=6, filename_digits=8, raw_index_start=1, "
                    "gt_index_start=1, lr=0.0002, nepoch=5)"
                )

            with patch.object(sys, "argv", ["train.py", "--weights", tmpdir, "--n_feats", "64"]):
                train_script = importlib.import_module("train")
                opt = train_script.options()

        self.assertEqual(opt.n_resgroups, 3)
        self.assertEqual(opt.reduction, 8)
        self.assertEqual(opt.gt_mapping_mode, "one_to_one")
        self.assertTrue(opt.use_fcb)
        self.assertEqual(opt.n_feats, 64)


class TestScriptTests(unittest.TestCase):
    def test_test_arg_parser_exposes_fcb_control_options(self):
        test_script = importlib.import_module("test")
        parser = test_script.build_arg_parser()
        action_dests = {action.dest for action in parser._actions}

        for key in [
            "fcb_init",
            "fcb_alpha",
            "fcb_gamma_init",
            "fcb_rho_min",
            "fcb_tau",
            "fcb_sigma_theta",
            "fcb_directions",
            "fcb_residual_scale",
        ]:
            self.assertIn(key, action_dests)

    def test_extract_sim_frames_accepts_3d_and_4d_stacks(self):
        test_script = importlib.import_module("test")
        stack3 = np.arange(10 * 4 * 4, dtype=np.float32).reshape(10, 4, 4)
        stack4 = np.stack([stack3, stack3 + 100], axis=0)

        self.assertTrue(np.array_equal(test_script.extract_sim_frames(stack3), stack3[:9]))
        self.assertTrue(
            np.array_equal(test_script.extract_sim_frames(stack4), stack4[1, :9])
        )

        with self.assertRaisesRegex(ValueError, r"Unsupported stack shape: \(4, 4\)"):
            test_script.extract_sim_frames(np.zeros((4, 4), dtype=np.float32))

    def test_normalize_sim_frames_uses_shared_raw_percentiles(self):
        test_script = importlib.import_module("test")
        frames = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
        vmin = np.percentile(frames, 0.1)
        vmax = np.percentile(frames, 99.9)
        denom = max(vmax - vmin, 1e-8)

        norm_frames = test_script.normalize_sim_frames(frames)

        self.assertEqual(norm_frames.dtype, np.float32)
        self.assertTrue(np.allclose(norm_frames, np.clip((frames - vmin) / denom, 0, 1)))

    def test_discover_test_groups_finds_level_subdirectories(self):
        test_script = importlib.import_module("test")

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "level_02"))
            os.makedirs(os.path.join(root, "level_01"))
            open(os.path.join(root, "level_02", "002.tif"), "wb").close()
            open(os.path.join(root, "level_01", "001.tif"), "wb").close()

            groups = test_script.discover_test_groups(root)

        self.assertEqual([name for name, _ in groups], ["level_01", "level_02"])

    def test_discover_test_groups_treats_flat_tif_directory_as_one_group(self):
        test_script = importlib.import_module("test")

        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, "002.tif"), "wb").close()
            open(os.path.join(root, "001.tif"), "wb").close()

            groups = test_script.discover_test_groups(root)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], os.path.basename(root))
        self.assertEqual(groups[0][1], root)

    def test_resolve_gt_path_supports_mirrored_level_or_flat_gt_root(self):
        test_script = importlib.import_module("test")

        with tempfile.TemporaryDirectory() as gt_root:
            os.makedirs(os.path.join(gt_root, "level_12"))
            mirrored = os.path.join(gt_root, "level_12", "001.tif")
            flat = os.path.join(gt_root, "002.tif")
            open(mirrored, "wb").close()
            open(flat, "wb").close()

            self.assertEqual(
                test_script.resolve_gt_path(gt_root, "level_12", "001.tif"),
                mirrored,
            )
            self.assertEqual(
                test_script.resolve_gt_path(gt_root, "level_12", "002.tif"),
                flat,
            )

    def test_compute_image_metrics_is_high_for_identical_images(self):
        test_script = importlib.import_module("test")
        image = np.arange(64, dtype=np.float32).reshape(8, 8)

        psnr, ssim = test_script.compute_image_metrics(image, image)

        self.assertTrue(np.isinf(psnr) or psnr > 90)
        self.assertAlmostEqual(ssim, 1.0, places=6)


class CheckpointTests(unittest.TestCase):
    def test_load_checkpoint_flexible_strips_module_prefix(self):
        from utils.util import load_checkpoint_flexible

        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        state = {"module." + k: v.detach().clone() for k, v in source.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "module_state.pth")
            torch.save({"state_dict": state}, ckpt_path)
            load_checkpoint_flexible(target, ckpt_path, torch.device("cpu"))

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_load_checkpoint_flexible_adds_module_prefix(self):
        from utils.util import load_checkpoint_flexible

        source = nn.Linear(3, 2)
        target = nn.DataParallel(nn.Linear(3, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "plain_state.pth")
            torch.save({"model": source.state_dict()}, ckpt_path)
            load_checkpoint_flexible(target, ckpt_path, torch.device("cpu"))

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.module.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
