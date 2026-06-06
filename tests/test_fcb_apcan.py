import importlib
import os
import tempfile
import unittest
from argparse import Namespace

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


class TestScriptTests(unittest.TestCase):
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
