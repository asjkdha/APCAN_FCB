import os
import re
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io


def _normalize_with_range(array, vmin, denom):
    normalized = np.clip((array.astype(np.float32) - vmin) / denom, 0, 1)
    return normalized.astype(np.float32)


def ensure_channel_first_gt(gt):
    gt = np.asarray(gt, dtype=np.float32)
    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=0)
    return gt.astype(np.float32)


def shared_percentile_normalize(input_9_frames, wide_field, gt):
    input_9_frames = np.asarray(input_9_frames, dtype=np.float32)
    wide_field = np.asarray(wide_field, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    # Use the raw SIM frames as the shared range so raw and GT are not
    # normalized independently, which would hide the real intensity mapping.
    vmin = np.percentile(input_9_frames, 0.1)
    vmax = np.percentile(input_9_frames, 99.9)
    denom = max(vmax - vmin, 1e-8)

    input_9_frames = _normalize_with_range(input_9_frames, vmin, denom)
    wide_field = _normalize_with_range(wide_field, vmin, denom)
    gt = ensure_channel_first_gt(_normalize_with_range(gt, vmin, denom))
    return input_9_frames, wide_field, gt


class SIMDataset:
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)
        return parser

    def __init__(self, opt, category):
        super(SIMDataset, self).__init__()
        self.images_path = []
        self.root = opt.root
        self.mode = category
        if category == 'train':
            inputs = os.path.join(opt.root, 'training')
        elif category == 'valid':
            inputs = os.path.join(opt.root, 'validate')
        else:
            raise ValueError('Unsupported dataset mode: {}'.format(category))
        images_input_path = glob.glob(inputs + '/*')
        self.images_path.extend(images_input_path)
        self.scale = opt.scale
        self.task = opt.task
        self.nch_in = opt.nch_in
        self.nch_out = opt.nch_out
        self.data_norm = opt.data_norm
        self.out = opt.out
        self.model = opt.model
        self.category = category
        self.gt_mapping_mode = getattr(opt, 'gt_mapping_mode', 'grouped')
        self.gt_group_size = getattr(opt, 'gt_group_size', 12)
        self.filename_digits = getattr(opt, 'filename_digits', 8)
        self.raw_index_start = getattr(opt, 'raw_index_start', 1)
        self.gt_index_start = getattr(opt, 'gt_index_start', 1)
        if category == 'valid':
            self.images_path = np.random.choice(self.images_path, size=opt.ntest)
        self.len = len(self.images_path)

    def get_gt_path(self, raw_folder_path):
        raw_name = os.path.basename(os.path.normpath(raw_folder_path))
        raw_idx = int(raw_name)

        if self.gt_mapping_mode == 'one_to_one':
            gt_idx = raw_idx
        elif self.gt_mapping_mode == 'grouped':
            gt_idx = ((raw_idx - self.raw_index_start) // self.gt_group_size) + self.gt_index_start
        else:
            raise ValueError('Unsupported gt_mapping_mode: {}'.format(self.gt_mapping_mode))

        gt_name = '{:0{}d}.tif'.format(gt_idx, self.filename_digits)
        if self.mode == 'train':
            gt_dir = os.path.join(self.root, 'training_gt')
        elif self.mode in ['val', 'valid', 'validate']:
            gt_dir = os.path.join(self.root, 'validate_gt')
        else:
            raise ValueError('Unsupported dataset mode: {}'.format(self.mode))

        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                'GT file not found: {}. raw_folder={}, gt_mapping_mode={}, gt_group_size={}'.format(
                    gt_path, raw_folder_path, self.gt_mapping_mode, self.gt_group_size
                )
            )
        return gt_path

    def __getitem__(self, index):
        img_path = glob.glob(self.images_path[index] + '/*.tif')
        img_path = sorted(img_path, key=lambda name: int(re.findall(r"\d+\d*", name)[-1]))
        stack = []
        for image_path in img_path:
            stack.append(io.imread(image_path))
        stack = np.array(stack).astype('float32')
        gt_path = self.get_gt_path(self.images_path[index])
        gt = io.imread(gt_path).astype('float32')
        input_9_frames = stack[:self.nch_in]
        if input_9_frames.shape[0] != self.nch_in:
            raise ValueError(
                'Expected {} SIM input frames, but got {} in {}'.format(
                    self.nch_in, input_9_frames.shape[0], self.images_path[index]
                )
            )
        if self.model == 'srcnn':
            inputs = []
            w, h = input_9_frames[0].shape
            for i in range(len(input_9_frames)):
                inputs.append(
                    np.array(Image.fromarray(input_9_frames[i]).resize((h * 2, w * 2), resample=Image.BICUBIC)))
            wide_field = np.mean(inputs, axis=0)
        else:
            wide_field = np.mean(input_9_frames, 0)

        # normalise
        if self.data_norm == 'minmax':
            input_9_frames, wide_field, gt = shared_percentile_normalize(input_9_frames, wide_field, gt)
        else:
            input_9_frames = input_9_frames.astype(np.float32)
            wide_field = wide_field.astype(np.float32)
            if self.nch_out == 1:
                gt = ensure_channel_first_gt(gt)
            else:
                gt = gt.astype(np.float32)
        input_9_frames = torch.from_numpy(input_9_frames).float()
        wide_field = torch.from_numpy(wide_field).unsqueeze(0).float()
        gt = torch.from_numpy(gt).float()
        return {'sim_inputs': input_9_frames, 'sim_gt': gt, 'wf': wide_field}

    def __len__(self):
        return self.len
