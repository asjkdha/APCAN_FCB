import os
import re
import torch
import csv
import glob
import argparse
import skimage
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models import get_model
from utils.util import load_checkpoint_flexible, prctile_norm

opt = argparse.Namespace()

opt.model = 'apcan_1_actin'
#opt.model = 'apcan_3_actin'

# opt.model = 'apcan_1_er'
# opt.model = 'apcan_3_er'


opt.weights = 'pretrain/{}.pth'.format(opt.model)

# input/output layer options
opt.imageSize = 502
opt.scale = 2
opt.nch_in = 9
opt.nch_out = 1
opt.fcb_rows = 502
opt.fcb_cols = 502
opt.use_fcb = True
opt.fcb_init = 'he'
opt.fcb_alpha = 0.7
opt.fcb_gamma_init = 1e-3
opt.fcb_rho_min = 0.25
opt.fcb_tau = 0.05
opt.fcb_sigma_theta = 0.17453292519943295
opt.fcb_directions = '0,60,120'
opt.fcb_residual_scale = 1e-3

# architecture options
opt.narch = 0
opt.n_resblocks = 4
opt.n_resgroups = 4
opt.reduction = 16
opt.n_feats = 64

# test options
opt.test = False
opt.cpu = False
opt.batchSize = 1
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')


def numeric_sort_key(path):
    numbers = re.findall(r"\d+\d*", os.path.basename(path))
    return int(numbers[-1]) if numbers else os.path.basename(path)


def list_tif_files(folder):
    files = glob.glob(os.path.join(folder, '*.tif'))
    return sorted(files, key=numeric_sort_key)


def discover_test_groups(root):
    root_tifs = list_tif_files(root)
    if root_tifs:
        return [(os.path.basename(os.path.normpath(root)), root)]

    groups = []
    for folder in sorted(glob.glob(os.path.join(root, '*')), key=numeric_sort_key):
        if os.path.isdir(folder) and list_tif_files(folder):
            groups.append((os.path.basename(os.path.normpath(folder)), folder))
    if not groups:
        raise FileNotFoundError(f"No tif files found under {root} or its immediate subfolders.")
    return groups


def resolve_gt_path(gt_root, group_name, filename):
    if not gt_root:
        return None
    candidates = [
        os.path.join(gt_root, group_name, filename),
        os.path.join(gt_root, filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"GT file not found for {group_name}/{filename}. Tried: {candidates}"
    )


def compute_image_metrics(gt, sr):
    gt_norm = prctile_norm(np.squeeze(gt).astype(np.float32))
    sr_norm = prctile_norm(np.squeeze(sr).astype(np.float32))
    return (
        peak_signal_noise_ratio(gt_norm, sr_norm, data_range=1.0),
        structural_similarity(gt_norm, sr_norm, data_range=1.0),
    )


def extract_sim_frames(stack):
    if stack.ndim == 3 and stack.shape[0] >= 9:
        return stack[:9]
    if stack.ndim == 3 and stack.shape[-1] >= 9:
        return np.moveaxis(stack[..., :9], -1, 0)
    if stack.ndim == 4 and stack.shape[0] > 1 and stack.shape[1] >= 9:
        return stack[1, :9]
    raise ValueError(f"Unsupported stack shape: {stack.shape}")


def normalize_sim_frames(sim_frames):
    sim_frames = np.asarray(sim_frames, dtype=np.float32)
    vmin = np.percentile(sim_frames, 0.1)
    vmax = np.percentile(sim_frames, 99.9)
    denom = max(vmax - vmin, 1e-8)
    return np.clip((sim_frames - vmin) / denom, 0, 1).astype(np.float32)


def LoadModel(opt):
    print('Loading model')
    print(opt)
    net = get_model(opt)
    print('loading checkpoint', opt.weights)
    load_checkpoint_flexible(net, opt.weights, opt.device)
    return net


def SIM_reconstruct9(model, opt):
    def prepimg(stack):
        input_9_frames = normalize_sim_frames(stack[:9])
        input_9_frames = torch.from_numpy(input_9_frames).float()
        return input_9_frames

    os.makedirs('%s' % opt.out, exist_ok=True)
    groups = discover_test_groups(opt.root)
    metric_rows = []
    summary_rows = []

    for group_name, group_dir in groups:
        files = list_tif_files(group_dir)
        save_dir = os.path.join(opt.out, group_name) if len(groups) > 1 else opt.out
        os.makedirs(save_dir, exist_ok=True)
        group_psnrs = []
        group_ssims = []

        print(f"Testing group {group_name}: {len(files)} tif files")
        for iidx, imgfile in enumerate(files):
            print('[%d/%d] Reconstructing %s' % (iidx + 1, len(files), imgfile))
            basename = os.path.basename(imgfile)

            stack = io.imread(imgfile)

            sim_frames = extract_sim_frames(stack)

            sim_input = prepimg(sim_frames)
            sim_input = sim_input.unsqueeze(0)
            with torch.no_grad():
                sr = model(sim_input.to(opt.device))
                sr = torch.clamp(sr.cpu(), min=0, max=1)
            sr = np.uint16(sr.squeeze().numpy() * 65535)
            save_path = os.path.join(save_dir, '%s_%s.tif' % (basename[:-4], opt.model))
            skimage.io.imsave(save_path, sr)

            if getattr(opt, 'gt_root', ''):
                gt_path = resolve_gt_path(opt.gt_root, group_name, basename)
                gt = io.imread(gt_path)
                psnr, ssim = compute_image_metrics(gt, sr)
                group_psnrs.append(psnr)
                group_ssims.append(ssim)
                metric_rows.append({
                    'group': group_name,
                    'file': basename,
                    'psnr': psnr,
                    'ssim': ssim,
                })

        if group_psnrs:
            mean_psnr = float(np.mean(group_psnrs))
            mean_ssim = float(np.mean(group_ssims))
            summary_rows.append({'group': group_name, 'mean_psnr': mean_psnr, 'mean_ssim': mean_ssim})
            print(f"{group_name} mean PSNR: {mean_psnr:.4f}, mean SSIM: {mean_ssim:.4f}")

    if metric_rows:
        metrics_path = os.path.join(opt.out, 'metrics.csv')
        summary_path = os.path.join(opt.out, 'metrics_summary.csv')
        with open(metrics_path, 'w', newline='') as fid:
            writer = csv.DictWriter(fid, fieldnames=['group', 'file', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(metric_rows)
        with open(summary_path, 'w', newline='') as fid:
            writer = csv.DictWriter(fid, fieldnames=['group', 'mean_psnr', 'mean_ssim'])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved summary metrics to {summary_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./testing/actin', help='test root or a folder containing level_* subfolders')
    parser.add_argument('--out', type=str, default='./output', help='output folder for reconstructed images')
    parser.add_argument('--gt_root', type=str, default='', help='optional GT root for PSNR/SSIM; supports flat or mirrored level folders')
    parser.add_argument('--weights', type=str, default=opt.weights, help='checkpoint path')
    parser.add_argument('--model', type=str, default=opt.model)
    parser.add_argument('--imageSize', type=int, default=opt.imageSize)
    parser.add_argument('--scale', type=int, default=opt.scale)
    parser.add_argument('--nch_in', type=int, default=opt.nch_in)
    parser.add_argument('--nch_out', type=int, default=opt.nch_out)
    parser.add_argument('--fcb_rows', type=int, default=opt.fcb_rows)
    parser.add_argument('--fcb_cols', type=int, default=opt.fcb_cols)
    parser.add_argument('--fcb_init', type=str, default=opt.fcb_init, choices=['he', 'glorot'])
    parser.add_argument('--fcb_alpha', type=float, default=opt.fcb_alpha)
    parser.add_argument('--fcb_gamma_init', type=float, default=opt.fcb_gamma_init)
    parser.add_argument('--fcb_rho_min', type=float, default=opt.fcb_rho_min)
    parser.add_argument('--fcb_tau', type=float, default=opt.fcb_tau)
    parser.add_argument('--fcb_sigma_theta', type=float, default=opt.fcb_sigma_theta)
    parser.add_argument('--fcb_directions', type=str, default=opt.fcb_directions)
    parser.add_argument('--fcb_residual_scale', type=float, default=opt.fcb_residual_scale)
    parser.add_argument('--use_fcb', dest='use_fcb', action='store_true')
    parser.add_argument('--no_use_fcb', dest='use_fcb', action='store_false')
    parser.set_defaults(use_fcb=opt.use_fcb)
    parser.add_argument('--narch', type=int, default=opt.narch)
    parser.add_argument('--n_resblocks', type=int, default=opt.n_resblocks)
    parser.add_argument('--n_resgroups', type=int, default=opt.n_resgroups)
    parser.add_argument('--reduction', type=int, default=opt.reduction)
    parser.add_argument('--n_feats', type=int, default=opt.n_feats)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


if __name__ == '__main__':
    opt = build_arg_parser().parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    net = LoadModel(opt)

    SIM_reconstruct9(net, opt)
