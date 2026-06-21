import os
import re
import torch
import csv
import glob
import argparse
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
opt.save_metric_plots = True
opt.metric_plot_dir = 'metric_plots'
opt.metrics_by_level_dir = 'metrics_by_level'


METRICS_FIELDNAMES = [
    'group',
    'image_index',
    'file',
    'sr_path',
    'gt_path',
    'psnr',
    'ssim',
]
SUMMARY_FIELDNAMES = [
    'group',
    'num_images',
    'mean_psnr',
    'std_psnr',
    'median_psnr',
    'min_psnr',
    'max_psnr',
    'mean_ssim',
    'std_ssim',
    'median_ssim',
    'min_ssim',
    'max_ssim',
]
LEVEL_METRICS_FIELDNAMES = [
    'image_index',
    'file',
    'sr_path',
    'gt_path',
    'psnr',
    'ssim',
]


def numeric_sort_key(path):
    basename = os.path.basename(path)
    numbers = re.findall(r"\d+\d*", basename)
    return (0, int(numbers[-1]), basename) if numbers else (1, basename)


def safe_filename(name):
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))
    safe = safe.strip('_')
    return safe if safe else 'group'


def summarize_metric_values(values):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    with np.errstate(invalid='ignore'):
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }


def save_rows_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_level_metric_csv(opt, group_name, rows):
    metrics_dir = os.path.join(opt.out, getattr(opt, 'metrics_by_level_dir', 'metrics_by_level'))
    path = os.path.join(metrics_dir, f'{safe_filename(group_name)}_metrics.csv')
    level_rows = [
        {key: row[key] for key in LEVEL_METRICS_FIELDNAMES}
        for row in rows
    ]
    save_rows_csv(path, level_rows, LEVEL_METRICS_FIELDNAMES)
    print(f"Saved per-level metrics to {path}")
    return path


def _finite_metric_pairs(rows, key):
    pairs = []
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            pairs.append((int(row['image_index']), value))
    return pairs


def _plot_hist_with_stats(axis, values, title, xlabel):
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel('count')
    if not finite_values:
        axis.text(0.5, 0.5, 'no finite values', ha='center', va='center', transform=axis.transAxes)
        return
    axis.hist(finite_values)
    mean_value = float(np.mean(finite_values))
    median_value = float(np.median(finite_values))
    axis.axvline(mean_value, linestyle='--', label='mean')
    axis.axvline(median_value, linestyle=':', label='median')
    axis.legend()


def _plot_metric_line(axis, rows, key, title, ylabel):
    pairs = _finite_metric_pairs(rows, key)
    axis.set_title(title)
    axis.set_xlabel('image index')
    axis.set_ylabel(ylabel)
    if not pairs:
        axis.text(0.5, 0.5, 'no finite values', ha='center', va='center', transform=axis.transAxes)
        return
    indexes, values = zip(*pairs)
    axis.plot(indexes, values, marker='o')


def save_level_metric_plot(opt, group_name, group_rows):
    try:
        plot_dir = os.path.join(opt.out, getattr(opt, 'metric_plot_dir', 'metric_plots'))
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, f'{safe_filename(group_name)}_metrics_distribution.png')
        psnrs = [row['psnr'] for row in group_rows]
        ssims = [row['ssim'] for row in group_rows]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        _plot_hist_with_stats(axes[0, 0], psnrs, f'{group_name} PSNR distribution', 'PSNR')
        _plot_hist_with_stats(axes[0, 1], ssims, f'{group_name} SSIM distribution', 'SSIM')
        _plot_metric_line(axes[1, 0], group_rows, 'psnr', 'Per-image PSNR', 'PSNR')
        _plot_metric_line(axes[1, 1], group_rows, 'ssim', 'Per-image SSIM', 'SSIM')
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved metric plot to {path}")
        return path
    except Exception as exc:
        print(f"Warning: failed to save metric plot for {group_name}: {exc}")
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def save_all_levels_boxplot(opt, metric_rows):
    try:
        grouped = {}
        for row in metric_rows:
            grouped.setdefault(row['group'], {'psnr': [], 'ssim': []})
            psnr = float(row['psnr'])
            ssim = float(row['ssim'])
            if np.isfinite(psnr):
                grouped[row['group']]['psnr'].append(psnr)
            if np.isfinite(ssim):
                grouped[row['group']]['ssim'].append(ssim)

        groups = [group for group, values in grouped.items() if values['psnr'] or values['ssim']]
        if not groups:
            print('Warning: no finite metrics available for all-level boxplot')
            return None

        plot_dir = os.path.join(opt.out, getattr(opt, 'metric_plot_dir', 'metric_plots'))
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, 'all_levels_metrics_boxplot.png')
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(groups) * 1.2), 5))

        psnr_data = [grouped[group]['psnr'] for group in groups]
        ssim_data = [grouped[group]['ssim'] for group in groups]
        axes[0].boxplot(psnr_data, labels=groups)
        axes[0].set_title('PSNR by level')
        axes[0].set_ylabel('PSNR')
        axes[1].boxplot(ssim_data, labels=groups)
        axes[1].set_title('SSIM by level')
        axes[1].set_ylabel('SSIM')
        for axis in axes:
            axis.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved all-level metric plot to {path}")
        return path
    except Exception as exc:
        print(f"Warning: failed to save all-level metric plot: {exc}")
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


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


def _build_summary_row(group_name, group_rows):
    psnr_stats = summarize_metric_values([row['psnr'] for row in group_rows])
    ssim_stats = summarize_metric_values([row['ssim'] for row in group_rows])
    return {
        'group': group_name,
        'num_images': len(group_rows),
        'mean_psnr': psnr_stats['mean'],
        'std_psnr': psnr_stats['std'],
        'median_psnr': psnr_stats['median'],
        'min_psnr': psnr_stats['min'],
        'max_psnr': psnr_stats['max'],
        'mean_ssim': ssim_stats['mean'],
        'std_ssim': ssim_stats['std'],
        'median_ssim': ssim_stats['median'],
        'min_ssim': ssim_stats['min'],
        'max_ssim': ssim_stats['max'],
    }


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
        group_metric_rows = []

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
                psnr = float(psnr)
                ssim = float(ssim)
                if not np.isfinite(psnr) or not np.isfinite(ssim):
                    print(
                        f"Warning: non-finite metric for {group_name}/{basename}: "
                        f"PSNR={psnr}, SSIM={ssim}"
                    )
                row = {
                    'group': group_name,
                    'image_index': iidx + 1,
                    'file': basename,
                    'sr_path': save_path,
                    'gt_path': gt_path,
                    'psnr': psnr,
                    'ssim': ssim,
                }
                metric_rows.append(row)
                group_metric_rows.append(row)

        if group_metric_rows:
            save_level_metric_csv(opt, group_name, group_metric_rows)
            if getattr(opt, 'save_metric_plots', True):
                save_level_metric_plot(opt, group_name, group_metric_rows)

            summary_row = _build_summary_row(group_name, group_metric_rows)
            summary_rows.append(summary_row)
            print(
                f"{group_name} mean PSNR: {summary_row['mean_psnr']:.4f} "
                f"std: {summary_row['std_psnr']:.4f}, "
                f"mean SSIM: {summary_row['mean_ssim']:.4f} "
                f"std: {summary_row['std_ssim']:.4f}"
            )

    if metric_rows:
        metrics_path = os.path.join(opt.out, 'metrics.csv')
        summary_path = os.path.join(opt.out, 'metrics_summary.csv')
        save_rows_csv(metrics_path, metric_rows, METRICS_FIELDNAMES)
        save_rows_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved summary metrics to {summary_path}")
        if getattr(opt, 'save_metric_plots', True):
            save_all_levels_boxplot(opt, metric_rows)


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
    parser.add_argument('--save_metric_plots', action='store_true', default=True,
                        help='save per-level PSNR/SSIM distribution plots when gt_root is provided')
    parser.add_argument('--no_save_metric_plots', dest='save_metric_plots', action='store_false')
    parser.add_argument('--metric_plot_dir', type=str, default='metric_plots',
                        help='subfolder under output directory for metric plots')
    parser.add_argument('--metrics_by_level_dir', type=str, default='metrics_by_level',
                        help='subfolder under output directory for per-level metric CSV files')
    return parser


if __name__ == '__main__':
    opt = build_arg_parser().parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    net = LoadModel(opt)

    SIM_reconstruct9(net, opt)
