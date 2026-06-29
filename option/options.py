import argparse
import sys

# training options
parser = argparse.ArgumentParser()
parser.add_argument('--preset', type=str, default='fcb_medium',
                    choices=['fcb_debug', 'fcb_small', 'fcb_medium', 'apcan_base'],
                    help='preset training configuration')
parser.add_argument('--model', type=str, default='apcan_1_actin', help='model to use')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_norm', type=str, default='minmax', help='if normalization should not be used')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--saveinterval', type=int, default=20, help='number of epochs between saves')
parser.add_argument('--ntrain', type=int, default=100, help='number of samples to train on')
parser.add_argument('--scheduler', type=str, default='25, 0.5', help='options for a scheduler, format: stepsize, gamma')
parser.add_argument('--log', action='store_true')
parser.add_argument('--task', type=str, default='simin_simout')
parser.add_argument('--norm_flag', type=int, default=1)
parser.add_argument('--gradient_clipping', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)

# data_preprocess
parser.add_argument('--dataset', type=str, default='F-actin', help='dataset to train')
parser.add_argument('--imageSize', type=int, default=502, help='the low resolution image size')
parser.add_argument('--weights', type=str, default='', help='model to retrain from')
parser.add_argument('--basedir', type=str, default='',
                    help='path to prepend to all others paths: root, output, weights')
parser.add_argument('--root', type=str, default='/data5/lyhe/python/FFTlearn_complex2_lotdata/dataset/', help='dataset to train')
parser.add_argument('--out', type=str, default='checkpoint', help='folder to output model training test_results')
parser.add_argument('--disposableTrainingData', action='store_true',
                    help='whether to delete training data_preprocess after training')
parser.add_argument('--gt_mapping_mode', type=str, default='grouped',
                    choices=['one_to_one', 'grouped'],
                    help='GT mapping mode')
parser.add_argument('--gt_group_size', type=int, default=12,
                    help='number of raw sample folders sharing one GT when gt_mapping_mode=grouped')
parser.add_argument('--filename_digits', type=int, default=8,
                    help='zero padding digits for folder and gt filenames')
parser.add_argument('--raw_index_start', type=int, default=1,
                    help='starting index of raw sample folders')
parser.add_argument('--gt_index_start', type=int, default=1,
                    help='starting index of gt files')

# computation 
parser.add_argument('--workers', type=int, default=4, help='number of data_preprocess loading workers')
parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
parser.add_argument('--distributed', action='store_true',
                    help='enable torch.distributed training launched by torchrun')
parser.add_argument('--dist_backend', type=str, default='fsdp', choices=['ddp', 'fsdp'],
                    help='distributed wrapper: ddp or fsdp')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank, normally provided by torchrun through env LOCAL_RANK')
parser.add_argument('--amp', action='store_true',
                    help='enable torch.cuda.amp mixed precision training')
parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                    help='AMP dtype')
parser.add_argument('--grad_accum_steps', type=int, default=1,
                    help='gradient accumulation steps')
parser.add_argument('--fsdp_cpu_offload', action='store_true',
                    help='enable FSDP CPU offload, slower but may reduce GPU memory')
parser.add_argument('--fsdp_mixed_precision', action='store_true',
                    help='enable FSDP mixed precision policy')
parser.add_argument('--activation_checkpoint', action='store_true',
                    help='enable activation checkpointing for selected model blocks')
parser.add_argument('--save_on_rank0_only', action='store_true', default=True,
                    help='save checkpoints and plots only on rank 0 in distributed training')

# restoration options
parser.add_argument('--scale', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nch_in', type=int, default=9, help='channels in input')
parser.add_argument('--nch_out', type=int, default=1, help='channels in output')
parser.add_argument('--use_fcb', action='store_true', help='replace APCALayer with FCBLayer')
parser.add_argument('--fcb_rows', type=int, default=502, help='FCB fixed input height')
parser.add_argument('--fcb_cols', type=int, default=502, help='FCB fixed input width')
parser.add_argument('--fcb_init', type=str, default='he', choices=['he', 'glorot'],
                    help='FCB complex kernel initialization')
parser.add_argument('--fcb_alpha', type=float, default=0.7,
                    help='mode rebalancing radial exponent')
parser.add_argument('--fcb_gamma_init', type=float, default=1e-3,
                    help='initial effective value for radial and SIM mode rebalancing gammas')
parser.add_argument('--fcb_rho_min', type=float, default=0.25,
                    help='SIM high-frequency gate threshold')
parser.add_argument('--fcb_tau', type=float, default=0.05,
                    help='SIM high-frequency gate softness')
parser.add_argument('--fcb_sigma_theta', type=float, default=0.17453292519943295,
                    help='SIM directional soft-sector width in radians')
parser.add_argument('--fcb_directions', type=str, default='-5.1966,54.9131,115.0424',
                    help='comma-separated SIM line-orientation directions in degrees')
parser.add_argument('--fcb_residual_scale', type=float, default=1e-2,
                    help='initial residual scale for FCBLayer output')
parser.add_argument('--fcb_diag', action='store_true',
                    help='record MRFCB diagnostic statistics during training')
parser.add_argument('--fcb_diag_interval', type=int, default=1,
                    help='epoch interval for MRFCB diagnostic logging')
parser.add_argument('--fft_loss_weight', type=float, default=0.0,
                    help='weight for FFT log-amplitude loss, default 0 disables it')
parser.add_argument('--fft_loss_warmup_epochs', type=int, default=0,
                    help='linearly ramp FFT loss weight during first N epochs')
parser.add_argument('--fft_loss_start_epoch', type=int, default=0,
                    help='start applying FFT loss after this epoch index, 0-based')
parser.add_argument('--fcb_use_sim_mask', action='store_true', default=True,
                    help='use SIM directional mask in mode rebalancing')
parser.add_argument('--fcb_no_sim_mask', dest='fcb_use_sim_mask', action='store_false',
                    help='disable SIM directional mask and use radial-only mode rebalancing')
parser.add_argument('--fcb_reparam', action='store_true',
                    help='enable FCB re-parameterization from local depth-wise conv to Fourier kernel')
parser.add_argument('--fcb_reparam_epoch', type=int, default=0,
                    help='epoch index at which to apply FCB re-parameterization; 0 means before training')
parser.add_argument('--fcb_reparam_source', type=str, default='local_dw1',
                    choices=['local_dw1', 'local_dw2'],
                    help='which local depth-wise conv to convert into DeepSparse Fourier kernel')
parser.add_argument('--fcb_reparam_freeze_global_before', action='store_true',
                    help='freeze global Fourier branch before re-parameterization warmup')

# architecture options 
parser.add_argument('--narch', type=int, default=0, help='architecture-dependent parameter')
parser.add_argument('--n_resblocks', type=int, default=4, help='number of residual blocks')
parser.add_argument('--n_resgroups', type=int, default=4, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')

# test options
parser.add_argument('--ntest', type=int, default=50, help='number of images to test per epoch or test run')
parser.add_argument('--testinterval', type=int, default=1, help='number of epochs between tests during training')
parser.add_argument('--test', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--batchSize_test', type=int, default=1, help='input batch size for test loader')
parser.add_argument('--plotinterval', type=int, default=1, help='number of epochs between plotting')
parser.add_argument('--nplot', type=int, default=4, help='number of plots in a test')


RESTORABLE_OPTION_KEYS = [
    'preset',
    'model',
    'root',
    'out',
    'use_fcb',
    'fcb_rows',
    'fcb_cols',
    'fcb_init',
    'fcb_alpha',
    'fcb_gamma_init',
    'fcb_rho_min',
    'fcb_tau',
    'fcb_sigma_theta',
    'fcb_directions',
    'fcb_residual_scale',
    'fcb_diag',
    'fcb_diag_interval',
    'fft_loss_weight',
    'fft_loss_warmup_epochs',
    'fft_loss_start_epoch',
    'fcb_use_sim_mask',
    'fcb_reparam',
    'fcb_reparam_epoch',
    'fcb_reparam_source',
    'fcb_reparam_freeze_global_before',
    'imageSize',
    'scale',
    'nch_in',
    'nch_out',
    'n_resgroups',
    'n_resblocks',
    'n_feats',
    'reduction',
    'batchSize',
    'distributed',
    'dist_backend',
    'local_rank',
    'amp',
    'amp_dtype',
    'grad_accum_steps',
    'fsdp_cpu_offload',
    'fsdp_mixed_precision',
    'activation_checkpoint',
    'save_on_rank0_only',
    'batchSize_test',
    'gt_mapping_mode',
    'gt_group_size',
    'filename_digits',
    'raw_index_start',
    'gt_index_start',
    'lr',
    'nepoch',
    'testinterval',
    'saveinterval',
    'gradient_clipping',
]


def get_cli_keys():
    keys = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            keys.add(key)
            if key == 'fcb_no_sim_mask':
                keys.add('fcb_use_sim_mask')
    return keys


def set_if_not_cli(opt, key, value, cli_keys):
    if key not in cli_keys:
        setattr(opt, key, value)


FCB_DEFAULTS = {
    'fcb_init': 'he',
    'fcb_alpha': 0.7,
    'fcb_gamma_init': 1e-3,
    'fcb_rho_min': 0.25,
    'fcb_tau': 0.05,
    'fcb_sigma_theta': 0.17453292519943295,
    'fcb_directions': '-5.1966,54.9131,115.0424',
    'fcb_residual_scale': 1e-2,
    'fcb_use_sim_mask': True,
}


def apply_preset(opt, cli_keys):
    presets = {
        'fcb_debug': {
            'use_fcb': True,
            'model': 'apcan_1_actin',
            'root': './dataset/F-actin',
            'out': 'checkpoint',
            'imageSize': 502,
            'fcb_rows': 502,
            'fcb_cols': 502,
            **FCB_DEFAULTS,
            'scale': 2,
            'nch_in': 9,
            'nch_out': 1,
            'batchSize': 1,
            'batchSize_test': 1,
            'n_resgroups': 1,
            'n_resblocks': 1,
            'n_feats': 32,
            'reduction': 16,
            'lr': 1e-4,
            'nepoch': 20,
            'testinterval': 1,
            'saveinterval': 10,
            'gradient_clipping': 0.1,
            'gt_mapping_mode': 'grouped',
            'gt_group_size': 12,
            'filename_digits': 8,
            'raw_index_start': 1,
            'gt_index_start': 1,
        },
        'fcb_small': {
            'use_fcb': True,
            'model': 'apcan_1_actin',
            'root': './dataset/F-actin',
            'out': 'checkpoint',
            'imageSize': 502,
            'fcb_rows': 502,
            'fcb_cols': 502,
            **FCB_DEFAULTS,
            'scale': 2,
            'nch_in': 9,
            'nch_out': 1,
            'batchSize': 1,
            'batchSize_test': 1,
            'n_resgroups': 2,
            'n_resblocks': 2,
            'n_feats': 32,
            'reduction': 16,
            'lr': 1e-4,
            'nepoch': 100,
            'testinterval': 1,
            'saveinterval': 20,
            'gradient_clipping': 0.1,
            'gt_mapping_mode': 'grouped',
            'gt_group_size': 12,
            'filename_digits': 8,
            'raw_index_start': 1,
            'gt_index_start': 1,
        },
        'fcb_medium': {
            'use_fcb': True,
            'model': 'apcan_1_actin',
            'root': '/data5/lyhe/python/FFTlearn_complex2_lotdata/dataset/F-actin/',
            'out': 'checkpoint',
            'imageSize': 502,
            'fcb_rows': 502,
            'fcb_cols': 502,
            **FCB_DEFAULTS,
            'scale': 2,
            'nch_in': 9,
            'nch_out': 1,
            'batchSize': 4,
            'batchSize_test': 1,
            'n_resgroups': 4,
            'n_resblocks': 6,
            'n_feats': 64,
            'reduction': 16,
            'lr': 1e-4,
            'nepoch': 200,
            'testinterval': 1,
            'saveinterval': 20,
            'gradient_clipping': 0.1,
            'gt_mapping_mode': 'grouped',
            'gt_group_size': 12,
            'filename_digits': 8,
            'raw_index_start': 1,
            'gt_index_start': 1,
        },
        'apcan_base': {
            'use_fcb': False,
            'model': 'apcan_1_actin',
            'root': './dataset/F-actin',
            'out': 'checkpoint',
            'imageSize': 502,
            'fcb_rows': 502,
            'fcb_cols': 502,
            **FCB_DEFAULTS,
            'scale': 2,
            'nch_in': 9,
            'nch_out': 1,
            'batchSize': 1,
            'batchSize_test': 1,
            'n_resgroups': 4,
            'n_resblocks': 4,
            'n_feats': 64,
            'reduction': 16,
            'lr': 1e-4,
            'nepoch': 100,
            'testinterval': 1,
            'saveinterval': 20,
            'gradient_clipping': 0.1,
            'gt_mapping_mode': 'grouped',
            'gt_group_size': 12,
            'filename_digits': 8,
            'raw_index_start': 1,
            'gt_index_start': 1,
        },
    }

    if opt.preset not in presets:
        raise ValueError(f'Unsupported preset: {opt.preset}')

    for key, value in presets[opt.preset].items():
        set_if_not_cli(opt, key, value, cli_keys)

    return opt
