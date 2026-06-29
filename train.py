import os
import sys
import time
import glob
import csv
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from models import get_model
from models.fcb import FCBLayer
from pytorch_ssim import SSIM
from utils.util import img_comp, load_checkpoint_flexible
from data import get_data_loader
from option.options import apply_preset, get_cli_keys, parser, RESTORABLE_OPTION_KEYS
from utils.plotting import testAndMakeCombinedPlots, generate_convergence_plots
from utils.distributed import (
    cleanup_distributed,
    get_amp_dtype,
    get_model_state_dict_for_save,
    is_main_process,
    rank0_print,
    setup_distributed,
    wrap_model_for_training,
)


def print_networks(net, verbose):
    if not is_main_process():
        return
    rank0_print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        rank0_print(net)
    rank0_print('[Network APCAN] Total number of parameters : %.3f M' % (num_params / 1e6))
    rank0_print('-----------------------------------------------')


def setup_tensorboard_logging(opt):
    if not getattr(opt, 'log', False) or not is_main_process():
        return
    from torch.utils.tensorboard import SummaryWriter

    opt.writer = SummaryWriter(log_dir=opt.out)
    opt.train_stats = open(os.path.join(opt.out, 'train_stats.txt'), 'a')


def close_tensorboard_logging(opt):
    if hasattr(opt, 'writer'):
        opt.writer.close()
    if hasattr(opt, 'train_stats'):
        opt.train_stats.close()


def _extract_logged_value(optstr, key):
    pattern = key + '='
    start = optstr.find(pattern)
    if start < 0:
        return None
    start += len(pattern)
    end = start
    quote = None
    while end < len(optstr):
        char = optstr[end]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in ["'", '"']:
            quote = char
        elif char in [',', ')', '\n']:
            break
        end += 1
    return optstr[start:end].strip()


def _convert_logged_value(raw_value, current_value):
    if raw_value is None:
        return current_value
    raw_value = raw_value.strip()
    if raw_value in ['None', 'none']:
        return None
    if isinstance(current_value, bool):
        return raw_value.lower() in ['true', '1', 'yes']
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if raw_value.startswith(("'", '"')) and raw_value.endswith(("'", '"')):
        return raw_value[1:-1]
    return raw_value


def restore_options_from_log(opt, logfile, cli_keys):
    with open(logfile, 'r') as fid:
        optstr = fid.read()
    for key in RESTORABLE_OPTION_KEYS:
        if key in cli_keys or not hasattr(opt, key):
            continue
        raw_value = _extract_logged_value(optstr, key)
        if raw_value is not None:
            setattr(opt, key, _convert_logged_value(raw_value, getattr(opt, key)))
    return opt


def options():
    cli_keys = get_cli_keys()
    opt = parser.parse_args()
    opt = apply_preset(opt, cli_keys)
    if opt.data_norm == '':
        opt.data_norm = opt.dataset
    elif opt.data_norm.lower() == 'none':
        opt.data_norm = None
    if opt.grad_accum_steps < 1:
        raise ValueError('--grad_accum_steps must be >= 1')
    if opt.fcb_diag_interval < 1:
        raise ValueError('--fcb_diag_interval must be >= 1')
    if opt.fft_loss_warmup_epochs < 0:
        raise ValueError('--fft_loss_warmup_epochs must be >= 0')
    if opt.fft_loss_start_epoch < 0:
        raise ValueError('--fft_loss_start_epoch must be >= 0')
    if opt.fcb_reparam_epoch < 0:
        raise ValueError('--fcb_reparam_epoch must be >= 0')
    if len(opt.basedir) > 0:
        opt.root = opt.root.replace('basedir', opt.basedir)
        opt.weights = opt.weights.replace('basedir', opt.basedir)
        opt.out = opt.out.replace('basedir', opt.basedir)
    if opt.out[:4] == 'root':
        opt.out = opt.out.replace('root', opt.root)
    if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
        logfile = opt.weights + '/{}.txt'.format(opt.model)
        opt.weights += '/best.pth'
        if not os.path.isfile(opt.weights):
            opt.weights = opt.weights.replace('best.pth', 'prelim.pth')
        if os.path.isfile(logfile):
            restore_options_from_log(opt, logfile, cli_keys)
    return opt


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def make_grad_scaler(opt, amp_enabled):
    scaler_enabled = amp_enabled and getattr(opt, 'amp_dtype', 'fp16') == 'fp16'
    amp_module = getattr(torch, 'amp', None)
    if amp_module is not None and hasattr(amp_module, 'GradScaler'):
        try:
            return amp_module.GradScaler('cuda', enabled=scaler_enabled)
        except TypeError:
            return amp_module.GradScaler(enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def make_autocast_context(amp_enabled, autocast_dtype):
    amp_module = getattr(torch, 'amp', None)
    if amp_module is not None and hasattr(amp_module, 'autocast'):
        try:
            return amp_module.autocast('cuda', enabled=amp_enabled, dtype=autocast_dtype)
        except TypeError:
            try:
                return amp_module.autocast(device_type='cuda', enabled=amp_enabled, dtype=autocast_dtype)
            except TypeError:
                return amp_module.autocast(enabled=amp_enabled, dtype=autocast_dtype)
    try:
        return torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=amp_enabled)


def _sync_bool_from_rank0(value, opt):
    if not getattr(opt, 'distributed', False):
        return value
    flag = torch.tensor([1 if value else 0], device=opt.device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def get_rank0_validation_model(net, opt):
    if (
        getattr(opt, 'distributed', False)
        and getattr(opt, 'dist_backend', 'fsdp') == 'ddp'
        and hasattr(net, 'module')
    ):
        return net.module
    return net


MRFCB_DIAGNOSTIC_FIELDS = [
    'epoch',
    'num_fcb_layers',
    'residual_scale_mean',
    'residual_scale_min',
    'residual_scale_max',
    'gamma_rad_mean',
    'gamma_sim_mean',
    'gate_mean_mean',
    'gate_std_mean',
    'gate_min_mean',
    'gate_max_mean',
    'global_std_mean',
    'local_std_mean',
    'fused_std_mean',
]


def fft_log_amp_loss(sr, hr):
    sr_fft = torch.fft.fft2(sr, dim=(-2, -1), norm='ortho')
    hr_fft = torch.fft.fft2(hr, dim=(-2, -1), norm='ortho')
    sr_amp = torch.log1p(torch.abs(sr_fft))
    hr_amp = torch.log1p(torch.abs(hr_fft))
    return F.l1_loss(sr_amp, hr_amp)


def current_fft_weight(opt, epoch):
    fft_loss_weight = getattr(opt, 'fft_loss_weight', 0.0)
    fft_loss_start_epoch = getattr(opt, 'fft_loss_start_epoch', 0)
    fft_loss_warmup_epochs = getattr(opt, 'fft_loss_warmup_epochs', 0)
    if fft_loss_weight <= 0:
        return 0.0
    if epoch < fft_loss_start_epoch:
        return 0.0
    active_epoch = epoch - fft_loss_start_epoch + 1
    if fft_loss_warmup_epochs > 0:
        ramp = min(1.0, active_epoch / float(fft_loss_warmup_epochs))
    else:
        ramp = 1.0
    return fft_loss_weight * ramp


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _min_or_zero(values):
    return float(np.min(values)) if values else 0.0


def _max_or_zero(values):
    return float(np.max(values)) if values else 0.0


def _with_fsdp_full_params(net, writeback, callback):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return callback()

    if isinstance(net, FSDP):
        with FSDP.summon_full_params(net, writeback=writeback, recurse=True):
            return callback()
    return callback()


def _iter_fcb_layers(net):
    for module in net.modules():
        if isinstance(module, FCBLayer) or hasattr(module, 'last_gate_mean'):
            yield module


def collect_mrfcb_diagnostics(net):
    def _collect():
        residual_scales = []
        gamma_rads = []
        gamma_sims = []
        gate_means = []
        gate_stds = []
        gate_mins = []
        gate_maxs = []
        global_stds = []
        local_stds = []
        fused_stds = []
        layers = list(_iter_fcb_layers(net))

        for layer in layers:
            residual_scale = getattr(layer, 'residual_scale', None)
            if residual_scale is not None:
                residual_scales.append(float(residual_scale.detach().cpu()))

            fourier = getattr(getattr(layer, 'global_branch', None), 'fourier', None)
            if fourier is not None and hasattr(fourier, 'get_gamma_values'):
                gamma_values = fourier.get_gamma_values()
                gamma_rads.append(gamma_values['gamma_rad'])
                gamma_sims.append(gamma_values['gamma_sim'])

            if getattr(layer, 'last_gate_mean', None) is not None:
                gate_means.append(float(layer.last_gate_mean))
                gate_stds.append(float(layer.last_gate_std))
                gate_mins.append(float(layer.last_gate_min))
                gate_maxs.append(float(layer.last_gate_max))
                global_stds.append(float(layer.last_global_std))
                local_stds.append(float(layer.last_local_std))
                fused_stds.append(float(layer.last_fused_std))

        return {
            'num_fcb_layers': len(layers),
            'residual_scale_mean': _mean_or_zero(residual_scales),
            'residual_scale_min': _min_or_zero(residual_scales),
            'residual_scale_max': _max_or_zero(residual_scales),
            'gamma_rad_mean': _mean_or_zero(gamma_rads),
            'gamma_sim_mean': _mean_or_zero(gamma_sims),
            'gate_mean_mean': _mean_or_zero(gate_means),
            'gate_std_mean': _mean_or_zero(gate_stds),
            'gate_min_mean': _mean_or_zero(gate_mins),
            'gate_max_mean': _mean_or_zero(gate_maxs),
            'global_std_mean': _mean_or_zero(global_stds),
            'local_std_mean': _mean_or_zero(local_stds),
            'fused_std_mean': _mean_or_zero(fused_stds),
        }

    return _with_fsdp_full_params(net, writeback=False, callback=_collect)


def _append_mrfcb_diagnostics_csv(opt, epoch, summary):
    path = os.path.join(opt.out, 'mrfcb_diagnostics.csv')
    exists = os.path.isfile(path)
    row = {'epoch': epoch + 1}
    row.update(summary)
    with open(path, 'a', newline='') as fid:
        writer = csv.DictWriter(fid, fieldnames=MRFCB_DIAGNOSTIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_mrfcb_diagnostics(opt, epoch, summary):
    if not is_main_process():
        return

    if summary['num_fcb_layers'] == 0:
        message = 'Warning: MRFCB diag requested but no FCBLayer modules were found.'
    else:
        lines = [
            'MRFCB diag:',
            'num_fcb_layers: %d' % summary['num_fcb_layers'],
            'residual_scale_mean: %.6g' % summary['residual_scale_mean'],
            'gamma_rad_mean: %.6g' % summary['gamma_rad_mean'],
            'gamma_sim_mean: %.6g' % summary['gamma_sim_mean'],
            'gate_mean_mean: %.6g' % summary['gate_mean_mean'],
            'gate_std_mean: %.6g' % summary['gate_std_mean'],
            'global_std_mean: %.6g' % summary['global_std_mean'],
            'local_std_mean: %.6g' % summary['local_std_mean'],
            'fused_std_mean: %.6g' % summary['fused_std_mean'],
        ]
        if not getattr(opt, 'fcb_use_sim_mask', True):
            lines.append('gamma_sim is recorded but not used because fcb_use_sim_mask=False.')
        message = '\n'.join(lines)

    print(message)
    print(message, file=opt.fid)
    opt.fid.flush()

    if getattr(opt, 'log', False) and hasattr(opt, 'writer'):
        for key, value in summary.items():
            opt.writer.add_scalar('mrfcb/%s' % key, value, epoch)

    _append_mrfcb_diagnostics_csv(opt, epoch, summary)


def apply_fcb_reparameterization(net, source='local_dw1'):
    def _apply():
        count = 0
        for module in net.modules():
            method = getattr(module, 'reparameterize_fourier_from_local', None)
            if callable(method):
                method(source)
                count += 1
        return count

    count = _with_fsdp_full_params(net, writeback=True, callback=_apply)
    if count == 0:
        rank0_print('Warning: FCB reparameterization requested but no FCBLayer modules were found.')
    else:
        rank0_print('Applied FCB reparameterization to %d FCBLayer modules from %s.' % (count, source))
    return count


def set_fcb_global_branch_trainable(net, trainable):
    count = 0
    for module in net.modules():
        method = getattr(module, 'set_global_branch_trainable', None)
        if callable(method):
            method(trainable)
            count += 1
    return count


def save_training_checkpoint(path, epoch, net, opt, optimizer, scheduler=None):
    state_dict = get_model_state_dict_for_save(net, opt)
    if not is_main_process():
        return

    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
    }
    if len(opt.scheduler) > 0 and scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, path)


def train(opt, trainloader, validloader, net, checkpoint=None):
    start_epoch = checkpoint.get('epoch', 0) if checkpoint is not None else 0
    validate_nrmse = [np.inf]
    loss_function = nn.L1Loss()
    ssim_function = SSIM()
    reparam_applied = bool(
        getattr(opt, 'fcb_reparam', False) and start_epoch > getattr(opt, 'fcb_reparam_epoch', 0)
    )

    if getattr(opt, 'fcb_reparam', False):
        if (
            opt.fcb_reparam_freeze_global_before
            and opt.fcb_reparam_epoch > 0
            and start_epoch < opt.fcb_reparam_epoch
        ):
            frozen_count = set_fcb_global_branch_trainable(net, False)
            rank0_print('Froze global Fourier branch in %d FCBLayer modules before reparameterization.' % frozen_count)
        if opt.fcb_reparam_epoch == 0 and start_epoch == 0:
            apply_fcb_reparameterization(net, opt.fcb_reparam_source)
            reparam_applied = True
        elif opt.fcb_reparam_epoch == 0:
            reparam_applied = True

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = None

    if checkpoint is not None:
        if opt.lr == 1 and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if len(opt.scheduler) > 0:
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if checkpoint is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    amp_enabled = bool(getattr(opt, 'amp', False) and opt.device.type == 'cuda')
    autocast_dtype = get_amp_dtype(opt)
    scaler = make_grad_scaler(opt, amp_enabled)
    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        if getattr(opt, 'distributed', False) and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)
        if (
            getattr(opt, 'fcb_reparam', False)
            and not reparam_applied
            and epoch >= opt.fcb_reparam_epoch
        ):
            apply_fcb_reparameterization(net, opt.fcb_reparam_source)
            trainable_count = set_fcb_global_branch_trainable(net, True)
            rank0_print('Enabled global Fourier branch training in %d FCBLayer modules.' % trainable_count)
            reparam_applied = True

        total_loss = 0.0
        count = 0
        fft_weight = current_fft_weight(opt, epoch)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(trainloader):
            lr = batch['sim_inputs'].to(opt.device, non_blocking=True)
            hr = batch['sim_gt'].to(opt.device, non_blocking=True)

            with make_autocast_context(amp_enabled, autocast_dtype):
                sr = net(lr)
                ssim = ssim_function(sr, hr)
                content_loss = loss_function(sr, hr)
                a = 0.84
                spatial_loss = a * content_loss + (1 - a) * (1 - ssim)
                if fft_weight > 0:
                    freq_loss = fft_log_amp_loss(sr.float(), hr.float())
                    loss = spatial_loss + fft_weight * freq_loss
                else:
                    freq_loss = None
                    loss = spatial_loss
                backward_loss = loss / opt.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            should_step = ((i + 1) % opt.grad_accum_steps == 0) or ((i + 1) == len(trainloader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_value = opt.gradient_clipping / get_lr(optimizer)
                nn.utils.clip_grad_value_(net.parameters(), clip_value)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach().item()
            count += 1

            if is_main_process():
                freq_loss_value = 0.0 if freq_loss is None else freq_loss.detach().item()
                print(
                    '\r[%d/%d][%d/%d] Total Loss: %0.6f L1 Loss: %0.6f SSIM: %0.6f FFT Loss: %0.6f FFT Weight: %0.6g' % (
                        epoch + 1, opt.nepoch, i + 1, len(trainloader),
                        loss.detach().item(), content_loss.detach().item(), ssim.detach().item(),
                        freq_loss_value, fft_weight), end='')
                if opt.log and count * opt.batchSize // 1000 > 0:
                    t1 = time.perf_counter() - opt.t0
                    mem = torch.cuda.memory_allocated() if opt.device.type == 'cuda' else 0
                    opt.writer.add_scalar('data/mean_loss_per_1000', total_loss / count, epoch)
                    opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                    print(epoch, count * opt.batchSize, t1, mem,
                          total_loss / count, file=opt.train_stats)
                    opt.train_stats.flush()
                    count = 0

        if len(opt.scheduler) > 0:
            scheduler.step()
            if is_main_process():
                for param_group in optimizer.param_groups:
                    print('\nLearning rate', param_group['lr'])
                    break

        epoch_count = max(count, 1)
        total_loss = total_loss / epoch_count
        if is_main_process():
            t1 = time.perf_counter() - opt.t0
            eta = (opt.nepoch - (epoch + 1)) * t1 / (epoch + 1)
            ostr = '\nEpoch [%d/%d] done, total loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
                epoch + 1, opt.nepoch, total_loss, t1, eta)
            print(ostr)
            print(ostr, file=opt.fid)
            opt.fid.flush()
            if opt.log:
                opt.writer.add_scalar('data/mean_loss', total_loss, epoch)

        if getattr(opt, 'fcb_diag', False) and (epoch + 1) % getattr(opt, 'fcb_diag_interval', 1) == 0:
            diag_summary = collect_mrfcb_diagnostics(net)
            log_mrfcb_diagnostics(opt, epoch, diag_summary)
            if getattr(opt, 'distributed', False):
                dist.barrier()

        if (epoch + 1) % opt.testinterval == 0:
            should_save_best = False
            if is_main_process():
                validation_net = get_rank0_validation_model(net, opt)
                should_save_best = validate(opt, validloader, validation_net, epoch, optimizer, scheduler, validate_nrmse)
            should_save_best = _sync_bool_from_rank0(should_save_best, opt)
            if should_save_best:
                save_training_checkpoint(opt.out + '/best.pth', epoch + 1, net, opt, optimizer, scheduler)
            if getattr(opt, 'distributed', False):
                dist.barrier()

        if (epoch + 1) % opt.saveinterval == 0:
            save_training_checkpoint('%s/prelim%d.pth' % (opt.out, epoch + 1), epoch + 1, net, opt, optimizer, scheduler)
            if getattr(opt, 'distributed', False):
                dist.barrier()


def validate(opt, validloader, net, epoch, optimizer, scheduler, validate_nrmse):
    mses, nrmses, psnrs, ssims = [], [], [], []
    count = 0
    net.eval()
    for i, batch in enumerate(validloader):
        lr_batch, hr_batch = batch['sim_inputs'], batch['sim_gt']
        with torch.no_grad():
            sr_batch = net(lr_batch.to(opt.device))
        for j in range(len(lr_batch)):
            sr_j = torch.clamp(sr_batch.data[j], min=0, max=1).cpu()
            hr_j = hr_batch.data[j].cpu()
            mses, nrmses, psnrs, ssims = img_comp(
                hr_j.numpy(), sr_j.numpy(), mses, nrmses, psnrs, ssims
            )
            count += 1
            if count == opt.ntest:
                break
        if count == opt.ntest:
            break
    net.train()

    mean_nrmse = np.mean(nrmses) if nrmses else np.inf
    improved = min(validate_nrmse) > mean_nrmse
    if improved:
        validate_nrmse.append(mean_nrmse)

    summarystr = ""
    if count == 0:
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. mse: %0.4f, nrmse: %0.4f, psnr: %0.2f, ssim: %0.4f' % (
        count, np.mean(mses), np.mean(nrmses), np.mean(psnrs), np.mean(ssims))
    print(summarystr)
    print(summarystr, file=opt.fid)
    opt.fid.flush()
    return improved


def _load_raw_checkpoint_if_needed(opt, net):
    checkpoint = None
    if len(opt.weights) > 0:
        rank0_print('loading checkpoint', opt.weights)
        checkpoint = torch.load(opt.weights, map_location=opt.device)
        load_checkpoint_flexible(net, opt.weights, opt.device)
    return checkpoint


def main(opt):
    setup_distributed(opt)
    try:
        opt.out = opt.out + '/' + opt.model
        if is_main_process():
            os.makedirs(opt.out, exist_ok=True)
            opt.fid = open(opt.out + '/{}.txt'.format(opt.model), 'w')
            ostr = 'ARGS: ' + ' '.join(sys.argv[:])
            print(opt, '\n')
            print(opt, '\n', file=opt.fid)
            print('\n%s\n' % ostr)
            print('\n%s\n' % ostr, file=opt.fid)
            setup_tensorboard_logging(opt)
            rank0_print('DDP keeps a full model replica per GPU; FSDP shards parameters, gradients, and optimizer state.')
            rank0_print('Distributed batchSize is per-GPU local batch. global_batch = batchSize * world_size * grad_accum_steps.')
        if getattr(opt, 'distributed', False):
            dist.barrier()

        rank0_print('getting dataloader', opt.root)
        trainloader, validloader = get_data_loader(opt)
        t0 = time.perf_counter()
        net = get_model(opt)
        checkpoint = _load_raw_checkpoint_if_needed(opt, net)
        net = wrap_model_for_training(net, opt)
        print_networks(net, False)

        if not opt.test:
            train(opt, trainloader, validloader, net, checkpoint=checkpoint)
        else:
            if is_main_process():
                rank0_print('time: %0.1f' % (time.perf_counter() - t0))
                testAndMakeCombinedPlots(net, validloader, opt)
            if getattr(opt, 'distributed', False):
                dist.barrier()

        if is_main_process():
            if not opt.test:
                generate_convergence_plots(opt, opt.out + '/{}.txt'.format(opt.model))
            print('time: %0.1f' % (time.perf_counter() - t0))
            if opt.disposableTrainingData and not opt.test:
                print('deleting training data')
                os.makedirs('%s/training_data_subset' % opt.out, exist_ok=True)
                samplecount = 0
                for file in glob.glob('%s/*' % opt.root):
                    if os.path.isfile(file):
                        basename = os.path.basename(file)
                        shutil.copy2(file, '%s/training_data_subset/%s' % (opt.out, basename))
                        samplecount += 1
                        if samplecount == 10:
                            break
                shutil.rmtree(opt.root)
    finally:
        if is_main_process() and hasattr(opt, 'fid'):
            opt.fid.close()
        if is_main_process():
            close_tensorboard_logging(opt)
        cleanup_distributed()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    opt = options()
    main(opt)
