import os
import sys
import time
import glob
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from PIL import Image
from models import get_model
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


def tensor_to_pil_image(tensor):
    array = tensor.detach().cpu().float().numpy()
    array = np.squeeze(array)
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    array = np.clip(array, 0, 1)
    return Image.fromarray((array * 65535).astype(np.uint16))


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


def _sync_bool_from_rank0(value, opt):
    if not getattr(opt, 'distributed', False):
        return value
    flag = torch.tensor([1 if value else 0], device=opt.device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


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
    start_epoch = 0
    validate_nrmse = [np.inf]
    loss_function = nn.L1Loss()
    ssim_function = SSIM()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = None

    if checkpoint is not None:
        if opt.lr == 1 and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)

    if len(opt.scheduler) > 0:
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if checkpoint is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    amp_enabled = bool(getattr(opt, 'amp', False) and opt.device.type == 'cuda')
    autocast_dtype = get_amp_dtype(opt)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and opt.amp_dtype == 'fp16')
    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        if getattr(opt, 'distributed', False) and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        total_loss = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(trainloader):
            lr = batch['sim_inputs'].to(opt.device, non_blocking=True)
            hr = batch['sim_gt'].to(opt.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=autocast_dtype):
                sr = net(lr)
                ssim = ssim_function(sr, hr)
                content_loss = loss_function(sr, hr)
                a = 0.84
                loss = a * content_loss + (1 - a) * (1 - ssim)
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
                print(
                    '\r[%d/%d][%d/%d] Total Loss: %0.6f L1 Loss: %0.6f SSIM: %0.6f' % (
                        epoch + 1, opt.nepoch, i + 1, len(trainloader),
                        loss.detach().item(), content_loss.detach().item(), ssim.detach().item()), end='')
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

        if (epoch + 1) % opt.testinterval == 0:
            should_save_best = False
            if is_main_process():
                should_save_best = validate(opt, validloader, net, epoch, optimizer, scheduler, validate_nrmse)
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
        lr_batch, hr_batch, wf_batch = batch['sim_inputs'], batch['sim_gt'], batch['wf']
        with torch.no_grad():
            sr_batch = net(lr_batch.to(opt.device))
        for j in range(len(lr_batch)):
            save_flag = (epoch < 5 or (
                    epoch + 1) % opt.plotinterval == 0 or epoch == opt.nepoch - 1) and count < opt.nplot
            sr_j = torch.clamp(sr_batch.data[j], min=0, max=1).cpu()
            hr_j = hr_batch.data[j].cpu()
            wf_j = wf_batch.data[j].cpu()
            if save_flag:
                tensor_to_pil_image(wf_j).save('%s/lr_epoch%d_%d.tif' % (opt.out, epoch + 1, count))
                tensor_to_pil_image(sr_j).save('%s/sr_epoch%d_%d.tif' % (opt.out, epoch + 1, count))
                tensor_to_pil_image(hr_j).save('%s/hr_epoch%d_%d.tif' % (opt.out, epoch + 1, count))
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
        cleanup_distributed()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    opt = options()
    main(opt)
