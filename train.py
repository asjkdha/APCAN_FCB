import os
import sys
import time
import glob
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from models import get_model
from pytorch_ssim import SSIM
from utils.util import img_comp, load_checkpoint_flexible
from data import get_data_loader
from option.options import apply_preset, get_cli_keys, parser, RESTORABLE_OPTION_KEYS
from utils.plotting import testAndMakeCombinedPlots, generate_convergence_plots


def print_networks(net, verbose):
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('[Network APCAN] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


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


def remove_dataparallel_wrapper(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]
        new_state_dict[name] = vl
    return new_state_dict


def train(opt, trainloader, validloader, net):
    start_epoch = 0
    validate_nrmse = [np.inf]
    loss_function = nn.L1Loss()
    ssim_function = SSIM()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    checkpoint = None
    scheduler = None
    if len(opt.weights) > 0:
        print('loading checkpoint', opt.weights)
        checkpoint = torch.load(opt.weights, map_location=opt.device)
        load_checkpoint_flexible(net, opt.weights, opt.device)
        if opt.lr == 1:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    if len(opt.scheduler) > 0:
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if len(opt.weights) > 0:
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
    opt.t0 = time.perf_counter()
    for epoch in range(start_epoch, opt.nepoch):
        total_loss = 0
        count = 0
        for i, batch in enumerate(trainloader):
            lr, hr = batch['sim_inputs'].to(opt.device), batch['sim_gt'].to(opt.device)
            sr = net(lr)
            ssim = ssim_function(sr, hr)
            content_loss = loss_function(sr, hr)
            a = 0.84
            loss = a * content_loss + (1 - a) * (1 - ssim)
            optimizer.zero_grad()
            loss.backward()

            clip_value = opt.gradient_clipping / get_lr(optimizer)
            nn.utils.clip_grad_value_(net.parameters(), clip_value)

            optimizer.step()
            ######### Status and display #########
            total_loss += loss.data.item()
            print(
                '\r[%d/%d][%d/%d] Total Loss: %0.6f L1 Loss: %0.6f SSIM: %0.6f' % (
                    epoch + 1, opt.nepoch, i + 1, len(trainloader),
                    loss.data.item(), content_loss.item(), ssim.item()), end='')
            count += 1
            if opt.log and count * opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                opt.writer.add_scalar(
                    'data/mean_loss_per_1000', total_loss / count, epoch)
                opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                print(epoch, count * opt.batchSize, t1, mem,
                      total_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate', param_group['lr'])
                break
        total_loss = total_loss / len(trainloader)
        t1 = time.perf_counter() - opt.t0
        eta = (opt.nepoch - (epoch + 1)) * t1 / (epoch + 1)
        ostr = '\nEpoch [%d/%d] done, total loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
            epoch + 1, opt.nepoch, total_loss, t1, eta)
        print(ostr)
        print(ostr, file=opt.fid)
        opt.fid.flush()
        if opt.log:
            opt.writer.add_scalar(
                'data/mean_loss', total_loss / len(trainloader), epoch)

        # ---------------- validate -----------------
        if (epoch + 1) % opt.testinterval == 0:
            validate(opt, validloader, net, epoch, optimizer, scheduler, validate_nrmse)
        if (epoch + 1) % opt.saveinterval == 0:
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if len(opt.scheduler) > 0:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch + 1))


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

    if min(validate_nrmse) > np.mean(nrmses):
        checkpoint = {'epoch': epoch + 1,
                      'state_dict': net.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if len(opt.scheduler) > 0 and scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()
        torch.save(checkpoint, opt.out + '/best.pth')
        validate_nrmse.append(np.mean(nrmses))

    summarystr = ""
    if count == 0:
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. mse: %0.4f, nrmse: %0.4f, psnr: %0.2f, ssim: %0.4f' % (
        count, np.mean(mses), np.mean(nrmses), np.mean(psnrs), np.mean(ssims))
    print(summarystr)
    print(summarystr, file=opt.fid)
    opt.fid.flush()


def main(opt):
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    opt.out = opt.out + '/' + opt.model
    os.makedirs(opt.out, exist_ok=True)
    opt.fid = open(opt.out + '/{}.txt'.format(opt.model), 'w')
    ostr = 'ARGS: ' + ' '.join(sys.argv[:])
    print(opt, '\n')
    print(opt, '\n', file=opt.fid)
    print('\n%s\n' % ostr)
    print('\n%s\n' % ostr, file=opt.fid)
    print('getting dataloader', opt.root)
    trainloader, validloader = get_data_loader(opt)
    t0 = time.perf_counter()
    net = get_model(opt)
    print_networks(net, False)
    if not opt.test:
        train(opt, trainloader, validloader, net)
    else:
        if len(opt.weights) > 0:
            print('loading checkpoint', opt.weights)
            load_checkpoint_flexible(net, opt.weights, opt.device)
            print('time: %0.1f' % (time.perf_counter() - t0))
        testAndMakeCombinedPlots(net, validloader, opt)
    opt.fid.close()
    if not opt.test:
        generate_convergence_plots(opt, opt.out + '/{}.txt'.format(opt.model))
    print('time: %0.1f' % (time.perf_counter() - t0))
    if opt.disposableTrainingData and not opt.test:
        print('deleting training data')
        # preserve a few samples
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


if __name__ == '__main__':
    # torch.manual_seed(123)
    torch.cuda.manual_seed(123)  # 为当前GPU设置随机种子
    opt = options()
    main(opt)
