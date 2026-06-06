import os
import re
import torch
import glob
import argparse
import skimage
import numpy as np
from skimage import io
from models import get_model
from utils.util import load_checkpoint_flexible

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


def extract_sim_frames(stack):
    if stack.ndim == 3 and stack.shape[0] >= 9:
        return stack[:9]
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
    files = glob.glob('%s/*.tif' % opt.root)
    files = sorted(files, key=lambda name: int(re.findall(r"\d+\d*", name)[-1]))

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
        skimage.io.imsave('%s/%s_%s.tif' % (opt.out, basename[:-4], opt.model), sr)


if __name__ == '__main__':
    net = LoadModel(opt)

    opt.root = './testing/actin'
    opt.out = './output'

    SIM_reconstruct9(net, opt)
