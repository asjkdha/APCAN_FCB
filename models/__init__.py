import torch
import torch.nn as nn
from torch.nn import init
from models.APCAN_1 import APCAN as APCAN_1


def init_net(net, init_type='kaiming', init_gain=0.02, debug=False):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


def get_model(opt):
    print("---------------------------------{}-------------------------".format(opt.model))

    if opt.model.lower()[0:7] != 'apcan_1':
        raise ValueError("Only apcan_1 is kept. Please set opt.model to apcan_1_actin or apcan_1_er.")

    net = APCAN_1(opt)

    if torch.cuda.is_available() and not getattr(opt, 'cpu', False):
        net.cuda()
        net = nn.DataParallel(net)

    return init_net(net, init_type='normal', init_gain=0.02)
