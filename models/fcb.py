import numpy as np
import torch
import torch.nn as nn


def complexinit(weights_real, weights_imag, criterion='he'):
    if weights_real.shape != weights_imag.shape:
        raise ValueError('weights_real and weights_imag must have the same shape')

    _, channels, _, _ = weights_real.shape
    fan_in = channels
    fan_out = weights_real.shape[0]
    if criterion == 'glorot':
        scale = 1.0 / np.sqrt(fan_in + fan_out) / 4.0
    elif criterion == 'he':
        scale = 1.0 / np.sqrt(fan_in) / 4.0
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    kernel_shape = tuple(weights_real.shape)
    modulus = np.random.rayleigh(scale=scale, size=kernel_shape)
    phase = np.random.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    with torch.no_grad():
        weights_real.copy_(
            torch.as_tensor(weight_real, dtype=weights_real.dtype, device=weights_real.device)
        )
        weights_imag.copy_(
            torch.as_tensor(weight_imag, dtype=weights_imag.dtype, device=weights_imag.device)
        )


class DeepSparse(nn.Module):
    def __init__(self, channels, num_rows=502, num_cols=502, init='he'):
        super(DeepSparse, self).__init__()
        self.weights_real = nn.Parameter(
            torch.Tensor(1, channels, num_rows, num_cols // 2 + 1)
        )
        self.weights_imag = nn.Parameter(
            torch.Tensor(1, channels, num_rows, num_cols // 2 + 1)
        )
        self.size = (num_rows, num_cols)
        complexinit(self.weights_real, self.weights_imag, init)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('FCB expects input with shape BxCxHxW, but got {}'.format(tuple(x.shape)))

        _, channels, height, width = x.shape
        if channels != self.weights_real.shape[1]:
            raise ValueError(
                'FCB expects {} channels, but got {}'.format(self.weights_real.shape[1], channels)
            )
        if (height, width) != self.size:
            raise ValueError(
                'FCB expects spatial size {}, but got {}'.format(self.size, (height, width))
            )

        x_fft = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        xr, xi = x_fft.real, x_fft.imag
        yr = xr * self.weights_real - xi * self.weights_imag
        yi = xr * self.weights_imag + xi * self.weights_real
        y_fft = torch.complex(yr, yi)
        return torch.fft.irfftn(y_fft, s=self.size, dim=(-2, -1), norm=None)


class FCBLayer(nn.Module):
    def __init__(self, n_feat, num_rows=502, num_cols=502, act=None):
        super(FCBLayer, self).__init__()
        self.fourier = DeepSparse(n_feat, num_rows, num_cols)
        self.pointwise = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.act = act if act is not None else nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.fourier(x)
        res = self.pointwise(res)
        res = self.act(res)
        return x + res
