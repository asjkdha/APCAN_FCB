import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def inverse_softplus(value):
    value = float(value)
    if value <= 0.0:
        raise ValueError('inverse_softplus expects a positive value')
    return math.log(math.expm1(value))


def clone_activation(act):
    if act is None:
        return nn.ReLU(inplace=True)
    return copy.deepcopy(act)


def build_rfft_frequency_grid(num_rows, num_cols, dtype=torch.float32):
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError('num_rows and num_cols must be positive')

    fy = torch.fft.fftfreq(num_rows).to(dtype=dtype)
    fx = torch.fft.rfftfreq(num_cols).to(dtype=dtype)
    fy_grid = fy[:, None]
    fx_grid = fx[None, :]

    rho = torch.sqrt(fx_grid.pow(2) + fy_grid.pow(2))
    rho_max = torch.max(rho)
    if float(rho_max) > 0.0:
        rho = rho / rho_max

    theta = torch.atan2(fy_grid.expand_as(rho), fx_grid.expand_as(rho))
    return rho.unsqueeze(0).unsqueeze(0), theta.unsqueeze(0).unsqueeze(0)


def parse_sim_directions(directions):
    if directions is None:
        return [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0]
    if isinstance(directions, str):
        parts = [part.strip() for part in directions.split(',') if part.strip()]
        if not parts:
            raise ValueError('directions must contain at least one angle')
        return [math.radians(float(part)) for part in parts]

    parsed = [float(direction) for direction in directions]
    if not parsed:
        raise ValueError('directions must contain at least one angle')
    return parsed


class SIMFrequencyPrior(nn.Module):
    """Caches the SIM directional prior for full rFFT modes.

    The mask is M_SIM(rho, theta) = H(rho) * mean_d(M_d), where
    H(rho) = sigmoid((rho - rho_min) / tau) and each directional sector uses
    a modulo-pi line-orientation distance. Implemented from the DiffFNO paper
    formula, because official DiffFNO source code is not available.
    """

    def __init__(
        self,
        num_rows,
        num_cols,
        rho_min=0.25,
        tau=0.05,
        sigma_theta=math.pi / 18.0,
        directions=None,
    ):
        super(SIMFrequencyPrior, self).__init__()
        if tau <= 0.0:
            raise ValueError('tau must be positive')
        if sigma_theta <= 0.0:
            raise ValueError('sigma_theta must be positive')

        self.rho_min = float(rho_min)
        self.tau = float(tau)
        self.sigma_theta = float(sigma_theta)
        self.directions = parse_sim_directions(directions)

        rho, theta = build_rfft_frequency_grid(num_rows, num_cols, dtype=torch.float32)
        direction_masks = []
        for direction in self.directions:
            delta = theta - direction
            distance = 0.5 * torch.abs(torch.atan2(torch.sin(2.0 * delta), torch.cos(2.0 * delta)))
            direction_masks.append(torch.exp(-0.5 * (distance / self.sigma_theta).pow(2)))

        directional_mask = torch.stack(direction_masks, dim=0).mean(dim=0)
        high_frequency_gate = torch.sigmoid((rho - self.rho_min) / self.tau)
        sim_mask = high_frequency_gate * directional_mask

        self.register_buffer('rho', rho)
        self.register_buffer('theta', theta)
        self.register_buffer('sim_mask', sim_mask)

    def get_weight_terms(self):
        return self.rho, self.sim_mask

    def forward(self):
        return self.get_weight_terms()


class DeepSparse(nn.Module):
    """Mode-rebalanced Fourier convolution over all rFFT modes.

    The FCB branch computes Y(k1, k2) = X(k1, k2) * P(k1, k2), followed by
    irFFT. It adds the paper-formula mode rebalancing term
    w_SIM(rho, theta) = 1 + gamma_rad * rho**alpha + gamma_sim * M_SIM.
    Implemented from the DiffFNO paper formula, because official DiffFNO
    source code is not available.
    """

    def __init__(
        self,
        channels,
        num_rows=502,
        num_cols=502,
        init='he',
        alpha=0.7,
        gamma_init=1e-3,
        rho_min=0.25,
        tau=0.05,
        sigma_theta=math.pi / 18.0,
        directions=None,
    ):
        super(DeepSparse, self).__init__()
        if alpha <= 0.0:
            raise ValueError('alpha must be positive')

        self.channels = channels
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.size = (num_rows, num_cols)
        self.alpha = float(alpha)

        self.weights_real = nn.Parameter(
            torch.Tensor(1, channels, num_rows, num_cols // 2 + 1)
        )
        self.weights_imag = nn.Parameter(
            torch.Tensor(1, channels, num_rows, num_cols // 2 + 1)
        )
        init_gamma = inverse_softplus(gamma_init)
        self.raw_gamma_rad = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.raw_gamma_sim = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.freq_prior = SIMFrequencyPrior(
            num_rows,
            num_cols,
            rho_min=rho_min,
            tau=tau,
            sigma_theta=sigma_theta,
            directions=directions,
        )

        complexinit(self.weights_real, self.weights_imag, init)

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'DeepSparse expects input shape [B, {self.channels}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}'
            )

        _, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f'DeepSparse expects input shape [B, {self.channels}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}'
            )
        if (height, width) != self.size:
            raise ValueError(
                f'DeepSparse expects input shape [B, {self.channels}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}; FCB expects spatial size {self.size}, '
                f'but got {(height, width)}'
            )

    def forward(self, x):
        self._check_input(x)

        x_fft = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        xr = x_fft.real
        xi = x_fft.imag

        wr = self.weights_real
        wi = self.weights_imag
        yr = xr * wr - xi * wi
        yi = xr * wi + xi * wr

        gamma_rad = F.softplus(self.raw_gamma_rad)
        gamma_sim = F.softplus(self.raw_gamma_sim)
        rho, sim_mask = self.freq_prior.get_weight_terms()
        rho = rho.to(dtype=yr.dtype)
        sim_mask = sim_mask.to(dtype=yr.dtype)
        mode_weight = 1.0 + gamma_rad * torch.pow(rho, self.alpha) + gamma_sim * sim_mask

        yr = yr * mode_weight
        yi = yi * mode_weight

        y_fft = torch.complex(yr, yi)
        return torch.fft.irfftn(y_fft, s=self.size, dim=(-2, -1), norm=None)

    def loadweight(self, ilayer):
        if not isinstance(ilayer, nn.Conv2d):
            raise ValueError('loadweight expects a torch.nn.Conv2d layer')
        if ilayer.groups != self.channels:
            raise ValueError(
                f'loadweight expects depth-wise conv groups={self.channels}, but got {ilayer.groups}'
            )
        if ilayer.in_channels != self.channels or ilayer.out_channels != self.channels:
            raise ValueError(
                f'loadweight expects in_channels=out_channels={self.channels}, '
                f'but got in_channels={ilayer.in_channels}, out_channels={ilayer.out_channels}'
            )
        if ilayer.weight.dim() != 4 or ilayer.weight.shape[1] != 1:
            raise ValueError(
                f'loadweight expects conv weight shape [C, 1, K, K], but got {tuple(ilayer.weight.shape)}'
            )

        weight = ilayer.weight.detach().to(device=self.weights_real.device, dtype=self.weights_real.dtype)
        weight = weight.squeeze(1)
        if weight.dim() != 3:
            raise ValueError(
                f'loadweight expects squeezed weight shape [C, K, K], but got {tuple(weight.shape)}'
            )

        _, kernel_rows, kernel_cols = weight.shape
        if kernel_rows > self.num_rows or kernel_cols > self.num_cols:
            raise ValueError(
                f'loadweight kernel size {(kernel_rows, kernel_cols)} does not fit Fourier size {self.size}'
            )

        weight = torch.flip(weight, dims=(-2, -1))
        weight = F.pad(weight, (0, self.num_cols - kernel_cols, 0, self.num_rows - kernel_rows))
        weight = torch.roll(weight, shifts=(-1, -1), dims=(-2, -1))
        weight_fft = torch.fft.rfftn(weight, dim=(-2, -1), norm=None)

        with torch.no_grad():
            self.weights_real[0].copy_(weight_fft.real)
            self.weights_imag[0].copy_(weight_fft.imag)

    def loadweight_from_depthwise_conv(self, conv_layer):
        return self.loadweight(conv_layer)


class LocalDWBranch(nn.Module):
    """Local spatial branch for details that a pure Fourier branch can miss.

    It uses circular depth-wise 3x3 convolutions, point-wise projections, and
    shortcuts while preserving the input feature shape.
    """

    def __init__(self, channels, act=None):
        super(LocalDWBranch, self).__init__()
        self.depthwise1 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False, padding_mode='circular'
        )
        self.pointwise1 = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = clone_activation(act)
        self.depthwise2 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False, padding_mode='circular'
        )
        self.pointwise2 = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        r1 = self.depthwise1(x) + x
        r1 = self.act1(self.pointwise1(r1))
        r2 = self.depthwise2(r1) + r1
        r2 = self.pointwise2(r2)
        return r2 + x


class GlobalMRFCBBranch(nn.Module):
    """Global branch made from DeepSparse plus a point-wise projection."""

    def __init__(
        self,
        channels,
        num_rows=502,
        num_cols=502,
        act=None,
        init='he',
        alpha=0.7,
        gamma_init=1e-3,
        rho_min=0.25,
        tau=0.05,
        sigma_theta=math.pi / 18.0,
        directions=None,
    ):
        super(GlobalMRFCBBranch, self).__init__()
        self.fourier = DeepSparse(
            channels,
            num_rows,
            num_cols,
            init=init,
            alpha=alpha,
            gamma_init=gamma_init,
            rho_min=rho_min,
            tau=tau,
            sigma_theta=sigma_theta,
            directions=directions,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = clone_activation(act)

    def forward(self, x):
        g = self.fourier(x)
        g = self.pointwise(g)
        g = self.act(g)
        return g


class GatedFusion(nn.Module):
    """Gated fusion for global and local branches.

    The GFM formula is G = sigmoid(Conv1x1([v_global, v_local])) and
    v_fused = G * v_global + (1 - G) * v_local, followed by a point-wise
    projection.
    """

    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Conv2d(2 * channels, 1, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.bias)

    def forward(self, global_feat, local_feat):
        gate_input = torch.cat([global_feat, local_feat], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * global_feat + (1.0 - gate) * local_feat
        return self.proj(fused), gate


class FCBLayer(nn.Module):
    """APCAN-SIM-MRFCB-GF replacement for the original FCB layer.

    The layer combines FCB Fourier convolution, mode rebalancing, a SIM
    directional frequency prior, a local depth-wise branch, and gated fusion.
    Its forward formula is y = x + eta * Proj(G * v_g + (1 - G) * v_l), where
    v_g is the global MRFCB branch and v_l is the local DW branch.
    """

    def __init__(
        self,
        n_feat,
        num_rows=502,
        num_cols=502,
        act=None,
        init='he',
        alpha=0.7,
        gamma_init=1e-3,
        rho_min=0.25,
        tau=0.05,
        sigma_theta=math.pi / 18.0,
        directions=None,
        residual_scale=1e-3,
    ):
        super(FCBLayer, self).__init__()
        self.n_feat = n_feat
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.size = (num_rows, num_cols)
        self.global_branch = GlobalMRFCBBranch(
            n_feat,
            num_rows,
            num_cols,
            act,
            init=init,
            alpha=alpha,
            gamma_init=gamma_init,
            rho_min=rho_min,
            tau=tau,
            sigma_theta=sigma_theta,
            directions=directions,
        )
        self.local_branch = LocalDWBranch(n_feat, act)
        self.fusion = GatedFusion(n_feat)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))

    @property
    def fourier(self):
        return self.global_branch.fourier

    @property
    def pointwise(self):
        return self.fusion.proj

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'FCBLayer expects input shape [B, {self.n_feat}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}'
            )

        _, channels, height, width = x.shape
        if channels != self.n_feat:
            raise ValueError(
                f'FCBLayer expects input shape [B, {self.n_feat}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}'
            )
        if (height, width) != self.size:
            raise ValueError(
                f'FCBLayer expects input shape [B, {self.n_feat}, {self.num_rows}, {self.num_cols}], '
                f'but got {tuple(x.shape)}; FCB expects spatial size {self.size}, '
                f'but got {(height, width)}'
            )

    def forward(self, x):
        self._check_input(x)
        v_global = self.global_branch(x)
        v_local = self.local_branch(x)
        fused, gate = self.fusion(v_global, v_local)
        return x + self.residual_scale * fused
