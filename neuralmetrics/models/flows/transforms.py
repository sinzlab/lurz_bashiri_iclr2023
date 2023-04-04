import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def mask_array(x, mask=None):
    if mask is not None:
        x = torch.where(mask, x, torch.zeros_like(x))
    return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def inv(self, x, mask=None):
        return mask_array(x, mask), torch.Tensor([0.0]).to(x.device)

    def forward(self, y, mask=None):
        return mask_array(y, mask), torch.Tensor([0.0]).to(y.device)


class SQRT(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, y, mask=None):
        x = mask_array(torch.sqrt(y), mask)
        logdet = mask_array(-np.log(2) - 0.5 * torch.log(y + self.eps), mask)
        return x, logdet

    def inv(self, x, mask=None):
        y = mask_array(x**2, mask)
        logdet = mask_array(np.log(2) + torch.log(x + self.eps), mask)
        return y, logdet


class Anscombe(nn.Module):
    def __init__(self):
        super().__init__()

    def inv(self, x, mask=None):
        """
        From Normal to Poisson
        """
        y = mask_array((x / 2) ** 2 - 3 / 8, mask)
        logdet = mask_array(-np.log(2) + torch.log(x), mask)
        return y, logdet

    @staticmethod
    def anscombe(x):
        return 2 * torch.sqrt(x + 3 / 8)

    def forward(self, y, mask=None):
        """
        From Poisson to Normal
        """
        x = mask_array(self.anscombe(y), mask)
        logdet = mask_array(np.log(2) - torch.log(self.anscombe(y)), mask)
        return x, logdet


class Log(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    # @inf_nan_check
    def inv(self, x, mask=None):
        y = mask_array(torch.exp(x), mask)
        logdet = mask_array(x, mask)
        return y, logdet

    # @inf_nan_check
    def forward(self, y, mask=None):
        x = mask_array(torch.log(y + self.eps), mask)
        logdet = mask_array(-torch.log(y + self.eps), mask)
        return x, logdet


class Exp(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x, mask=None):
        y = mask_array(torch.log(x + self.eps), mask)
        logdet = mask_array(-torch.log(x + self.eps), mask)
        return y, logdet

    def forward(self, y, mask=None):
        x = mask_array(torch.exp(y), mask)
        logdet = mask_array(y, mask)
        return x, logdet


class ELU(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.register_buffer("alpha", torch.tensor([1.0]))
        self.eps = eps

    def inv_elu(self, x):
        device = x.device
        return torch.maximum(torch.zeros(1).to(device), x) + torch.minimum(
            torch.zeros(1).to(device), torch.log(x / self.alpha + 1.0 + self.eps)
        )

    def inv(self, x, mask=None):
        device = x.device
        assert not (x <= -1.0).any(), "Inverse Elu is not defined for values smaller or equal to -1"
        y = mask_array(self.inv_elu(x), mask)
        logdet = mask_array(torch.maximum(torch.zeros(1).to(device), torch.log(1 / (x + 1) + self.eps)), mask)
        return y, logdet

    def forward(self, y, mask=None):
        device = y.device

        x = mask_array(F.elu(y), mask)
        logdet = mask_array(torch.minimum(torch.zeros(1).to(device), y), mask)
        return x, logdet


class Tanh(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x, mask=None):
        y = mask_array(0.5 * torch.log((1 + x) / (1 - x) + self.eps), mask)
        logdet = mask_array(-torch.log(1 - x**2 + self.eps), mask)
        return y, logdet

    def forward(self, y, mask=None):
        x = mask_array(torch.tanh(y), mask)
        logdet = mask_array(torch.log((1 - torch.tanh(y) ** 2).abs() + self.eps), mask)
        return x, logdet


class Sigmoid(nn.Module):
    def __init__(self, temp=1.0, eps=1e-12):
        super().__init__()
        self.temp = temp
        self.eps = eps

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x / self.temp))

    def logit(self, x):
        return self.temp * torch.log(x / (1 - x) + self.eps)

    # @inf_nan_check
    def forward(self, y, mask=None):
        x = mask_array(self.sigmoid(y), mask)
        logdet = mask_array(torch.log((self.sigmoid(y) * (1 - self.sigmoid(y))) + self.eps), mask)
        return x, logdet

    # @inf_nan_check
    def inv(self, x, mask=None):
        y = mask_array(self.logit(x), mask)
        logdet = mask_array(torch.log((1 / x + 1 / (1 - x)).abs() + self.eps), mask)
        return y, logdet


class Logit(nn.Module):
    def __init__(self, temp=1.0, eps=1e-12):
        super().__init__()
        self.temp = temp
        self.eps = eps

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x / self.temp))

    def logit(self, x):
        return self.temp * torch.log(x / (1 - x) + self.eps)

    # @inf_nan_check
    def forward(self, y, mask=None):
        x = mask_array(self.logit(y), mask)
        logdet = mask_array(torch.log((1 / y + 1 / (1 - y)).abs() + self.eps), mask)
        return x, logdet

    # @inf_nan_check
    def inv(self, x, mask=None):
        y = mask_array(self.sigmoid(x), mask)
        logdet = mask_array(torch.log((self.sigmoid(x) * (1 - self.sigmoid(x))) + self.eps), mask)
        return y, logdet


class Affine(nn.Module):
    def __init__(
        self,
        n_dims=1,
        only_positive_shift=False,
        learn_t=True,
        learn_a=True,
        init_t=0.0,
        init_a=1.0,
        eps=1e-12,
    ):
        """
        Affine transformation layer with positively-constrained scale.
        """
        super().__init__()
        self._a = nn.Parameter(torch.zeros(1, n_dims) + init_a, requires_grad=learn_a)
        self._t = nn.Parameter(torch.zeros(1, n_dims) + init_t, requires_grad=learn_t)
        self.only_positive_shift = only_positive_shift
        self.eps = eps

    @property
    def a(self):
        self._a.data.clamp_(min=0.0)
        return self._a

    @property
    def t(self):
        if self.only_positive_shift:
            self._t.data.clamp_(min=0.0)
        return self._t

    # @inf_nan_check
    def forward(self, y, mask=None):
        x = mask_array(self.a * y + self.t, mask)
        logdet = mask_array(torch.log(torch.abs(self.a) + self.eps), mask)
        return x, logdet

    # @inf_nan_check
    def inv(self, x, mask=None):
        y = mask_array((x - self.t) / self.a, mask)
        logdet = mask_array(torch.log(1 / self.a + self.eps), mask)
        return y, logdet


class MonotonicRationalQuadratic(nn.Module):
    def __init__(self, k, n_dims):
        super().__init__()

        self._x_knots = nn.Parameter(torch.ones(n_dims, k + 1) / (k + 1))
        self._y_knots = nn.Parameter(torch.ones(n_dims, k + 1) / (k + 1))
        self._derivatives = nn.Parameter(torch.ones(n_dims, k + 1) * torch.log(torch.exp(torch.tensor(1.0)) - 1.0))

    @property
    def x_knots(self):
        x_knots = nn.Softplus()(self._x_knots).cumsum(dim=1)
        x_knots = x_knots - x_knots.min(dim=1, keepdim=True)[0]
        return x_knots / x_knots.max(dim=1, keepdim=True)[0]

    @property
    def y_knots(self):
        y_knots = nn.Softplus()(self._y_knots).cumsum(dim=1)
        y_knots = y_knots - y_knots.min(dim=1, keepdim=True)[0]
        return y_knots / y_knots.max(dim=1, keepdim=True)[0]

    @property
    def derivatives(self):
        return nn.Softplus()(self._derivatives)

    # @inf_nan_check
    def forward(self, y, mask=None):
        x, logdet = monotonic_rational_quadratic_transform_forward(y, self.x_knots, self.y_knots, self.derivatives)
        x = mask_array(x, mask)
        logdet = mask_array(logdet, mask)
        return x, logdet

    # @inf_nan_check
    def inv(self, x, mask=None):
        y, logdet = monotonic_rational_quadratic_transform_inverse(x, self.x_knots, self.y_knots, self.derivatives)
        y = mask_array(y, mask)
        logdet = mask_array(logdet, mask)
        return y, logdet


class MarginalFlow(nn.Module):
    def __init__(
        self,
        transforms,
        d_out,
        unit_variance_constraint=True,
        init_psi_diag_coef=1.0,
        logloc=0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(transforms)
        self.unit_variance_constraint = unit_variance_constraint

        self._mu = nn.Parameter(torch.zeros(d_out), requires_grad=False)
        self.logpsi_diag = nn.Parameter(torch.log(torch.ones(d_out) * init_psi_diag_coef), requires_grad=False)
        self.register_buffer("logloc", torch.tensor([logloc]))

    @property
    def mu(self):
        return self._mu

    @property
    def psi_diag(self):
        return torch.exp(self.logpsi_diag)

    @property
    def base_dist(self):
        return Normal(self.mu, self.psi_diag)

    def inv(self, x, mask=None):
        y, log_abs_jacobian = x, 0.0
        for l in reversed(self.layers):
            y, l_log_abs_jacobian = l.inv(y, mask)
            log_abs_jacobian = log_abs_jacobian + l_log_abs_jacobian
        return y, log_abs_jacobian

    def forward(self, y, mask=None, logdet_per_dimension=False):
        x, log_abs_jacobian = y, 0.0
        for l in self.layers:
            x, l_log_abs_jacobian = l(x, mask)
            log_abs_jacobian = log_abs_jacobian + l_log_abs_jacobian

            # if torch.isnan(log_abs_jacobian).any() or torch.isinf(log_abs_jacobian).any():
            #     breakpoint()
        return x, log_abs_jacobian if logdet_per_dimension else log_abs_jacobian.sum(1, keepdim=True)

    def log_prob(self, y, mask=None, per_dimension=False):
        x, log_abs_det = self.forward(y, mask=mask, logdet_per_dimension=True)
        loglikelihood = mask_array(self.base_dist.log_prob(x), mask)
        log_prob = loglikelihood + log_abs_det

        return log_prob if per_dimension else log_prob.sum(1, keepdim=True)

    def sample(self, n):
        base_samples = self.base_dist.sample((n,))
        return self.inv(base_samples)[0]
