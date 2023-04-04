import torch.nn as nn
import torch
import numpy as np


def elu1(x):
    return F.elu(x) + 1.0


class NormalInverseGamma(nn.Module):
    def __init__(self, n_dims, apply_elu=True, eps=1e-8, alpha_greater_one=False):
        super().__init__()
        self.n_dims = n_dims
        self.apply_elu = apply_elu
        self.alpha_greater_one = alpha_greater_one
        self.eps = eps
        self._mu = nn.Parameter(torch.randn(1, n_dims))
        self._lambd = nn.Parameter(torch.rand(1, n_dims))
        self._alpha = nn.Parameter(torch.rand(1, n_dims))
        self._beta = nn.Parameter(torch.rand(1, n_dims))

        self.current_loss = 1.0e30

    @property
    def mu(self):
        return self._mu

    @property
    def lambd(self):
        return elu1(self._lambd) + self.eps if self.apply_elu else self._lambd

    @property
    def alpha(self):
        out = elu1(self._alpha) + self.eps if self.apply_elu else self._alpha
        return out + 1.0 if self.alpha_greater_one else out

    @property
    def beta(self):
        return elu1(self._beta) + self.eps if self.apply_elu else self._beta

    def log_prob(self, x, sigma_squared, mask=None):
        if mask is not None:
            log_prob = (
                0.5 * torch.log(self.lambd[:, mask])
                - 0.5 * torch.log(2 * np.pi * sigma_squared)
                + self.alpha[:, mask] * torch.log(self.beta[:, mask])
                - torch.lgamma(self.alpha[:, mask])
                - (self.alpha[:, mask] + 1) * torch.log(sigma_squared)
                - (2 * self.beta[:, mask] + self.lambd[:, mask] * (x - self.mu[:, mask]) ** 2) / (2 * sigma_squared)
            )

        else:
            log_prob = (
                0.5 * torch.log(self.lambd)
                - 0.5 * torch.log(2 * np.pi * sigma_squared)
                + self.alpha * torch.log(self.beta)
                - torch.lgamma(self.alpha)
                - (self.alpha + 1) * torch.log(sigma_squared)
                - (2 * self.beta + self.lambd * (x - self.mu) ** 2) / (2 * sigma_squared)
            )

        if log_prob.isnan().any() or log_prob.isinf().any():
            import ipdb

            ipdb.set_trace()
        return log_prob

    def sample(self):
        raise NotImplementedError()

    def forward(self):
        return self.mu, self.lambd, self.alpha, self.beta
