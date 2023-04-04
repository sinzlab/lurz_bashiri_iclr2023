import numpy as np
import torch
from torch.nn import functional as F


def get_slab_mean_and_variance_from_moments(mean, variance, q, loc, sigma_clamp_value=None):
    q = q.clone()
    q[q == 0.0] = torch.nan

    threshold = loc * (q + (1 - q) / 2)
    q[mean <= threshold] = torch.nan

    E_slab = (mean - (1 - q) * loc / 2) / q - loc
    Var_slab = (variance - (1 - q) * (loc**2) / 12 - q * (1 - q) * (E_slab - loc / 2) ** 2) / q

    if sigma_clamp_value is not None:
        Var_slab = torch.clamp(Var_slab, min=sigma_clamp_value)  # 0.0001

    return E_slab, Var_slab


def get_zig_params_from_moments(mean, variance, q, loc):
    E_slab, Var_slab = get_slab_mean_and_variance_from_moments(mean, variance, q, loc)
    theta = Var_slab / E_slab
    k = (E_slab**2) / Var_slab

    assert (theta[torch.where(~torch.isnan(theta))[0]] > 0.0).all(), "theta must be positive"
    assert (k[torch.where(~torch.isnan(k))[0]] > 0.0).all(), "k must be positive"
    return theta, k


def get_zil_params_from_moments(mean, variance, q, loc, eps=1.0e-3, sigma_clamp_value=None):
    E_slab, Var_slab = get_slab_mean_and_variance_from_moments(
        mean, variance, q, loc, sigma_clamp_value=sigma_clamp_value
    )
    A = Var_slab / (E_slab**2) + 1
    mu = torch.log(E_slab / torch.sqrt(A))
    sigma2 = torch.log(A)
    assert (sigma2[torch.where(~torch.isnan(sigma2))[0]] > 0.0).all(), "variance must be positive"

    return mu, sigma2


def lognormal_mean_transform(dist_params, use_torch=True):
    exp = torch.exp if use_torch else np.exp

    mu, sigma2 = dist_params["mean"], dist_params["variance"]
    mean = exp(mu + sigma2 / 2)
    assert not (mean.isnan().any() or mean.isinf().any())
    return mean


def lognormal_variance_transform(dist_params, use_torch=True):
    exp = torch.exp if use_torch else np.exp

    mu, sigma2 = dist_params["mean"], dist_params["variance"]
    variance = (exp(sigma2) - 1) * exp(2 * mu + sigma2)
    assert not (variance.isnan().any() or variance.isinf().any() or (variance <= 0).any())
    return variance
