import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, StudentT
from torch.nn import functional as F


class Base_GS(nn.Module):
    def __init__(self):
        super().__init__()

    def posterior_predictive(self, responses_i_j_n, responses_i_notj_n, neuron_idx=slice(None)):
        assert (
            responses_i_j_n.ndim == 2 and responses_i_j_n.shape[0] == 1
        ), "responses_i_j_n must be of shape (1, neurons)"
        pp_stim_pdf, params = self.posterior_predictive_stim(data=responses_i_notj_n, neuron_idx=neuron_idx)
        pp_stim = pp_stim_pdf.log_prob(responses_i_j_n[:, neuron_idx]).exp()
        if torch.isnan(pp_stim).any() or torch.isinf(pp_stim).any() or (pp_stim == 0.0).any():
            breakpoint()
        _ = torch.Tensor(
            [0.0]
        )  # this is for compatibility with the zero-inflation model where a transform on the data can be applied
        return pp_stim, _, params

    def forward(self, responses_i_j_n, responses_i_notj_n):
        posterior_predictive, params = self.posterior_predictive(responses_i_j_n, responses_i_notj_n)
        return torch.log(posterior_predictive), params

    @staticmethod
    def get_number_of_repeats(x):
        assert len(x.shape) == 2, "Array must be of shape (repeats, neurons)"
        n_missing_trials = torch.isnan(x).sum().item() / x.shape[1]
        assert n_missing_trials.is_integer(), "n_missing_trials must be an integer value but is {}".format(
            n_missing_trials
        )
        n = len(x) - n_missing_trials
        return n


class Gaussian_GS(Base_GS):
    def __init__(
        self,
        mu_prior,
        nu_prior,
        alpha_prior,
        beta_prior,
        loc=0.0,
        train_prior_hyperparams=False,
        alpha_greater_one=False,
    ):
        super().__init__()
        alpha_prior = alpha_prior - 1 if alpha_greater_one else alpha_prior
        self.mu_prior = nn.Parameter(mu_prior, requires_grad=train_prior_hyperparams)
        self.nu_prior_log = nn.Parameter(nu_prior.log(), requires_grad=train_prior_hyperparams)
        self.alpha_prior_log = nn.Parameter(alpha_prior.log(), requires_grad=train_prior_hyperparams)
        self.beta_prior_log = nn.Parameter(beta_prior.log(), requires_grad=train_prior_hyperparams)
        self.loc = loc
        self.alpha_greater_one = alpha_greater_one
        self.current_loss = 1.0e30
        self.train_prior_hyperparams = train_prior_hyperparams

    @property
    def nu_prior(self):
        return self.nu_prior_log.exp()

    @property
    def alpha_prior(self):
        return self.alpha_prior_log.exp() + 1.0 if self.alpha_greater_one else self.alpha_prior_log.exp()

    @property
    def beta_prior(self):
        return self.beta_prior_log.exp()

    def posterior_predictive_stim(self, data, neuron_idx=slice(None)):
        assert data.ndim == 2, "data must be of shape (repeats, neurons)"

        data = data[:, neuron_idx]

        n = torch.nansum(~data.isnan(), axis=0, keepdim=True)

        data_mean = torch.nanmean(data, axis=0, keepdim=True)
        data_mean = torch.where(n > 0, data_mean, torch.zeros_like(data_mean))

        # Compute posterior parameters
        mu_posterior = (self.nu_prior[:, neuron_idx] * self.mu_prior[:, neuron_idx] + n * data_mean) / (
            self.nu_prior[:, neuron_idx] + n
        )
        nu_posterior = self.nu_prior[:, neuron_idx] + n
        alpha_posterior = self.alpha_prior[:, neuron_idx] + n / 2
        beta_posterior = (
            self.beta_prior[:, neuron_idx]
            + 0.5 * torch.nansum((data - data_mean) ** 2, axis=0, keepdim=True)
            + (n * self.nu_prior[:, neuron_idx] * (data_mean - self.mu_prior[:, neuron_idx]) ** 2)
            / ((self.nu_prior[:, neuron_idx] + n) * 2)
        )
        posterior_params = {
            "mu_posterior": mu_posterior,
            "nu_posterior": nu_posterior,
            "alpha_posterior": alpha_posterior,
            "beta_posterior": beta_posterior,
        }

        # Compute posterior predictive parameters
        nu_pp = 2 * alpha_posterior
        mu_pp = mu_posterior

        variance_pp = (beta_posterior * (nu_posterior + 1)) / (alpha_posterior * nu_posterior)

        # assert not ((variance_pp > 50).any() or (variance_pp <= 0).any())
        assert not (variance_pp <= 0).any()

        # Store distribution parameters and moments
        mean = mu_pp.clone()
        mean[nu_pp < 1] = torch.nan
        variance = variance_pp * nu_pp / (nu_pp - 2)
        variance[nu_pp < 2] = torch.nan

        variance_pp = variance_pp

        assert ~(
            torch.isnan(nu_pp).any()
            or torch.isnan(mu_pp).any()
            or torch.isnan(variance_pp).any()
            or torch.isnan(mean).any()
            or torch.isnan(variance).any()
        )
        params = {
            "df": nu_pp,
            "location": mu_pp,
            "scale": torch.sqrt(variance_pp),
            "mean": mean,
            "variance": variance,
            "posterior_params": posterior_params,
        }

        return StudentT(df=nu_pp, loc=mu_pp, scale=torch.sqrt(variance_pp)), params


class Gamma_GS(Base_GS):
    def __init__(self, alpha, alpha_zero, beta_zero, loc=0.0, train_prior_hyperparams=False):
        super().__init__()

        self.alpha_log = nn.Parameter(torch.Tensor(alpha).log(), requires_grad=train_prior_hyperparams)
        self.alpha_zero_log = nn.Parameter(torch.Tensor(alpha_zero).log(), requires_grad=train_prior_hyperparams)
        self.beta_zero_log = nn.Parameter(torch.Tensor(beta_zero).log(), requires_grad=train_prior_hyperparams)
        self.loc = loc
        self.current_loss = 1.0e30
        self.train_prior_hyperparams = train_prior_hyperparams

    @property
    def alpha(self):
        return self.alpha_log.exp()

    @property
    def alpha_zero(self):
        return self.alpha_zero_log.exp()

    @property
    def beta_zero(self):
        return self.beta_zero_log.exp()

    def posterior_predictive_stim(self, data, neuron_idx=slice(None)):

        assert data.ndim == 2, "data must be of shape (repeats, neurons)"

        data = data[:, neuron_idx]
        n = torch.nansum(~data.isnan(), axis=0, keepdim=True)

        # Compute posterior parameters
        alpha_prime = self.alpha_zero[:, neuron_idx] + n * self.alpha[:, neuron_idx]
        beta_prime = self.beta_zero[:, neuron_idx] + torch.nansum(data, axis=0)

        # Store distribution parameters and moments
        mean = self.alpha[:, neuron_idx] * beta_prime / (alpha_prime - 1)
        variance = None
        params = {
            "alpha": self.alpha[:, neuron_idx],
            "beta": alpha_prime,
            "p": 1,
            "q": beta_prime,
            "mean": mean,
            "variance": variance,
        }

        return (
            Generalized_beta_prime_distribution(self.alpha[:, neuron_idx], alpha_prime, 1, beta_prime),
            params,
        )


class BrainState_GS(Gaussian_GS):
    def __init__(self, *args, C, psi, train_bs_hyperparams=False, **kwargs):
        super().__init__(*args, **kwargs)

        assert C.ndim == 2, "C must be a 2D Tensor with shape (neurons, latent_dimensions)"
        n_neurons = C.shape[0]
        self.C = nn.Parameter(C, requires_grad=train_bs_hyperparams)

        assert psi.ndim == 1, "psi must be a 1D Tensor (either of length 1 or neurons)"
        if len(psi) == n_neurons:
            self.logpsi_diag = nn.Parameter(psi, requires_grad=train_bs_hyperparams)
            self.logvar = nn.Parameter(torch.zeros(1), requires_grad=False)

        elif len(psi) == 1:
            self.logpsi_diag = nn.Parameter(torch.zeros(n_neurons), requires_grad=False)
            self.logvar = nn.Parameter(psi, requires_grad=train_bs_hyperparams)

        else:
            raise ValueError("psi must be of length 1 or neurons")

    @property
    def psi(self):
        return (self.logpsi_diag + self.logvar).exp().diag()

    def posterior_predictive_bs(self, ind, data):
        mean, variance = self.compute_conditional_mean_and_variance(ind=ind, samples=data)
        return Normal(loc=mean, scale=torch.sqrt(variance))

    def posterior_predictive(self, responses_i_j_n, responses_i_notj_n, neuron_idx=slice(None)):
        assert responses_i_notj_n.ndim == 2, "responses_i_notj_n must be of shape (latents, neurons)"
        assert (
            responses_i_j_n.ndim == 2 and responses_i_j_n.shape[0] == 1
        ), "responses_i_j_n must be of shape (1, neurons)"
        # check if the diagonal is not too low
        if (sum(torch.diag(self.psi) <= 0.01).item() / torch.diag(self.psi).shape[0]) >= 0.5:
            warnings.warn(
                "More than 50% of the diagonal entries are smaller than 0.01. The results may not be accurate enough."
            )

        # get pp of stimulus part
        pp_stim_pdfs = self.posterior_predictive_stim(data=responses_i_notj_n, neuron_idx=neuron_idx)

        pps = []
        for idx in range(responses_i_j_n.shape[1]):
            # get pp of stimulus part
            df_stim, loc_stim, scale_stim = (
                pp_stim_pdfs.df[:, idx],
                pp_stim_pdfs.loc[:, idx],
                pp_stim_pdfs.scale[:, idx],
            )
            pp_stim_pdf = StudentT(df=df_stim, loc=loc_stim, scale=scale_stim)

            # get pp of brain-state part
            pp_bs_pdf = self.posterior_predictive_bs(idx, data=responses_i_j_n)

            # discretize pdfs
            grid, grid_delta = self.build_convolution_grid(pp_stim_pdf, pp_bs_pdf)
            pp_stim_pdf = pp_stim_pdf.log_prob(grid).exp()
            pp_bs_pdf = pp_bs_pdf.log_prob(grid).exp()

            # get convolution pdf
            pp_stim_pmf = pp_stim_pdf * grid_delta
            pp_bs_pmf = pp_bs_pdf * grid_delta
            conv_pmf = self.convolution(pp_stim_pmf, pp_bs_pmf, grid)
            conv_pdf = conv_pmf / grid_delta
            pp = self.evaluate_discretized_pdf(responses_i_j_n[:, idx].item(), conv_pdf, grid)
            pps.append(pp)
        return torch.Tensor(pps)

    def evaluate_discretized_pdf(self, x, pdf, grid):
        diffs, indices = torch.abs(x - grid).sort()

        if diffs[0] == 0.0:
            return pdf[indices[0]]
        else:
            # interpolate between the two nearest points to get the density value
            left_pdf = pdf[indices[0]]
            right_pdf = pdf[indices[1]]
            return (diffs[0] / (diffs[0] + diffs[1])) * (right_pdf - left_pdf) + left_pdf

    def build_convolution_grid(self, pp_stim_pdf, pp_bs_pdf, grid_delta=1e-2):
        device = pp_stim_pdf.loc.device
        grid_spread = (4 * torch.max(pp_stim_pdf.scale, pp_bs_pdf.scale)).item()
        grid_min = (torch.min(pp_stim_pdf.loc, pp_bs_pdf.loc) - grid_spread).item()
        grid_max = (torch.max(pp_stim_pdf.loc, pp_bs_pdf.loc) + grid_spread).item()
        grid = torch.arange(grid_min, grid_max, grid_delta, device=device)
        return grid, grid_delta

    def compute_conditional_mean_and_variance(self, ind, samples):
        cov_mat = self.psi + self.C @ self.C.T
        not_ind = ~np.isin(torch.arange(cov_mat.shape[0]), ind)
        cov_others_others_inv = self.woodbury(
            self.psi[:, not_ind][not_ind, :], self.C[not_ind, :], self.C[not_ind, :].T
        )
        cov_self_others = cov_mat[ind, not_ind]

        mean = cov_self_others @ cov_others_others_inv @ samples.T[not_ind]
        variance = cov_mat[ind, [ind]] - cov_self_others @ cov_others_others_inv @ cov_self_others.T

        return mean, variance

    def woodbury(self, A, U, V):
        k = V.shape[0]
        A_inv = torch.diag(1.0 / torch.diag(A))
        B_inv = torch.inverse(torch.eye(k).to(A.device) + V @ A_inv @ U)
        return A_inv - (A_inv @ U @ B_inv @ V @ A_inv)

    def convolution(self, x, y, grid):
        x_ = x.reshape(1, 1, -1)
        y_ = y.reshape(1, 1, -1)
        delta = grid[1] - grid[0]
        shift = int((torch.abs(grid[-1]) - torch.abs(grid[0])) / 2 / delta)
        conv_full = F.conv1d(x_, y_.flip(2), padding=len(y)).roll(shift, 2)
        conv_same = conv_full[:, :, conv_full.shape[2] // 4 : -conv_full.shape[2] // 4].flatten()
        return conv_same


class FADecomposition(nn.Module):
    def __init__(self, d_neurons, d_latent, psi_diag_same_variance=True):
        super().__init__()
        self.C = nn.Parameter(torch.rand(d_neurons, d_latent) * 0.1)
        if psi_diag_same_variance:
            self.logpsi_diag = nn.Parameter(torch.zeros(d_neurons), requires_grad=False)
            self.logvar = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.logpsi_diag = nn.Parameter(torch.zeros(d_neurons), requires_grad=True)
            self.logvar = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def psi(self):
        return (self.logpsi_diag + self.logvar).exp().diag()

    def forward(self):
        return self.C @ self.C.T + self.psi


class Generalized_beta_prime_distribution:
    def __init__(self, alpha, beta, p, q, loc=0.0, epsilon=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.loc = loc
        self.epsilon = epsilon

    def pdf(self, x):
        normalization = (
            self.q * (torch.lgamma(self.alpha) + torch.lgamma(self.beta) - torch.lgamma(self.alpha + self.beta)).exp()
        )
        out = (
            self.p
            * ((x + self.loc) / self.q) ** (self.alpha * self.p - 1)
            * (1 + ((x + self.loc) / self.q) ** self.p) ** (-self.alpha - self.beta)
            / normalization
        )
        return out

    def log_prob(self, x):
        return torch.log(self.pdf(x) + self.epsilon)


class Mixture_Model(nn.Module):
    """
    This is the mixture model suggested by reviewer 3 of the ICLR submission.
    """

    def __init__(self, n_images, n_neurons, train_weight=True):
        super().__init__()
        self.w_ = nn.Parameter(torch.zeros((1, n_images, n_neurons)), requires_grad=train_weight)

    @property
    def w(self):
        return torch.sigmoid(self.w_)

    def forward(self, l_gs, l_null):
        mixture = self.w * l_gs + (1 - self.w) * l_null
        return torch.log(mixture)
