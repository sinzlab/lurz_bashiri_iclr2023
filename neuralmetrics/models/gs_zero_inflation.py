from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal, StudentT

from .flows.transforms import Identity


class Zero_Inflation_Base(nn.Module):
    def __init__(
        self,
        loc,
        dist_slab,
        alpha_q,
        beta_q,
        possible_number_of_loo_repeats,
        transform=None,
        n_integral_steps_for_q=100,
    ):
        super().__init__()
        self.slab = dist_slab
        self.loc = loc
        self.transform = transform if transform is not None else Identity()
        self.n_integral_steps_for_q = n_integral_steps_for_q
        self.train_prior_hyperparams = self.slab.train_prior_hyperparams
        if self.train_prior_hyperparams:
            warn(
                "Hyperparameter optimization is set to True."
                + " Do not forget to recompute the integral over q after each training epoch in the training loop!"
            )

        self.possible_number_of_loo_repeats = possible_number_of_loo_repeats

        self.alpha_q_log = nn.Parameter(alpha_q.log(), requires_grad=self.train_prior_hyperparams)
        self.beta_q_log = nn.Parameter(beta_q.log(), requires_grad=self.train_prior_hyperparams)

    @property
    def alpha_q(self):
        return self.alpha_q_log.exp()

    @property
    def beta_q(self):
        return self.beta_q_log.exp()

    def transform_data(self, resps):
        data = resps.reshape(-1, resps.shape[-1])
        mask = ~torch.isnan(data)
        transformed_resps, logdet = self.transform(data, mask)
        transformed_resps = torch.where(mask, transformed_resps, torch.zeros_like(transformed_resps) * torch.nan)
        return transformed_resps.reshape(resps.shape), logdet

    def mean_variance_slab(self, dist, n_samples=10000):
        samples = dist.sample((n_samples,))
        samples, _ = self.transform.inv(samples)
        return samples.mean(dim=0), samples.var(dim=0)

    def get_integrals_over_q(self):
        """

        Args:
            resps (torch.Tensor): Neural responses of size (repeats, images, neurons)
            n_integral_steps (int): number of slices to split the integral into, i.e. the precision of the approximation of the integral
        Returns:


        """

        q_range = torch.linspace(0.001, 0.999, self.n_integral_steps_for_q).to(self.alpha_q.device)
        dq = q_range[1] - q_range[0]

        integral_dict = {}
        for n in self.possible_number_of_loo_repeats:

            integrals_slab, integrals_spike = [], []
            for n_positive in range(int(n + 1)):
                n_negative = n - n_positive

                integral_slab, integral_spike = 0, 0
                for q in q_range:
                    post = Beta(self.alpha_q + n_positive, self.beta_q + n_negative).log_prob(q).exp()
                    assert ~(torch.isnan(post).any() or torch.isinf(post).any()), "None or inf value encountered"

                    integral_slab += post * q * dq
                    integral_spike += post * (1 - q) * dq

                integrals_slab.append(integral_slab)
                integrals_spike.append(integral_spike)
            integral_dict[n] = {
                "integrals_slab": torch.vstack(integrals_slab),
                "integrals_spike": torch.vstack(integrals_spike),
            }
        return integral_dict

    def posterior_predictive(self, responses_i_j_n, responses_i_notj_n):

        n = self.slab.get_number_of_repeats(responses_i_notj_n)
        n_neurons = responses_i_j_n.shape[1]
        n_positive_responses = torch.sum(responses_i_notj_n > self.loc, axis=0)
        posterior_predictive = torch.zeros(n_neurons).to(responses_i_j_n.device)

        idx_spike = torch.where(responses_i_j_n <= self.loc)[1]  # neuron indicies
        idx_slab = torch.where(responses_i_j_n > self.loc)[1]
        slab_mask_target_repeat = torch.ones_like(responses_i_j_n)
        slab_mask_target_repeat[:, idx_spike] = torch.nan

        slab_mask_other_repeat = torch.ones_like(responses_i_notj_n)
        slab_mask_other_repeat[responses_i_notj_n <= self.loc] = torch.nan
        slab_mask_other_repeat[torch.where(responses_i_notj_n.isnan())] = torch.nan

        responses_i_j_n_slab = responses_i_j_n * slab_mask_target_repeat
        responses_i_notj_n_slab = responses_i_notj_n * slab_mask_other_repeat

        # Go to transformed space
        y_n_transformed, logdet = self.transform_data(responses_i_j_n_slab - self.loc)
        y_not_n_transformed, _ = self.transform_data(responses_i_notj_n_slab - self.loc)

        # Compute pp for slab
        pp_slab_dist, params_slab = self.slab.posterior_predictive_stim(y_not_n_transformed)  # return the distribution
        pp_slab, *_ = self.slab.posterior_predictive(
            y_n_transformed, y_not_n_transformed, idx_slab
        )  # returns the likelihood value

        # Get integral over q
        slab_integral_over_q = self.integrals_over_q_dict[n]["integrals_slab"][
            n_positive_responses, np.arange(n_neurons)
        ]
        slab_integral_over_q = torch.clamp(slab_integral_over_q, 1e-2, 1 - 1e-2)
        spike_integral_over_q = 1 - slab_integral_over_q

        posterior_predictive[idx_spike] = (spike_integral_over_q / self.loc)[idx_spike]
        posterior_predictive[idx_slab] = slab_integral_over_q[idx_slab] * pp_slab.flatten()

        # transform the mean and variance
        mean_slab, variance_slab = (
            self.mean_variance_slab(pp_slab_dist)
            if not isinstance(self.transform, Identity)
            else (params_slab["mean"], params_slab["variance"])
        )
        mean_slab = mean_slab + self.loc
        mean = (spike_integral_over_q * self.loc / 2) + (slab_integral_over_q * mean_slab)

        params = {
            "mean": mean,
            "q": slab_integral_over_q,
            "slab_mean": mean_slab,
            "slab_variance": variance_slab,
            "slab_params": params_slab,
            "n_pos": n_positive_responses,
            "n_neg": n - n_positive_responses,
        }
        if variance_slab is not None:
            variance = (
                spike_integral_over_q * self.loc**2 / 12
                + slab_integral_over_q * variance_slab
                + spike_integral_over_q * slab_integral_over_q * (self.loc / 2 - slab_integral_over_q * mean_slab) ** 2
            )
            params["variance"] = variance

        return posterior_predictive, logdet, params
