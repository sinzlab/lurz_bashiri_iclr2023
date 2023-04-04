import numpy as np
import torch
from torch.distributions import Normal

from neuralpredictors.layers.encoders.mean_variance_functions import (fitted_zig_mean, fitted_zil_mean,
                                                                      fitted_zil_variance)
from neuralpredictors.measures.zero_inflated_losses import ZIGLoss, ZILLoss

from .gs_models import Gamma_GS, Gaussian_GS
from .gs_zero_inflation import Zero_Inflation_Base
from .utils import get_zig_params_from_moments, get_zil_params_from_moments


def compute_gs_loss_over_images(gs_model, resps, target_repeat_idx, use_map):
    if not isinstance(gs_model.slab, Gaussian_GS) and use_map:
        raise NotImplementedError("THE MAP estimator is so far only implemented for the ZIL loss.")

    n_images = resps.shape[1]
    n_neurons = resps.shape[2]
    device = list(gs_model.named_parameters())[0][-1].device
    try:
        loc = torch.Tensor([gs_model.loc]).to(device)
    except:
        pass

    # Compute losses and means
    loo_responses = np.vstack([resps[0:target_repeat_idx], resps[target_repeat_idx + 1 :]])

    loss = 0
    for image_idx in range(n_images):
        loo_responses_per_image = torch.Tensor(loo_responses[:, image_idx, :])
        left_out_response_per_image = torch.Tensor(resps[target_repeat_idx, image_idx, :][None, :])

        if np.isnan(left_out_response_per_image).sum() == n_neurons:
            continue

        elif np.isnan(left_out_response_per_image).sum() == 0:

            # bayesian
            likelihood, logdet, params = gs_model.posterior_predictive(
                left_out_response_per_image.to(device), loo_responses_per_image.to(device)
            )
            if use_map:
                mu_post = params["slab_params"]["posterior_params"]["mu_posterior"]
                alpha_post = params["slab_params"]["posterior_params"]["alpha_posterior"]
                beta_post = params["slab_params"]["posterior_params"]["beta_posterior"]
                n_pos = params["n_pos"]
                n_neg = params["n_neg"]
                alpha_q = gs_model.alpha_q
                beta_q = gs_model.beta_q

                mu_map = mu_post
                sigma2_map = beta_post / (alpha_post + 3 / 2)
                q_map = (n_pos + alpha_q - 1) / (n_neg + n_pos + alpha_q + beta_q - 2)
                # Mode of Beta dist is bimodal for alpha, beta < 1
                # q_map[torch.where(((alpha_q + n_pos > 1) & (beta_q + n_neg <= 1)))[0]] = 1
                # q_map[torch.where(((alpha_q + n_pos <= 1) & (beta_q + n_neg > 1)))[0]] = 0
                q_map = torch.clamp(q_map, 0.01, 0.99)

                loss += ZILLoss(per_neuron=True)(
                    left_out_response_per_image.to(device), (mu_map, sigma2_map, loc, q_map)
                )
            else:
                log_prob = torch.log(likelihood) + logdet
                loss += -log_prob
        else:
            raise ValueError("Wrong assignment of None values!")
    return loss.sum()


def compute_gs_loss_over_target_repeats(resps, gs_model, use_map):
    repeat_indices = np.arange(resps.shape[0])

    loss = 0.0
    for target_repeat_idx in repeat_indices:
        loss_ = compute_gs_loss_over_images(gs_model, resps, target_repeat_idx, use_map=use_map)
        loss += loss_
    return loss


def compute_null_loss(resps, params_from_moments_function, loss_function, loc, device="cuda"):
    n_neurons = resps.shape[-1]
    if not isinstance(loc, torch.Tensor):
        loc = torch.Tensor([loc]).to(device)
    else:
        loc.to(device)

    null_data = resps.reshape(-1, resps.shape[-1])
    mean_null = torch.from_numpy(np.nanmean(null_data, axis=0)).to(device)
    var_null = torch.from_numpy(np.nanvar(null_data, axis=0)).to(device)
    q_null = torch.sum(torch.from_numpy(null_data).to(device) > loc, axis=0).to(device) / null_data.shape[0]
    slab_params = params_from_moments_function(mean_null, var_null, q_null, loc)

    loss = 0.0
    for repeat_idx in range(resps.shape[0]):
        for image_idx in range(resps.shape[1]):
            left_out_response_per_image = torch.Tensor(resps[repeat_idx, image_idx, :][None, :]).to(device)
            if not torch.isnan(left_out_response_per_image).sum() == n_neurons:
                loss += loss_function(left_out_response_per_image, (*slab_params, loc, q_null))
    return loss


def compute_score(
    gs_model,
    resps,
    images=None,
    trained_model=None,
    eps=0.0,
    sigma_clamp_value=None,
    gt_params=None,
    device="cuda",
):
    n_images = resps.shape[1]
    n_neurons = resps.shape[2]
    try:
        loc = torch.Tensor([gs_model.loc])
    except:
        pass

    # Initialize Tensors
    losses = torch.empty((n_images, n_neurons)) * np.nan
    losses_pe = torch.empty((n_images, n_neurons)) * np.nan
    losses_mean_matched = torch.empty((n_images, n_neurons)) * np.nan
    losses_var_matched = torch.empty((n_images, n_neurons)) * np.nan
    losses_q_matched = torch.empty((n_images, n_neurons)) * np.nan
    losses_all_matched = torch.empty((n_images, n_neurons)) * np.nan
    losses_map = torch.empty((n_images, n_neurons)) * np.nan
    losses_gt = torch.empty((n_images, n_neurons)) * np.nan
    losses_null = torch.empty((n_images, n_neurons)) * np.nan
    means_bayes = torch.empty((n_images, n_neurons)) * np.nan
    vars_bayes = torch.empty((n_images, n_neurons)) * np.nan
    means_pe = torch.empty((n_images, n_neurons)) * np.nan
    vars_pe = torch.empty((n_images, n_neurons)) * np.nan
    target_responses = torch.empty((n_images, n_neurons)) * np.nan

    # Compute params for Null model
    null_data = resps.reshape(-1, resps.shape[-1])
    mean_null = torch.from_numpy(np.nanmean(null_data, axis=0))
    var_null = torch.from_numpy(np.nanvar(null_data, axis=0))
    q_null = torch.sum(torch.from_numpy(null_data) > loc, axis=0) / null_data.shape[0]
    m_null, s_null = get_zil_params_from_moments(mean_null, var_null, q_null, loc, sigma_clamp_value=None)

    if trained_model is not None:
        data_keys = list(trained_model.readout.keys())
        data_key = data_keys[0]
        assert len(data_keys) == 1, "This is only implemented for one data set."
        m_model, s_model, _, q_model = trained_model(images.to(device), data_key=data_key)
        m_model = m_model.cpu().data
        s_model = s_model.cpu().data
        q_model = q_model.cpu().data

    # Compute losses and means
    with torch.no_grad():
        loo_responses = resps[1:, :, :]

        for image_idx in range(n_images):
            loo_responses_per_image = torch.Tensor(loo_responses[:, image_idx, :])
            left_out_response_per_image = torch.Tensor(resps[0, image_idx, :][None, :])

            if np.isnan(left_out_response_per_image).sum() == n_neurons:
                continue

            elif np.isnan(left_out_response_per_image).sum() == 0:

                # bayesian
                likelihood, logdet, params = gs_model.posterior_predictive(
                    left_out_response_per_image.to(device), loo_responses_per_image.to(device)
                )
                log_prob = torch.log(likelihood.cpu()) + logdet.cpu()

                losses[image_idx, :] = -log_prob
                means_bayes[image_idx, :] = params["mean"]
                vars_bayes[image_idx, :] = params["variance"]

                # point estimate
                mean_pe = torch.nanmean(loo_responses_per_image, axis=0)
                variance_pe = torch.from_numpy(np.nanvar(loo_responses_per_image, axis=0)) + eps

                if isinstance(gs_model, Zero_Inflation_Base) and isinstance(gs_model.slab, Gamma_GS):

                    ## point estimate from the data
                    q_pe = torch.sum(loo_responses_per_image > loc, axis=0) / loo_responses_per_image.shape[0]
                    theta_pe, k_pe = get_zig_params_from_moments(mean_pe, variance_pe, q_pe, loc)
                    theta_pe[torch.isnan(theta_pe)] = 0.01
                    k_pe[torch.isnan(k_pe)] = 0.01
                    q_pe = torch.clamp(q_pe, 0.01, 0.99)
                    means_pe[image_idx, :] = fitted_zig_mean(theta_pe, k_pe, loc, q_pe)

                    loss_pe = ZIGLoss(per_neuron=True)(left_out_response_per_image, theta_pe, k_pe, loc, q_pe)
                    losses_pe[image_idx, :] = loss_pe

                elif isinstance(gs_model, Zero_Inflation_Base) and isinstance(gs_model.slab, Gaussian_GS):

                    if trained_model is None:
                        ## point estimate from the data
                        q_pe = torch.sum(loo_responses_per_image > loc, axis=0) / loo_responses_per_image.shape[0]
                        m_pe, s_pe = get_zil_params_from_moments(
                            mean_pe, variance_pe, q_pe, loc, sigma_clamp_value=sigma_clamp_value
                        )
                        m_pe[torch.isnan(m_pe)] = loc + 0.1 + eps
                        s_pe[torch.isnan(s_pe)] = sigma_clamp_value
                        q_pe = torch.clamp(q_pe, 0.01, 0.99)
                    else:
                        m_pe = m_model[image_idx]
                        s_pe = s_model[image_idx]
                        q_pe = q_model[image_idx]
                    means_pe[image_idx, :] = fitted_zil_mean(m_pe, s_pe, q_pe, loc)
                    vars_pe[image_idx, :] = fitted_zil_variance(m_pe, s_pe, q_pe, loc)

                    loss_pe = ZILLoss(per_neuron=True)(left_out_response_per_image, m_pe, s_pe, loc, q_pe)
                    losses_pe[image_idx, :] = loss_pe

                    ## match the variance and q to only compare the mean (likelihood-based)
                    loss_mean_matched = ZILLoss(per_neuron=True)(
                        left_out_response_per_image, params["slab_params"]["mean"].cpu().data, s_pe, loc, q_pe
                    )
                    losses_mean_matched[image_idx, :] = loss_mean_matched

                    loss_var_matched = ZILLoss(per_neuron=True)(
                        left_out_response_per_image, m_pe, params["slab_params"]["variance"].cpu().data, loc, q_pe
                    )
                    losses_var_matched[image_idx, :] = loss_var_matched

                    loss_q_matched = ZILLoss(per_neuron=True)(
                        left_out_response_per_image, m_pe, s_pe, loc, params["q"].cpu().data
                    )
                    losses_q_matched[image_idx, :] = loss_q_matched

                    loss_all_matched = ZILLoss(per_neuron=True)(
                        left_out_response_per_image,
                        params["slab_params"]["mean"].cpu().data,
                        params["slab_params"]["variance"].cpu().data,
                        loc,
                        params["q"].cpu().data,
                    )
                    losses_all_matched[image_idx, :] = loss_all_matched

                    ## null model
                    loss_null = ZILLoss(per_neuron=True)(left_out_response_per_image, m_null, s_null, loc, q_null)
                    losses_null[image_idx, :] = loss_null

                    # MAP model
                    mu_post = params["slab_params"]["posterior_params"]["mu_posterior"].cpu().data
                    nu_post = (
                        params["slab_params"]["posterior_params"]["nu_posterior"].cpu().data
                    )  # not really needed for MAP but here for completeness
                    alpha_post = params["slab_params"]["posterior_params"]["alpha_posterior"].cpu().data
                    beta_post = params["slab_params"]["posterior_params"]["beta_posterior"].cpu().data
                    n_pos = params["n_pos"].cpu().data
                    n_neg = params["n_neg"].cpu().data
                    alpha_q = gs_model.alpha_q.cpu().data
                    beta_q = gs_model.beta_q.cpu().data

                    mu_map = mu_post
                    sigma2_map = beta_post / (alpha_post + 3 / 2)
                    q_map = (n_pos + alpha_q - 1) / (n_neg + n_pos + alpha_q + beta_q - 2)
                    # Mode of Beta dist is bimodal for alpha, beta < 1
                    q_map[torch.where(((alpha_q + n_pos > 1) & (beta_q + n_neg <= 1)))[0]] = 1
                    q_map[torch.where(((alpha_q + n_pos <= 1) & (beta_q + n_neg > 1)))[0]] = 0
                    q_map = torch.clamp(q_map, 0.01, 0.99)

                    loss_map = ZILLoss(per_neuron=True)(left_out_response_per_image, mu_map, sigma2_map, loc, q_map)
                    losses_map[image_idx, :] = loss_map

                    ## ground truth loss
                    if gt_params is not None:
                        m_gt = torch.from_numpy(gt_params["mu"][image_idx])
                        s_gt = torch.from_numpy(gt_params["sigma2"][image_idx])
                        loc = torch.tensor(gt_params["loc"])  # [image_idx]
                        q_gt = torch.tensor(gt_params["q"])  # [image_idx]
                        loss_gt = ZILLoss(per_neuron=True)(left_out_response_per_image, m_gt, s_gt, loc, q_gt)
                        losses_gt[image_idx, :] = loss_gt

                elif isinstance(gs_model, Gaussian_GS):
                    losses_pe[image_idx, :] = -Normal(mean_pe, variance_pe).log_prob(left_out_response_per_image)

                else:
                    raise ValueError("pp not implemented for this gs_model")

            else:
                raise ValueError("Wrong assignment of None values!")

            target_responses[image_idx, :] = left_out_response_per_image

    out = dict(
        losses=losses.cpu().data.numpy(),
        losses_pe=losses_pe.cpu().data.numpy(),
        losses_mean_matched=losses_mean_matched.cpu().data.numpy(),
        losses_var_matched=losses_var_matched.cpu().data.numpy(),
        losses_q_matched=losses_q_matched.cpu().data.numpy(),
        losses_all_matched=losses_all_matched.cpu().data.numpy(),
        losses_map=losses_map.cpu().data.numpy(),
        losses_gt=losses_gt.cpu().data.numpy(),
        losses_null=losses_null.cpu().data.numpy(),
        means_bayes=means_bayes.cpu().data.numpy(),
        means_pe=means_pe.cpu().data.numpy(),
        target_responses=target_responses.cpu().data.numpy(),
        vars_bayes=vars_bayes.cpu().data.numpy(),
        vars_pe=vars_pe.cpu().data.numpy(),
    )

    return out
