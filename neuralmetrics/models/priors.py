from collections import Counter

import numpy as np
import torch
import wandb
from torch import nn, optim

from neuralpredictors.training.early_stopping import early_stopping

from .distributions import NormalInverseGamma
from .score_functions import compute_gs_loss_over_target_repeats


def get_prior_for_q(resps, loc):
    """

    Args:
        resps (torch.Tensor): Neural responses of size (repeats, images, neurons)
        loc (float or torch.Tensor): Zero threshold

    Returns:
        Parameters alpha and beta for the prior over q
    """
    assert resps.ndim == 3, "data must be of size (repeats, images, neurons) "

    positive_responses = resps > loc
    positive_responses[torch.where(torch.isnan(resps))] = torch.nan

    n_positive_responses = torch.nansum(positive_responses, axis=0)
    n_responses = torch.sum(~torch.isnan(resps), axis=0)

    qs = n_positive_responses / n_responses

    mean = torch.mean(qs, axis=0)
    variance = torch.var(qs, axis=0)

    alpha = mean**2 * (1 - mean) / variance - mean
    beta = alpha * (1 / mean - 1)

    assert (alpha > 0.0).all() and (beta > 0.0).all(), "alpha and beta must be positive"
    return alpha, beta


def get_prior_for_gaussian(
    resps,
    lr=1e-1,
    per_neuron=False,
    mask=None,
    patience=10,
    lr_decay_factor=0.5,
    tolerance=1.0e-6,
    verbose=True,
    lr_decay_steps=3,
):
    min_lr = lr / 1000
    maximize = False

    mask = np.ones_like(resps) if mask is None else mask
    n_repeats, n_images, n_neurons = resps.shape

    resps = resps * mask

    if per_neuron:
        variances = np.nanvar(resps, axis=0)
        variances = np.where(variances == 0.0, np.nan, variances)
        image_idx = np.argsort(variances, axis=0)
        variances = np.take_along_axis(variances, image_idx, axis=0)
        neuron_idx = np.argsort(np.isnan(variances).sum(axis=0))
        variances = variances[:, neuron_idx]

        lambdas = np.nanmean(resps, axis=0)
        lambdas = np.take_along_axis(lambdas, image_idx, axis=0)
        lambdas = lambdas[:, neuron_idx]
        n_dims = n_neurons

    else:
        lambdas = np.nanmean(np.nanmean(resps, axis=0), axis=0)[:, None]
        variances = np.nanmean(np.nanvar(resps, axis=0), axis=0)[:, None] + 1e-7
        n_dims = 1

    # initialize loss
    prior_model = NormalInverseGamma(n_dims=n_dims, alpha_greater_one=True)
    prior_model.current_loss = 1.0e20
    optimizer = optim.Adam(prior_model.parameters(), lr=lr)

    prior_model.train()
    losses = []

    def stop_closure(model):
        return model.current_loss

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )
    for epoch, val_obj in early_stopping(
        prior_model,
        stop_closure,
        interval=1,
        patience=patience,
        start=0,
        max_iter=1.0e100,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=True,
        tracker=None,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        optimizer.zero_grad()

        loss = 0
        n_neurons_cumsum = 0
        for n_nans, n_neurons_without_nans in Counter(np.isnan(variances).sum(axis=0)).items():
            mask = np.zeros(variances.shape[-1])
            mask[n_neurons_cumsum:n_neurons_without_nans] = 1
            mask = mask.astype(bool)
            var = torch.from_numpy(variances[0 : variances.shape[0] - n_nans, mask])
            lambd = torch.from_numpy(lambdas[0 : variances.shape[0] - n_nans, mask])
            n_neurons_cumsum += n_neurons_without_nans

            log_prob = prior_model.log_prob(lambd, var, mask=mask)
            loss += -log_prob.sum()

        loss += -log_prob.sum()

        if loss.isnan().any():
            breakpoint()
        prior_model.current_loss = loss.item()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("Loss: {:.4f}, Epochs: {}".format(loss, epoch), end="\r", flush=True)

    prior_model.eval()

    if not per_neuron:
        mu_0 = prior_model.mu.cpu().data.repeat((1, n_neurons))
        nu = prior_model.lambd.cpu().data.repeat((1, n_neurons))
        alpha = prior_model.alpha.cpu().data.repeat((1, n_neurons))
        beta = prior_model.beta.cpu().data.repeat((1, n_neurons))

    else:
        mu_0 = prior_model.mu.cpu().data
        nu = prior_model.lambd.cpu().data
        alpha = prior_model.alpha.cpu().data
        beta = prior_model.beta.cpu().data

    return mu_0, nu, alpha, beta


def get_prior_for_gamma(resps, per_neuron=False, mask=None):
    mask = np.ones_like(resps) if mask is None else mask
    n_repeats, n_images, n_neurons = resps.shape

    resps = resps * mask

    idx_not_enough_positive_values = np.where(np.sum(~np.isnan(resps), axis=0) <= 1)
    resps[:, idx_not_enough_positive_values[0], idx_not_enough_positive_values[1]] = np.nan

    means = np.nanmean(resps, axis=0)
    variances = np.nanvar(resps, axis=0)

    if per_neuron:
        alpha = means**2 / variances
        beta = alpha / means
        alpha_zero = np.nanmean(beta, axis=0, keepdims=True) ** 2 / np.nanvar(beta, axis=0, keepdims=True)
        beta_zero = alpha_zero / np.nanmean(beta, axis=0, keepdims=True)
        alpha = np.nanmean(means, axis=0, keepdims=True) ** 2 / np.nanmean(variances, axis=0, keepdims=True)
    else:
        alpha = np.nanmean(means, axis=0, keepdims=True) ** 2 / np.nanmean(variances, axis=0, keepdims=True)
        beta = alpha / np.nanmean(means, axis=0, keepdims=True)
        alpha_zero = np.nanmean(beta, axis=1, keepdims=True) ** 2 / np.nanvar(beta, axis=1, keepdims=True)
        beta_zero = alpha_zero / np.nanmean(beta, axis=1, keepdims=True)

        alpha_zero = np.repeat(alpha_zero, n_neurons, axis=1)
        beta_zero = np.repeat(beta_zero, n_neurons, axis=1)

    return alpha, alpha_zero, beta_zero


def train_prior_for_gaussian(
    resps,
    gs_model,
    use_map=False,  # whether to train the prior for the MAP estimator
    lr=1e-1,
    patience=3,
    lr_decay_factor=0.5,
    tolerance=1.0e-6,
    verbose=True,
    lr_decay_steps=3,
    max_iter=1.0e100,
    logger=False,
):
    if logger:
        config = {}
        run = wandb.init(project="prior_optimization", config=config)

    min_lr = lr / 1000
    maximize = False

    # initialize loss
    optimizer = optim.Adam(gs_model.parameters(), lr=lr)

    gs_model.train()
    losses = []

    def stop_closure(model):
        return model.current_loss

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )
    for epoch, val_obj in early_stopping(
        gs_model.slab,
        stop_closure,
        interval=1,
        patience=patience,
        start=0,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=True,
        tracker=None,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        optimizer.zero_grad()
        gs_model.integrals_over_q_dict = gs_model.get_integrals_over_q()

        loss = compute_gs_loss_over_target_repeats(resps, gs_model, use_map=use_map)

        if logger:
            wandb.log({k: v for k, v in gs_model.named_parameters()})

        if loss.isnan().any():
            breakpoint()
        gs_model.slab.current_loss = loss.item()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print("Loss: {:.4f}, Epochs: {}".format(loss, epoch), end="\r", flush=True)

    gs_model.eval()
    return gs_model, loss
