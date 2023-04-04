import numpy as np
import torch
from scipy.special import digamma
from scipy.special import gamma as gamma_fn
from scipy.stats import gamma as gamma_distribution
from scipy.stats import invgamma
from torch.distributions import Gamma, MultivariateNormal
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralpredictors.layers.encoders.mean_variance_functions import fitted_zil_mean, fitted_zil_variance


class Dataset_from_dict(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return_dict = {}
        for dataset_name, data_set in self.data.items():
            return_dict[dataset_name] = data_set[index]
        return return_dict

    def __len__(self):
        return len(list(self.data.values())[0])


def make_gaussian_data(mean, cov, n_samples=1000):
    distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
    samples = distrib.sample((n_samples,))
    return samples


def make_gamma_data(ks, thetas, n_samples=1000):
    samples = []
    for i in range(len(thetas)):
        distrib = Gamma(concentration=ks[i], rate=1 / thetas[i])
        samples.append(distrib.sample((n_samples,)))
    return torch.stack(samples).T


def gaussian_entropy(covariance):
    N = np.linalg.matrix_rank(covariance)
    return N / 2 * np.log(2 * np.pi * np.e) + 0.5 * np.log(np.linalg.det(covariance))


def gamma_entropy(theta, k):
    entropy = 0
    for i in range(len(theta)):
        entropy += k[i] + np.log(theta[i]) + np.log(gamma_fn(k[i])) + (1 - k[i]) * digamma(k[i])
    return entropy


def simulate_neuron_data(
    n_images,
    n_repeats,
    n_neurons,
    zero_inflation_level=0.0,
    exp_data=False,
    loc=None,
    mu_bounds=[-2.0, -1.0],
    lambda_bounds=[0.5, 2.0],
    alpha_bounds=[15, 20],
    beta_bounds=[0.1, 0.2],
    random_state=None,
):

    np.random.seed(random_state)

    if exp_data and (loc is None):
        raise ValueError("If you are simulating from LogNormal dist, please specify a loc parameter.")

    gt_mus = np.random.uniform(mu_bounds[0], mu_bounds[1], n_neurons)
    gt_lambdas = np.random.uniform(lambda_bounds[0], lambda_bounds[1], n_neurons)
    gt_alphas = np.random.uniform(alpha_bounds[0], alpha_bounds[1], n_neurons)
    gt_betas = np.random.uniform(beta_bounds[0], beta_bounds[1], n_neurons)

    images_means, images_variances = [], []
    for gt_mu, gt_lambda, gt_alpha, gt_beta in tqdm(zip(gt_mus, gt_lambdas, gt_alphas, gt_betas), total=len(gt_mus)):
        images_variance = invgamma(a=gt_alpha, loc=0, scale=1 / gt_beta).rvs(n_images)
        m = np.ones(len(images_variance)) * gt_mu
        v = images_variance / gt_lambda
        images_mean = np.random.normal(m, np.sqrt(v), size=(len(m)))

        images_means.append(images_mean[:, None])
        images_variances.append(images_variance[:, None])

    images_means = np.hstack(images_means)
    images_variances = np.hstack(images_variances)

    simulated_data = []
    for image_mean, image_variance in tqdm(zip(images_means, images_variances), total=len(images_means)):
        samples_per_image = np.random.normal(loc=image_mean, scale=np.sqrt(image_variance), size=(n_repeats, n_neurons))
        samples_per_image = np.exp(samples_per_image) if exp_data else samples_per_image
        simulated_data.append(samples_per_image)
    simulated_data = np.stack(simulated_data, axis=1) + loc

    zero_response_indices = np.random.rand(n_repeats, n_images, n_neurons) < zero_inflation_level
    simulated_data[zero_response_indices] = 0.0

    # full distribution mean and variance
    q = 1 - zero_inflation_level
    means = fitted_zil_mean(images_means, images_variances, q, loc, use_torch=False)
    variances = fitted_zil_variance(images_means, images_variances, q, loc, use_torch=False)

    zil_params = {"mean": images_means, "variance": images_variances, "q": q, "loc": loc}

    return simulated_data, means, variances, zil_params


def simulate_neuron_data_advanced(
    n_images,
    n_repeats,
    n_neurons,
    zero_inflation_level=0.0,
    loc=np.exp(-10),
    alpha_for_lambda=8.29,
    beta_for_lambda=7.32,
    alpha_for_alpha=27.81,
    beta_for_alpha=0.8,
    image_mean_shift=0.0,
    image_variance_scale=0.2,
    mu_mu_spread=1.2,
    random_state=None,
):
    np.random.seed(random_state)
    gt_lambdas = gamma_distribution(a=alpha_for_lambda, scale=1 / beta_for_lambda).rvs(n_neurons)
    gt_alphas = gamma_distribution(a=alpha_for_alpha, scale=1 / beta_for_alpha).rvs(n_neurons)

    mean = np.array([-2.61 * mu_mu_spread + image_mean_shift, 1.81 * image_variance_scale])
    cov_mat = np.array(
        [
            [0.11 * mu_mu_spread**2, -0.07 * mu_mu_spread * image_variance_scale],
            [-0.07 * mu_mu_spread * image_variance_scale, 0.08 * image_variance_scale**2],
        ]
    )

    gt_mus, avg_variances = np.random.multivariate_normal(mean, cov_mat, size=n_neurons).T
    gt_betas = gt_alphas / avg_variances

    # In order to improve SNR: increase v and/or decrease images_variance
    images_means, images_variances = [], []
    for gt_mu, gt_lambda, gt_alpha, gt_beta in tqdm(zip(gt_mus, gt_lambdas, gt_alphas, gt_betas), total=len(gt_mus)):
        # images_variance = invgamma(a=gt_alpha, loc=0, scale=1 / gt_beta).rvs(n_images)
        images_variance = gamma_distribution(a=gt_alpha, loc=0, scale=1 / gt_beta).rvs(n_images)
        m = np.ones(len(images_variance)) * gt_mu
        v = images_variance / gt_lambda
        images_mean = np.random.normal(m, np.sqrt(v), size=(len(m)))

        images_means.append(images_mean[:, None])
        images_variances.append(images_variance[:, None])

    images_means = np.hstack(images_means)
    images_variances = np.hstack(images_variances)

    simulated_data = []
    for image_mean, image_variance in tqdm(zip(images_means, images_variances), total=len(images_means)):
        # samples_per_image = np.random.normal(loc=image_mean, scale=np.sqrt(image_variance), size=(n_repeats, n_neurons))
        # samples_per_image = (np.random.randn(n_repeats, n_neurons) + image_mean[None, :]) * np.sqrt(image_variance)[None, :]
        # samples_per_image = np.exp(samples_per_image)
        samples_per_image = np.random.lognormal(
            mean=image_mean, sigma=np.sqrt(image_variance), size=(n_repeats, n_neurons)
        )
        if (samples_per_image > 10000).any():
            breakpoint()
        simulated_data.append(samples_per_image)
    simulated_data = np.stack(simulated_data, axis=1) + loc

    zero_response_indices = np.random.rand(n_repeats, n_images, n_neurons) < zero_inflation_level
    simulated_data[zero_response_indices] = 0.0

    # full distribution mean and variance
    q = 1 - zero_inflation_level
    means = fitted_zil_mean(images_means, images_variances, q, loc, use_torch=False)
    variances = fitted_zil_variance(images_means, images_variances, q, loc, use_torch=False)

    zil_params = {"mu": images_means, "sigma2": images_variances, "q": q, "loc": loc}

    return simulated_data, means, variances, zil_params
