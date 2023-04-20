import torch
import numpy as np 

from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.utils import gridapply
from tqdm import tqdm
from matplotlib import pyplot as plt

def compute_coverage(model, benchmark, config):
    """Compute the empirical expected coverage of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        (Tensor, Tensor): a tuple (levels, coverages) containing the coverages associated to different levels
    """

    dataset = benchmark.get_coverage_set(config["coverage_set_size"])
    if model.sampling_enabled():
        return expected_coverage_mc(model.get_posterior_fct(), dataset, n=benchmark.get_nb_cov_samples())
    else:
        return expected_coverage_ni(model, dataset, benchmark.get_domain(), bins=benchmark.get_cov_bins())

def compute_normalized_entropy_log_posterior(model, benchmark, config):
    """Compute the average entropy and normalized log posterior associated to nominal parameter value of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        (float, float): a tuple (entropies, nominal_log_probs) containing the average entropy and normalized nominal log posterior density
    """

    dataset = benchmark.get_coverage_set(config["coverage_set_size"])
    domain = benchmark.get_domain()
    bins = benchmark.get_cov_bins()

    nominal_log_probs = []
    entropies = []
    
    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit='pair'):
            _, log_probs = gridapply(lambda theta: model(theta, x), domain, bins=bins)
            
            lower, upper = domain
            dims = len(lower)

            if type(bins) is int:
                bins = [bins] * dims

            log_volume = np.log(np.prod([(u-l)/b for u, l, b in zip(upper.numpy(), lower.numpy(), bins)]))

            nominal_log_prob = model(nominal_theta, x).item()
            normalizing_constant = log_volume + log_probs.flatten().logsumexp(dim=0).item()
            log_probs = log_probs - normalizing_constant
            nominal_log_prob = nominal_log_prob - normalizing_constant
            
            log_bin_probs = log_probs + log_volume
            entropy = -torch.sum(torch.exp(log_bin_probs) * log_bin_probs).item()

            nominal_log_probs.append(nominal_log_prob)
            entropies.append(entropy)

    return np.mean(entropies), np.mean(nominal_log_probs)

def compute_log_posterior(model, benchmark, config):
    """Compute the average log posterior associated to nominal parameter value of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the average nominal log posterior density
    """

    nominal_log_probs = []
    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])

    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit='pair'):
            nominal_log_prob = model(nominal_theta, x)
            nominal_log_probs.append(nominal_log_prob)
            
    return torch.mean(torch.cat(nominal_log_probs)).item()

def compute_balancing_error(model, benchmark, config):
    """Compute the average balancing error of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the average balancing error
    """

    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])
    d_joints = []
    d_marginals = []
    prior = benchmark.get_prior()

    with torch.no_grad():
        for theta_joint, x in tqdm(dataset, unit='pair'):
            theta_marginal = torch.roll(theta_joint, 1, dims=0)
            d_joint = torch.sigmoid(model(theta_joint, x).cpu() - prior.log_prob(theta_joint))
            d_marginal = torch.sigmoid(model(theta_marginal, x).cpu() - prior.log_prob(theta_marginal))
            d_joints.append(d_joint)
            d_marginals.append(d_marginal)

        balancing = torch.mean(torch.cat(d_joints)).item() + torch.mean(torch.cat(d_marginals)).item()

    return np.absolute(1-balancing)

def compute_prior_mixture_coef(model, benchmark, config):
    """Compute the mixture coef of prior augmented models.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the mixture coef
    """

    prior_mixture_coefs = []
    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])

    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit='pair'):
            prior_mixture_coef = torch.sigmoid(model.prior_mixture(model.embedding(x)).squeeze())
            prior_mixture_coefs.append(prior_mixture_coef)
    
    return torch.cat(prior_mixture_coefs)

def plot_1d_posterior(model, benchmark, config, x, nominal_theta, ax=None):
    """Plot the posterior for 1d benchmarks

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration
        x (Tensor): the observation
        nominal_theta (Tensor): the parameters used to simulate the observation
        ax: the Axes on which to produce the plot

    Returns:
        Tensor: the posterior
    """

    pass

def plot_2d_posterior(model, benchmark, config, x, nominal_theta, ax=None, vmax=None, save_name=None):
    """Plot the posterior for 2d benchmarks

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration
        x (Tensor): the observation
        nominal_theta (Tensor): the parameters used to simulate the observation
        ax: the Axes on which to produce the plot
        vmax: The maximal value on the colormap
        save_name: The filename used to save the plot

    Returns:
        Tensor: the posterior
    """

    @torch.no_grad()
    def posterior(theta):
        return model(theta, x).exp()
    
    theta, probs = gridapply(posterior, benchmark.get_domain(), bins=benchmark.get_cov_bins())

    if ax is None:
        ax = plt.gca()
        set_colorbar = True
    else:
        set_colorbar = False
    
    if vmax is not None:
        cplot = ax.contourf(theta[..., 0], theta[..., 1], probs, cmap='Blues', vmax=vmax)
    else:
        cplot = ax.contourf(theta[..., 0], theta[..., 1], probs, cmap='Blues')
    
    ax.plot(nominal_theta[0], nominal_theta[1], '*', markersize=8, color='k')

    if set_colorbar:
        plt.colorbar(cplot)

    if save_name is not None:
        plt.savefig(save_name)

    return probs

def plot_nd_posterior(model, benchmark, config, x, nominal_theta, ax=None):
    """Plot the posterior for >2d benchmarks

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration
        x (Tensor): the observation
        nominal_theta (Tensor): the parameters used to simulate the observation
        ax: the Axes on which to produce the plot

    Returns:
        Tensor: the posterior
    """

    pass