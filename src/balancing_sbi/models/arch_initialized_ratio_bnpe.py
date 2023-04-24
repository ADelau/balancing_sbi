from .ratio_bnpe import RatioBNPEModel
from .base import ModelFactory

import torch
import os
from torch import nn as nn
import numpy as np 
import torch.optim as optim
import torch.nn.functional as F
import zuko
from tqdm import tqdm
from lampe.utils import GDStep
from torch.distributions import Normal

class ArchInitializedRatioBNPEFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, ArchInitializedRatioBNPEModel)

    def get_train_time(self, benchmark_time, epochs):
        return 2*super().get_train_time(benchmark_time, epochs)
    
class GaussianScaledFlowBasedRatio(nn.Module):
    def __init__(self, npe, prior, lower, upper):
        super().__init__()
        self.npe = npe
        self.prior = prior

        # We assume uniform prior [lower, upper]
        self.lower = lower
        self.upper = upper
        self.shift = self.lower
        self.scale = self.upper - self.lower

        # The base distribution
        self.normal = Normal(0., 1.)

    def forward(self, theta, x):
        theta_device = theta.get_device()
        return self.log_prob(theta, x) - self.prior.log_prob(theta.cpu()).to(theta_device if theta_device >= 0 else "cpu")

    def log_prob(self, theta, x):
        theta_device = theta.get_device()

        #transform prior U(lower, upper) to U(0,1)
        transformed_theta = (theta - self.shift)/self.scale

        # jacobian mapping U(lower, upper) to U(0,1)
        npe_probs = -torch.sum(torch.log(self.scale))

        # transform U(0,1) to N(0,1)
        transformed_theta = self.normal.icdf(transformed_theta.cpu()).to(theta_device if theta_device >= 0 else "cpu")

        # the posterior in transformed space
        npe_probs = npe_probs + self.npe(transformed_theta, x)

        # jacobian mapping U(0,1) to N(0,1)
        npe_probs = npe_probs - torch.sum(self.normal.log_prob(transformed_theta.cpu()).to(theta_device if theta_device >= 0 else "cpu"), axis=-1)
        npe_probs[torch.logical_not(torch.all(torch.isfinite(transformed_theta), dim=-1))] = -float("Inf")
        return npe_probs

    def sample(self, x, shape):
        return self.normal.cdf(self.npe.sample(x, shape)) * self.scale + self.shift

class UniformScaledFlowBasedRatio(nn.Module):
    def __init__(self, npe, prior, shift, scale):
        super().__init__()
        self.npe = npe
        self.prior = prior
        self.shift = shift
        self.scale = scale

    def forward(self, theta, x):
        theta_device = theta.get_device()
        return self.log_prob(theta, x) - self.prior.log_prob(theta.cpu()).to(theta_device if theta_device >= 0 else "cpu")

    def log_prob(self, theta, x):
        return self.npe((theta - self.shift) / self.scale, x) - torch.sum(torch.log(self.scale))

    def sample(self, x, shape):
        return self.npe.sample(x, shape) * self.scale + self.shift

class WeightedBNRELoss(nn.Module):
    def __init__(self, estimator, lmbda=100.0):
        super().__init__()

        self.estimator = estimator
        self.lmbda = lmbda
        self.running_l = None
        self.running_lb = None

    def forward(self, theta, x):
        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()
        l = (l1 + l0) / 2
        
        if self.running_l is None or self.running_lb is None:
            self.running_l = l.detach().item() + 1e-3
            self.running_lb = lb.detach().item() + 1e-3
            return l + self.lmbda * lb
        else:
            self.running_l = 0.9 * self.running_l + 0.1 * l.detach().item()
            self.running_lb = 0.9 * self.running_lb + 0.1 * lb.detach().item()
            return l/self.running_l + (self.lmbda/10) * (lb/self.running_lb)

class ArchInitializedRatioBNPEModel(RatioBNPEModel):
    # This only works for NSF architectures and for uniform priors.

    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)
        self.benchmark = benchmark
        self.lower, self.upper = self.benchmark.get_domain()
        self.lower = self.lower.to(self.device)
        self.upper = self.upper.to(self.device)

        if config["not_modify_base"]:
            # Keep gaussian base distribution
            self.model = GaussianScaledFlowBasedRatio(self.npe, self.prior, self.lower, self.upper)
        else:
            # Use U(-5, 5) as base distribution and add a transformation to the prior

            # Shift and scale to map U[-5, 5] to the prior
            self.shift = (self.lower + self.upper)/2
            self.scale = (self.upper - self.lower)/10

            self.model = UniformScaledFlowBasedRatio(self.npe, self.prior, self.shift, self.scale)

            self.flow.flow.base = zuko.flows.Unconditional(
                zuko.distributions.BoxUniform,
                -5*torch.ones(self.benchmark.get_parameter_dim()),
                5*torch.ones(self.benchmark.get_parameter_dim()),
                buffer=True,
            )

    def initialize(self, config):
        """Initialize the normalizing flow to have transformations closeto identity"""
        
        for transform in self.flow.flow.transforms:
            layers = list(transform.hyper.children())
            for layer in layers[:-1]:
                if isinstance(layer, zuko.nn.MaskedLinear):
                    # divide weights by weight scale
                    layer.weight = nn.Parameter(layer.weight/config["weight_scale"])
                    if config["init_bias"]:
                        # divide biases by weight scale
                        layer.bias = nn.Parameter(layer.bias/config["weight_scale"])
                    else:
                        # set biases to 0
                        layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

            # divide weights by weight scale  
            layers[-1].weight = nn.Parameter(layers[-1].weight/config["weight_scale"])
            if config["init_bias"]:
                # divide biases by weight scale
                layers[-1].bias = nn.Parameter(layers[-1].bias/config["weight_scale"])
            else:
                # set biases to 0
                layers[-1].bias = nn.Parameter(torch.zeros_like(layers[-1].bias))

    def train_models(self, train_set, val_set, config):
        if not config["no_initialize"]:
            self.initialize(config)

        if config["use_super_train"]:
            super().train_models(train_set, val_set, config)
        else:
            learning_rate = float(config["learning_rate"])
            epochs = config["epochs"]
            self.train()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            if config["schedule"]:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=float(config["min_lr"]))

            step = GDStep(optimizer)

            best_loss = float('inf')
            best_weights = self.model.state_dict()

            loss = self.get_loss_fct(config, config["weight_losses"])(self.model)

            train_losses = []
            val_losses = []

            with tqdm(range(epochs), unit='epoch') as tq:

                for epoch in tq:

                    if config["weight_losses"] and epoch >= config["weight_losses_max_epoch"]:
                        # Weight the loss (not used for paper results)
                        loss = self.get_loss_fct(config, False)(self.model)
                
                    self.train()
                    
                    # Perform train steps
                    train_loss = torch.stack([
                        step(loss(theta.to(self.device), x.to(self.device)))
                        for theta, x in train_set
                    ]).mean().item()

                    self.eval()

                    # Evaluate performance on validation set
                    with torch.no_grad():
                        val_loss = torch.stack([
                            loss(theta.to(self.device), x.to(self.device))
                            for theta, x in val_set
                        ]).mean().item()
                    
                    if not config["weight_losses"] or epoch >= config["weight_losses_max_epoch"]:
                        # Save the weights if they achieve the best validation loss
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_weights = self.model.state_dict()

                        if config["schedule"]:
                            scheduler.step(val_loss)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    tq.set_postfix(train_loss=train_loss, val_loss=val_loss)

            self.model.load_state_dict(best_weights)
            torch.save(train_losses, os.path.join(self.model_path, "train_losses.pt"))
            torch.save(val_losses, os.path.join(self.model_path, "val_losses.pt"))

    def get_loss_fct(self, config, weighted=False):
        if weighted:
            return lambda estimator: WeightedBNRELoss(estimator, lmbda=config["regularization_strength"])
        else:
            return super().get_loss_fct(config)

    def log_prob(self, theta, x):
        return self.model.log_prob(theta.to(self.device), x.to(self.device))

    def get_posterior_fct(self):
        def get_posterior(x):
            class Posterior():
                def __init__(self, sampling_fct, log_prob_fct):
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Posterior(lambda shape: self.model.sample(x.to(self.device), shape).cpu(), lambda theta: self.model.log_prob(theta.to(self.device), x.to(self.device)).cpu())

        return get_posterior
    