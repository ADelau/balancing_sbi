import os
import torch
from lampe.inference import NPE, NRELoss
import torch.nn as nn

from .base import Model, ModelFactory
from .npe import NPEModel, NPEWithEmbedding

class RatioNPEFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, RatioNPEModel)

    def get_train_time(self, benchmark_time, epochs):
        return 2*super().get_train_time(benchmark_time, epochs)

class FlowBasedRatio(nn.Module):
    def __init__(self, npe, prior):
        super().__init__()
        self.npe = npe
        self.prior = prior

    def forward(self, theta, x):
        theta_device = theta.get_device()

        # log ratio = log posterior - log prior
        return self.npe(theta, x) - self.prior.log_prob(theta.cpu()).to(theta_device if theta_device >= 0 else "cpu")

    def sample(self, x, shape):
        return self.npe.sample(x, shape)

class RatioNPEModel(NPEModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)
        self.npe = self.model
        self.model = FlowBasedRatio(self.npe, self.prior)

    def get_loss_fct(self, config):
        return NRELoss

    def log_prob(self, theta, x):
        x = x.to(self.device)
        theta = theta.to(self.device)
        return self.npe(theta, x)

    def get_posterior_fct(self):
        def get_posterior(x):
            class Posterior():
                def __init__(self, sampling_fct, log_prob_fct):
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Posterior(lambda shape: self.npe.sample(x.to(self.device), shape), lambda theta: self.npe(theta.to(self.device), x.to(self.device)))

        return get_posterior