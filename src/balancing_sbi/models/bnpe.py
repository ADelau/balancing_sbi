import os
import torch
import torch.nn as nn

from .base import ModelFactory
from.npe import NPEModel

class BNPELoss(nn.Module):
    def __init__(self, estimator, prior, lmbda=100.0):
        super().__init__()

        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda

    def forward(self, theta, x):

        theta_prime = torch.roll(theta, 1, dims=0)

        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l0 = -log_p.mean()
        
        # balancing criterion
        # discriminator output = sigmoid(log ratio) = sigmoid(log posterior - log prior)
        lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()

        return l0 + self.lmbda * lb

class BNPEFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, BNPEModel)

    def get_train_time(self, benchmark_time, epochs):
        return 4*super().get_train_time(benchmark_time, epochs)

class BNPEModel(NPEModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)

    def get_loss_fct(self, config):
        return lambda estimator: BNPELoss(estimator, self.prior, lmbda=config["regularization_strength"])