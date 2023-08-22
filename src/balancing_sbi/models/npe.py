import os
import torch
from lampe.inference import NPE, NPELoss
import torch.nn as nn

from .base import Model, ModelFactory

class NPEFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, NPEModel)

    def get_train_time(self, benchmark_time, epochs):
        return 2*super().get_train_time(benchmark_time, epochs)

class NPEWithEmbedding(nn.Module):
    def __init__(self, npe, embedding):
        super().__init__()
        self.npe = npe
        self.embedding = embedding

    def forward(self, theta, x):
        return self.npe(theta, self.embedding(x))

    def sample(self, x, shape):
        return self.npe.sample(self.embedding(x), shape)

class NPEModel(Model):
    def __init__(self, benchmark, model_path, config):

        self.observable_shape = benchmark.get_observable_shape()
        self.embedding_dim = benchmark.get_embedding_dim()
        self.parameter_dim = benchmark.get_parameter_dim()
        self.device = benchmark.get_device()

        self.model_path = model_path

        self.prior = benchmark.get_prior()

        embedding_build = benchmark.get_embedding_build()
        self.embedding = embedding_build(self.embedding_dim, self.observable_shape).to(self.device)

        flow_build, flow_kwargs = benchmark.get_flow_build()
        self.flow = NPE(self.parameter_dim, self.embedding_dim, build=flow_build, **flow_kwargs).to(self.device)
        self.model = NPEWithEmbedding(self.flow, self.embedding)

    @classmethod
    def is_trained(cls, model_path):
        return (os.path.exists(os.path.join(model_path, "embedding.pt")) and os.path.exists(os.path.join(model_path, "flow.pt")))

    def get_loss_fct(self, config):
        return NPELoss

    def log_prob(self, theta, x):
        x = x.to(self.device)
        theta = theta.to(self.device)
        return self.model(theta, x)

    def get_posterior_fct(self):
        def get_posterior(x):
            class Posterior():
                def __init__(self, sampling_fct, log_prob_fct):
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Posterior(lambda shape: self.model.sample(x.to(self.device), shape).cpu(), lambda theta: self.model(theta.to(self.device), x.to(self.device)).cpu())

        return get_posterior

    def __call__(self, theta, x):
        return self.log_prob(theta, x)

    def sampling_enabled(self):
        return True

    def save(self):
        torch.save(self.embedding.state_dict(), os.path.join(self.model_path, "embedding.pt"))
        torch.save(self.flow.state_dict(), os.path.join(self.model_path, "flow.pt"))

    def load(self):
        self.embedding.load_state_dict(torch.load(os.path.join(self.model_path, "embedding.pt")))
        self.flow.load_state_dict(torch.load(os.path.join(self.model_path, "flow.pt")))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()