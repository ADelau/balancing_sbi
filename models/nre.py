import os
import torch
from lampe.inference import NRE, NRELoss
import torch.nn as nn

from .base import Model, ModelFactory

class NREFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, NREModel)

class ClassifierWithEmbedding(nn.Module):
    def __init__(self, classifier, embedding):
        super().__init__()
        self.classifier = classifier
        self.embedding = embedding

    def forward(self, theta, x):
        return self.classifier(theta, self.embedding(x))
 
class NREModel(Model):
    def __init__(self, benchmark, model_path, config):

        self.observable_shape = benchmark.get_observable_shape()
        self.embedding_dim = benchmark.get_embedding_dim()
        self.parameter_dim = benchmark.get_parameter_dim()
        self.device = benchmark.get_device()

        self.model_path = model_path

        self.prior = benchmark.get_prior()

        embedding_build = benchmark.get_embedding_build()
        self.embedding = embedding_build(self.embedding_dim, self.observable_shape).to(self.device)

        classifier_build, classifier_kwargs = benchmark.get_classifier_build()
        self.classifier = NRE(self.parameter_dim, self.embedding_dim, build=classifier_build, **classifier_kwargs).to(self.device)
        self.model = ClassifierWithEmbedding(self.classifier, self.embedding)

    @classmethod
    def is_trained(cls, model_path):
        return (os.path.exists(os.path.join(model_path, "embedding.pt")) and os.path.exists(os.path.join(model_path, "classifier.pt")))

    def get_loss_fct(self, config):
        return NRELoss

    def log_prob(self, theta, x):
        return self.prior.log_prob(theta.cpu()) + self.model(theta.to(self.device), x.to(self.device)).cpu()

    def __call__(self, theta, x):
        return self.log_prob(theta, x)

    def sampling_enabled(self):
        return False

    def save(self):
        torch.save(self.embedding.state_dict(), os.path.join(self.model_path, "embedding.pt"))
        torch.save(self.classifier.state_dict(), os.path.join(self.model_path, "classifier.pt"))

    def load(self):
        self.embedding.load_state_dict(torch.load(os.path.join(self.model_path, "embedding.pt")))
        self.classifier.load_state_dict(torch.load(os.path.join(self.model_path, "classifier.pt")))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()