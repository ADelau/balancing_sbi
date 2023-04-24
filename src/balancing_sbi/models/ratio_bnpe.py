import os

from .ratio_npe import RatioNPEModel
from .base import Model, ModelFactory
from lampe.inference import BNRELoss

class RatioBNPEFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, RatioBNPEModel)

    def get_train_time(self, benchmark_time, epochs):
        return 2*super().get_train_time(benchmark_time, epochs)

class RatioBNPEModel(RatioNPEModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)

    def get_loss_fct(self, config):
        return lambda estimator: BNRELoss(estimator, lmbda=config["regularization_strength"])
