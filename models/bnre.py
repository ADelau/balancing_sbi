import os

from .nre import NREFactory, NREModel
from .base import ModelFactory
from lampe.inference import BNRELoss

class BNREFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, BNREModel)

class BNREModel(NREModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)

    def get_loss_fct(self, config):
        return lambda estimator: BNRELoss(estimator, lmbda=config["regularization_strength"])