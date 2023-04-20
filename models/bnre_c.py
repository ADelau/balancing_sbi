from .base import Model, ModelFactory
from .nre import NREFactory, NREModel
from lampe.inference import BinaryBalancedCNRELoss

class BNRECFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, BNRECModel)

class BNRECModel(NREModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)

    def get_loss_fct(self, config):
        return lambda estimator: BinaryBalancedCNRELoss(estimator, num_classes=config["num_classes"], gamma=config["gamma"], lmbda=config["regularization_strength"])