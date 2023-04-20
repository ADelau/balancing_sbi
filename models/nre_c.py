from .base import Model, ModelFactory
from .nre import NREFactory, NREModel
from lampe.inference import CNRELoss

class NRECFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, NRECModel)

class NRECModel(NREModel):
    def __init__(self, benchmark, model_path, config):
        super().__init__(benchmark, model_path, config)

    def get_loss_fct(self, config):
        return lambda estimator: CNRELoss(estimator, num_classes=config["num_classes"], gamma=config["gamma"])