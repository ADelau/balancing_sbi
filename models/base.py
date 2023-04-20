import os
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from lampe.utils import GDStep

class ModelFactory(ABC):
    def __init__(self, config, benchmark, simulation_budget, model_class):
        self.experience_dir = os.path.join(config["experience_dir"], str(simulation_budget))
        if not os.path.exists(self.experience_dir):
            os.makedirs(self.experience_dir)

        self.model_class = model_class
        self.benchmark = benchmark
        self.config = config

    def is_trained(self, id):
        return self.model_class.is_trained(self.get_model_path(id))

    def instantiate_model(self, id):
        model_path = self.get_model_path(id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = self.model_class(self.benchmark, model_path, self.config)
        model.to(self.benchmark.get_device())
        return model

    def get_model_path(self, id):
        return os.path.join(self.experience_dir, "model_{}".format(id))

    def get_train_nb_cpus(self, benchmark_nb_cpus):
        return benchmark_nb_cpus

    def get_train_nb_gpus(self, benchmark_nb_gpus):
        return benchmark_nb_gpus

    def get_train_ram(self, benchmark_ram):
        return benchmark_ram

    def get_train_time(self, benchmark_time, epochs):
        return benchmark_time*epochs + 1800

    def get_test_nb_cpus(self, benchmark_nb_cpus):
        return self.get_train_nb_cpus(benchmark_nb_cpus)

    def get_test_nb_gpus(self, benchmark_nb_gpus):
        return self.get_train_nb_gpus(benchmark_nb_gpus)

    def get_test_ram(self, benchmark_ram):
        return self.get_train_ram(benchmark_ram)

    def get_test_time(self, benchmark_time):
        return self.get_train_time(benchmark_time, 1)

    def get_coverage_nb_cpus(self, benchmark_nb_cpus):
        return self.get_train_nb_cpus(benchmark_nb_cpus)

    def get_coverage_nb_gpus(self, benchmark_nb_gpus):
        return self.get_train_nb_gpus(benchmark_nb_gpus)

    def get_coverage_ram(self, benchmark_ram):
        return benchmark_ram

    def get_coverage_time(self, benchmark_time):
        return benchmark_time

class Model(ABC):
    def __init__(self):
        pass

    def get_sampling_fct(self):
        raise NotImplementedError()

    def to(self, device):
        self.model.to(device)

    def train_models(self, train_set, val_set, config):
        learning_rate = float(config["learning_rate"])
        epochs = config["epochs"]
        self.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if config["schedule"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=float(config["min_lr"]))

        step = GDStep(optimizer)

        best_loss = float('inf')
        best_weights = self.model.state_dict()

        loss = self.get_loss_fct(config)(self.model)

        train_losses = []
        val_losses = []

        with tqdm(range(epochs), unit='epoch') as tq:

            for epoch in tq:
            
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

    @abstractmethod
    def get_loss_fct(self):
        pass
    
    @abstractmethod
    def log_prob(self, x, theta):
        pass
    
    @abstractmethod
    def sampling_enabled(self):
        return False

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass