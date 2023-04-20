# Reproduced from https://github.com/montefiore-ai/hypothesis

import torch
import zuko
import numpy as np
import math
import torch.nn as nn

from .base import Benchmark

class LotkaVolterra(Benchmark):
    def __init__(self):
        super().__init__("lotka_volterra")

        epsilon = 0.00001
        self.LOWER = -4 * torch.ones(2).float()
        self.UPPER = torch.ones(2).float()
        self.UPPER += epsilon  # Account for half-open interval

        self._initial_state = np.array([50, 100])
        self._duration = 50.
        self._dt = 0.05
        self._prey_prior = self.get_prior()

    @torch.no_grad()
    def get_prior(self):
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    @torch.no_grad()
    def simulate(self, parameters):
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """
        
        latents = self._prey_prior.sample()
        parameters = torch.cat([parameters, latents]).exp().numpy()

        steps = int(self._duration / self._dt) + 1
        states = np.zeros((steps, 2))
        state = np.copy(self._initial_state)
        for step in range(steps):
            x, y = state
            xy = x * y
            propensities = np.array([xy, x, y, xy])
            rates = parameters * propensities
            total_rate = sum(rates)
            if total_rate <= 0.00001:
                break
            normalized_rates = rates / total_rate
            transition = np.random.choice([0, 1, 2, 3], p=normalized_rates)
            if transition == 0:
                state[0] += 1  # Increase predator population by 1
            elif transition == 1:
                state[0] -= 1  # Decrease predator population by 1
            elif transition == 2:
                state[1] += 1  # Increase prey population by 1
            else:
                state[1] -= 1  # Decrease prey population by 1
            states[step, :] = np.copy(state)

        return torch.from_numpy(states).float()

    def get_simulation_batch_size(self):
        return 128

    def is_vectorized(self):
        return False

    def get_parameter_dim(self):
        return 2

    def get_observable_shape(self):
        return(2001, 2)

    def get_embedding_dim(self):
        return 16*32

    def get_embedding_build(self):
        class Prepare(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                '''
                Reshapes the input according to the shape saved in the view data structure.
                '''
                x = x.view([-1, 1001, 2]).permute((0, 2, 1)) / 100
                return x

        def get_embedding(embedding_dim, observable_shape):
            nb_channels = 16
            nb_conv_layers = 10
            shrink_every = 2
            final_shape = 1001

            for i in range(nb_conv_layers):
                if i%shrink_every == 0:
                    final_shape = math.floor((final_shape - 1)/2 + 1)
                else:
                    final_shape = final_shape

            cnn = [Prepare(), nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)]

            for i in range(nb_conv_layers):
                if i%shrink_every == 0:
                    stride=2
                else:
                    stride=1

                cnn.append(nn.Conv1d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1))
                cnn.append(nn.SELU())
                cnn.append(nn.MaxPool1d(3, stride=stride, padding=1))

            cnn.append(nn.Flatten())

            return nn.Sequential(*cnn)
        
        return get_embedding

    def get_classifier_build(self):
        return zuko.nn.MLP, {"hidden_features": [256]*6, "activation": torch.nn.SELU}

    def get_flow_build(self):
        return zuko.flows.NSF, {"hidden_features": [256]*3, "activation": torch.nn.SELU}

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_domain(self):
        return self.LOWER, self.UPPER

    def get_nb_cov_samples(self):
        return 1000

    def get_cov_bins(self):
        return 100
    
    def get_simulate_nb_gpus(self):
        return 0

    def get_simulate_nb_cpus(self):
        return 1

    def get_simulate_ram(self, block_size):
        return 0.0001*block_size

    def get_simulate_time(self, block_size):
        return 10*block_size

    def get_merge_nb_gpus(self):
        return 0

    def get_merge_nb_cpus(self):
        return 1

    def get_merge_ram(self, dataset_size):
        return 0.0005*dataset_size + 2

    def get_merge_time(self, dataset_size):
        return 0.1*dataset_size + 500

    def get_train_nb_gpus(self):
        return 1

    def get_train_nb_cpus(self):
        return 1

    def get_train_ram(self):
        return 8

    def get_train_time(self, dataset_size):
        return 0.002*dataset_size + 60

    def get_test_nb_gpus(self):
        return 1

    def get_test_nb_cpus(self):
        return 1

    def get_test_ram(self):
        return 8

    def get_test_time(self, dataset_size):
        return 0.002*dataset_size + 600

    def get_coverage_nb_gpus(self):
        return 1

    def get_coverage_nb_cpus(self):
        return 1

    def get_coverage_ram(self):
        return 8

    def get_coverage_time(self, dataset_size):
        return 10 * dataset_size + 1800