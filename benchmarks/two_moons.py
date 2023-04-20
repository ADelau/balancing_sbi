# Reproduced from https://github.com/sbi-benchmark/sbibm

import torch
import zuko
import math

from .base import Benchmark

class TwoMoons(Benchmark):
    def __init__(self):
        super().__init__("two_moons")

        self.a_dist = torch.distributions.uniform.Uniform(-math.pi / 2.0, math.pi / 2.0)
        self.r_dist = torch.distributions.normal.Normal(0.1, 0.01)

        self.LOWER = -1 * torch.ones(2).float()
        self.UPPER = 1 * torch.ones(2).float()
        
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
        
        a = self.a_dist.sample()
        r = self.r_dist.sample()
        
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)

        z0 = (c * parameters[0] - s * parameters[1])
        z1 = (s * parameters[0] + c * parameters[1])

        return torch.Tensor([torch.cos(a) * r + 0.25 -torch.abs(z0), torch.sin(a) * r + z1])

    def get_simulation_batch_size(self):
        return 128

    def is_vectorized(self):
        return False

    def get_parameter_dim(self):
        return 2

    def get_observable_shape(self):
        return(2,)

    def get_classifier_build(self):
        return zuko.nn.MLP, {"hidden_features": [256]*6, "activation": torch.nn.SELU}

    def get_flow_build(self):
        return zuko.flows.NSF, {"hidden_features": [256]*6, "activation": torch.nn.SELU}

    def get_device(self):
        return "cpu"

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
        return 0.0001*dataset_size

    def get_merge_time(self, dataset_size):
        return 0.1*dataset_size

    def get_train_nb_gpus(self):
        return 1

    def get_train_nb_cpus(self):
        return 1

    def get_train_ram(self):
        return 4

    def get_train_time(self, dataset_size):
        return 0.005*dataset_size + 60

    def get_test_nb_gpus(self):
        return 1

    def get_test_nb_cpus(self):
        return 1

    def get_test_ram(self):
        return 4

    def get_test_time(self, dataset_size):
        return 0.002*dataset_size + 600

    def get_coverage_nb_gpus(self):
        return 1

    def get_coverage_nb_cpus(self):
        return 1

    def get_coverage_ram(self):
        return 4

    def get_coverage_time(self, dataset_size):
        return 10 * dataset_size + 1800