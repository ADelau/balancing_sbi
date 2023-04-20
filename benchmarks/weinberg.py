# Reproduced from https://github.com/montefiore-ai/hypothesis

import torch
import zuko
import numpy as np

from .base import Benchmark

class Weinberg(Benchmark):

    MZ = int(90)
    GFNom = float(1)

    def __init__(self):
        super().__init__("weinberg")

        self._num_samples = 20
        self._beam_energy = 40.

        self.LOWER = 0.25 * torch.ones(1).float()
        self.UPPER = 2.0 * torch.ones(1).float()
        
    @torch.no_grad()
    def get_prior(self):
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    def _a_fb(self, sqrtshalf, gf):
        sqrts = sqrtshalf * 2.
        A_FB_EN = np.tanh((sqrts - self.MZ) / self.MZ * 10)
        A_FB_GF = gf / self.GFNom

        return 2 * A_FB_EN * A_FB_GF

    def _diffxsec(self, costheta, sqrtshalf, gf):
        norm = 2. * ((1. + 1. / 3.))

        return ((1 + costheta**2) + self._a_fb(sqrtshalf, gf) * costheta) / norm

    @torch.no_grad()
    def simulate(self, parameters):
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """

        theta = parameters.item()
        psi = self._beam_energy
        samples = []

        for _ in range(self._num_samples):
            sample = None
            x = np.linspace(-1, 1, 10000)
            maxval = np.max(self._diffxsec(x, psi, theta))
            while sample is None:
                xprop = np.random.uniform(-1, 1)
                ycut = np.random.random()
                yprop = self._diffxsec(xprop, psi, theta) / maxval
                if yprop / maxval < ycut:
                    continue
                sample = xprop
            sample = torch.tensor(sample)
            samples.append(sample)

        x = torch.stack(samples)

        return x

    def get_simulation_batch_size(self):
        return 128

    def is_vectorized(self):
        return False

    def get_parameter_dim(self):
        return 1

    def get_observable_shape(self):
        return(20,)

    def get_classifier_build(self):
        return zuko.nn.MLP, {"hidden_features": [256]*6, "activation": torch.nn.SELU}

    def get_flow_build(self):
        return zuko.flows.NSF, {"hidden_features": [256]*3, "activation": torch.nn.SELU}

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
        return 0.002*dataset_size + 60

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