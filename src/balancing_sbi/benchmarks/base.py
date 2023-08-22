from abc import ABC, abstractmethod
import lampe
import os
import torch.nn as nn

from enum import Enum

class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    COVERAGE = "coverage"

class Benchmark(ABC):
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name
        self.create_data_dirs()
    
    @abstractmethod
    def get_prior(self):
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        pass
    
    @abstractmethod
    def simulate(self, parameters):
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """
        
        pass
    
    @abstractmethod
    def get_simulation_batch_size(self):
        pass

    @abstractmethod
    def is_vectorized(self):
        pass
    
    def create_data_dirs(self):
        for dataset in Datasets:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), os.path.join(os.path.join("data", self.benchmark_name), dataset.value))):
                os.makedirs(os.path.join(os.path.dirname(__file__), os.path.join(os.path.join("data", self.benchmark_name), dataset.value)))

    def get_store_path(self, dataset, id=None):
        if id is not None:
            dataset_name = os.path.join(dataset.value, str(id))
        else:
            dataset_name = dataset.value

        return os.path.join(os.path.dirname(__file__), os.path.join(os.path.join("data", self.benchmark_name), dataset_name))

    def simulate_block(self, config, dataset, block_id, dataset_id=None):
        prior = self.get_prior()
        loader = lampe.data.JointLoader(prior, self.simulate, batch_size=self.get_simulation_batch_size(),
                                        vectorized=self.is_vectorized())
        
        lampe.data.H5Dataset.store(loader, os.path.join(self.get_store_path(dataset, id=dataset_id), 'block_{}.h5'.format(block_id)), size=config["block_size"])

    def is_block_simulated(self, config, dataset, block_id, dataset_id=None):
        return os.path.exists(os.path.join(self.get_store_path(dataset, id=dataset_id), 'block_{}.h5'.format(block_id)))

    def merge_blocks(self, config, dataset, dataset_size, block_ids, dataset_id=None):
        data = lampe.data.H5Dataset(*[os.path.join(self.get_store_path(dataset, id=dataset_id), 'block_{}.h5'.format(block_id)) for block_id in block_ids], 
                                    batch_size=self.get_simulation_batch_size())

        lampe.data.H5Dataset.store(data, os.path.join(self.get_store_path(dataset, id=dataset_id), 'dataset_{}.h5'.format(dataset_size)), size=dataset_size)

    def are_blocks_merged(self, config, dataset, dataset_size, dataset_id=None):
        return os.path.exists(os.path.join(self.get_store_path(dataset, id=dataset_id), 'dataset_{}.h5'.format(dataset_size)))

    @abstractmethod
    def get_parameter_dim(self):
        pass
    
    @abstractmethod
    def get_observable_shape(self):
        pass

    def get_embedding_dim(self):
        return self.get_observable_shape()[0]

    @abstractmethod
    def get_classifier_build(self):
        pass
    
    @abstractmethod
    def get_flow_build(self):
        pass

    def get_embedding_build(self):
        def get_embedding(embedding_dim, observable_shape):
            return nn.Identity()
        
        return get_embedding

    def get_train_set(self, dataset_size, batch_size, id):
        data = lampe.data.H5Dataset(os.path.join(self.get_store_path(Datasets.TRAIN, id), 'dataset_{}.h5'.format(dataset_size)), batch_size=batch_size, shuffle=True)
        return data

    def get_val_set(self, dataset_size, batch_size, id):
        data = lampe.data.H5Dataset(os.path.join(self.get_store_path(Datasets.VAL, id), 'dataset_{}.h5'.format(dataset_size)), batch_size=batch_size)
        return data

    def get_test_set(self, dataset_size, batch_size):
        data = lampe.data.H5Dataset(os.path.join(self.get_store_path(Datasets.TEST), 'dataset_{}.h5'.format(dataset_size)), batch_size=batch_size)
        return data

    def get_coverage_set(self, dataset_size):
        data = lampe.data.H5Dataset(os.path.join(self.get_store_path(Datasets.COVERAGE), 'dataset_{}.h5'.format(dataset_size)))
        return data
    
    @abstractmethod
    def get_device(self):
        pass
    
    @abstractmethod
    def get_nb_cov_samples(self):
        pass

    @abstractmethod
    def get_cov_bins(self):
        pass

    @abstractmethod
    def get_simulate_nb_gpus(self):
        pass
    
    @abstractmethod
    def get_simulate_nb_cpus(self):
        pass

    @abstractmethod
    def get_simulate_ram(self, block_size):
        pass

    @abstractmethod
    def get_simulate_time(self, block_size):
        pass

    @abstractmethod
    def get_merge_nb_gpus(self):
        pass

    @abstractmethod
    def get_merge_nb_cpus(self):
        pass

    @abstractmethod
    def get_merge_ram(self, dataset_size):
        pass

    @abstractmethod
    def get_merge_time(self, dataset_size):
        pass

    @abstractmethod
    def get_train_nb_gpus(self):
        pass

    @abstractmethod
    def get_train_nb_cpus(self):
        pass

    @abstractmethod
    def get_train_ram(self):
        pass

    @abstractmethod
    def get_train_time(self, dataset_size):
        pass

    @abstractmethod
    def get_test_nb_gpus(self):
        pass

    @abstractmethod
    def get_test_nb_cpus(self):
        pass

    @abstractmethod
    def get_test_ram(self):
        pass

    @abstractmethod
    def get_test_time(self, dataset_size):
        pass