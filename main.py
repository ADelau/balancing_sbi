import sys
import argparse
import math
import torch
import numpy as np
import random
import math
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

from dawgz import job, after, waitfor, ensure, schedule, context
from config_files import read_config
from benchmarks import load_benchmark, Datasets
from models import load_model_factory
from diagnostics import compute_coverage, compute_normalized_entropy_log_posterior, compute_log_posterior, compute_balancing_error, compute_prior_mixture_coef, plot_1d_posterior, plot_2d_posterior, plot_nd_posterior

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def seconds_to_time(seconds):
    seconds = math.ceil(seconds)
    minutes = seconds//60
    seconds = seconds%60

    hours = minutes//60
    minutes = minutes%60

    return "{}:{:02d}:{:02d}".format(hours, minutes, seconds)

if __name__ == '__main__':
    # Increase recursion depth (large workflow)
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument("--slurm", action="store_true", help="Executes the workflow on a Slurm-enabled HPC system (default: false).")

    arguments, _ = parser.parse_known_args()

    config = read_config(arguments.config_file)
    benchmark = load_benchmark(config)

    nb_train_blocks = math.ceil(config["train_set_size"]/config["block_size"])
    nb_val_blocks = math.ceil(config["val_set_size"]/config["block_size"])
    nb_test_blocks = math.ceil(config["test_set_size"]/config["block_size"])
    nb_coverage_blocks = math.ceil(config["coverage_set_size"]/config["block_size"])

    jobs = []
    diagnostics = config["diagnostics"]

    @ensure(lambda id: benchmark.is_block_simulated(config, Datasets.TRAIN, id%nb_train_blocks, dataset_id=id//nb_train_blocks))
    @job(array=nb_train_blocks*config["nb_runs"], cpus=benchmark.get_simulate_nb_cpus(), gpus=benchmark.get_simulate_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))), 
         time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])))
    def simulate_train(id):
        if config["seed"] is not None:
            set_seed(config["seed"] + id)

        benchmark.simulate_block(config, Datasets.TRAIN, id%nb_train_blocks, dataset_id=id//nb_train_blocks)

    jobs.append(simulate_train)

    @ensure(lambda id: benchmark.is_block_simulated(config, Datasets.VAL, id%nb_val_blocks, dataset_id=id//nb_val_blocks))
    @job(array=nb_val_blocks*config["nb_runs"], cpus=benchmark.get_simulate_nb_cpus(), gpus=benchmark.get_simulate_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))), 
         time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])))
    def simulate_val(id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(1e8) + id)

        benchmark.simulate_block(config, Datasets.VAL, id%nb_val_blocks, dataset_id=id//nb_val_blocks)

    jobs.append(simulate_val)

    @ensure(lambda block_id: benchmark.is_block_simulated(config, Datasets.TEST, block_id))
    @job(array=nb_test_blocks, cpus=benchmark.get_simulate_nb_cpus(), gpus=benchmark.get_simulate_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))), 
         time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])))
    def simulate_test(block_id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(2e8) + block_id)

        benchmark.simulate_block(config, Datasets.TEST, block_id)

    jobs.append(simulate_test)

    @ensure(lambda block_id: benchmark.is_block_simulated(config, Datasets.COVERAGE, block_id))
    @job(array=nb_test_blocks, cpus=benchmark.get_simulate_nb_cpus(), gpus=benchmark.get_simulate_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))), 
         time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])))
    def simulate_coverage(block_id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(3e8) + block_id)

        benchmark.simulate_block(config, Datasets.COVERAGE, block_id)

    jobs.append(simulate_coverage)

    @after(simulate_test)
    @ensure(lambda: benchmark.are_blocks_merged(config, Datasets.TEST, config["test_set_size"]))
    @job(cpus=benchmark.get_merge_nb_cpus(), gpus=benchmark.get_merge_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_merge_ram(config["test_set_size"]))), 
         time=seconds_to_time(benchmark.get_merge_time(config["test_set_size"])))
    def merge_test():
        benchmark.merge_blocks(config, Datasets.TEST, config["test_set_size"], [x for x in range(nb_test_blocks)])

    jobs.append(merge_test)

    @after(simulate_coverage)
    @ensure(lambda: benchmark.are_blocks_merged(config, Datasets.COVERAGE, config["coverage_set_size"]))
    @job(cpus=benchmark.get_merge_nb_cpus(), gpus=benchmark.get_merge_nb_gpus(), 
         ram="{}GB".format(math.ceil(benchmark.get_merge_ram(config["coverage_set_size"]))), 
         time=seconds_to_time(benchmark.get_merge_time(config["coverage_set_size"])))
    def merge_coverage():
        benchmark.merge_blocks(config, Datasets.COVERAGE, config["coverage_set_size"], [x for x in range(nb_test_blocks)])

    jobs.append(merge_coverage)

    simulation_budgets = config["simulation_budgets"]

    for simulation_budget in simulation_budgets:
        val_fraction = config["val_fraction"]
        train_size = math.floor(simulation_budget*(1-val_fraction))
        val_size = math.floor(simulation_budget*val_fraction)

        model_factory = load_model_factory(config, benchmark, simulation_budget)

        @after(simulate_train)
        @context(train_size=train_size)
        @ensure(lambda dataset_id: benchmark.are_blocks_merged(config, Datasets.TRAIN, train_size, dataset_id=dataset_id))
        @job(array=config["nb_runs"], cpus=benchmark.get_merge_nb_cpus(), gpus=benchmark.get_merge_nb_gpus(), 
             ram="{}GB".format(math.ceil(benchmark.get_merge_ram(train_size))), 
             time=seconds_to_time(benchmark.get_merge_time(train_size)))
        def merge_train(dataset_id):
            benchmark.merge_blocks(config, Datasets.TRAIN, train_size, [x for x in range(nb_train_blocks)], dataset_id=dataset_id)

        jobs.append(merge_train)

        @after(simulate_val)
        @context(val_size=val_size)
        @ensure(lambda dataset_id: benchmark.are_blocks_merged(config, Datasets.VAL, val_size, dataset_id=dataset_id))
        @job(array=config["nb_runs"], cpus=benchmark.get_merge_nb_cpus(), gpus=benchmark.get_merge_nb_gpus(), 
             ram="{}GB".format(math.ceil(benchmark.get_merge_ram(val_size))), 
             time=seconds_to_time(benchmark.get_merge_time(val_size)))
        def merge_val(dataset_id):
            benchmark.merge_blocks(config, Datasets.VAL, val_size, [x for x in range(nb_val_blocks)], dataset_id=dataset_id)

        jobs.append(merge_val)

        @after(merge_train, merge_val)
        @context(train_size=train_size, val_size=val_size, model_factory=model_factory)
        @ensure(lambda job_id: model_factory.is_trained(job_id))
        @job(array=config["nb_runs"], cpus=model_factory.get_train_nb_cpus(benchmark.get_train_nb_cpus()), 
             gpus=model_factory.get_train_nb_gpus(benchmark.get_train_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_train_ram(benchmark.get_train_ram()))), 
             time=seconds_to_time(model_factory.get_train_time(benchmark.get_train_time(train_size + val_size), config["epochs"])))
        def train(job_id):
            if config["seed"] is not None:
                set_seed(config["seed"] + simulation_budget + job_id)

            
            batch_size = config["train_batch_size"]

            model = model_factory.instantiate_model(job_id)
            model.train_models(benchmark.get_train_set(train_size, batch_size, job_id), 
                               benchmark.get_val_set(val_size, batch_size, job_id),
                               config)
            model.save()

        jobs.append(train)

        @after(train, merge_coverage)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: os.path.exists(os.path.join(model_factory.get_model_path(job_id), "levels.pt")) and 
                os.path.exists(os.path.join(model_factory.get_model_path(job_id), "coverages.pt")))
        @job(array=config["nb_runs"], cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()), 
             gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))),
             time=seconds_to_time(model_factory.get_coverage_time(benchmark.get_coverage_time(config["coverage_set_size"]))))
        def coverage(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            levels, coverages = compute_coverage(model, benchmark, config)
            torch.save(levels, os.path.join(model_factory.get_model_path(job_id), "levels.pt"))
            torch.save(coverages, os.path.join(model_factory.get_model_path(job_id), "coverages.pt"))

        if "coverage" in diagnostics:
            jobs.append(coverage)

        @after(train, merge_coverage)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: os.path.exists(os.path.join(model_factory.get_model_path(job_id), "entropy.pt")) and 
                os.path.exists(os.path.join(model_factory.get_model_path(job_id), "normalized_nominal_log_prob.pt")))
        @job(array=config["nb_runs"], cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()), 
             gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))),
             time=seconds_to_time(model_factory.get_coverage_time(benchmark.get_coverage_time(config["coverage_set_size"]))))
        def normalized_entropy_log_posterior(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            if job_id == 0:
                entropy, normalized_nominal_log_prob = compute_normalized_entropy_log_posterior(model, benchmark, config)
            else:
                entropy, normalized_nominal_log_prob = compute_normalized_entropy_log_posterior(model, benchmark, config)
            if "entropy" in diagnostics:
                torch.save(entropy, os.path.join(model_factory.get_model_path(job_id), "entropy.pt"))
            if "normalized_nominal_log_prob" in diagnostics:
                torch.save(normalized_nominal_log_prob, os.path.join(model_factory.get_model_path(job_id), "normalized_nominal_log_prob.pt"))

        if "entropy" in diagnostics or "normalized_nominal_log_prob" in diagnostics:
            jobs.append(normalized_entropy_log_posterior)

        @after(train, merge_test)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: os.path.exists(os.path.join(model_factory.get_model_path(job_id), "nominal_log_prob.pt")))
        @job(array=config["nb_runs"], cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()), 
             gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))),
             time=seconds_to_time(model_factory.get_test_time(benchmark.get_test_time(config["test_set_size"]))))
        def log_posterior(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            nominal_log_prob = compute_log_posterior(model, benchmark, config)
            torch.save(nominal_log_prob, os.path.join(model_factory.get_model_path(job_id), "nominal_log_prob.pt"))

        if "nominal_log_prob" in diagnostics:
            jobs.append(log_posterior)

        @after(train, merge_test)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: os.path.exists(os.path.join(model_factory.get_model_path(job_id), "balancing_error.pt")))
        @job(array=config["nb_runs"], cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()), 
             gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))),
             time=seconds_to_time(model_factory.get_test_time(benchmark.get_test_time(config["test_set_size"]))))
        def balancing_error(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            balancing_error = compute_balancing_error(model, benchmark, config)
            torch.save(balancing_error, os.path.join(model_factory.get_model_path(job_id), "balancing_error.pt"))

        if "balancing_error" in diagnostics:
            jobs.append(balancing_error)

        @after(train, merge_test)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: os.path.exists(os.path.join(model_factory.get_model_path(job_id), "prior_mixture_coef.pt")))
        @job(array=config["nb_runs"], cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()), 
             gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))),
             time=seconds_to_time(model_factory.get_test_time(benchmark.get_test_time(config["test_set_size"]))))
        def prior_mixture_coef(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            prior_mixture_coef = compute_prior_mixture_coef(model, benchmark, config)
            torch.save(prior_mixture_coef, os.path.join(model_factory.get_model_path(job_id), "prior_mixture_coef.pt"))

        @after(train, merge_test)
        @context(model_factory=model_factory)
        @ensure(lambda job_id: len(glob.glob(os.path.join(model_factory.get_model_path(job_id), "posterior_*.pdf"))) == 50)
        @job(array=config["nb_runs"], cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()), 
             gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()), 
             ram="{}GB".format(math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))),
             time=seconds_to_time(model_factory.get_coverage_time(benchmark.get_coverage_time(50))))
        def plot(job_id):
            model = model_factory.instantiate_model(job_id)
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            parameter_dim = benchmark.get_parameter_dim()

            dataset = benchmark.get_coverage_set(config["coverage_set_size"])
            nb_plotted = 0

            for nominal_theta, x in tqdm(dataset, unit='pair'):

                plt.figure()

                if parameter_dim == 1:
                    plot_1d_posterior(model, benchmark, config, x, nominal_theta, 
                                      save_name=os.path.join(model_factory.get_model_path(job_id), "posterior_{}.pdf".format(nb_plotted)))
                elif parameter_dim == 2:
                    plot_2d_posterior(model, benchmark, config, x, nominal_theta, 
                                      save_name=os.path.join(model_factory.get_model_path(job_id), "posterior_{}.pdf".format(nb_plotted)))
                else:
                    plot_nd_posterior(model, benchmark, config, x, nominal_theta, 
                                      save_name=os.path.join(model_factory.get_model_path(job_id), "posterior_{}.pdf".format(nb_plotted)))

                plt.close()

                nb_plotted += 1
                if nb_plotted >= 50:
                    break
        
        if "plot" in diagnostics:
            jobs.append(plot)

    backend = 'slurm' if arguments.slurm else 'async'
    schedule(*jobs, name='simple.py', backend=backend, prune=True)