# Balancing Simulation-based Inference for Conservative Posteriors

## Abstract
Conservative inference is a major concern in simulation-based inference. It has been shown that commonly used algorithms can produce overconfident posterior approximations. Balancing has empirically proven to be an effective way to mitigate this issue. However, its application remains limited to neural ratio estimation. In this work, we extend balancing to any algorithm that provides a posterior density. In particular, we introduce a balanced version of both neural posterior estimation and contrastive neural ratio estimation. We show empirically that the balanced versions tend to produce conservative posterior approximations on a wide variety of benchmarks. In addition, we provide an alternative interpretation of the balancing condition in terms of the χ2 divergence.

A PDF render of the manuscript is available on [`ArXiV`](https://arxiv.org/pdf/2304.10978.pdf).

<img src=".github/coverage_2.png">

## Reproducing the experiments
First, install all the dependencies from the [requirements.txt](requirements.txt) file. The pipelines performing the experiments can then be executed by running the following command
```
python main.py --config_file <config_file_path>
```
Config files can be found [here](config_files). Optionally, the `--slurm` argument can be added to run on a slurm cluster.
