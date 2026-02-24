# FedVG


Using validation gradient-based aggregation in Federated Learning.

> **Note 1:** This repository is forked from [FL-bench](https://github.com/username/original-repo), with modifications for FedVG.

> **Note 2:** The term `FedGrad` is used analogous to `FedVG` in this repository.

> **Note 3:** Combination of FedVG and baseline methods are named `fedgrad<baseline>` inside `src/server` directory.

## Getting Started
### Environment Preparation

Create a python environment with python 3.11 (recommended) using conda.
```shell
conda create --name fedvg python=3.11
```
Activate the conda environment:
```shell
conda activate fedvg
```

In your new virtual environment (conda or pip), with python 3.11, install the required packages:
```shell
pip install -r requirements.txt
```

### Easy Run
ALL classes of methods are inherited from `FedAvgServer` and `FedAvgClient`. [`src/server/fedavg.py`](src/server/fedavg.py) and [`src/client/fedavg.py`](src/client/fedavg.py) can be explored to figure out the entire workflow.

### Step 1. Generate FL Dataset
Partition the CIFAR-10 according to Dir(0.1) for 100 clients
```shell
python generate_data.py -d cifar10 -a 0.1 -cn 100 --val_ratio 0.1
```
[`data/README.md`](data/#readme) provides full details for generating federated dastaset.

The generated datset is stored in the format `<dataset>_<client_num>_<alpha>` inside the `data` folder, and this full name should be provided as argument while running experiments. Make sure to copy raw contents of downloaded folder to this created folder.

### Step 2. Run Experiment

To quickly run FedVG on CIFAR-10 dataset on ResNet18, run the following:
```shell
python -u main.py dataset.name=cifar10_nc100_alpha0.1 model.name=res18 method=fedgrad common.global_epoch=200 common.seed=0
```

```sh
python main.py [--config-path, --config-name] [method=<METHOD_NAME> args...]
```

- `method`: The algorithm's name, e.g., `method=fedavg`. 
>   \[!NOTE\]
>   `method` should be identical to the `.py` file name in `src/server`.

- `--config-path`: Relative path to the directory of the config file. Defaults to `config`.
- `--config-name`: Name of `.yaml` config file (w/o the `.yaml` extension). Defaults to `defaults`, which points to `config/defaults.yaml`.

Such as running FedAvg with all defaults. 
```sh
python main.py method=fedavg
```
Defaults are set in both [`config/defaults.yaml`](config/defaults.yaml) and [`src/utils/constants.py`](src/utils/constants.py).

### How To Customize FL method Arguments
- By modifying config file.
- By explicitly setting in CLI, e.g., `python main.py --config-name my_cfg.yaml method=fedprox fedprox.mu=0.01`.
- By modifying the default value in `config/defaults.yaml` or `get_hyperparams()` in `src/server/<method>.py`

> \[!NOTE\]
> For the same FL method argument, the priority of argument setting is **CLI > Config file > Default value**. 
> 
> For example, the default value of `fedprox.mu` is `1`, 
> ```python
> # src/server/fedprox.py
> class FedProxServer(FedAvgServer):
> 
>     @staticmethod
>     def get_hyperparams(args_list=None) -> Namespace:
>         parser = ArgumentParser()
>         parser.add_argument("--mu", type=float, default=1.0)
>         return parser.parse_args(args_list)
> 
> ```
> and your `.yaml` config file has
> ```yaml
> # config/your_config.yaml
> ...
> fedprox:
>   mu: 0.01
> ```
> 
> ```shell
> python main.py method=fedprox                                            # fedprox.mu = 1
> python main.py --config-name your_config method=fedprox                  # fedprox.mu = 0.01
> python main.py --config-name your_config method=fedprox fedprox.mu=0.001 # fedprox.mu = 0.001
> ``` 