from argparse import ArgumentParser, Namespace

import torch
from omegaconf import DictConfig

from src.server.fedgradbase import FedGradbaseServer


class FedGradAvgServer(FedGradbaseServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--server_lr", type=float, default=1e-1)
        parser.add_argument("--tau", type=float, default=1e-3)
        parser.add_argument("--server_momentum", type=float, default=0.9)
        parser.add_argument("--type", type=str, default="avg")
        parser.add_argument("--mode", type=str, default="clientwise")
        parser.add_argument("--use_adaptive_opt", type=bool, default=True)
        return parser.parse_args(args_list)

    algo_names = {
        "adagrad": "FedAdagrad",
        "yogi": "FedYogi",
        "adam": "FedAdam",
        "avg": "FedAvgNew",
        "avgm": "FedAvgMNew",
    }

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedGradAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        # algo = self.algo_names[args.fedacc.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.use_adaptive_opt = self.args.fedgradavg.use_adaptive_opt

    def train_one_round(self):
        client_packages = self.trainer.train()

        clients_model_params_diff = []
        grad_norms = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            regular_model_params = package["regular_model_params"]
            grad_norm = self.get_val_gradient_score(regular_model_params)
            grad_norms.append(grad_norm)

        client_weights_grad = self.get_weights_agg(grad_norms)
        fedavg_weights = torch.tensor(
            [package["weight"] for package in client_packages.values()]
        )
        fedavg_weights = fedavg_weights / fedavg_weights.sum()

        client_weights = (client_weights_grad + fedavg_weights) / 2

        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=client_weights,
        )
