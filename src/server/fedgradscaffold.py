from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch
from omegaconf import DictConfig

from src.client.scaffold import SCAFFOLDClient
from src.server.fedgradbase import FedGradbaseServer


class FedGradSCAFFOLDServer(FedGradbaseServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--global_lr", type=float, default=1.0)
        parser.add_argument("--type", type=str, default="avg")
        parser.add_argument("--mode", type=str, default="clientwise")
        parser.add_argument("--use_adaptive_opt", type=bool, default=False)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedGradSCAFFOLD",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.c_global = [
            torch.zeros_like(param) for param in self.public_model_params.values()
        ]
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]
        self.init_trainer(SCAFFOLDClient)

        self.use_adaptive_opt = self.args.fedgradscaffold.use_adaptive_opt

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]
        return server_package

    def aggregate_clients(self, weights, client_packages):
        c_delta_list = [package["c_delta"] for package in client_packages.values()]
        y_delta_list = [package["y_delta"] for package in client_packages.values()]
        for param, y_delta in zip(
            self.public_model_params.values(), zip(*y_delta_list)
        ):
            param.data += self.args.fedgradscaffold.global_lr * torch.sum(
                torch.stack(y_delta, dim=-1) * weights, dim=-1
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_list)):
            c_global.data += torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num
