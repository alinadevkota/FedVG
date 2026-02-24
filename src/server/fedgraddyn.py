from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
from omegaconf import DictConfig

from src.client.fedgraddyn import FedGradDynClient
from src.server.fedgradbase import FedGradbaseServer
from src.utils.tools import vectorize


# Fixed according to FedDyn implementation in FL-Simulator (issue #133)
class FedGradDynServer(FedGradbaseServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--max_grad_norm", type=float, default=10)
        parser.add_argument("--type", type=str, default="avg")
        parser.add_argument("--mode", type=str, default="clientwise")
        parser.add_argument("--use_adaptive_opt", type=bool, default=True)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedGradDyn",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedGradDynClient)
        param_numel = vectorize(self.public_model_params).numel()
        self.nabla = torch.zeros(size=(self.client_num, param_numel))

        self.use_adaptive_opt = self.args.fedgraddyn.use_adaptive_opt
        self.need_step2 = True

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["local_dual_correction"] = self.nabla[client_id] - vectorize(
            self.public_model_params
        )
        return server_package

    def aggregate_step2(self, client_packages):
        # super().aggregate_client_updates(client_packages)
        param_shapes = [
            (param.numel(), param.shape) for param in self.public_model_params.values()
        ]

        for client_id, package in client_packages.items():
            # model difference in FL-bench is like diff = param_old - param_new
            # so we do the negative here
            self.nabla[client_id] -= vectorize(package["model_params_diff"])

        flatten_new_params = vectorize(self.public_model_params) + self.nabla.mean(
            dim=0
        )

        # reshape
        new_params = []
        i = 0
        for numel, shape in param_shapes:
            new_params.append(flatten_new_params[i : i + numel].reshape(shape))
            i += numel
        self.public_model_params = OrderedDict(
            zip(self.public_model_params.keys(), new_params)
        )
