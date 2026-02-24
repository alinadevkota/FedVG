from argparse import ArgumentParser, Namespace

import torch
from omegaconf import DictConfig

from src.client.fedgradelastic import FedGradElasticClient
from src.server.fedgradbase import FedGradbaseServer


class FedGradElasticServer(FedGradbaseServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--sample_ratio", type=float, default=0.3)  # opacue
        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--mu", type=float, default=0.95)
        parser.add_argument("--type", type=str, default="avg")
        parser.add_argument("--mode", type=str, default="clientwise")
        parser.add_argument("--use_adaptive_opt", type=bool, default=False)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedGradElastic",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedGradElasticClient)

        self.use_adaptive_opt = self.args.fedgradelastic.use_adaptive_opt

    def aggregate_clients(self, weights, client_packages):
        sensitivities = []

        for package in client_packages.values():
            sensitivities.append(package["sensitivity"])

        sensitivities = torch.stack(sensitivities, dim=-1)

        aggregated_sensitivity = torch.sum(sensitivities * weights, dim=-1)
        max_sensitivity = sensitivities.max(dim=-1)[0]

        zeta = (
            1 + self.args.fedgradelastic.tau - aggregated_sensitivity / max_sensitivity
        )

        for (key, global_param), coef in zip(self.public_model_params.items(), zeta):
            diffs = torch.stack(
                [
                    package["model_params_diff"][key]
                    for package in client_packages.values()
                ],
                dim=-1,
            )
            aggregated = torch.sum(diffs * weights, dim=-1)
            global_param.data -= coef * aggregated
