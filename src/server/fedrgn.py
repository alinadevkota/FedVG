from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer
from src.server.fedopt import AdaptiveOptimizer


class FedRGNServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--type",
            choices=["adagrad", "yogi", "adam", "avg"],
            type=str,
            default="adam",
        )
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--server_lr", type=float, default=1e-1)
        parser.add_argument("--tau", type=float, default=1e-3)
        return parser.parse_args(args_list)

    algo_names = {
        "adagrad": "FedAdagrad",
        "yogi": "FedYogi",
        "adam": "FedAdam",
        "avg": "FedAvgNew",
    }

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        # algo = self.algo_names[args.fedopt.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.adaptive_optimizer = AdaptiveOptimizerGrad(
            optimizer_type=self.args.fedrgn.type,
            params_dict=self.public_model_params,
            beta1=self.args.fedrgn.beta1,
            beta2=self.args.fedrgn.beta2,
            lr=self.args.fedrgn.server_lr,
            tau=self.args.fedrgn.tau,
            _type=self.args.fedrgn.type,
            mode=self.args.fedrgn.mode,
        )

        self.mode = self.args.fedrgn.mode

    def get_gradients(self, model, dataloader, criterion, device):
        model.train()
        model.to(device)
        gradients = {}
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:  # handling layers that do not have gradients
                    gradients[name] = None
                else:
                    if name not in gradients:
                        gradients[name] = param.grad.clone().detach()
                    else:
                        gradients[name] += param.grad.clone().detach()

        for param_name in gradients:
            if gradients[param_name] is not None:  # if layers have gradients
                gradients[param_name] = gradients[param_name] / len(dataloader)

        return gradients

    def get_val_gradient_score(self, regular_model_params, order=1):
        target_model = deepcopy(self.model)
        target_model.load_state_dict(regular_model_params, strict=False)
        self.dataset.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.valset) <= 0:
            print("NO VAL SET FOUND!!")
            print("STOPPING THE TRAINING")
            exit()

        gradients = self.get_gradients(
            model=target_model,
            dataloader=self.valloader,
            criterion=criterion,
            device=self.device,
        )
        gradient_norm = self.get_norm(gradients, order=order)

        return gradient_norm

    def get_norm(self, layers, order):
        """
        order is for the norm order
        """
        if self.mode == "clientwise":
            norm = 0.0
            for _, l in layers.items():
                if l is not None:  # if layers have gradients
                    norm += torch.norm(l, p=order).item()
            return norm
        elif self.mode == "layerwise":
            norms = {}
            for layer_name, l in layers.items():
                if l is not None:
                    norm = torch.norm(l, p=order).item()
                    norms[layer_name] = norm
            return norms

    def get_relative_gradient_norm(self, regular_model_params, order=1):
        grad_norm = self.get_val_gradient_score(regular_model_params, order=order)
        param_norm = self.get_norm(regular_model_params, order=order)

        rgns = {
            k: 1 / (grad_norm[k] / param_norm[k])
            for k in param_norm.keys()
            if k in grad_norm.keys()
        }
        return rgns

    def get_weights_agg(self, client_rgns, eps=1e-8):
        """
        client_rgns: list of dictionaries containing rgn values
        """
        if self.model == "clientwise":
            raise NotImplementedError(
                "Clientwise aggrestion is not yet implemented for FedRGN"
            )

        elif self.mode == "layerwise":
            weights_dict = {}
            for layer_name in client_rgns[0].keys():
                layer_rgns = []
                for client in client_rgns:
                    layer_rgns.append(client[layer_name])
                layer_rgn_total = sum(layer_rgns)
                weights = [rgn / layer_rgn_total for rgn in layer_rgns]
                weights_dict[layer_name] = torch.tensor(weights)

        return weights_dict

    def train_one_round(self):
        client_packages = self.trainer.train()

        clients_model_params_diff = []
        client_rgns = []
        for package in client_packages.values():
            clients_model_params_diff.append(
                package["model_params_diff"]
            )  # \Delta \theta_i
            client_rgns.append(
                self.get_relative_gradient_norm(
                    package["regular_model_params"], order=2
                )
            )

        weights = self.get_weights_agg(client_rgns)

        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=weights,
        )


class AdaptiveOptimizerGrad(AdaptiveOptimizer):
    def __init__(
        self,
        optimizer_type: str,
        params_dict: OrderedDict[str, torch.Tensor],
        beta1: float,
        beta2: float,
        lr: float,
        tau: float,
        _type: str,
        mode: str,
    ):
        super().__init__(optimizer_type, params_dict, beta1, beta2, lr, tau, _type)
        self.mode = mode

    @torch.no_grad()
    def step(
        self,
        clients_model_params_diff: list[OrderedDict[str, torch.Tensor]],
        weights: torch.Tensor,
    ):
        params_diff = []

        # compute weighted delta
        if self.mode == "clientwise":
            list_clients_model_params_diff = [
                [-diff for diff in diff_dict.values()]
                for diff_dict in clients_model_params_diff
            ]
            for diff in zip(*list_clients_model_params_diff):
                params_diff.append(
                    torch.sum(torch.stack(diff, dim=-1) * weights, dim=-1)
                )

        elif self.mode == "layerwise":
            num_clients = len(clients_model_params_diff)

            for layer in clients_model_params_diff[0].keys():  # Iterate over layers
                layer_diffs = torch.stack(
                    [
                        client_diff[layer] * -1
                        for client_diff in clients_model_params_diff
                    ],
                    dim=-1,
                )
                if layer in weights:
                    layer_weights = torch.tensor(
                        [
                            weights[layer][i]
                            for i in range(len(clients_model_params_diff))
                        ],
                        device=layer_diffs.device,
                        dtype=layer_diffs.dtype,
                    )
                else:
                    layer_weights = torch.full(
                        (num_clients,),
                        1.0 / num_clients,
                        device=layer_diffs.device,
                        dtype=layer_diffs.dtype,
                    )

                params_diff.append(torch.sum(layer_diffs * layer_weights, dim=-1))

        if self._type != "avg":
            # update momentums
            for m, diff in zip(self.momentums, params_diff):
                m.data = self.beta1 * m + (1 - self.beta1) * diff

            # update velocities according to different rules
            self.update(params_diff)

            # update model parameters
            for param, m, v in zip(
                self.params_dict.values(), self.momentums, self.velocities
            ):
                param.data = param.data + self.lr * (m / (v.sqrt() + self.tau))

        else:
            for param, diff in zip(self.params_dict.values(), params_diff):
                param.data = param.data + diff
