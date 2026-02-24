from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

from src.server.fedavg import FedAvgServer
from src.server.fedopt import AdaptiveOptimizer


class FedAccServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--type",
            choices=["adagrad", "yogi", "adam", "avg", "avgm"],
            type=str,
            default="adam",
        )
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--server_lr", type=float, default=1e-1)
        parser.add_argument("--tau", type=float, default=1e-3)
        parser.add_argument("--server_momentum", type=float, default=0.9)
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
        algorithm_name: str = "FedAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        # algo = self.algo_names[args.fedacc.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.adaptive_optimizer = AdaptiveOptimizerGrad(
            optimizer_type=self.args.fedacc.type,
            params_dict=self.public_model_params,
            beta1=self.args.fedacc.beta1,
            beta2=self.args.fedacc.beta2,
            lr=self.args.fedacc.server_lr,
            tau=self.args.fedacc.tau,
            _type=self.args.fedacc.type,
        )

    def get_val_acc(
        self,
        regular_model_params,
    ):
        target_model = deepcopy(self.model)
        target_model.load_state_dict(regular_model_params, strict=False)
        self.dataset.eval()

        target_model.eval()
        target_model.to(self.device)
        all_preds = []
        all_labels = []
        with torch.no_grad():  # Disable gradient calculation for validation
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = target_model(x)
                predicted = logits.argmax(1)  # Use argmax to get the predicted class
                all_preds.extend(predicted.cpu().numpy())  # Store predictions on CPU
                all_labels.extend(y.cpu().numpy())  # Store actual labels on CPU

        # Use accuracy_score from sklearn to calculate the accuracy
        avg_accuracy = accuracy_score(all_labels, all_preds)
        return avg_accuracy

    def train_one_round(self):
        client_packages = self.trainer.train()

        clients_model_params_diff = []
        accuracies = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            regular_model_params = package["regular_model_params"]
            acc = self.get_val_acc(regular_model_params)
            accuracies.append(acc)

        sum_acc = sum(accuracies)
        client_weights = [x / sum_acc for x in accuracies]
        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=torch.tensor(client_weights),
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
    ):
        super().__init__(optimizer_type, params_dict, beta1, beta2, lr, tau, _type)

    @torch.no_grad()
    def step(
        self,
        clients_model_params_diff: list[OrderedDict[str, torch.Tensor]],
        weights: torch.Tensor,
    ):
        params_diff = []

        # compute weighted delta
        list_clients_model_params_diff = [
            [-diff for diff in diff_dict.values()]
            for diff_dict in clients_model_params_diff
        ]
        for diff in zip(*list_clients_model_params_diff):
            params_diff.append(torch.sum(torch.stack(diff, dim=-1) * weights, dim=-1))

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
