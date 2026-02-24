from argparse import ArgumentParser, Namespace

# from collections import OrderedDict
from copy import deepcopy

import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

from src.server.fedgradbase import FedGradbaseServer


class FedGradAccServer(FedGradbaseServer):
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
        algorithm_name: str = "FedGradAcc",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        # algo = self.algo_names[args.fedacc.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.use_adaptive_opt = self.args.fedgradacc.use_adaptive_opt

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
        grad_norms = []
        accuracies = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            regular_model_params = package["regular_model_params"]
            grad_norm = self.get_val_gradient_score(regular_model_params)
            grad_norms.append(grad_norm)
            acc = self.get_val_acc(regular_model_params)
            accuracies.append(acc)

        client_weights_grad = self.get_weights_agg(grad_norms)
        sum_acc = sum(accuracies)
        client_weights_acc = [x / sum_acc for x in accuracies]
        client_weights_acc = torch.tensor(client_weights_acc)

        client_weights = (client_weights_grad + client_weights_acc) / 2

        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=client_weights,
        )
