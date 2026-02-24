from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
from copy import deepcopy

import torch
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer
from src.server.fedopt import AdaptiveOptimizer


class FedGradServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--type",
            choices=["adagrad", "yogi", "adam", "avg", "avgm"],
            type=str,
            default="avg",
        )
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--server_lr", type=float, default=1e-1)
        parser.add_argument("--tau", type=float, default=1e-3)
        parser.add_argument("--server_momentum", type=float, default=0.9)
        parser.add_argument(
            "--mode",
            type=str,
            default="clientwise",
            choices=["clientwise", "layerwise", "spectral"],
        )
        parser.add_argument("--layer_as_block", type=bool, default=False)

        parser.add_argument("--norm_order", type=float, default=1)
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
        # algo = self.algo_names[args.fedgrad.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.mode = self.args.fedgrad.mode
        self.layer_as_block = self.args.fedgrad.layer_as_block
        self.norms_over_time = []
        self.ns = 2  # number of singular values to take
        self.norm_order = self.args.fedgrad.norm_order

        self.adaptive_optimizer = AdaptiveOptimizerGrad(
            optimizer_type=self.args.fedgrad.type,
            params_dict=self.public_model_params,
            beta1=self.args.fedgrad.beta1,
            beta2=self.args.fedgrad.beta2,
            lr=self.args.fedgrad.server_lr,
            tau=self.args.fedgrad.tau,
            _type=self.args.fedgrad.type,
            mode=self.args.fedgrad.mode,
            layer_as_block=self.layer_as_block,
            model_name=self.args.model.name,
        )

    def get_gradients(self, model, dataloader, criterion, device):
        model.train()
        model.to(device)
        gradients = {}
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
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

    def get_gradient_norm(self, gradients, order):
        """
        order is for the norm order
        """
        if self.mode == "clientwise":
            gradient_norm = 0.0
            for _, grad in gradients.items():
                if grad is not None:  # if layers have gradients
                    gradient_norm += torch.norm(grad, p=order).item()
            return gradient_norm
        elif self.mode == "layerwise" or self.mode == "blockwise":
            gradient_norms = {}
            for layer_name, grad in gradients.items():
                if grad is not None:
                    grad_norm = torch.norm(grad, p=order).item()
                    gradient_norms[layer_name] = grad_norm
            return gradient_norms

    def get_val_gradient_score(self, regular_model_params):
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
        if self.mode != "spectral":
            gradient_norm = self.get_gradient_norm(gradients, order=self.norm_order)
            return gradient_norm
        else:
            return {
                layer_name: grad
                for layer_name, grad in gradients.items()
                if grad is not None
            }

    def get_weights_agg(
        self, grad_norms, eps=1e-8
    ):  # eps needed to avoid divison by zero
        if self.mode == "clientwise":
            inv_grad_norms = [1 / (gn + eps) for gn in grad_norms]
            sum_inv_norm = sum(inv_grad_norms)
            weights = [inv_norm / sum_inv_norm for inv_norm in inv_grad_norms]
            return torch.tensor(weights)
        elif self.mode == "layerwise" or self.mode == "spectral":
            weights_dict = {}
            layer_names = grad_norms[0].keys()
            if self.mode == "spectral":
                for layer_name in layer_names:
                    inv_sigma1s = []
                    for idx in range(len(grad_norms)):
                        grad_matrix = grad_norms[idx][layer_name]
                        if "bias" in layer_name or "bn" in layer_name:
                            inv_sigma1s.append(
                                1 / (torch.linalg.vector_norm(grad_matrix).item() + eps)
                            )
                            continue
                        if "conv" in layer_name or "downsample" in layer_name:
                            shape = grad_matrix.shape
                            grad_matrix = grad_matrix.reshape(shape[0], -1)
                        # print(layer_name, grad_matrix.shape, grad_matrix)
                        U, S, Vt = torch.linalg.svd(grad_matrix, full_matrices=False)
                        sigma1 = S[0]
                        inv_sigma1s.append(1 / (sigma1 + eps))
                    sum_inv_sigma1 = sum(inv_sigma1s)
                    weights = [
                        inv_sigma1 / sum_inv_sigma1 for inv_sigma1 in inv_sigma1s
                    ]
                    weights_dict[layer_name] = torch.tensor(weights)
            else:
                for layer_name in layer_names:
                    inv_grad_norms = []
                    for idx in range(len(grad_norms)):
                        inv_grad_norms.append(1 / (grad_norms[idx][layer_name] + eps))
                    sum_inv_norm = sum(inv_grad_norms)
                    weights = [inv_norm / sum_inv_norm for inv_norm in inv_grad_norms]
                    weights_dict[layer_name] = torch.tensor(weights)
            return weights_dict
        elif self.mode == "blockwise":
            block_map = get_block_map(
                grad_norms[0].keys(), self.layer_as_block, self.args.model.name
            )  # returns {block_name: [layer_names]}

            weights_dict = {}

            for block, layers in block_map.items():
                inv_grad_norms = []
                for client_idx in range(len(grad_norms)):
                    # Sum norms of layers in the block for this client
                    block_norm = sum(
                        grad_norms[client_idx][ln]
                        for ln in layers
                        if ln in grad_norms[client_idx]
                    )
                    inv_grad_norms.append(1 / (block_norm + eps))
                sum_inv_norm = sum(inv_grad_norms)
                weights = [inv / sum_inv_norm for inv in inv_grad_norms]
                weights_dict[block] = torch.tensor(weights)

            return weights_dict

    def train_one_round(self):
        client_packages = self.trainer.train()

        clients_model_params_diff = []
        grad_norms = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            regular_model_params = package["regular_model_params"]
            grad_norm = self.get_val_gradient_score(regular_model_params)
            grad_norms.append(grad_norm)

        client_weights = self.get_weights_agg(grad_norms)  #
        self.norms_over_time.append(client_weights)
        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=client_weights,
        )


def get_block_map(layer_names, layer_as_block=False, model_name=None):
    """
    Groups layers into blocks based on name prefixes (e.g., 'base.conv1', 'base.layer1.0.conv1')
    Returns a dict: {block_name: [layer_names]}
    """
    block_map = defaultdict(list)
    # print(layer_names)
    for name in layer_names:
        if "classifier" in name:
            block = "classifier"
        if name.startswith("base.features") and model_name.startswith("vit_swin"):
            parts = name.split(".")
            block = ".".join(parts[:3]) if not layer_as_block else ".".join(parts[:-2])
        elif "base.encoder.layers" in name and model_name.startswith("vit"):
            parts = name.split(".")
            block = ".".join(parts[:4]) if not layer_as_block else ".".join(parts[:-3])
        elif "blocks" in name and model_name == "vit_s_16":
            parts = name.split(".")
            block = ".".join(parts[:3]) if not layer_as_block else ".".join(parts[:-2])
        elif "layer" in name:
            parts = name.split(".")
            block = ".".join(parts[:2]) if not layer_as_block else ".".join(parts[:-1])
        else:
            parts = name.split(".")
            block = parts[0] if not layer_as_block else ".".join(parts[:-1])
            # raise Exception("Something went wrong with getting layer maps")
        block_map[block].append(name)

    return dict(block_map)


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
        layer_as_block: bool,
        model_name: str = None,
    ):
        super().__init__(optimizer_type, params_dict, beta1, beta2, lr, tau, _type)
        self.mode = mode
        self.layer_as_block = layer_as_block
        self.model_name = model_name

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

        elif self.mode == "layerwise" or self.mode == "spectral":
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
        elif self.mode == "blockwise":
            num_clients = len(clients_model_params_diff)
            block_map = get_block_map(
                clients_model_params_diff[0].keys(),
                self.layer_as_block,
                self.model_name,
            )
            layer_to_block = {
                layer_name: block
                for block, layer_list in block_map.items()
                for layer_name in layer_list
            }
            for layer in clients_model_params_diff[0].keys():  # Iterate over layers
                layer_diffs = torch.stack(
                    [
                        client_diff[layer] * -1
                        for client_diff in clients_model_params_diff
                    ],
                    dim=-1,
                )
                # Determine block name for this layer
                block = layer_to_block.get(layer, None)
                if block is None:
                    raise Exception(f"Layer '{layer}' not found in any block.")

                if block in weights:
                    block_weights = torch.tensor(
                        [weights[block][i] for i in range(num_clients)],
                        device=layer_diffs.device,
                        dtype=layer_diffs.dtype,
                    )
                else:
                    block_weights = torch.full(
                        (num_clients,),
                        1.0 / num_clients,
                        device=layer_diffs.device,
                        dtype=layer_diffs.dtype,
                    )

                params_diff.append(torch.sum(layer_diffs * block_weights, dim=-1))

        if self._type == "avg":
            for param, diff in zip(self.params_dict.values(), params_diff):
                param.data = param.data + diff
        elif self._type == "avgm":
            self.avgm_optimizer.zero_grad()
            for param, diff in zip(self.params_dict.values(), params_diff):
                param.grad = -diff
            self.avgm_optimizer.step()
        else:
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
