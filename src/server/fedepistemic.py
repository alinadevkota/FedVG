from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from omegaconf import DictConfig

# from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.server.fedavg import FedAvgServer
from src.server.fedopt import AdaptiveOptimizer


class FedEpistemicServer(FedAvgServer):
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
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.adaptive_optimizer = AdaptiveOptimizerGrad(
            optimizer_type=self.args.fedepistemic.type,
            params_dict=self.public_model_params,
            beta1=self.args.fedepistemic.beta1,
            beta2=self.args.fedepistemic.beta2,
            lr=self.args.fedepistemic.server_lr,
            tau=self.args.fedepistemic.tau,
            _type=self.args.fedepistemic.type,
            mode="layerwise",
        )

    def _get_gradients_batch(self, model, x_batch, y_batch):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        B = x_batch.size(0)

        grads_per_sample = []

        for i in range(B):
            x = x_batch[i].unsqueeze(0).clone().detach().requires_grad_(True)
            y = y_batch[i].unsqueeze(0)
            model.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            sample_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    sample_grads[name] = param.grad.detach().clone()
            grads_per_sample.append(sample_grads)

        return grads_per_sample  # List of dicts: [B] -> {layer_name: grad_tensor}

    def get_certainty_score(self, model, dataloader, noise_std=0.01, num_samples=5):
        device = self.device
        model.to(device)
        model.eval()

        layerwise_uncertainties_norm = {}
        layer_names = None
        num_batches = 0

        for x_batch, y_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)

            # Step 1: Compute original gradients (returns list of dicts: [B][layer_name -> tensor])
            orig_grads = self._get_gradients_batch(model, x_batch, y_batch)

            if layer_names is None:
                layer_names = orig_grads[0].keys()
                for name in layer_names:
                    layerwise_uncertainties_norm[name] = []

            # Step 2: Perturb input
            x_perturb_all = x_batch.repeat(num_samples, 1, 1, 1)
            noise = torch.randn_like(x_perturb_all) * noise_std
            x_perturb_all += noise
            y_perturb_all = y_batch.repeat(num_samples)

            # Step 3: Compute perturbed gradients
            grads_pert = self._get_gradients_batch(model, x_perturb_all, y_perturb_all)

            # Step 4: Compute per-layer metrics
            for name in layer_names:
                # Stack original grads [B, ...]
                orig_grads_tensor = torch.stack([g[name] for g in orig_grads], dim=0)

                # Stack perturbed grads: [num_samples*B, ...]
                grads_pert_tensor = torch.stack([g[name] for g in grads_pert], dim=0)

                # Reshape to [num_samples, B, ...]
                grads_pert_tensor = grads_pert_tensor.view(
                    num_samples, B, *grads_pert_tensor.shape[1:]
                )

                # Stack for variance and norm calculations: [num_samples+1, B, ...]
                all_grads_tensor = torch.cat(
                    (orig_grads_tensor.unsqueeze(0), grads_pert_tensor), dim=0
                )
                stacked = all_grads_tensor.view(
                    num_samples + 1, B, -1
                )  # [num_samples+1, B, D]

                norm = stacked.mean(dim=0).norm(p=1, dim=1)  # [B]

                layerwise_uncertainties_norm[name].extend(norm.tolist())

            num_batches += 1

        # Step 6: Compute average uncertainty across all batches for each layer
        for name in layer_names:
            layerwise_uncertainties_norm[name] = sum(
                layerwise_uncertainties_norm[name]
            ) / len(layerwise_uncertainties_norm[name])

        return layerwise_uncertainties_norm

    def normalize_scores(self, uncertainties, eps=1e-8):
        weights_dict = {}
        layer_names = uncertainties[0].keys()
        for layer_name in layer_names:
            inv_grad_norms = []
            for idx in range(len(uncertainties)):
                inv_grad_norms.append(1 / (uncertainties[idx][layer_name] + eps))
            sum_inv_norm = sum(inv_grad_norms)
            weights = [inv_norm / sum_inv_norm for inv_norm in inv_grad_norms]
            weights_dict[layer_name] = torch.tensor(weights)

        return weights_dict

    def train_one_round(self):
        client_packages = self.trainer.train()

        target_model = deepcopy(self.model)

        clients_model_params_diff = []
        uncertainties = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            regular_model_params = package["regular_model_params"]
            target_model.load_state_dict(regular_model_params, strict=False)
            uncertainty = self.get_certainty_score(target_model, self.valsubloader)
            uncertainties.append(uncertainty)

        client_weights = self.normalize_scores(uncertainties)
        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=client_weights,
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
