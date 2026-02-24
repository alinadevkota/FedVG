import os
from argparse import ArgumentParser, Namespace

# from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.server.fedavg import FedAvgServer


class FedMultipleServer(FedAvgServer):
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
        # self.stats = []
        self.save_path = f"/home/Desktop/FL-grad-aggregation/out/debug_prelim_analysis_federated_{self.args.dataset.name}_{self.args.model.name}"  # noqa: E501
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.num_epochs = self.args.common.global_epoch

    def get_gradients_batch(self, model, x_batch, y_batch):
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

    def compute_uncertainty_for_dataset(
        self, model, dataloader, noise_std=0.01, num_samples=5
    ):
        device = self.device
        model.to(device)
        model.eval()

        layerwise_uncertainties_cosinesim = (
            {}
        )  # key: layer name, value: list of uncertainties
        layerwise_uncertainties_var = {}
        layerwise_uncertainties_norm = {}
        layer_names = None
        num_batches = 0

        for x_batch, y_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)

            # Step 1: Compute original gradients
            orig_grads = self.get_gradients_batch(
                model, x_batch, y_batch
            )  # list of dicts: [B][layer_name -> tensor]
            if layer_names is None:
                layer_names = orig_grads[0].keys()
                for name in layer_names:
                    layerwise_uncertainties_cosinesim[name] = []
                    layerwise_uncertainties_var[name] = []
                    layerwise_uncertainties_norm[name] = []

            # Step 2: Create a single perturbed input batch
            x_perturb_all = x_batch.repeat(num_samples, 1, 1, 1)
            noise = torch.randn_like(x_perturb_all) * noise_std
            x_perturb_all += noise

            y_perturb_all = y_batch.repeat(num_samples)

            # Step 3: Compute perturbed gradients
            grads_pert = self.get_gradients_batch(
                model, x_perturb_all, y_perturb_all
            )  # list of dicts: [num_samples * B][layer_name -> tensor]

            # Step 4: Group perturbed grads by original sample
            grads_pert_grouped = [
                grads_pert[
                    i::B
                ]  # gets [i, i+B, i+2B, ...] => num_samples items for sample i
                for i in range(B)
            ]

            # Step 5: Compute cosine similarities per sample per layer
            for i in range(B):
                for name in layer_names:
                    orig_grad = orig_grads[i][name]
                    sims = []
                    grads = [orig_grad]
                    for j in range(num_samples):
                        pert_grad = grads_pert_grouped[i][j][name]
                        # print(orig_grad.shape, pert_grad.shape, name)
                        sim = F.cosine_similarity(
                            orig_grad.view(-1).unsqueeze(0),
                            pert_grad.view(-1).unsqueeze(0),
                            dim=1,
                        ).item()
                        sims.append(sim)
                        grads.append(pert_grad)

                    avg_sim = sum(sims) / len(sims)
                    uncertainty = 1.0 - avg_sim
                    layerwise_uncertainties_cosinesim[name].append(uncertainty)

                    stacked_grads = torch.stack(grads, dim=0)
                    avg_grad = stacked_grads.mean(dim=0)
                    grad_var = stacked_grads.var(dim=0)
                    grad_var_mean = grad_var.mean()
                    avg_grad_norm = torch.norm(avg_grad, p=1)

                    layerwise_uncertainties_var[name].append(grad_var_mean)
                    layerwise_uncertainties_norm[name].append(avg_grad_norm)

            num_batches += 1

        # Step 6: Compute average uncertainty across all batches for each layer
        for name in layer_names:
            layerwise_uncertainties_cosinesim[name] = (
                sum(layerwise_uncertainties_cosinesim[name]) / num_batches
            )
            layerwise_uncertainties_var[name] = (
                sum(layerwise_uncertainties_var[name]) / num_batches
            )
            layerwise_uncertainties_norm[name] = (
                sum(layerwise_uncertainties_norm[name]) / num_batches
            )

        return (
            layerwise_uncertainties_cosinesim,
            layerwise_uncertainties_var,
            layerwise_uncertainties_norm,
        )
        # dict: layer_name -> avg uncertainty score

    def get_gradients(self, model, dataloader):
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        model.to(self.device)
        gradients = {}
        for x, y in dataloader:
            model.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
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
        gradient_norms = {}
        for layer_name, grad in gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad, p=order).item()
                gradient_norms[layer_name] = grad_norm
            else:
                gradient_norms[layer_name] = None
        return gradient_norms

    def get_spectral_sigma(self, gradients):
        sigmas = {}
        for layer_name, grad_matrix in gradients.items():
            if grad_matrix is None or grad_matrix.ndim <= 1:
                sigmas[layer_name] = None
                continue
            if "conv" in layer_name or "downsample" in layer_name:
                shape = grad_matrix.shape
                grad_matrix = grad_matrix.reshape(shape[0], -1)
            # print(layer_name, grad_matrix.shape, grad_matrix)
            U, S, Vt = torch.linalg.svd(grad_matrix, full_matrices=False)
            sigmas[layer_name] = S[0]

        return sigmas

    def get_accuracy(self, model, dataloader):
        model.eval()
        model.to(self.device)
        all_preds = []
        all_labels = []
        with torch.no_grad():  # Disable gradient calculation for validation
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                predicted = logits.argmax(1)  # Use argmax to get the predicted class
                all_preds.extend(predicted.cpu().numpy())  # Store predictions on CPU
                all_labels.extend(y.cpu().numpy())  # Store actual labels on CPU

        # Use accuracy_score from sklearn to calculate the accuracy
        avg_accuracy = accuracy_score(all_labels, all_preds)
        return avg_accuracy

    def get_stats_for_single_model(self, params, delta_params, target_model):
        val_gradients = self.get_gradients(target_model, self.valsubloader)
        param_norm = self.get_gradient_norm(params, order=1)
        param_grad_norm = self.get_gradient_norm(delta_params, order=1)
        val_grad_norm = self.get_gradient_norm(val_gradients, order=1)
        sigmas_params = self.get_spectral_sigma(params)
        sigmas_param_grad = self.get_spectral_sigma(delta_params)
        sigmas_val_grad = self.get_spectral_sigma(val_gradients)
        print("Getting Performance")
        test_accuracy = self.get_accuracy(target_model, self.testloader)
        val_accuracy = self.get_accuracy(target_model, self.valloader)
        train_accuracy = self.get_accuracy(target_model, self.trainloader)
        print("Getting Uncertainties")
        us_cosinesim, us_var, us_norm = self.compute_uncertainty_for_dataset(
            target_model, self.valsubloader
        )
        return {
            "delta_params": delta_params,
            "params": params,
            "val_gradients": val_gradients,
            "param_norm": param_norm,
            "param_grad_norm": param_grad_norm,
            "val_grad_norm": val_grad_norm,
            "sigmas_params": sigmas_params,
            "sigmas_param_grad": sigmas_param_grad,
            "sigmas_val_grad": sigmas_val_grad,
            "test_accuracy": test_accuracy,
            "val_accuracy": val_accuracy,
            "train_accuracy": train_accuracy,
            "us_cosinesim": us_cosinesim,
            "us_var": us_var,
            "us_norm": us_norm,
        }

    def get_param_diff(self, params, previous_params):
        return {
            key: param_old - param_new
            for (key, param_new), param_old in zip(
                params.items(),
                previous_params.values(),
            )
        }

    def get_stats(self, client_packages):
        print("Getting Stats")
        stats = {}
        target_model = deepcopy(self.model)

        params_global = self.model.state_dict()
        params_global_prev = self.prev_model.state_dict()
        delta_params_global = self.get_param_diff(params_global, params_global_prev)
        stats["global"] = self.get_stats_for_single_model(
            params_global, delta_params_global, self.model
        )

        for client_id, package in client_packages.items():
            params = package["regular_model_params"]
            delta_params = package["model_params_diff"]
            target_model.load_state_dict(params, strict=False)
            print(f"Getting Gradients for client {client_id}")
            row = self.get_stats_for_single_model(params, delta_params, target_model)
            stats[client_id] = row

        return stats

    def train_one_round(self):
        self.prev_model = deepcopy(self.model)
        client_packages = self.trainer.train()
        self.aggregate_client_updates(client_packages)
        stats = self.get_stats(client_packages)
        torch.save(
            stats, os.path.join(self.save_path, f"stats_{self.current_epoch}.pt")
        )
