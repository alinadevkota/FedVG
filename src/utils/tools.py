import os
import random
from argparse import Namespace
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import pynvml
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from rich.console import Console
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.constants import DEFAULTS
from src.utils.metrics import Metrics
from src.utils.simulate_realistic import get_realistic_valset


def fix_random_seed(seed: int, use_cuda=False) -> None:
    """Fix the random seed of FL training.

    Args:
        seed: Any number you like as the random seed.
        use_cuda: Flag indicates if using cuda.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    """Dynamically select CUDA device (has the most memory) for running FL
    experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        print(torch.cuda.device_count())  # temporary: to run in different GPU locally
        # assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")


def vectorize(
    src: OrderedDict[str, torch.Tensor] | list[torch.Tensor] | torch.nn.Module,
    detach=True,
) -> torch.Tensor:
    """Vectorize(Flatten) and concatenate all tensors in `src`.

    Args:
        `src`: The source of tensors.
        `detach`: Set as `True` to return `tensor.detach().clone()`. Defaults to `True`.

    Returns:
        The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict) or isinstance(src, dict):
        return torch.cat([func(param).flatten() for param in src.values()])
    elif isinstance(src, torch.nn.Module):
        return torch.cat([func(param).flatten() for param in src.state_dict().values()])
    elif isinstance(src, Iterator):
        return torch.cat([func(param).flatten() for param in src])


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    model_in_train_mode: bool = False,
) -> Metrics:
    """For evaluating the `model` over `dataloader` and return metrics.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").
        model_in_eval_mode (bool, optional): Set as `True` to switch model to eval mode. Defaults to `True`.

    Returns:
        Metrics: The metrics objective.
    """
    if model_in_train_mode:
        model.train()
    else:
        model.eval()
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics


def parse_args(
    config: DictConfig,
    method_name: str,
    get_method_args_func: Callable[[Sequence[str] | None], Namespace] | None,
) -> DictConfig:
    """Purge arguments from default args dict, config file and CLI and produce
    the final arguments.

    Args:
        config: DictConfig set from .yaml config file.
        method_name: The FL method's name.
        get_method_args_func: The callable function of parsing FL method `method_name`'s spec arguments.
    Returns:
        DictConfig: The final argument namespace.
    """
    final_args = DictConfig(DEFAULTS)

    def _merge_configs(defaults: DictConfig, config: DictConfig) -> DictConfig:
        merged = DictConfig({})
        for key, default_value in defaults.items():
            if key in config:
                if isinstance(default_value, DictConfig) and isinstance(
                    config[key], DictConfig
                ):
                    merged[key] = _merge_configs(default_value, config[key])
                else:
                    merged[key] = config[key]
            else:
                merged[key] = default_value
        return merged

    final_args = _merge_configs(final_args, config)

    if hasattr(config, method_name):
        final_args[method_name] = config[method_name]

    if get_method_args_func is not None:
        default_method_args = DictConfig(get_method_args_func([]).__dict__)
        if hasattr(final_args, method_name):
            for key in default_method_args.keys():
                if key not in final_args[method_name].keys():
                    final_args[method_name][key] = default_method_args[key]
        else:
            final_args[method_name] = default_method_args

    assert final_args.mode in [
        "serial",
        "parallel",
    ], f"Unrecongnized mode: {final_args.mode}"
    if final_args.mode == "parallel":
        import ray

        num_available_gpus = final_args.parallel.num_gpus
        num_available_cpus = final_args.parallel.num_cpus
        if num_available_gpus is None:
            pynvml.nvmlInit()
            num_total_gpus = pynvml.nvmlDeviceGetCount()
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
            else:
                num_available_gpus = num_total_gpus
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()

        try:
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                num_cpus=num_available_cpus,
                num_gpus=num_available_gpus,
                ignore_reinit_error=True,
            )
        except ValueError:
            # have existing cluster
            # then ignore num_cpus and num_gpus
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                ignore_reinit_error=True,
            )

        cluster_resources = ray.cluster_resources()
        final_args.parallel.num_cpus = cluster_resources["CPU"]
        final_args.parallel.num_gpus = cluster_resources["GPU"]
        if final_args.parallel.num_workers < 2:
            print(
                f"num_workers is less than 2: {final_args.parallel.num_workers}, "
                "mode fallbacks to serial."
            )
            final_args.mode = "serial"
            del final_args.parallel

    return final_args


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress
        bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_output_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_output_stream = open(logfile_path, "w")
            self.logfile_logger = Console(
                file=self.logfile_output_stream,
                record=True,
                log_path=False,
                log_time=False,
                soft_wrap=True,
                tab_size=4,
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logfile_logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_output_stream:
            self.logfile_output_stream.close()


def initialize_data_loaders(
    dataset: Dataset,
    data_indices: List[Dict[str, List[int]]],
    batch_size: int = 32,
    val_noise_rate=None,
    val_subset_ratio=None,
    val_imbalance_ratio=None,
    realistic_split=False,
    dataset_name=None,
    **dataloader_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, Subset]:
    """Initialize data loaders for training, validation, and testing.

    Args:
        dataset: The dataset to be used for creating subsets.
        data_indices: A list of dictionaries, where each dictionary contains
            the indices for 'train', 'val', and 'test' splits for a client.
        batch_size: The batch size for the data loaders. Defaults to 32.
        **dataloader_kwargs: Additional keyword arguments for the data loaders.

    Returns:
        A tuple containing:
        - trainloader: DataLoader for the training set.
        - testloader: DataLoader for the test set.
        - valloader: DataLoader for the validation set.
        - trainset: Subset of the dataset for training.
        - testset: Subset of the dataset for testing.
        - valset: Subset of the dataset for validation.
    """
    val_indices = np.concatenate(
        [client_i_indices["val"] for client_i_indices in data_indices]
    )
    test_indices = np.concatenate(
        [client_i_indices["test"] for client_i_indices in data_indices]
    )
    train_indices = np.concatenate(
        [client_i_indices["train"] for client_i_indices in data_indices]
    )

    sampled_indices_val = random.sample(
        list(val_indices),
        500 if dataset_name in ["covid19", "tiny_imagenet", "dermamnist"] else 500,
    )  # 100 for covid19 else 1000
    if realistic_split is False:
        valset = Subset(dataset, val_indices)
        valsubset = Subset(dataset, sampled_indices_val)
    else:
        valset = get_realistic_valset(dataset_name)
        valsubset = valset

    sampled_indices_train = random.sample(
        list(train_indices), 10
    )  # 100 for covid19 else 1000
    testset = Subset(dataset, test_indices)
    trainset = Subset(dataset, train_indices)
    trainsubset = Subset(dataset, sampled_indices_train)

    if val_subset_ratio is not None and val_subset_ratio != "None":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("initial size: ", len(valset))
        labels = np.array([valset[i][1] for i in range(len(valset))])

        from collections import Counter

        orig_dist = Counter(labels)
        print("Original class distribution:")
        for cls, count in sorted(orig_dist.items()):
            print(f"  Class {cls}: {count}")

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=42
        )  # use 0.3 for 30% subset

        for subset_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            stratified_subset = Subset(valset, subset_idx)

            break
        valset = stratified_subset
        print("final size: ", len(valset))

        # Print subset class distribution
        subset_labels = np.array([valset[i][1] for i in range(len(valset))])
        subset_dist = Counter(subset_labels)
        print("Subset class distribution:")
        for cls, count in sorted(subset_dist.items()):
            print(f"  Class {cls}: {count}")

    if val_noise_rate is not None and val_noise_rate != "None":
        valset = add_noise_once(valset, sigma=val_noise_rate)

    if val_imbalance_ratio is not None and val_imbalance_ratio != "None":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMBALAMNCE CASEEEEEE")
        print("initial size: ", len(valset))
        labels = np.array([valset[i][1] for i in range(len(valset))])

        from collections import Counter

        orig_dist = Counter(labels)
        print("Original class distribution:")
        for cls, count in sorted(orig_dist.items()):
            print(f"  Class {cls}: {count}")

        valset = create_imbalanced_subset(valset, val_imbalance_ratio)

        print("final size: ", len(valset))

        # Print subset class distribution
        subset_labels = np.array([valset[i][1] for i in range(len(valset))])
        subset_dist = Counter(subset_labels)
        print("Subset class distribution:")
        for cls, count in sorted(subset_dist.items()):
            print(f"  Class {cls}: {count}")

        # visualize_valset_class_distribution(valset, val_imbalance_ratio)
        # exit()
        valloader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs,
        )
    else:
        valloader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
        )

    valsubloader = DataLoader(
        valsubset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )
    trainsubloader = DataLoader(
        trainsubset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
    )

    return (
        trainloader,
        testloader,
        valloader,
        # valsubloader,
        trainset,
        testset,
        valset,
        trainsubloader,
        valsubloader,
    )


def create_imbalanced_subset(valset, val_imbalance_ratio):
    labels = np.array([valset[i][1] for i in range(len(valset))])
    class_indices = defaultdict(list)

    # Collect indices for each class
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    num_classes = len(class_indices)
    total_size = len(valset) // 2  # Halve the original size

    # Create long-tailed class distribution (power-law)
    # class_proportions will sum to 1
    class_proportions = np.array([val_imbalance_ratio**i for i in range(num_classes)])
    class_proportions /= class_proportions.sum()

    # Calculate how many samples to take per class
    samples_per_class = (class_proportions * total_size).astype(int)

    selected_indices = []

    for i, count in enumerate(samples_per_class):
        indices = class_indices[i]
        if len(indices) == 0 or count == 0:
            continue
        count = min(count, len(indices))  # don't oversample
        selected = random.sample(indices, count)
        selected_indices.extend(selected)

    # Final imbalanced subset
    imbalanced_subset = Subset(valset, selected_indices)
    return imbalanced_subset


# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.noise = None

    def __call__(self, tensor):
        if self.noise is None:
            self.noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + self.noise, 0.0, 1.0)


def add_noise_once(valset, sigma, seed=42):
    from torch.utils.data import TensorDataset

    g = torch.Generator().manual_seed(seed)  # reproducibility

    noisy_images = []
    labels = []
    for i in range(len(valset)):
        x, y = valset[i]  # this applies the transform already
        noise = torch.normal(0.0, sigma, size=x.shape, generator=g)
        x_noisy = torch.clamp(noise, 0.0, 1.0)
        noisy_images.append(x_noisy)
        labels.append(y)

    noisy_images = torch.stack(noisy_images)
    labels = torch.tensor(labels)

    # Return a new dataset with noisy images
    return TensorDataset(noisy_images, labels)


def create_noisy_transform(noise_rate):
    transform_with_noise = transforms.Compose(
        [
            AddGaussianNoise(mean=0.0, std=noise_rate),  # Add Gaussian noise
        ]
    )
    return transform_with_noise


class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def visualize_valset(valset, noise_ratio):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    # Load 10 samples from the transformed validation set
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(4):
        image, label = valset[i]

        # Undo normalization if needed
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    fig.suptitle(f"Epsilon = {noise_ratio}", fontsize=24)
    plt.tight_layout()
    print(f"valsets_noise_ratio{noise_ratio}.png")
    plt.savefig(f"valsets_noise_ratio{noise_ratio}.png")
