from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from medmnist import DermaMNIST
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


def build_cifar10_to_cifar100_map():
    """
    Clean, non-overlapping, semantically meaningful CIFAR-10 → CIFAR-100 mapping.
    Uses only actual CIFAR-100 class names (no imaginary labels).
    """
    return {
        "airplane": [],  # aerial / rail vehicles
        "automobile": ["bus", "motorcycle"],  # ground vehicles
        "bird": [],  # animals related to 'bird'-type category (aquatic mammals)
        "cat": ["leopard", "lion", "tiger"],  # felines & small carnivores
        "deer": [],  # large herbivorous mammals
        "dog": ["wolf", "fox"],  # canine-like / small mammals
        "frog": [],  # reptiles/amphibians
        "horse": [],  # large upright animals
        "ship": [],  # water/structure-related
        "truck": ["pickup truck"],  # heavy machinery
    }


def build_cifar10_to_stl10_map():
    """STL-10 has exact same classes as CIFAR-10"""
    return {
        "airplane": ["airplane"],
        "automobile": ["car"],
        "bird": ["bird"],
        "cat": ["cat"],
        "deer": ["deer"],
        "dog": ["dog"],
        "frog": [],
        "horse": ["horse"],
        "ship": ["ship"],
        "truck": ["truck"],
    }


def build_label_maps_cifar100(dataset_root):
    """Returns CIFAR-100 class name -> index map and class names list."""
    cifar100_labels = datasets.CIFAR100(
        root=dataset_root, train=True, download=True
    ).classes
    label_map = {name: idx for idx, name in enumerate(cifar100_labels)}
    return label_map, cifar100_labels


def build_label_maps(dataset_root):
    """Returns STL-10 class name -> index map and class names list"""
    stl10_labels = datasets.STL10(
        root=dataset_root, split="train", download=True
    ).classes
    label_map = {name: idx for idx, name in enumerate(stl10_labels)}
    return label_map, stl10_labels


def build_superclass_remap(cifar10_to_dataset, label_map):
    """Mapping from STL-10 label index -> CIFAR-10 superclass index (0–9)"""
    orig_to_super = {}
    for super_idx, (super_name, member_names) in enumerate(cifar10_to_dataset.items()):
        for m in member_names:
            if m in label_map:
                orig_to_super[label_map[m]] = super_idx
    return orig_to_super


class SubsetRemapped(Dataset):
    def __init__(self, base_dataset, indices, remap_dict):
        self.base = base_dataset
        self.indices = indices
        self.remap_dict = remap_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        return x, self.remap_dict[y]


def print_class_distribution(labels, class_names, title="Class Distribution"):
    counts = Counter(labels)
    print(f"\n{title}:")
    for class_idx, count in sorted(counts.items()):
        name = (
            class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        )
        print(f"{name:<15s}: {count}")
    print(f"Total samples: {sum(counts.values())}")


def get_dermamnist_isic_ids():
    derm_train = DermaMNIST(split="train", download=True)
    derm_val = DermaMNIST(split="val", download=True)
    derm_test = DermaMNIST(split="test", download=True)

    all_ids = []

    for ds in [derm_train, derm_val, derm_test]:
        # Use indices as unique IDs
        all_ids.extend([f"{ds.__class__.__name__}_{i}" for i in range(len(ds))])

    return set(all_ids)


class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019 (filtered for DermMNIST overlap exclusion).
    Converts one-hot labels to integers 0-6 for DermMNIST classes.
    """

    def __init__(self, root, transform=None, target_transform=None, exclude_ids=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.exclude_ids = set(exclude_ids) if exclude_ids else set()

        csv_path = self.root / "ISIC_2019_Training_GroundTruth.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing ISIC CSV file: {csv_path}")

        df = pd.read_csv(csv_path)

        # Only consider these 7 classes for DermMNIST mapping
        all_labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]
        dx_to_derma = {lbl: idx for idx, lbl in enumerate(all_labels)}

        self.image_paths = []
        self.labels = []

        for _, row in df.iterrows():
            image_id = row["image"]
            if image_id in self.exclude_ids:
                continue  # skip overlaps

            img_path = self.root / "ISIC_2019_Training_Input" / f"{image_id}.jpg"
            if not img_path.exists():
                continue

            # Find the one-hot label among the 7 classes
            found = False
            for lbl in all_labels:
                if lbl in row and row[lbl] == 1:
                    self.image_paths.append(img_path)
                    self.labels.append(dx_to_derma[lbl])
                    found = True
                    break
            if not found:
                # skip SCC, UNK, or unlabeled
                continue

        print(
            f"✅ Loaded {len(self.image_paths)} ISIC 2019 samples "
            f"(excluded {len(self.exclude_ids)} overlaps)."
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_realistic_valset_cifar10_cifar100(min_per_class, total_subset_size):
    CURRENT_DIR = Path("/home/FL-grad-aggregation")
    dataset_root = CURRENT_DIR / "data_realistic/cifar100"

    cifar10_to_dataset = build_cifar10_to_cifar100_map()
    label_map, dataset_labels = build_label_maps_cifar100(dataset_root)
    orig_to_super = build_superclass_remap(cifar10_to_dataset, label_map)

    transform = transforms.Compose([transforms.ToTensor()])
    cifar100 = datasets.CIFAR100(root=dataset_root, train=True, transform=transform)

    # Group indices by superclass
    super_to_indices = defaultdict(list)
    for idx, t in enumerate(cifar100.targets):
        if t in orig_to_super:
            super_idx = orig_to_super[t]
            super_to_indices[super_idx].append(idx)

    g = torch.Generator().manual_seed(42)
    selected_indices = []

    # Step 1: ensure min_per_class per superclass
    for super_idx in range(10):
        indices = super_to_indices.get(super_idx, [])
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for superclass {super_idx}")
            continue
        n_samples = min(len(indices), min_per_class)
        subset = torch.randperm(len(indices), generator=g)[:n_samples]
        selected_indices.extend([indices[i] for i in subset])
        print(f"Superclass {super_idx}: selected {n_samples}/{len(indices)}")

    # Step 2: fill remaining budget uniformly
    remaining_budget = total_subset_size - len(selected_indices)
    if remaining_budget > 0:
        all_pool = [i for lst in super_to_indices.values() for i in lst]
        extra = torch.randperm(len(all_pool), generator=g)[:remaining_budget]
        selected_indices.extend([all_pool[i] for i in extra])

    # Step 3: print original STL-10 distribution
    val_targets = [cifar100.targets[i] for i in selected_indices]
    print_class_distribution(
        val_targets, dataset_labels, title="Original STL-10 class distribution"
    )

    # Step 4: wrap in remapped dataset
    val_subset = SubsetRemapped(cifar100, selected_indices, orig_to_super)

    # Step 5: print CIFAR-10-level distribution
    remapped_targets = [orig_to_super[t] for t in val_targets if t in orig_to_super]
    print_class_distribution(
        remapped_targets,
        list(cifar10_to_dataset.keys()),
        title="Remapped CIFAR-10 superclass distribution",
    )

    print(f"\n✅ Final subset size: {len(selected_indices)}")
    return val_subset


def get_realistic_valset_cifar10(min_per_class, total_subset_size):
    CURRENT_DIR = Path("/home/FL-grad-aggregation")
    dataset_root = CURRENT_DIR / "data_realistic/stl10"

    # Mapping & labels
    cifar10_to_dataset = build_cifar10_to_stl10_map()
    label_map, dataset_labels = build_label_maps(dataset_root)
    orig_to_super = build_superclass_remap(cifar10_to_dataset, label_map)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # downsample from 96x96 -> 32x32
            transforms.ToTensor(),
        ]
    )
    stl10 = datasets.STL10(root=dataset_root, split="train", transform=transform)

    # Group indices by superclass
    super_to_indices = defaultdict(list)
    for idx, t in enumerate(stl10.labels):
        if t in orig_to_super:
            super_idx = orig_to_super[t]
            super_to_indices[super_idx].append(idx)

    g = torch.Generator().manual_seed(42)
    selected_indices = []

    # Step 1: ensure min_per_class per superclass
    for super_idx in range(10):
        indices = super_to_indices.get(super_idx, [])
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for superclass {super_idx}")
            continue
        n_samples = min(len(indices), min_per_class)
        subset = torch.randperm(len(indices), generator=g)[:n_samples]
        selected_indices.extend([indices[i] for i in subset])
        print(f"Superclass {super_idx}: selected {n_samples}/{len(indices)}")

    # Step 2: fill remaining budget uniformly
    remaining_budget = total_subset_size - len(selected_indices)
    if remaining_budget > 0:
        all_pool = [i for lst in super_to_indices.values() for i in lst]
        extra = torch.randperm(len(all_pool), generator=g)[:remaining_budget]
        selected_indices.extend([all_pool[i] for i in extra])

    # Step 3: print original STL-10 distribution
    val_targets = [stl10.labels[i] for i in selected_indices]
    print_class_distribution(
        val_targets, dataset_labels, title="Original STL-10 class distribution"
    )

    # Step 4: wrap in remapped dataset
    val_subset = SubsetRemapped(stl10, selected_indices, orig_to_super)

    # Step 5: print CIFAR-10-level distribution
    remapped_targets = [orig_to_super[t] for t in val_targets if t in orig_to_super]
    print_class_distribution(
        remapped_targets,
        list(cifar10_to_dataset.keys()),
        title="Remapped CIFAR-10 superclass distribution",
    )

    print(f"\n✅ Final subset size: {len(selected_indices)}")
    return val_subset


def get_realistic_valset_dermamnist(min_per_class, total_subset_size):
    CURRENT_DIR = Path("/home/FL-grad-aggregation/")
    dataset_root = CURRENT_DIR / "data_realistic/isic2019_images"

    # Reverse map for printing
    derma_labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Exclude overlapping IDs if you have a function for it
    exclude_ids = None  # or get_dermamnist_isic_ids()

    isic = ISIC2019Dataset(
        root=dataset_root, transform=transform, exclude_ids=exclude_ids
    )

    # Group indices by class
    super_to_indices = defaultdict(list)
    for idx, label in enumerate(isic.labels):
        super_to_indices[label].append(idx)

    g = torch.Generator().manual_seed(42)
    selected_indices = []

    # Step 1: ensure min_per_class coverage
    for cls_idx in range(len(derma_labels)):
        indices = super_to_indices.get(cls_idx, [])
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for class {derma_labels[cls_idx]}")
            continue
        n_samples = min(len(indices), min_per_class)
        subset = torch.randperm(len(indices), generator=g)[:n_samples]
        selected_indices.extend([indices[i] for i in subset])
        print(f"Class {derma_labels[cls_idx]}: selected {n_samples}/{len(indices)}")

    # Step 2: fill remaining budget
    remaining_budget = total_subset_size - len(selected_indices)
    if remaining_budget > 0:
        all_pool = [i for lst in super_to_indices.values() for i in lst]
        extra = torch.randperm(len(all_pool), generator=g)[:remaining_budget]
        selected_indices.extend([all_pool[i] for i in extra])

    # Step 3: print class distribution
    val_targets = [isic.labels[i] for i in selected_indices]
    print_class_distribution(
        val_targets, derma_labels, title="DermMNIST-equivalent ISIC 2019 distribution"
    )

    # Step 4: wrap in SubsetRemapped if you have it (or just return a Subset)
    val_subset = SubsetRemapped(isic, selected_indices, {i: i for i in range(7)})

    print(f"\n✅ Final subset size: {len(selected_indices)}")
    return val_subset


def get_realistic_valset(dataset_name):
    if dataset_name == "cifar10":
        val_subset = get_realistic_valset_cifar10_cifar100(
            min_per_class=50, total_subset_size=5000
        )
    elif dataset_name == "dermamnist":
        val_subset = get_realistic_valset_dermamnist(
            min_per_class=50, total_subset_size=1000
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    return val_subset
