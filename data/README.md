# Download Datasets
Most of the datasets supported by this benchmark are integrated into `torchvision.datasets`, expect *Tiny-ImageNet-200*, *Covid-19*, *Organ-S/A/CMNIST*, *DermaMNIST*. 

For those datasets, download scripts are in folder [`data/download`](download).

e.g.

```shell
cd data/download
sh tiny_imagenet.sh
```

# Generic Arguments 🔧
📢 All arguments have their default value.
| Arguments for general datasets | Description                                                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset, -d`                | The name of dataset.                                                                                                                                     |
| `--iid`                        | Non-zero value for randomly partitioning data and disabling all other Non-IID partition methods.                                                         |
| `--client_num, -cn`            | The amount of clients.                                                                                                                                   |
| `--split, -sp`                 | Chooses from `[sample, user]`.  `user`: partition clients into train-test groups; `sample`: partition each client's data samples into train-test groups. |
| `--val_ratio, -vr`             | Propotion of valset data/clients.                                                                                                                        |
| `--test_ratio, -tr`            | Propotion of testset data/clients.                                                                                                                       |
| `--plot_distribution, -pd`     | Non-zero value for saving data distribution image.                                                                                                       |

⭐ For *CIFAR-100* specifically, this benchmark supports partitioning it into the superclass category (*CIFAR-100*'s 100 classes can also be classified into 20 superclasses) by setting `--super_class` to non-zero.

# Partition Schemes 🌌

## IID


Partition data evenly. Client data distributions are similar to each other. Note that this setting has the **highest priority**, means that activating this scheme will disable all others.

✨ IID partition can only process partial datasets and combines other Non-IID schemes.

- `--iid`: Need to set in `[0, 1]`. 

```shell
python generate_data.py -d cifar10 --iid 1 -cn 20
```

<img src="../.github/images/distributions/iid.png" alt="Image" width="350"/>

```shell
# 50% data are partitioned IID, and the rest 50% are partitioned according to dirichlet parititon scheme: Dir(0.1) 
python generate_data.py -d cifar10 --iid 0.5 --alpha 0.1 -cn 20
```

<img src="../.github/images/distributions/iid0.5-a0.1.png" alt="Image" width="350"/>


## Dirichlet


Refers to [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (*FedAvgM*)](https://arxiv.org/abs/1909.06335). Dataset would be splitted according to $Dir(\alpha)$. Smaller $\alpha$ means stronger label heterogeneity.
  
- `--alpha, -a`: The parameter for controlling intensity of label heterogeneity.
- `--min_samples_per_client, -ms`: The parameter for defining the minimum number of samples each client would be distributed. *A small `--min_samples_per_client` along with small `--alpha` or big `--client_num` might considerablely prolong the partition.*

```shell
python generate_data.py -d cifar10 -a 0.1 -cn 20
``` 
<img src="../.github/images/distributions/dir0.1.png" alt="Image" width="350"/>




## Note

FL-bench ignores the test set of *Tiny-ImageNet-200* due to it is unlabeled.

