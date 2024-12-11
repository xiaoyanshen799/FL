"""MNIST dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fedprox.dataset_preparation import _partition_data


import os
import pickle
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> DataLoader:
    """Create the test DataLoader and save train/val datasets for each client.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    DataLoader
        The DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    datasets, testset = _partition_data(
        num_clients,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        seed=seed,
    )

    # Create a directory to save the client datasets
    os.makedirs("client_datasets", exist_ok=True)

    # Split each partition into train/val and save them to files
    for client_idx, dataset in enumerate(datasets):
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )

        # Save train and validation datasets to files
        with open(f"client_datasets/trainloaders_{client_idx + 1}.pkl", "wb") as train_file:
            pickle.dump(ds_train, train_file)
        with open(f"client_datasets/valloaders_{client_idx + 1}.pkl", "wb") as val_file:
            pickle.dump(ds_val, val_file)

    # Return only the testset DataLoader
    return DataLoader(testset, batch_size=batch_size)


def load_client_dataloader(client_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load the train and validation DataLoader for a specific client.

    Parameters
    ----------
    client_id : int
        The ID of the client whose data should be loaded
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The DataLoader for training and the DataLoader for validation.
    """
    # Load train and validation datasets from files
    with open(f"client_datasets/trainloaders_{client_id}.pkl", "rb") as train_file:
        ds_train = pickle.load(train_file)
    with open(f"client_datasets/valloaders_{client_id}.pkl", "rb") as val_file:
        ds_val = pickle.load(val_file)

    # Create DataLoaders from the datasets
    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(ds_val, batch_size=batch_size)

    return trainloader, valloader
