"""Custom transform functions."""
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm


def pre_transform_in_memory(
    dataset: Data, transform_func: Callable, show_progress: bool = False
) -> Data | None:
    """Apply a transform function to InMemoryDataset in pre_transform stage.

    Parameters
    ----------
    dataset : Data
        Dataset to pre-transform.
    transform_func : Callable
        Transform function to apply.
    show_progress : bool
        Whether to show progress in tqdm.

    Returns
    -------
    Data | None
        Dataset if no transform_func specified, None otherwise.
    """
    if transform_func is None:
        return dataset

    data_list = [
        transform_func(dataset.get(i))
        for i in tqdm(
            range(len(dataset)),
            disable=not show_progress,
            mininterval=10,
            miniters=len(dataset) // 20,
        )
    ]
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def resample_citation_network(
    data: Data,
    num_val: int = 500,
    num_test: int = 1000,
    train_examples_per_class: int = 20,
) -> Data:
    """Adapted from https://github.com/rampasek/HGNet/ and modified.

    Find a set such that all selected nodes are at least k + 1
    hops apart. Intersect it with all masks of the input data. Draw a random
    node, and make sure none of its k-hop neighbors are selected for any
    split. Therefore, any k-hop neighborhood of a labeled node will be
    sanitized of labels. It is recommended to test for k = 1 and k = 2.

    Parameters
    ----------
    data : Data
        PyG Data object from Planetoid datasets.
    num_val : int
        Number of validation examples in split.
    num_test : int
        Number of test examples in split.
    train_examples_per_class : int
        Number of examples per class in the training split.

    Returns
    -------
    Data
        Modified PyG Data object.
    """
    _N = data.num_nodes
    adj = [set() for _ in range(_N)]  # Initialize empty adjacency matrix

    # adj = list of sets, adj[node] = node's neighbors
    for u, v in data.edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)

    buffer = cfg.dataset.citation_buffer

    # Get k-hop neighborhood
    for _ in range(buffer - 1):
        k_hop_neighbors = [set() for _ in range(_N)]  # Initialize neighborhood

        for u in range(_N):
            for v in adj[u]:  # Looping through unique neighbors
                if adj[v] != u:
                    k_hop_neighbors[u].update(adj[v])

        adj = k_hop_neighbors

    rng = np.random.default_rng(seed=cfg.seed)
    random_node = [rng.choice(_N)]
    k_hop_nb = set.union(
        *[adj[v] for v in random_node]
    )  # Get neighbors that these nodes share
    indep_nodes = list(random_node)
    non_neighbors = set(range(_N)).difference(
        k_hop_nb.union(random_node)
    )  # Independent nodes

    while non_neighbors:
        node = rng.choice(list(non_neighbors))
        indep_nodes.append(node)
        non_neighbors.difference_update(list(adj[node]) + [node])

    print(f"Found {buffer}-hop independent set of size {len(indep_nodes)}")
    indep_nodes = np.asarray(indep_nodes)

    train_key = (
        data.train_mask.numpy().astype(int).sum()
    )  # For Cora: 20 examples per class
    val_key = data.val_mask.numpy().astype(int).sum()  # For Cora: 500
    test_key = data.test_mask.numpy().astype(int).sum()  # For Cora: 1000

    indep_mask = data.train_mask.new_empty(
        data.train_mask.size(0), dtype=torch.bool
    ).fill_(True)
    indep_mask[indep_nodes] = False

    y = data.y.clone().detach()
    y[indep_mask] = -1  # Independent nodes to have no labels
    num_classes = y.max().item() + 1

    data.train_mask.fill_(False)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[
            rng.permutation(idx.size(0))[:train_examples_per_class]
        ]  # Cora: shuffle 20 examples per class for training set
        data.train_mask[idx] = True

    used = data.train_mask.clone().detach()
    used[indep_mask] = True
    rm = (~used).nonzero(as_tuple=False).view(-1)
    num_remaining = rm.size(0)
    rm = rm[rng.permutation(num_remaining)]  # Shuffle
    num_needed = num_val + num_test
    print(f"Remaining: {num_remaining}, needed: {num_needed}")

    if num_needed > num_remaining:
        num_val = num_val // 3
        num_test = num_test * 2 // 3
        print(f"num_val = {num_val}, num_test = {num_test}")

    # Reassign labels in validation and test splits
    data.val_mask.fill_(False)
    data.val_mask[rm[:num_val]] = True
    num_prev = num_val
    data.test_mask.fill_(False)
    data.test_mask[rm[num_prev : num_prev + num_test]] = True

    new_train_key = data.train_mask.numpy().astype(int).sum()
    new_val_key = data.val_mask.numpy().astype(int).sum()
    new_test_key = data.test_mask.numpy().astype(int).sum()

    print(f"\n> Train nodes:  original = {train_key}, new = {new_train_key}")
    print(f"> Val nodes:    original = {val_key}, new = {new_val_key}")
    print(f"> Test nodes:   original = {test_key}, new = {new_test_key}")

    return data
