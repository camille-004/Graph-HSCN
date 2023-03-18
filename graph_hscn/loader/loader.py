import time
from functools import partial
from typing import Any, Sequence

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hscn.config.config import DataConfig, PEConfig
from graph_hscn.loader.dataset.peptides_functional import (
    PeptidesFunctionalDataset,
)
from graph_hscn.loader.dataset.peptides_structural import (
    PeptidesStructuralDataset,
)
from graph_hscn.logger import CustomLogger
from graph_hscn.transform.posenc import compute_posenc_stats
from graph_hscn.transform.pre_transform import pre_transform_in_memory


def set_dataset_attr(
    dataset: PygGraphPropPredDataset,
    name: str,
    value: Any,
    size: int,
) -> None:
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def set_splits(
    dataset: PygGraphPropPredDataset,
    data_cfg: DataConfig,
) -> None:
    assert (
        data_cfg.task_level == "graph"
    ), "Only graph-level OGB datasets supported."
    splits = dataset.get_idx_split()
    split_names = ["train_graph_index", "val_graph_index", "test_graph_index"]
    for i, key in enumerate(splits.keys()):
        idx = splits[key]
        set_dataset_attr(dataset, split_names[i], idx, len(idx))


def get_loader(
    dataset: PygGraphPropPredDataset | Sequence[Data],
    data_cfg: DataConfig,
    shuffle: bool,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        data_cfg.batch_size,
        shuffle,
        num_workers=data_cfg.num_workers,
        persistent_workers=data_cfg.num_workers > 0,
    )
    return loader


def load_dataset(
    logger: CustomLogger, data_cfg: DataConfig, pe_cfg: PEConfig
) -> tuple[list[DataLoader], PygGraphPropPredDataset]:
    if data_cfg.dataset_name == "peptides_func":
        dataset = PeptidesFunctionalDataset()
    elif data_cfg.dataset_name == "peptides_struct":
        dataset = PeptidesStructuralDataset()
    else:
        raise ValueError(
            f"Unknown or unsupported dataset: {data_cfg.dataset_name}"
        )

    if data_cfg.pe:
        start = time.perf_counter()
        logger.info("Precomputing SignNet statistics for all graphs...")
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logger.info(f"...Estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(
            dataset,
            partial(
                compute_posenc_stats, is_undirected=is_undirected, cfg=pe_cfg
            ),
            show_progress=True,
        )
        elapsed = time.perf_counter() - start
        time_str = (
            time.strftime("%H:%M:%S", time.gmtime(elapsed))
            + f"{elapsed:.2f}"[-3:]
        )
        logger.info(f"Done! Took {time_str}")

    if data_cfg.task_level == "graph":
        # Set dataset attributes
        set_splits(dataset, data_cfg)
        idx = dataset.data["train_graph_index"]
        loaders = [get_loader(dataset[idx], data_cfg, shuffle=True)]
        delattr(dataset.data, "train_graph_index")

        for split_name in ["val_graph_index", "test_graph_index"]:
            idx = dataset.data[split_name]
            loaders.append(get_loader(dataset[idx], data_cfg, shuffle=False))
            delattr(dataset.data, split_name)

        return loaders, dataset
    else:
        raise NotImplementedError
