"""Defining custom loaders."""
import logging
import os
import time
from functools import partial

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg
from torch_geometric.graphgym.register import register_loader
from torch_geometric.utils import degree

# isort: off
from graph_hscn.loader.dataset.peptides_functional import (
    PeptidesFunctionalDataset,
)
from graph_hscn.loader.dataset.peptides_structural import (
    PeptidesStructuralDataset,
)
from graph_hscn.loader.split_generator import (
    prepare_splits,
    set_dataset_splits,
)
from graph_hscn.transform.posenc_stats import compute_posenc_stats
from graph_hscn.transform.transforms import (
    pre_transform_in_memory,
    resample_citation_network,
)

# isort: on


def log_loaded_dataset(dataset: Data, _format: str, name: str) -> None:
    """Log basic dataset statistics.

    Parameters
    ----------
    dataset : Data
        Dataset to use.
    _format : str
        Specified dataset format.
    name : str
        Name of the dataset.

    Returns
    -------
    None
    """
    logging.info(f"[*] Loaded dataset '{name}' from '{_format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  Undirected: {dataset[0].is_undirected()}")
    logging.info(f"  Num. graphs: {len(dataset)}")

    total_num_nodes = 0

    if hasattr(dataset.data, "num_nodes"):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, "x"):
        total_num_nodes = dataset.data.x.size(0)

    logging.info(
        f"  Avg num. nodes / graph: " f"{total_num_nodes // len(dataset)}"
    )
    logging.info(f"  Num. node features: {dataset.num_node_features}")
    logging.info(f"  Num. edge features: {dataset.num_edge_features}")

    if hasattr(dataset, "num_tasks"):
        logging.info(f"  Num. tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, "y") and dataset.data.y is not None:
        if dataset.data.y.numel() == dataset.data.y.size(
            0
        ) and torch.is_floating_point(dataset.data.y):
            logging.info("  Num. classes: (appears to be a regression task)")
        else:
            logging.info(f"  Num. classes: {dataset.num_classes}")


@register_loader("custom_loader")
def load_dataset(_format: str, name: str, data_dir: str) -> Data:
    """
    Load specified dataset.

    Load and resample the citation networks dataset or load and preformat the
    Peptides datasets.

    Parameters
    ----------
    _format : str
        Specified dataset format.
    name : str
        Specified dataset name.
    data_dir : str
        Directory in which to save dataset.

    Returns
    -------
    Data
        Loaded and preformatted dataset.
    """
    if _format.startswith("PyG-"):
        pyg_dataset_id = _format.split("-", 1)[1]
        data_dir = os.path.join(data_dir, pyg_dataset_id)
        if pyg_dataset_id == "Planetoid":
            dataset = Planetoid(
                data_dir,
                name,
                pre_transform=T.NormalizeFeatures(),
            )
            pre_transform_in_memory(dataset, resample_citation_network)
        else:
            raise ValueError(
                f"Unexpected PyG dataset identifier: {_format}."
                f"Only Planetoid dataset is supported."
            )
    elif _format == "PyG":
        dataset = load_pyg(name, data_dir)
    elif _format == "OGB":
        if name.startswith("peptides-"):
            dataset = preformat_peptides(data_dir, name)
        else:
            raise ValueError(
                f"Unsupported OGB-derived dataset: {name}. Only"
                f"peptides datasets are currently supported."
            )
    else:
        raise ValueError(f"Unknown data format: {_format}")
    print(_format)

    log_loaded_dataset(dataset, _format, name)
    pe_enabled_list = []

    for k, pe_cfg in cfg.items():
        if k.startswith("posenc_") and pe_cfg.enable:
            pe_name = k.split("_", 1)[1]
            pe_enabled_list.append(pe_name)

            if hasattr(pe_cfg, "kernel"):
                if pe_cfg.kernel.times_func:
                    pe_cfg.kernel.times = list(eval(pe_cfg.kernel.times_func))

                logging.info(
                    f"Parsed {pe_name} PE kernel times / steps: "
                    f"{pe_cfg.kernel.times}"
                )

    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(
            f"Precomputing Positional Encoding statistics: "
            f"{pe_enabled_list} for all graphs..."
        )

        is_undirected = all(d.is_directed() for d in dataset[:10])
        logging.info(f"f  ...Estimated to be undirected: {is_undirected}")

        pre_transform_in_memory(
            dataset,
            partial(
                compute_posenc_stats,
                pe_types=pe_enabled_list,
                is_undirected=is_undirected,
                cfg=cfg,
            ),
            show_progress=True,
        )
        elapsed = time.perf_counter() - start
        time_str = (
            time.strftime("%H:%M:%S", time.gmtime(elapsed))
            + f"{elapsed:.2f}"[-3:]
        )
        logging.info(f"Done! Took {time_str}")

    if hasattr(dataset, "split_idxs"):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, "split_idxs")

    prepare_splits(dataset)

    if (
        cfg.gt.layer_type.startswith("PNAConv")
        and len(cfg.gt.pna_degrees) == 0
    ):
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data["train_graph_index"]]
        )

    return dataset


def preformat_peptides(
    data_dir: str, name: str
) -> PeptidesStructuralDataset | PeptidesFunctionalDataset:
    """Load a dataset and set its splits.

    Parameters
    ----------
    data_dir : str
        Directory in which to save the dataset.
    name : str
        Dataset name.

    Returns
    -------
    PeptidesStructuralDataset | PeptidesFunctionalDataset
        PReformatted peptides dataset.
    """
    dataset_type = name.split("-", 1)[1]

    match dataset_type:
        case "functional":
            dataset = PeptidesFunctionalDataset(data_dir)
        case "structural":
            dataset = PeptidesStructuralDataset(data_dir)

    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ["train", "val", "test"]]

    return dataset


def compute_indegree_histogram(dataset: Data) -> list:
    """Compute the in-degree histogram tensor for PNAConv layer.

    Parameters
    ----------
    dataset : Data
        Dataset to use.

    Returns
    -------
    list
        In-degree histogram.
    """
    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0

    for data in dataset:
        d = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())

    return deg.numpy().tolist()[: max_degree + 1]
