from typing import Sequence

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm

from graph_hscn.config.config import DataConfig, HSCNConfig
from graph_hscn.loader.loader import get_loader
from graph_hscn.logger import CustomLogger


def generate_hetero_data(
    cluster_lst: list,
    dataset: PygGraphPropPredDataset | Sequence[Data],
    split_idx: dict[str, torch.Tensor],
    data_cfg: DataConfig,
    model_cfg: HSCNConfig,
    logger: CustomLogger,
) -> list[HeteroData]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_data_lst = []
    if data_cfg.task_level == "graph":
        if not isinstance(dataset, list):
            train_dataset = dataset[split_idx["train"]]
            val_dataset = dataset[split_idx["val"]]
            test_dataset = dataset[split_idx["test"]]
        else:
            train_dataset = [dataset[i] for i in split_idx["train"]]
            val_dataset = [dataset[i] for i in split_idx["val"]]
            test_dataset = [dataset[i] for i in split_idx["test"]]

    for split_name, split_data in zip(
        ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
    ):
        logger.info(
            f"Generating heterogeneous dataset with virtual nodes for "
            f"{split_name} split..."
        )
        clusters_split = [cluster_lst[i] for i in split_idx[split_name]]
        for idx in tqdm(range(len(split_data))):
            data = split_data[idx].to(device)
            clust_node = [[] for idx in range(model_cfg.num_clusters)]
            clusters = clusters_split[idx]
            unique_clusters = np.unique(clusters)
            clust_map = {
                unique_clusters[idx]: idx
                for idx in range(len(unique_clusters))
            }
            clusters = [clust_map[val] for val in clusters]
            for ix in range(data.num_nodes):
                clust_num = clusters[ix] - 1
                clust_node[clust_num].append(data.x[ix].tolist())
            clust_node = [lst for lst in clust_node if len(lst) != 0]
            clust_mean = [
                np.mean(clust_lst, axis=0) for clust_lst in clust_node
            ]
            clust_mean = np.array(clust_mean)
            num_clust = len(clust_mean)

            # Create HeteroData
            h_data = HeteroData()
            h_data["local"].x = data.x.float()
            h_data["local"].y = data.y
            h_data["virtual"].x = torch.FloatTensor(clust_mean)
            h_data["local", "to", "local"].edge_index = data.edge_index
            col = np.concatenate(
                [[idx] * (num_clust - idx) for idx in range(num_clust)]
            )
            row = np.concatenate(
                [
                    [idx for idx in range(num_clust - ix)]
                    for ix in range(num_clust)
                ]
            )
            h_data["virtual", "to", "virtual"].edge_index = torch.LongTensor(
                [list(col), list(row)]
            )
            edge_lst = []
            for ix in range(len(clusters)):
                clust_num = clusters[ix]
                edge_lst.append([ix, clust_num])
            h_data["local", "to", "virtual"].edge_index = torch.LongTensor(
                edge_lst
            ).T
            h_data_lst.append(h_data)
    return h_data_lst


def hetero_loaders(
    data_cfg: DataConfig,
    hetero_dataset: list[HeteroData],
    split_idx: dict[str, torch.Tensor],
) -> list:
    if data_cfg.task_level == "graph":
        train_dataset = [hetero_dataset[i] for i in split_idx["train"]]
        val_dataset = [hetero_dataset[i] for i in split_idx["val"]]
        test_dataset = [hetero_dataset[i] for i in split_idx["test"]]
        return [
            get_loader(train_dataset, data_cfg, shuffle=True),
            get_loader(val_dataset, data_cfg, shuffle=False),
            get_loader(test_dataset, data_cfg, shuffle=False),
        ]
    else:
        raise NotImplementedError
