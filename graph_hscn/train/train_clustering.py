from typing import Sequence

import torch
import wandb
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from tqdm import tqdm

from graph_hscn.config.config import (
    OPTIM_DICT,
    HSCNConfig,
    OptimConfig,
    TrainingConfig,
)
from graph_hscn.logger import CustomLogger
from graph_hscn.model.hscn import SCN


def train_clustering(
    logger: CustomLogger,
    dataset: PygGraphPropPredDataset | Sequence[Data],
    model: SCN,
    model_cfg: HSCNConfig,
    optim_cfg: OptimConfig,
    training_cfg: TrainingConfig,
) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = OPTIM_DICT[optim_cfg.optim_type](
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        params=model.parameters(),
    )
    for epoch in range(model_cfg.cluster_epochs):
        logger.info(f"Fitting clustering, epoch {epoch}...")
        for data in tqdm(dataset):
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index,
                data.edge_weight,
                data.num_nodes,
                add_self_loops=True,
            )
            data = data.to(device)
            optimizer.zero_grad()
            _, mc_loss, o_loss, adj = model(
                data.x.float(), data.edge_index, data.edge_weight
            )
            loss = mc_loss + o_loss
            loss.backward()
            optimizer.step()

        if training_cfg.use_wandb:
            wandb.log({"cluster_loss": loss})

    cluster_all_lst = []
    logger.info("Generating cluster assignments...")
    for data in tqdm(dataset):
        data.edge_index, data.edge_weight = gcn_norm(
            data.edge_index,
            data.edge_weight,
            data.num_nodes,
            add_self_loops=True,
        )
        data = data.to(device)
        clust, _, _, adj = model(
            data.x.float(), data.edge_index, data.edge_weight
        )
        clusters = clust.max(1)[1].cpu().numpy()
        cluster_all_lst.append(clusters)
    return cluster_all_lst
