"""Customized training pipeline."""
from typing import Literal

import torch
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch_geometric.data import Data
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.transforms import gcn_norm

from ca_net.logger import CustomLogger

Split = Literal["train", "val", "test"]


def train_epoch(
    logger: CustomLogger,
    loader: Data,
    model: GraphGymModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    batch_accumulation: int,
) -> None:
    """Define a training epoch.

    Parameters
    ----------
    logger : CustomLogger
        Logger to use.
    loader : Data
        Data to be loaded and trained on.
    model : GraphGymModule
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer used by model.
    scheduler : torch.optim.lr_scheduler.StepLR
        LR scheduler used by model.
    batch_accumulation : int
        Threshold for clipping the gradient norm of the parameters iterable.

    Returns
    -------
    None
    """
    data = loader
    # Normalized adjacency matrix
    data.edge_index, data.edge_weight = gcn_norm(
        data.edge_index,
        data.edge_weight,
        data.num_nodes,
        add_self_loops=False,
        dtype=data.x.dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train():
        model.train()
        optimizer.zero_grad()
        _, mc_loss, o_loss = model(data.x, data.edge_index, data.edge_weight)
        loss = mc_loss + o_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
        return NMI(clust.max(1)[1].cpu(), data.y.cpu())

    patience = 50
    best_nmi = 0
    for epoch in range(1, 10000):
        train_loss = train()
        nmi = test()
        print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}")
        if nmi > best_nmi:
            best_nmi = nmi
            patience = 50
        else:
            patience -= 1
        if patience == 0:
            break
