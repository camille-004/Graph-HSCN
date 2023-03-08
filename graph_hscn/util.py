"""Utility and helper functions."""
import logging
from functools import reduce
from typing import Any, Literal

import numpy as np
import torch
from scipy.stats import stats
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from yacs.config import CfgNode

EPS = 1e-5

Reduction = Literal["elementwise_mean", "sum", "none"]


def reformat(x: Any) -> float:
    """Reformatting for metrics.

    Parameters
    ----------
    x : Any
        Value to round.

    Returns
    -------
    float
        Rounded value.
    """
    return round(float(x), cfg.round)


def eval_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Spearman Rho averaged across tasks.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Prediction labels.

    Returns
    -------
    dict[str, float]
        A dict with the name of the metric and the value.
    """
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # Ignore NaNs
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(
                stats.spearmanr(y_true[is_labeled, i], y_pred[is_labeled, i])[
                    0
                ]
            )

    return {"spearmanr": sum(res_list) / len(res_list)}


def _get_rank(values: torch.Tensor) -> torch.Tensor:
    """Compute the rank of a tensor.

    Parameters
    ----------
    values : torch.Tensor
        Tensor of which to compute the rank.

    Returns
    -------
    torch.Tensor
        A tensor with the same shape and dtype as the input tensor, where each
        element is the rank of the corresponding element in the input tensor.
    """
    arange = torch.arange(
        values.shape[0], dtype=values.dtype, device=values.device
    )
    val_sorter = torch.argsort(values, dimm=0)
    val_rank = torch.empty_like(values)

    if values.ndim == 1:
        val_rank[val_sorter] = arange
    elif values.ndim == 2:
        for ii in range(val_rank.shape[1]):
            val_rank[val_sorter[:, ii], ii] = arange
    else:
        raise ValueError(
            f"Only supports tensors of dimensions 1 and 2. Provided dim="
            f"`{values.ndim}`."
        )

    return val_rank


def pearsonr(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: Reduction = "elementwise_mean",
) -> torch.Tensor:
    """Compute the Pearson R correlation.

    Parameters
    ----------
    y_pred : torch.Tensor
        Estimated labels.
    y_true : torch.Tensor
        Ground truth labels.
    reduction : Reduction
        A method to reduce the metric score over labels.
        - ``'elementwise_mean'``: Takes the mean (default).
        - ``'sum'``: Takes the sum.
        - ``'none'``: No reduction.

    Returns
    -------
    torch.Tensor
        Tensor with the Pearson's R.
    """
    pred, true = y_pred.to(torch.float32), y_true.to(torch.float32)

    shifted_x = pred - torch.mean(pred, dim=0)
    shifted_y = true - torch.mean(true, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x**2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y**2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (
        sigma_x * sigma_y + EPS
    )
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def spearmanr(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: Reduction = "elementwise_mean",
) -> torch.Tensor:
    """Calculate the Spearman's rho correlation.

    Parameters
    ----------
    y_pred : torch.Tensor
        Estimated labels.
    y_true : torch.Tensor
        Ground truth labels.
    reduction : Reduction
        A method to reduce the metric score over labels.
        - ``'elementwise_mean'``: Takes the mean (default).
        - ``'sum'``: Takes the sum.
        - ``'none'``: No reduction.

    Returns
    -------
    torch.Tensor
        Tensor with the Spearman's rho.
    """
    spearman = pearsonr(
        _get_rank(y_pred), _get_rank(y_true), reduction=reduction
    )
    return spearman


def make_wandb_name(cfg: CfgNode) -> str:
    """Make a name for the WandB run.

    Parameters
    ----------
    cfg : Yacs config used by GraphGym.

    Returns
    -------
    str
        Name to be used by WandB.
    """
    dataset_name = cfg.dataset.format

    if dataset_name.startswith("OGB"):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith("PyG-"):
        dataset_name = dataset_name[4:]
    if cfg.dataset.name != "none":
        dataset_name += "-" if dataset_name != "" else ""
        dataset_name += cfg.dataset.name

    model_name = cfg.model.type

    if cfg.model.type in ["gnn", "custom_gnn"]:
        model_name += f"{cfg.gnn.layer_type}"

    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name


def flatten_dict(metrics: list[dict] | list[list]) -> dict:
    """Flatten a list of train/val/test metrics into one dict for WandB.

    Parameters
    ----------
    metrics : list[dict] | list[list]
        List of dictionaries or lists with metrics.

    Returns
    -------
    dict
        A flat dictionary, names prefixed with "train - ", "val - ", "test - "
    """
    prefixes = ["train", "val", "test"]
    result = {}

    for i in range(len(metrics)):
        _stats = metrics[i][-1]
        result.update({f"{prefixes[i]} - {k}": v for k, v in _stats.items()})

    return result


def cfg_to_dict(cfg_node: CfgNode, key_list: list = []) -> CfgNode | dict:
    """Convert a config node to a dictionary.

    Yacs doesn't have a default function to convert a CfgNode to a plain Python
    dict. Adapted from https://github.com/rbgirshick/yacs/issues/19.

    Parameters
    ----------
    cfg_node : CfgNode
        Main config to convert.
    key_list : list
        Keys to have in dict, and whose type to check.

    Returns
    -------
    CfgNode | dict
        Converted config or original CfgNode if key type is not valid.
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(
                f"Key {'.'.join(key_list)} with value {type(cfg_node)} is not"
                f"a valid type; valid types: {_VALID_TYPES}"
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])

        return cfg_dict


def set_new_cfg_allowed(config: CfgNode, is_new_allowed: bool) -> None:
    """Allow merging new keys from other configs.

    Set YACS config (and recursively its subconfigs) to allow merging new keys
    from other configs.

    Parameters
    ----------
    config : CfgNode
        Config to merge.
    is_new_allowed : bool
        Whether a new config is allowed.

    Returns
    -------
    None
    """
    config.__dict__[CfgNode.NEW_ALLOWED] = is_new_allowed
    # Recursively set new_allowed state
    for v in config.__dict__.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)
    for v in config.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)


def negate_edge_index(
    edge_index: torch.Tensor, batch: Data | None = None
) -> torch.Tensor:
    """Obtain a tensor representing the non-existent nodes in the graph.

    Useful for multi-head attention, modeling the absense of connections
    between nodes. Learn to distinguish between existing and missing edges in
    the graph, better capturing the structure and relationships between nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge_index.
    batch : Data
        Input batch.

    Returns
    -------
    torch.Tensor
        Concatenated negative edge indices.
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="sum")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix.
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(
            _idx1.numel(), dtype=torch.short, device=edge_index.device
        )
        adj = scatter(zero, idx, dim=0, dim_size=flattened_size, reduce="mul")

        # Convert to edge index format.
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative
