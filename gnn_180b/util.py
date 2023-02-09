"""Utility and helper functions."""
import logging
from functools import reduce
from typing import Any, Literal

import numpy as np
import torch
from scipy.stats import stats
from torch_geometric.graphgym.config import cfg
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
    """Compute Spearman Rho averaged across tasks."""
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


def _get_rank(values):
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


def flatten_dict(metrics: list[dict]) -> dict:
    """Flatten a list of train/val/test metrics into one dict for WandB.

    Parameters
    ----------
    metrics : list[dict]
        List of dictionaries with metrics.

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
