from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, root_validator, validator
from torch.optim import Adagrad, Adam, AdamW, Optimizer
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn.conv import MessagePassing

import graph_hscn.config.defaults as defaults

ACT_DICT: dict[str, Callable] = {
    "elu": F.elu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "identity": nn.Identity(),
}
CONV_DICT: dict[str, type[MessagePassing]] = {
    "gcn": GCNConv,
    "gat": GATConv,
    "gin": GINConv,
}
OPTIM_DICT: [str, type[Optimizer]] = {
    "adagrad": Adagrad,
    "adam": Adam,
    "adamW": AdamW,
}
DATASETS_NUM_FEATURES: [str, int] = {"peptides_func": 9, "peptides_struct": 9}


class DataConfig(BaseModel):
    """Model for data location."""

    dataset_name: str
    pe: bool = True
    batch_size: int = defaults.BATCH_SIZE
    num_workers: int = defaults.NUM_WORKERS
    task_level: str | None

    @validator("task_level", always=True)
    def set_task_level(cls, v, values):
        if "peptides" in values["dataset_name"]:
            return "graph"
        return "node"


class MPNNConfig(BaseModel):
    """Configuration for MPNN experiment."""

    conv_type: str
    activation: str
    hidden_channels: int = defaults.HIDDEN_CHANNELS
    num_layers: int = defaults.NUM_LAYERS
    dropout: float = defaults.DROPOUT
    use_batch_norm: bool = defaults.USE_BATCH_NORM
    use_layer_norm: bool = defaults.USE_LAYER_NORM

    @validator("dropout")
    def check_range(cls, v: float):
        """Check for entries not in [0, 1] range."""
        if v:
            if 0.0 <= v <= 1.0:
                return v
            else:
                raise ValueError(f"{v} must be between 0.0 and 1.0.")

    @validator("num_layers", "hidden_channels")
    def check_non_negative(cls, v: float):
        """Check for negative entries."""
        if v < 0:
            raise ValueError(f"{v} must be non-negative.")
        return v


class HSCNConfig(BaseModel):
    """Configuration for HSCN experiment."""

    activation: str
    lv_conv_type: str = "GAT"  # Must be GAT first or will error out.
    ll_conv_type: str = "GCN"
    vv_conv_type: str = "GCN"
    hidden_channels: int = defaults.HIDDEN_CHANNELS
    num_layers: int = defaults.NUM_LAYERS
    num_clusters: int = defaults.NUM_CLUSTERS
    cluster_epochs: int = defaults.CLUSTER_EPOCHS

    @validator("num_layers", "hidden_channels")
    def check_non_negative(cls, v: float):
        """Check for negative entries."""
        if v < 0:
            raise ValueError(f"{v} must be non-negative.")
        return v


class OptimConfig(BaseModel):
    """Optimizer configuration."""

    optim_type: str
    batch_accumulation: int = defaults.BATCH_ACCUMULATION
    clip_grad_norm: bool = defaults.CLIP_GRAD_NORM
    lr: float = defaults.LR
    weight_decay: float = defaults.WEIGHT_DECAY

    @validator("lr", "weight_decay")
    def check_range(cls, v: float):
        """Check for entries not in [0, 1] range."""
        if v:
            if 0.0 <= v <= 1.0:
                return v
            else:
                raise ValueError(f"{v} must be between 0.0 and 1.0.")


class PEConfig(BaseModel):
    """Positional encoding configuration."""

    dim_in: int
    dim_emb: int
    dim_pe: int
    model: str = defaults.PE_MODEL
    layers: int = defaults.PE_LAYERS
    post_layers: int = defaults.POST_LAYERS
    eigen_max_freqs: int = defaults.EIG_MAX_FREQS
    eigvec_norm: str = defaults.EIGVEC_NORM
    eigen_laplacian_norm: str = defaults.EIG_LAP_NORM
    phi_hidden_dim: int = defaults.PHI_HIDDEN_DIM
    phi_out_dim: int = defaults.PHI_OUT_DIM
    pass_as_var: bool = defaults.PASS_AS_VAR
    use_bn: bool = defaults.PE_USE_BN


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_type: str
    loss_fn: str
    metric: str
    epochs: int = defaults.EPOCHS
    eval_period: int = defaults.EVAL_PERIOD
    min_delta: float = defaults.MIN_DELTA
    patience: int = defaults.PATIENCE
    use_wandb: bool = defaults.USE_WANDB
    wandb_proj_name: str | None

    @root_validator
    def check_use_wandb(cls, values):
        if not values["use_wandb"]:
            raise ValueError(
                "WandB project name provided but use_wandb set to False."
            )
        return values
