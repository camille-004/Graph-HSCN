import argparse
import os
from pathlib import Path

import torch

import graph_hscn.config.defaults as defaults
import wandb
from graph_hscn.config.config import (
    DATASETS_NUM_FEATURES,
    DataConfig,
    HSCNConfig,
    MPNNConfig,
    OptimConfig,
    PEConfig,
    TrainingConfig,
)
from graph_hscn.constants import DATASETS_DIR, LOGS_DIR
from graph_hscn.loader.hetero_data import generate_hetero_data, hetero_loaders
from graph_hscn.loader.loader import load_dataset
from graph_hscn.logger import CustomLogger
from graph_hscn.model.hscn import HSCN, SCN, build_hscn
from graph_hscn.model.mpnn import build_mpnn
from graph_hscn.train.train import compute_posenc, train
from graph_hscn.train.train_clustering import train_clustering
from graph_hscn.train.utils import get_each_data_from_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphHSCN CLI")

    # Dataset args
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name.")

    # Model args
    parser.add_argument("-m", "--model_type", type=str, help="Model to train.")
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        help="Activation function to use.",
        default="relu",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of hidden layers.",
        default=defaults.NUM_LAYERS,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability",
        default=defaults.DROPOUT,
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        help="Hidden channels.",
        default=defaults.HIDDEN_CHANNELS,
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use WandB."
    )

    # HSCN args
    parser.add_argument(
        "-n",
        "--num_clusters",
        type=int,
        help="Number of clusters for coarsening.",
        default=5,
    )
    parser.add_argument(
        "--cluster_epochs",
        type=int,
        help="Number of epochs for clustering.",
        default=10,
    )
    parser.add_argument(
        "--mp_units",
        type=list,
        help="Input and output units for MP layers (list).",
        default=[16],
    )

    # MPNN args
    parser.add_argument(
        "--mp_conv",
        type=str,
        help="MPNN convolution layer type.",
        default="gcn",
    )

    # PE args
    parser.add_argument(
        "--use_pe",
        action="store_true",
        help="Whether to use SignNet positional encoding.",
    )
    parser.add_argument(
        "--pe_dim_in", type=int, help="In-dimension for SignNet."
    )
    parser.add_argument(
        "--pe_dim", type=int, help="Dimension for SignNet encoding."
    )
    parser.add_argument(
        "--pe_use_bn",
        action="store_true",
        help="Whether to use batch normalization for PE.",
    )

    # Optimizer args
    parser.add_argument(
        "-o",
        "--optim_type",
        type=str,
        help="Type of optimizer to use.",
        default="adamW",
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate.", default=defaults.LR
    )

    args = parser.parse_args()

    # Set up data config
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        dataset_name=args.dataset,
        root_dir=DATASETS_DIR,
        pe=args.use_pe,
    )

    # Set metrics
    match data_cfg.dataset_name:
        case "peptides_func":
            metric = "ap"
            loss = "cross_entropy"
        case "peptides_struct":
            metric = "mae"
            loss = "l1"
        case other:
            raise NotImplementedError

    # Set up PE config
    if args.use_pe:
        assert all(
            [
                args.pe_dim_in is not None,
                args.pe_dim is not None,
                args.pe_use_bn is not None,
            ]
        )
        pe_cfg = PEConfig(
            args.pe_dim_in,
            DATASETS_NUM_FEATURES[data_cfg.dataset_name],
            args.pe_dim,
            args.pe_use_bn,
        )
    else:
        pe_cfg = None

    # Set up optimizer config
    optim_cfg = OptimConfig(optim_type=args.optim_type, lr=args.lr)

    if args.model_type.lower() == "hscn":
        assert all(
            [
                args.num_clusters is not None,
                args.cluster_epochs is not None,
                args.mp_units is not None,
            ]
        )
        model_cfg = HSCNConfig(
            activation=args.activation,
            cluster_epochs=args.cluster_epochs,
            num_layers=args.num_layers,
            hidden_channels=args.hidden_channels,
        )
        proj_name = f"{data_cfg.dataset_name}_HSCN_{model_cfg.num_clusters}"
    elif args.model_type.lower() in ["gcn", "gat", "gin"]:
        model_cfg = MPNNConfig(
            conv_type=args.mp_conv,
            activation=args.activation,
            num_layers=args.num_layers,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
        )
        proj_name = f"{data_cfg.dataset_name}_{model_cfg.conv_type}_{model_cfg.num_layers}"
    else:
        raise NotImplementedError

    # Set up training config
    training_cfg = TrainingConfig(
        loss_fn=loss,
        metric=metric,
        use_wandb=args.use_wandb,
        wandb_proj_name=proj_name,
    )

    # Set up logger
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = CustomLogger(
        Path(LOGS_DIR, f"{data_cfg.dataset_name}_{args.model_type}.log"),
        metric_name=training_cfg.metric,
    )

    loaders, dataset = load_dataset(logger, data_cfg, pe_cfg)
    split_idx = dataset.get_idx_split()
    num_features = dataset[0].num_features
    num_classes = dataset[0].y.shape[1]

    if data_cfg.pe:
        loaders, dataset = compute_posenc(
            loaders, data_cfg, num_features, pe_cfg, logger
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_cfg.pe:
        dataset = get_each_data_from_batch(dataset)

    if args.model_type == "hscn":
        cluster_model = SCN(
            args.mp_units,
            args.activation,
            num_features,
            model_cfg.num_clusters,
        ).to(device)
        cluster_all_lst = train_clustering(
            logger, dataset, cluster_model, model_cfg, optim_cfg, training_cfg
        )
        hetero_data = generate_hetero_data(
            cluster_all_lst, dataset, split_idx, data_cfg, model_cfg, logger
        )
        loaders = hetero_loaders(data_cfg, hetero_data, split_idx)
        model = build_hscn(model_cfg, num_features, num_classes)
        model = model.to(device)
    else:
        model = build_mpnn(model_cfg, num_features, num_classes)

    train(logger, optim_cfg, training_cfg, loaders, model)


if __name__ == "__main__":
    main()
