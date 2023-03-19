import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

import wandb
from graph_hscn.config.config import (
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
from graph_hscn.model.hscn import SCN, build_hscn
from graph_hscn.model.mpnn import build_mpnn
from graph_hscn.train.train import compute_posenc, train
from graph_hscn.train.train_clustering import train_clustering
from graph_hscn.train.utils import get_each_data_from_batch


def setup() -> list[Any | None]:
    parser = argparse.ArgumentParser(description="GraphHSCN CLI")
    parser.add_argument("--cfg", type=str, help="Config file to use.")
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Set up data config.
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    data_cfg = DataConfig.parse_obj(cfg["data"])

    # Set up PE config.
    if cfg["pe"]["use"]:
        pe_cfg = PEConfig.parse_obj(cfg["pe"])
    else:
        pe_cfg = None

    # Set up optimizer config.
    optim_cfg = OptimConfig.parse_obj(cfg["optim"])

    # Set up model config.
    if "mp" in cfg.keys():
        model_cfg = MPNNConfig.parse_obj(cfg["mp"])
        proj_name = (
            f"{data_cfg.dataset_name}_{model_cfg.conv_type}"
            f"_{model_cfg.num_layers}"
        )
    elif "hscn" in cfg.keys():
        model_cfg = HSCNConfig.parse_obj(cfg["hscn"])
        proj_name = f"{data_cfg.dataset_name}_HSCN_{model_cfg.num_clusters}"
    else:
        raise ValueError("Need either `mp` or `hscn` settings in config file.")

    # Set up training config
    training_cfg = TrainingConfig.parse_obj(cfg["training"])

    # Set up logger
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = CustomLogger(
        Path(
            LOGS_DIR, f"{data_cfg.dataset_name}_{training_cfg.model_type}.log"
        ),
        metric_name=training_cfg.metric,
    )
    return [
        data_cfg,
        pe_cfg,
        optim_cfg,
        model_cfg,
        training_cfg,
        logger,
        proj_name,
    ]


def run_train(
    data_cfg, pe_cfg, optim_cfg, model_cfg, training_cfg, logger, proj_name
) -> None:
    loaders, dataset = load_dataset(logger, data_cfg, pe_cfg)
    split_idx = dataset.get_idx_split()
    num_features = dataset[0].num_features
    num_classes = dataset[0].y.shape[1]
    if data_cfg.pe:
        loaders, dataset = compute_posenc(
            loaders, data_cfg, num_features, pe_cfg, logger
        )
        dataset = get_each_data_from_batch(dataset)
    if training_cfg.use_wandb:
        wandb.init(project=proj_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training_cfg.model_type == "hscn":
        cluster_model = SCN(
            model_cfg.mp_units,
            model_cfg.activation,
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
    wandb.finish()


def main() -> None:
    (
        data_cfg,
        pe_cfg,
        optim_cfg,
        model_cfg,
        training_cfg,
        logger,
        proj_name,
    ) = setup()
    run_train(
        data_cfg, pe_cfg, optim_cfg, model_cfg, training_cfg, logger, proj_name
    )


if __name__ == "__main__":
    main()
