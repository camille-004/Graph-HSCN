# from pathlib import Path
#
# import torch
# import wandb
#
# from graph_hscn.config.config import (
#     DataConfig,
#     HSCNConfig,
#     MPNNConfig,
#     OptimConfig,
#     PEConfig,
#     TrainingConfig,
# )
# from graph_hscn.loader.hetero_data import generate_hetero_data, hetero_loaders
# from graph_hscn.loader.loader import load_dataset
# from graph_hscn.logger import CustomLogger
# from graph_hscn.model.hscn import HSCN, SCN, build_hscn
# from graph_hscn.train.train import compute_posenc, train
# from graph_hscn.train.train_clustering import train_clustering
# from graph_hscn.train.utils import get_each_data_from_batch
#
# data_cfg = DataConfig(
#     dataset_name="peptides_struct",
#     task_level="graph",
#     root_dir="../../datasets",
#     pe=False,
# )
# pe_cfg = PEConfig(dim_in=16, dim_emb=9, dim_pe=4, use_bn=False)
# # model_cfg = MPNNConfig(
# #     conv_type="GCN", activation="relu", loss_fn="cross_entropy"
# # )
# model_cfg = HSCNConfig(activation="relu", cluster_epochs=1)
# if isinstance(model_cfg, HSCNConfig):
#     proj_name = f"{data_cfg.dataset_name}_HSCN_{model_cfg.num_clusters}"
# elif isinstance(model_cfg, MPNNConfig):
#     proj_name = f"{data_cfg.dataset_name}_{model_cfg.conv_type}"
# else:
#     raise NotImplementedError
# optim_cfg = OptimConfig(optim_type="adamW", lr=0.01)
# training_cfg = TrainingConfig(
#     loss_fn="l1", metric="mae", wandb_proj_name=proj_name
# )
# # training_cfg = TrainingConfig(
# #     loss_fn="cross_entropy", metric="ap", wandb_proj_name=proj_name
# # )
# if training_cfg.use_wandb:
#     assert training_cfg.wandb_proj_name, "WandB project name not provided."
#     wandb.init(
#         project=training_cfg.wandb_proj_name,
#         config={"architecture": "HSCN"},
#     )
#
# logger = CustomLogger(Path("trains.log"), metric_name=training_cfg.metric)
# loaders, dataset = load_dataset(logger, data_cfg, pe_cfg)
# split_idx = dataset.get_idx_split()
# num_features = loaders[0].dataset.data.num_features
# num_classes = loaders[0].dataset[0].y.shape[1]
#
# if data_cfg.pe:
#     loaders, dataset = compute_posenc(
#         loaders, data_cfg, num_features, pe_cfg, logger
#     )
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cluster_model = SCN([16], "elu", num_features, model_cfg.num_clusters).to(
#     device
# )
# if data_cfg.pe:
#     dataset = get_each_data_from_batch(dataset)
#
# cluster_all_lst = train_clustering(
#     logger, dataset, cluster_model, model_cfg, optim_cfg, training_cfg
# )
# hetero_data = generate_hetero_data(
#     cluster_all_lst, dataset, split_idx, data_cfg, model_cfg, logger
# )
# loaders = hetero_loaders(data_cfg, hetero_data, split_idx)
# model = build_hscn(model_cfg, num_features, num_classes)
# model = model.to(device)
# train(logger, optim_cfg, training_cfg, loaders, model)
