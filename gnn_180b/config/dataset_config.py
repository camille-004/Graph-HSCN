from torch_geometric.graphgym.register import register_config


@register_config("dataset")
def dataset_cfg(cfg):
    cfg.dataset.num_classes = 10
