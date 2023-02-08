from torch_geometric.graphgym.register import register_config


@register_config("split_cfg")
def set_cfg_split(cfg):
    # Default to selecting the standard split that ships with the dataset
    cfg.dataset.split_node = "standard"

    # Choose a particular split to use if multiple splits are available
    cfg.dataset.split_index = 0

    # Dir to cache cross-validation splits
    cfg.dataset.split_dir = "./splits"

    # Choose to run multiple splits in one program execution, if set,
    # Takes the precedence over cfg.dataset.split_index for split selection
    cfg.run_multiple_splits = []
