from torch_geometric.graphgym.register import register_config


@register_config("overwrite_defaults")
def overwrite_defaults_cfg(cfg):
    # Overwrite default dataset_name
    cfg.dataset_name = "none"

    # Overwrite default rounding precision
    cfg.round = 5


@register_config("extended")
def extended_cfg(cfg):
    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""
