"""Custom position encoding config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("posenc")
def set_cfg_posenc(cfg: CN) -> None:
    """Define a custom config to extend positional encoding options.

    Parameters
    ----------
    cfg : CN
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_EquivStableLapPE = CN()

    # Common arguments to all PE types.
    for name in ["posenc_LapPE", "posenc_SignNet"]:
        pe_cfg = getattr(cfg, name)

        # Use extended positional encodings
        pe_cfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pe_cfg.model = "none"

        # Size of Positional Encoding embedding
        pe_cfg.dim_pe = 16

        # Number of layers in PE encoder model
        pe_cfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pe_cfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pe_cfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pe_cfg.raw_norm_type = "none"

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pe_cfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = "none"

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ["posenc_LapPE", "posenc_SignNet", "posenc_EquivStableLapPE"]:
        pe_cfg = getattr(cfg, name)
        pe_cfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or
        # 'rw'
        pe_cfg.eigen.laplacian_norm = "sym"

        # The normalization scheme for the eigen vectors of the Laplacian
        pe_cfg.eigen.eigvec_norm = "L2"

        # Maximum number of top smallest frequencies & eigenvectors to use
        pe_cfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64
