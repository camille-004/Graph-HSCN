import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
)

from graph_hscn.config.config import PEConfig


def compute_posenc_stats(
    data: Data, is_undirected: bool, cfg: PEConfig
) -> Data:
    if hasattr(data, "num_nodes"):
        N = data.num_nodes
    else:
        N = data.x.shape[0]

    laplacian_norm_type = cfg.eigen_laplacian_norm.lower()

    if laplacian_norm_type == "none":
        laplacian_norm_type = None

    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    norm_type = cfg.eigen_laplacian_norm.lower()

    if norm_type == "none":
        norm_type = None

    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=norm_type, num_nodes=N)
    )
    evals_sn, evects_sn = np.linalg.eigh(L.toarray())
    data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
        evals=evals_sn,
        evects=evects_sn,
        max_freqs=cfg.eigen_max_freqs,
        eigvec_norm=cfg.eigvec_norm,
    )

    return data


def get_lap_decomp_stats(
    evals: torch.Tensor,
    evects: torch.Tensor,
    max_freqs: int,
    eigvec_norm: str = "L2",
) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(evals)  # Number of nodes, including disconnected nodes

    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigenvectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)

    if N < max_freqs:
        eig_vecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
    else:
        eig_vecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        eig_vals = F.pad(
            evals, (0, max_freqs - N), value=float("nan")
        ).unsqueeze(0)
    else:
        eig_vals = evals.unsqueeze(0)

    eig_vals = eig_vals.repeat(N, 1).unsqueeze(2)

    return eig_vals, eig_vecs


def eigvec_normalizer(
    eig_vecs: torch.Tensor,
    eig_vals: torch.Tensor,
    normalization: str = "L2",
    eps: float = 1e-12,
) -> torch.Tensor:
    match normalization:
        case "L1":
            # eigvec / sum(abs(eigvec))
            denom = eig_vecs.norm(p=1, dim=0, keepdim=True)
        case "L2":
            # eigvec / sqrt(sum(eigvec^2))
            denom = eig_vecs.norm(p=2, dim=0, keepdim=True)
        case "abs-max":
            # eigvec / max(|eigvec|)
            denom = torch.max(eig_vecs.abs(), dim=0, keepdim=True).values
        case other:
            raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(eig_vecs)
    eig_vecs = eig_vecs / denom

    return eig_vecs
