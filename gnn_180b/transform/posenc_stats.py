import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
)


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    for t in pe_types:
        if t not in [
            "LapPE",
            "EquivStableLapPE",
            "SignNet",
        ]:
            raise ValueError(
                f"Unexpected PE stats selection {t} in {pe_types}"
            )

    if hasattr(data, "num_nodes"):
        N = data.num_nodes
    else:
        N = data.x.shape[0]

    laplacian_norm_type = cfg.posenc_LaPE.eigen.laplacian_norm.lower()

    if laplacian_norm_type == "none":
        laplacian_norm_type = None

    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigenvalues and eigenvectors
    evals, evects = None, None

    if "LaPE" in pe_types or "EquivStableLapPE" in pe_types:
        L = to_scipy_sparse_matrix(
            undir_edge_index, normalization=laplacian_norm_type, num_nodes=N
        )
        evals, evects = np.linalg.eigh(L.toarray())

        max_freqs, eigvec_norm = None, None

        if "LapPE" in pe_types:
            max_freqs = cfg.posenc_LaPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_LaPE.eigen.eigvec_norm
        elif "EquivStableLapPE" in pe_types:
            max_freqs = cfg.posenc_EquivStableLLapPE.eigen.nax_freqs
            eigvec_norm = cfg.posenc_EquivStableLapPE.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals,
            evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm,
        )

    if "SignNet" in pe_types:
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()

        if norm_type == "none":
            norm_type = None

        L = to_scipy_sparse_matrix(
            *get_laplacian(
                undir_edge_index, normalization=norm_type, num_nodes=N
            )
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn,
            evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm,
        )

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
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


def eigvec_normalizer(eig_vecs, normalization="L2", eps=1e-12):
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
            raise ValueError(f"Unsupported normalization '{normalization}'")

    denom = denom.clamp_min(eps).expand_as(eig_vecs)
    eig_vecs = eig_vecs / denom

    return eig_vecs
