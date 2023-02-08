import numpy as np
from scipy.stats import stats


def eval_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Spearman Rho averaged across tasks."""
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # Ignore NaNs
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(
                stats.spearmanr(y_true[is_labeled, i], y_pred[is_labeled, i])[
                    0
                ]
            )

    return {"spearmanr": sum(res_list) / len(res_list)}
