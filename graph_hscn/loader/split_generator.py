"""Functions for splitting the dataset."""
import json
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index_to_mask, set_dataset_attr


def prepare_splits(dataset: Data) -> None:
    """Specify the split to use in loading the dataset.

    Parameters
    ----------
    dataset : Data
        Dataset to use.

    Returns
    -------
    None
    """
    split_mode = cfg.dataset.split_mode

    match split_mode:
        case "standard":
            setup_standard_split(dataset)
        case "random":
            setup_random_split(dataset)
        case other:
            if split_mode.startswith("cv-"):
                cv_type, k = split_mode.split("-")[1:]
                setup_cv_split(dataset, cv_type, int(k))
            else:
                raise ValueError(f"Unknown split mode: {split_mode}")


def setup_standard_split(dataset: Data) -> None:
    """Set up the dataset for a standard split (supplied in the dataset).

    Parameters
    ----------
    dataset : Data
        Dataset to use.

    Returns
    -------
    None
    """
    split_index = cfg.dataset.split_index
    task_level = cfg.dataset.task

    if task_level == "node":
        for split_name in "train_mask", "val_mask", "test_mask":
            mask = getattr(dataset.data, split_name, None)

            if mask is None:
                raise ValueError(f"Missing '{split_name}' for standard split.")

            if mask.dim() == 2:
                if split_index >= mask.shape[1]:
                    raise IndexError(
                        f"Specified split index ({split_index}) is out of"
                        f"range of the number of available splits"
                        f"({mask.shape[1]}) for {split_name}"
                    )

                set_dataset_attr(
                    dataset,
                    split_name,
                    mask[:, split_index],
                    len(mask[:, split_index]),
                )
    else:  # Graph-level
        for split_name in (
            "train_graph_index",
            "val_graph_index",
            "test_graph_index",
        ):
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split.")

        if split_index != 0:
            raise NotImplementedError(
                f"Multiple standard splits not supported for dataset task"
                f"level: {task_level}"
            )


def setup_random_split(dataset: Data) -> None:
    """Set up the dataset for a random split based on the split ratio.

    Parameters
    ----------
    dataset : Data
        Dataset to use.

    Returns
    -------
    None
    """
    split_ratios = cfg.dataset.split

    if len(split_ratios) != 3:
        raise ValueError(
            f"Three split ratios is expected for train/val/test, received "
            f"{len(split_ratios)} split ratios: {repr(split_ratios)}"
        )
    elif sum(split_ratios) != 1:
        raise ValueError(
            f"The train/val/test split ratios must sum up to 1, input ratios"
            f"sum up to {sum(split_ratios):.2f} instead: {repr(split_ratios)}"
        )

    train_index, val_test_index = next(
        ShuffleSplit(train_size=split_ratios[0], random_state=cfg.seed).split(
            dataset.data.y, dataset.data.y
        )
    )
    val_index, test_index = next(
        ShuffleSplit(
            train_size=split_ratios[1] / (1 - split_ratios[0]),
            random_state=cfg.seed,
        ).split(dataset.data.y[val_test_index], dataset.data.y[val_test_index])
    )
    val_index = val_test_index[val_index]
    test_index = val_test_index[test_index]

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def set_dataset_splits(dataset: Data, splits: list[torch.Tensor]) -> None:
    """Set the dataset splits depending on the task level..

    Parameters
    ----------
    dataset : Data
        Dataset to use.
    splits : list[torch.Tensor]
        List of indices for splitting.

    Returns
    -------
    None
    """
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))

            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: split #{i} (n"
                    f" = {len(splits[i])}) and split #{j} (n = "
                    f"{len(splits[j])}) have {n_intersect} intersecting"
                    f"indices."
                )
    task_level = cfg.dataset.task

    match task_level:
        case "node":
            split_names = ["train_mask", "val_mask", "test_mask"]

            for split_name, split_index in zip(split_names, splits):
                mask = index_to_mask(
                    torch.from_numpy(split_index), size=dataset.data.y.shape[0]
                )
                set_dataset_attr(dataset, split_name, mask, len(mask))
        case "graph":
            split_names = [
                "train_graph_index",
                "val_graph_index",
                "test_graph_index",
            ]

            for split_name, split_index in zip(split_names, splits):
                set_dataset_attr(
                    dataset, split_name, split_index, len(split_index)
                )
        case other:
            raise ValueError(f"Unsupported dataset task level: {task_level}")


def setup_cv_split(dataset: Data, cv_type: str, k: int) -> None:
    """Set up splits for CV.

    Parameters
    ----------
    dataset : Data
        Dataset to use.
    cv_type : str
        Type of CV to use.
    k : int
        Value of k for CV.

    Returns
    -------
    None
    """
    split_index = cfg.dataset.split_index
    split_dir = cfg.dataset.split_dir

    if split_index >= k:
        raise IndexError(
            f"Specified split_index={split_index} is out of range of the"
            f"number of folds k={k}"
        )

    os.makedirs(split_dir, exist_ok=True)
    save_file = os.path.join(
        split_dir, f"{cfg.dataset.format}_{dataset.name}_{cv_type}-{k}.json"
    )

    if not os.path.isfile(save_file):
        create_cv_splits(dataset, cv_type, k, save_file)

    with open(save_file) as f:
        cv = json.load(f)

    assert cv["dataset"] == dataset.name, "Unexpected dataset CV splits."
    assert cv["n_samples"] == len(dataset), "Dataset length does not match."
    assert cv["n_splits"] > split_index, "Fold selection out of range."
    assert k == cv["n_splits"], f"Expected k={k}, but {cv['n_splits']} found."

    test_ids = cv[str(split_index)]
    val_ids = cv[str((split_index + 1) % k)]
    train_ids = []

    for i in range(k):
        if i != split_index and i != (split_index + 1) % k:
            train_ids.extend(cv[str(i)])

    set_dataset_splits(dataset, [train_ids, val_ids, test_ids])


def create_cv_splits(
    dataset: Data, cv_type: str, k: int, file_name: str
) -> None:
    """Create cross-validation splits.

    Parameters
    ----------
    dataset : Data
        Dataset to use.
    cv_type : str
        Type of CV to use.
    k : int
        Value of K for CV.
    file_name : str
        Name of file in which to save splits.

    Returns
    -------
    None
    """
    n_samples = len(dataset)

    if cv_type == "stratifiedKfold":
        kf = StratifiedKFold(n_split=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples), dataset.data.y)
    elif cv_type == "kfold":
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples))
    else:
        raise ValueError(f"Unexpected cross-validation type: {cv_type}")

    splits = {
        "n_samples": n_samples,
        "n_pslits": k,
        "cross_validator": kf.__str__(),
        "dataset": dataset.name,
    }

    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()

    with open(file_name, "w") as f:
        json.dump(splits, f)

    logging.info(f"[*] Saved newly generated CV splits by {kf} to {file_name}")
