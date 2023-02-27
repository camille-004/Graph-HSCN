"""https://github.com/vijaydwivedi75/lrgb/blob/d48dd4462019bd26c367a1ab303a1c07349e40d8/aggr_results/agg_runs.py"""  # noqa
import argparse
import logging
import os.path
import sys

import numpy as np
from torch_geometric.graphgym.config import assert_cfg, cfg, set_cfg
from torch_geometric.graphgym.utils.agg_runs import (  # noqa
    agg_dict_list,
    is_seed,
    is_split,
)
from torch_geometric.graphgym.utils.io import (  # noqa
    dict_list_to_json,
    dict_to_json,
    json_to_dict_list,
    makedirs_rm_exist,
)

from ca_net.util import set_new_cfg_allowed

sys.path.append(".")
sys.path.append("..")


def parse_args() -> argparse.Namespace:
    """Parse the arguments.

    Returns
    -------
    argparse.Namespace
        Argument values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        dest="dir",
        help="Dir with multiple seed results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        help="metric to select best epoch",
        required=False,
        type=str,
        default="auto",
    )
    return parser.parse_args()


def join_list(l1: list, l2: list) -> list:
    """Join two lists together.

    Parameters
    ----------
    l1 : list
        First list.
    l2 : list
        Second list.

    Returns
    -------
    list
        Joined list.
    """
    if len(l1) > len(l2):
        print(
            f">> W: Padding the second list (len={len(l2)}) with the last "
            f"item to match len={len(l1)} of the first list."
        )
        while len(l1) > len(l2):
            l2.append(l2[-1])

    if len(l1) < len(l2):
        print(
            f">> W: Padding the first list (len={len(l1)}) with the last "
            f"item to match len={len(l2)} of the second list."
        )
        while len(l1) < len(l2):
            l1.append(l1[-1])

    assert len(l1) == len(
        l2
    ), "Results with different seeds must have the save format"

    for i in range(len(l1)):
        l1[i] += l2[i]

    return l1


def agg_runs(_dir: str, metric_best: str = "auto") -> None:
    """Aggregate over different random seeds of a single experiment.

    NOTE: This is an unchanged copy from GraphGym, only `join_list` function
    had to be modified to pad list to process incomplete runs.

    Parameters
    ----------
    _dir : str
        Directory of the results, containing 1 experiment.
    metric_best : str
        The metric for selecting the best
        validation performance. Options: auto, accuracy, ap, mae.

    Returns
    -------
    None
    """
    results = {"train": None, "val": None, "test": None}
    results_best = {"train": None, "val": None, "test": None}
    for seed in os.listdir(_dir):
        if is_seed(seed):
            dir_seed = os.path.join(_dir, seed)
            split = "val"

            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, "stats.json")
                stats_list = json_to_dict_list(fname_stats)

                if metric_best == "auto":
                    if "accuracy" in stats_list[0]:
                        metric = "accuracy"
                    elif "ap" in stats_list[0]:
                        metric = "ap"
                    else:
                        metric = "mae"
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list]
                )
                best_epoch = stats_list[
                    eval(f"performance_np.{cfg.metric_agg}()")
                ]["epoch"]
                print(best_epoch)

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, "stats.json")
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [
                        stats
                        for stats in stats_list
                        if stats["epoch"] == best_epoch
                    ][0]
                    print(stats_best)
                    stats_list = [[stats] for stats in stats_list]

                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)

                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]

    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {
        k: v for k, v in results_best.items() if v is not None
    }  # rm None

    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])

    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])

    # Save aggregated results.
    for key, value in results.items():
        dir_out = os.path.join(_dir, key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, "stats.json")
        dict_list_to_json(value, fname)

    for key, value in results_best.items():
        dir_out = os.path.join(_dir, key)
        fname = os.path.join(dir_out, "best.json")
        dict_to_json(value, fname)
    logging.info(f"Results aggregated across runs saved in {_dir}")


if __name__ == "__main__":
    args = parse_args()

    set_cfg(cfg)
    set_new_cfg_allowed(cfg, True)
    cfg.merge_from_file(os.path.join(args.dir, "config.yaml"))
    assert_cfg(cfg)

    if args.metric == "auto":
        args.metric = cfg.metric_best
    print(f"metric:   {args.metric}")
    print(f"agg_type: {cfg.metric_agg}")

    # Aggregate results from different seeds
    agg_runs(args.dir, args.metric)
