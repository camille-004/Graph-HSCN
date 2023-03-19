import hashlib
import os
import pickle
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

import graph_hscn.loader.dataset.constants as const
from graph_hscn.constants import DATASETS_DIR


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path = DATASETS_DIR,
        _smiles2graph: Callable[[str], dict] = smiles2graph,
        transform: BaseTransform = None,
        pre_transform: BaseTransform = None,
    ) -> None:
        self.original_root = root
        self.smiles2graph = _smiles2graph
        self.folder = os.path.join(root, "peptides_functional")
        self.url = const.FUNCTIONAL_URL
        self.version = const.FUNCTIONAL_VERSION
        self.url_stratified_split = const.FUNCTIONAL_URL_STRATIFIED_SPLIT
        self.md5sum_stratified_split = const.FUNCTIONAL_MD5SUM_STRATIFIED_SPLIT
        release_tag = os.path.join(self.folder, self.version)

        if os.path.isdir(self.folder) and (not os.path.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if (
                input("Will you update the dataset now? (y/N)\n").lower()
                == "y"
            ):
                shutil.rmtree(self.folder)
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return const.FUNCTIONAL_RAW_FILE_NAME

    @property
    def processed_file_names(self) -> str:
        return const.FUNCTIONAL_PROCESSED_FILE_NAMES

    def _md5sum(self, path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            _hash = self._md5sum(path)

            if _hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file.")
            open(os.path.join(self.root, _hash), "w").close()
            path_split = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self) -> None:
        df = pd.read_csv(
            os.path.join(self.raw_dir, const.FUNCTIONAL_RAW_FILE_NAME)
        )
        smiles_list = df["smiles"]
        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            graph = smiles2graph(smiles)
            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                torch.int64
            )
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                torch.int64
            )
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([eval(df["labels"].iloc[i])])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self) -> dict[str, torch.Tensor]:
        split_file = os.path.join(
            self.root, const.FUNCTIONAL_SPLIT_PICKLE_FILE
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict
