import hashlib
import os
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        _smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        self.original_root = root
        self.smiles2graph = _smiles2graph
        self.folder = os.path.join(root, "peptides_functional")
        self.url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = "701eb743e899f4d793f0e13c8fa5a1b4"
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

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
        return "peptide_multi_class_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
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

    def process(self):
        df = pd.read_csv(
            os.path.join(self.raw_dir, "peptide_multi_class_dataset.csv.gz")
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

    def get_idx_split(self):
        split_file = os.path.join(
            self.root, "splits_random_stratified_peptide.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)

        split_dict = replace_numpy_with_torchtensor(splits)

        return split_dict
