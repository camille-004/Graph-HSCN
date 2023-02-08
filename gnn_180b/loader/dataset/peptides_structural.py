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


class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        _smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        self.original_root = root
        self.smiles2graph = _smiles2graph
        self.folder = os.path.join(root, "peptides_structural")
        self.url = "https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1"
        self.version = "9786061a34298a0684150f2e4ff13f47"
        self.url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
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
        return "peptide_structure_dataset.csv.gz"

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
            os.path.join(self.raw_dir, "peptide_structure_dataset.csv.gz")
        )
        smiles_list = df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        df.loc[:, target_names] = df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = df.iloc[i][target_names]
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
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_file = os.path.join(
            self.root, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)

        split_dict = replace_numpy_with_torchtensor(splits)

        return split_dict
