"""Class for peptides structural dataset."""
import hashlib
import os
import pickle
import shutil
from typing import Callable

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

import ca_net.loader.dataset.constants as Const


class PeptidesStructuralDataset(InMemoryDataset):
    """Class for peptides functional dataset.

    11-target regression.

    Parameters
    ----------
    root : str, optional
        Root directory of the dataset. Default is "datasets".
    _smiles2graph : Callable[[str], dict], optional
        Function to convert SMILES strings into graphs. Default is the
        `smiles2graph` function.
    transform : BaseTransform or Callable , optional
        Transform to apply on the data. Default is None.
    pre_transform : BaseTransform or Callable, optional
        Transform to apply before saving the processed data. Default is None.

    Attributes
    ----------
    original_root : str
        Root directory of the dataset.
    smiles2graph : Callable[[str], dict]
        Function to convert SMILES strings into graphs.
    folder : str
        Folder for the dataset.
    url : str
        URL for the dataset.
    version : str
        Version of the dataset.
    url_stratified_split : str
        URL for the stratified split.
    md5sum_stratified_split : str
        MD5 hash for the stratified split.
    data : Tensor
        Processed data.

    Methods
    -------
    download()
        Download the peptides data from the URL.
    process()
        Perform necessary processing on the raw Peptides data.
    _md5sum(path)
        Get the MD5 of a path.
    get_idx_split()
        Get split indices of the dataset.
    """

    def __init__(
        self,
        root: str = "datasets",
        _smiles2graph: Callable[[str], dict] = smiles2graph,
        transform: BaseTransform | Callable = None,
        pre_transform: BaseTransform | Callable = None,
    ) -> None:
        self.original_root = root
        self.smiles2graph = _smiles2graph
        self.folder = os.path.join(root, "peptides_structural")
        self.url = Const.STRUCTURAL_URL
        self.version = Const.STRUCTURAL_VERSION
        self.url_stratified_split = Const.STRUCTURAL_URL_STRATIFIED_SPLIT
        self.md5sum_stratified_split = Const.STRUCTURAL_MD5SUM_STRATIFIED_SPLIT

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
        """Raw file names.

        Returns
        -------
        Raw file names.
        """
        return Const.STRUCTURAL_RAW_FILE_NAME

    @property
    def processed_file_names(self) -> str:
        """Processed file names.

        Returns
        -------
        Processed file names.
        """
        return Const.STRUCTURAL_PROCESSED_FILE_NAMES

    def _md5sum(self, path: str) -> str:
        """Get the MD5 of a path.

        Parameters
        ----------
        path : str
            Input path as a string.

        Returns
        -------
        The MD5 of the binary file.
        """
        hash_md5 = hashlib.md5()

        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)

        return hash_md5.hexdigest()

    def download(self) -> None:
        """Download the peptides data from the URL.

        Returns
        -------
        None
        """
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
        """Perform necessary processing on the raw peptides data.

        Returns
        -------
        None
        """
        df = pd.read_csv(
            os.path.join(self.raw_dir, Const.STRUCTURAL_RAW_FILE_NAME)
        )
        smiles_list = df["smiles"]
        target_names = Const.STRUCTURAL_TARGET_NAMES
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

    def get_idx_split(self) -> dict[str, torch.Tensor]:
        """Get split indices of the dataset.

        Read the splits from the split file and return the dictionary of
        splits.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of splits.
        """
        split_file = os.path.join(
            self.root, Const.STRUCTURAL_SPLIT_PICKLE_FILE
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)

        split_dict = replace_numpy_with_torchtensor(splits)

        return split_dict
