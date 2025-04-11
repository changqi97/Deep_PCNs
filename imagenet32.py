import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class ImageNet32(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    train_list = [
        ['train_data_batch_9', '58417149b5ce31688f805341e7f57b4f'], ['train_data_batch_4', '876f7e6d6ddb1f52ecb654ffdc8293f6'], ['train_data_batch_3', '03d3dc4e850e23e1d526f268a0196296'], ['train_data_batch_7', '32ecc8ad6c55b1c9cb6cf79a2cc46094'], ['train_data_batch_1', '6c4495e65cd24a8c3019857ef9b758ee'], ['train_data_batch_5', 'c789bcdd1feed34a9bc58598a1a1bf7d'], ['train_data_batch_8', 'bdeb6da3d05771121992b48c59c69439'], ['train_data_batch_6', '8ce5344cb1e11f31bc507cae4c856083'], ['train_data_batch_2', '3dd727de4155836807dfc19f98c9fe70'], ['train_data_batch_10', '46ad60a1144aaf97a143914453b0dabb']
    ]

    test_list = [
       ['val_data', '06c02b8b4c8de8b3a36b07859a49de6f'],
    ]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target-1)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"