import numpy as np
import torch
from tqdm.auto import tqdm
import os
from pathlib import Path
import tifffile as tiff

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class VesuviusDataset(BaseDataset):
    def __init__(
        self,
        part,
        val_size=None,
        *args,
        **kwargs,
    ):
        index_name = f"{part}_index.json"
        self.index_path = ROOT_PATH / "data" / index_name
        assert part in ['train', 'val', 'test']
        self.val_size = val_size
        self.is_train = part != 'test'
        self.is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
        if self.is_kaggle_env:
            self.data_path = Path('/kaggle/input/vesuvius-challenge-surface-detection')
        else:
            self.data_path = ROOT_PATH / 'data'

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, part):
        index = []
        if self.is_train:
            images_path = self.data_path / 'train_images'
            target_path = self.data_path / 'train_labels'
        else:
            images_path = self.data_path / 'test_images'

        num_images = len([*images_path.iterdir()])
        if self.val_size is not None:
            num_val_images = self.val_size * num_images
        else:
            num_val_images = num_images

        for i, image_path in enumerate(images_path.iterdir()):
            item = {
                'image_path': str(image_path),
            }
            if self.is_train:
                item.update({
                    'target_path': target_path / image_path.name
                })
            is_item_in_dataset = (
                (part == 'train' and i >= num_val_images) or
                (part == 'val' and i < num_val_images) or
                part == 'test'
            )
            if is_item_in_dataset:
                index.append(item)

        write_json(index, str(self.index_path))

        return index

    def load_object(self, path):
        volume = tiff.imread(path)
        volume = torch.from_numpy(volume).long()
        return volume

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """

        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if '@' in transform_name: # transform applied for several tensors
                    transform_names = transform_name.split('@')
                    data_dict = {name: instance_data[name] for name in transform_names}
                    transform_result = self.instance_transforms[transform_name](**data_dict)
                    for name, value in transform_result:
                        instance_data[name] = value
                else:
                    instance_data[transform_name] = self.instance_transforms[transform_name](
                        instance_data[transform_name]
                    )
        return instance_data

    def __getitem__(self, ind):
        item = self._index[ind]
        instance_data = {
            'volume': self.load_object(item['image_path']),
        }
        if self.is_train:
            instance_data.update({
                'target': self.load_object(item['target_path'])
            })
        instance_data = self.preprocess_data(instance_data)
        return instance_data
