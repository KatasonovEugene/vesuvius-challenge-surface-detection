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
        part='train',
        val_size=None,
        override=False,
        load_in_memory=False,
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

        if self.index_path.exists() and not override:
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
            num_val_images = int(self.val_size * num_images)
        else:
            num_val_images = num_images

        for i, image_path in enumerate(images_path.iterdir()):
            item = {
                'image_path': str(image_path),
            }
            if self.is_train:
                item.update({
                    'target_path': str(target_path / image_path.name)
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

    def load_object(self, path, dtype):
        volume = tiff.imread(path)
        volume = torch.from_numpy(volume).to(dtype).unsqueeze(0)
        return volume

    def __getitem__(self, ind):
        item = self._index[ind]
        instance_data = {
            'volume': self.load_object(item['image_path'], torch.float32),
            'image_id': torch.tensor(int(os.path.splitext(os.path.basename(item['image_path']))[0]))
        }
        if self.is_train:
            target = self.load_object(item['target_path'], torch.int64)
            instance_data.update({
                'gt_mask': target,
                'gt_skel': target,
            })
        instance_data = self.preprocess_data(instance_data)
        return instance_data
