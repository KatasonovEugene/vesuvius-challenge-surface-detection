import numpy as np
import torch
from tqdm.auto import tqdm
import os
from pathlib import Path
import tifffile as tiff
from torch.utils.data import Dataset

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from vesuvius import Volume

class SSLDataset(Dataset):
    def __init__(
        self,
        instance_transforms,
        crop_size,
        part='train',
        *args,
        **kwargs,
    ):
        self.volumes = [
            Volume(type="zarr", path="https://data.aws.ash2txt.org/samples/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0"),
            Volume(type="zarr", path="https://data.aws.ash2txt.org/samples/PHerc0009B/volumes/20250521125136-8.640um-1.2m-116keV-masked.zarr/0"),
        ]

        self.shapes = [volume.shape() for volume in self.volumes]

        assert part in ['train']
        self.is_train = True
        self.instance_transforms = instance_transforms
        self.crop_size = crop_size

        super().__init__(*args, **kwargs)

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return 1000

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
                if transform_name == 'contrastive':
                    instance_data['temp'] = instance_data['volume']

                    instance_data['volume'] = instance_data['volume_sem1']
                    result = self.instance_transforms[transform_name](**instance_data)
                    instance_data['volume_sem1'] = result['volume']

                    instance_data['volume'] = instance_data['volume_sem2']
                    result = self.instance_transforms[transform_name](**instance_data)
                    instance_data['volume_sem2'] = result['volume']

                    instance_data['volume'] = instance_data['temp']
                    instance_data.pop('temp')
                elif transform_name == 'struct':
                    instance_data['temp'] = instance_data['volume']

                    instance_data['volume'] = instance_data['volume_struct']
                    result = self.instance_transforms[transform_name](**instance_data)
                    instance_data['volume_struct'] = result['volume']

                    instance_data['volume'] = instance_data['temp']
                    instance_data.pop('temp')
                else:
                    result = self.instance_transforms[transform_name](**instance_data)
                    instance_data.update(result)
        return instance_data
    
    def sample_crop(self):
        idx = np.random.randint(len(self.volumes))
        volume = self.volumes[idx]

        D, H, W = self.shapes[idx]

        cd, ch, cw = self.crop_size

        z = np.random.randint(0, D - cd)
        y = np.random.randint(0, H - ch)
        x = np.random.randint(0, W - cw)

        full_id = f'{idx}_{z}_{y}_{x}'

        return volume[z:z+cd, y:y+ch, x:x+cw], full_id

    def __getitem__(self, ind):
        volume, image_id = self.sample_crop()

        instance_data = {
            'volume': volume,
            'image_id': image_id,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data
