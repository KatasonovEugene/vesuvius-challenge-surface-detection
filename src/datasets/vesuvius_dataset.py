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
        images_path=None,
        pseudotarget_path=None,
        teacher_probs_path=None,
        mix_target_with_ps=True,
        *args,
        **kwargs,
    ):
        index_name = f"{part}_index.json"
        if pseudotarget_path is not None:
            index_name = "pseudotarget" + index_name
        if teacher_probs_path is not None:
            index_name = "teacher" + index_name

        self.is_kaggle_env = 'KAGGLE_URL_BASE' in os.environ
        if self.is_kaggle_env:
            self.index_path = ROOT_PATH / "data"
        else:
            self.index_path = Path("data")
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_path / index_name

        assert part in ['train', 'val', 'test']
        self.val_size = val_size
        self.is_train = part != 'test'
        if self.is_kaggle_env:
            self.data_path = Path('/kaggle/input/vesuvius-challenge-surface-detection')
        else:
            self.data_path = ROOT_PATH / 'data'

        if images_path is None:
            self.images_path = None
        else:
            self.images_path = ROOT_PATH / images_path

        if pseudotarget_path is None:
            self.pseudotarget_path = None
        else:
            self.pseudotarget_path = ROOT_PATH / pseudotarget_path

        if teacher_probs_path is None:
            self.teacher_probs_path = None
        else:
            self.teacher_probs_path = ROOT_PATH / teacher_probs_path

        self.mix_target_with_ps = mix_target_with_ps

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
            if self.images_path is not None:
                images_path = self.images_path
            pseudotarget_path = self.pseudotarget_path
            teacher_probs_path = self.teacher_probs_path
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
                    'target_path': str(target_path / image_path.name),
                })
                if pseudotarget_path is not None:
                    item.update({
                        'pseudotarget_path': str(pseudotarget_path / image_path.name),
                    })
                if teacher_probs_path is not None:
                    item.update({
                        'teacher_probs_path': str(teacher_probs_path / image_path.name),
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
        volume = volume.astype(dtype)[None]
        return volume

    def __getitem__(self, ind):
        item = self._index[ind]
        image_path = item['image_path']
        instance_data = {
            'volume': self.load_object(image_path, np.float32),
            'image_id': Path(image_path).stem,
        }
        if self.is_train:
            target = self.load_object(item['target_path'], np.int8)
            if 'pseudotarget_path' in item:
                pseudotarget = self.load_object(item['pseudotarget_path'], np.int8)
                old_target = target.copy()

                if self.mix_target_with_ps:
                    mask = (target == 2)
                    target[mask] = pseudotarget[mask]
                    instance_data.update({
                        'old_gt_mask': old_target,
                    })
                else:
                    target = pseudotarget

            if 'teacher_probs_path' in item:
                teacher_probs = self.load_object(item['teacher_probs_path'], np.float32)
                instance_data.update({
                    'teacher_probs': teacher_probs,
                })

            instance_data.update({
                'gt_mask': target,
                'gt_skel': target,
            })
        instance_data = self.preprocess_data(instance_data)
        return instance_data
