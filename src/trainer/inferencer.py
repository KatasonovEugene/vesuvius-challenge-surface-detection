import torch
from tqdm.auto import tqdm
import numpy as np
import tifffile as tiff
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
from src.utils.post_process_utils import build_anisotropic_struct_

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in metrics["inference"]:
                metrics.update(met.name, met(**batch))

        batch_size = batch["outputs"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995

            sample = batch["outputs"][i].clone()
            post_processed_sample = self.sample_post_process(sample)

            output_id = current_id + i

            if self.save_path is not None:
                tiff.imsave(self.save_path / part / f"{output_id}.tif", post_processed_sample)

        return batch


    def sample_post_process(self, volume):
        '''
        Applying post processing to one sample containing probabilites.

        Expected shape: [D, H, W, 1]

        Args:
            volume (Tensor): tensor containing probabilities [0, 1] of class 1
        '''

        volume = volume.squeeze(-1) # [D, H, W, 1] -> [D, H, W]
        volume = volume.cpu().numpy()

        # --- Parameters ---

        T_low=0.50
        T_high=0.90
        z_radius=1
        xy_radius=0
        dust_min_size=100


        # --- Step 1: 3D Hysteresis ---
        strong = volume >= T_high
        weak   = volume >= T_low

        if not strong.any():
            return np.zeros_like(volume, dtype=np.uint8)

        struct_hyst = ndi.generate_binary_structure(3, 3)
        mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

        if not mask.any():
            return np.zeros_like(volume, dtype=np.uint8)

        # --- Step 2: 3D Anisotropic Closing ---
        if z_radius > 0 or xy_radius > 0:
            struct_close = build_anisotropic_struct_(z_radius, xy_radius)
            if struct_close is not None:
                mask = ndi.binary_closing(mask, structure=struct_close)

        # --- Step 3: Dust Removal ---
        if dust_min_size > 0:
            mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

        return mask.astype(np.uint8)


    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """

        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type) if self.batch_transforms else None
        if transforms is not None:
            for transform_name in transforms.keys():
                if '@' in transform_name: # transform applied for several tensors
                    transform_names = transform_name.split('@')
                    data_dict = {name: batch[name] for name in transform_names}
                    transform_result = transforms[transform_name](**data_dict)
                    for name, value in transform_result:
                        batch[name] = value
                else:
                    batch[transform_name] = transforms[transform_name](
                        batch[transform_name]
                    )
        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result() if self.evaluation_metrics else None
