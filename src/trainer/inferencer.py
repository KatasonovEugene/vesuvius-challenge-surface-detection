import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
import tifffile as tiff
import copy

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.model import Ensemble

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
        tta_transforms=None,
        postprocess_transforms=None,
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
        self.is_ensemble = isinstance(self.model.get_inner_model(), Ensemble)

        self.batch_transforms = batch_transforms
        self.tta_transforms = tta_transforms
        self.postprocess_transforms = postprocess_transforms

        self._init_amp_config(self.cfg_trainer)

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[key for met in self.metrics["inference"] for key in met.keys_full_list()],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
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

        with self._autocast_context():
            outputs = self.model(**batch)
        batch.update(outputs)

        if self.tta_transforms:
            samples_num = 1
            for transform_name in self.tta_transforms.keys():
                batch_copy = copy.deepcopy(batch)

                transform_result = self.tta_transforms[transform_name](**batch_copy)
                batch_copy.update(transform_result)

                with self._autocast_context():
                    transformed_outputs = self.model(**batch_copy)
                batch_copy.update(transformed_outputs)

                real_logits = self.tta_transforms[transform_name].detransform(**batch_copy)
                batch_copy.update(real_logits)

                if self.is_ensemble:
                    batch['probs'] += batch_copy['probs']
                else:
                    batch['logits'] += batch_copy['logits']
                samples_num += 1

            if self.is_ensemble:
                batch['probs'] /= samples_num
            else:
                batch['logits'] /= samples_num

        if self.is_ensemble:
            batch['outputs'] = batch['probs'][:, 1]
        else:
            batch['outputs'] = F.softmax(batch['logits'], dim=1)[:, 1]

        batch = self.apply_transforms(self.postprocess_transforms, batch)

        if metrics is not None and self.metrics is not None:
            for met in self.metrics["inference"]:
                metric_result = met(**batch) # make sure metric works with 'outputs'
                if isinstance(metric_result, dict):
                    for key in metric_result.keys():
                        metrics.update(met.name + '_' + key, metric_result[key])
                else:
                    metrics.update(met.name, metric_result)

        batch_size = batch['outputs'].shape[0]
        for i in range(batch_size):
            post_processed_sample = batch["outputs"][i].clone()
            output_image_id = batch['image_id'][i]
            if self.save_path is not None:
                tiff.imwrite(self.save_path / part / f'{output_image_id}.tif', post_processed_sample.cpu().numpy())

        if self.evaluation_metrics:
            print(f'Batch: {batch_idx}. Metric accumulated results:')
            for key, value in self.evaluation_metrics.result().items():
                print(f"    {key:15s}: {value}")

        return batch

    def apply_transforms(self, transforms, batch):
        if transforms is not None:
            for transform_name in transforms.keys():
                if '@' in transform_name: # transform applied for several tensors
                    transform_names = transform_name.split('@')
                    data_dict = {name: batch[name] for name in transform_names}
                    transform_result = transforms[transform_name](**data_dict)
                    for name, value in transform_result:
                        batch[name] = value
                else:
                    transform_result = transforms[transform_name](**batch)
                    batch.update(transform_result)
        return batch

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
        return self.apply_transforms(transforms, batch)

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
