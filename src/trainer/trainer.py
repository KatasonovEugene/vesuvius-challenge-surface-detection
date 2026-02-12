import torch
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.plot_utils import plot_batch, view_batch_3d
import numpy as np
import matplotlib.cm as cm


class Trainer(BaseTrainer):
    def __init__(self, *args, log_batch_plots=False, view_3d_online=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_batch_plots = log_batch_plots
        self.view_3d_online = view_3d_online
        self.training_steps = 0

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with self._autocast_context():
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(training_steps=self.training_steps, **batch)
        batch.update(all_losses)

        if self.is_train:
            if self.scaler is not None:
                self.scaler.scale(batch["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch["loss"].backward()
                self._clip_grad_norm()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.training_steps += 1

        if self.log_batch_plots:
            if 'outputs' not in batch:
                batch['outputs'] = torch.softmax(batch['logits'], dim=1)[:, 1].detach()
            if not hasattr(self, 'batch_count'):
                self.batch_count = 0
            plot_batch(**batch, name=f'batch_plot_{self.batch_count}')
            self.batch_count += 1

        if self.view_3d_online:
            if 'outputs' not in batch:
                batch['outputs'] = torch.softmax(batch['logits'], dim=1)[:, 1].detach()
            view_batch_3d(**batch)

        for loss_name in self.criterion.names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metric_result = met(**batch)
            if isinstance(metric_result, dict):
                for key in metric_result.keys():
                    value = metric_result[key]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metrics.update(met.name + '_' + key, value)
            else:
                if isinstance(metric_result, torch.Tensor):
                    metric_result = metric_result.item()
                metrics.update(met.name, metric_result)

        return batch
    
    def convert_image(self, img):
        img = img.detach().long().cpu().numpy()
        out = np.zeros((*img.shape, 3), dtype=np.uint8)

        out[img == 0] = [0, 0, 0]
        out[img == 1] = [255, 255, 255]
        out[img == 2] = [127, 127, 127]

        return out

    def convert_heatmap(self, img):
        img = img.detach().cpu().numpy()
        cmap = cm.get_cmap("inferno")
        colored = cmap(img)[..., :3]
        return (colored * 255).astype(np.uint8)

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode != "train":
            indices = np.arange(31, 160, step=32)
            logits = batch['logits'][0]
            prob = torch.softmax(logits, dim=0)[1]
            pred = logits.argmax(dim=0)
            pred = self.convert_image(pred)
            prob = self.convert_heatmap(prob)
            slices = {
                "pred_axial_z": [pred[i, :, :] for i in indices],
                "prob_axial_z": [prob[i, :, :] for i in indices],
                "pred_coronal_y": [pred[:, i, :] for i in indices],
                "prob_coronal_y": [prob[:, i, :] for i in indices],
                "pred_sagittal_x": [pred[:, :, i] for i in indices],
                "prob_sagittal_x": [prob[:, :, i] for i in indices],
            }
            self.writer.add_slices(slices)
