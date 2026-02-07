import torch
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.plot_utils import plot_batch, view_batch_3d
import numpy as np


class Trainer(BaseTrainer):
    def __init__(self, *args, log_batch_plots=False, view_3d_online=False, log_batch_to_writer=True, grad_accum_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_batch_plots = log_batch_plots
        self.log_batch_to_writer = log_batch_to_writer
        self.view_3d_online = view_3d_online
        self.grad_accum_steps = grad_accum_steps

    def process_batch(self, batch, batch_idx, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if batch_idx % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()

        with self._autocast_context():
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            if self.is_train:
                for loss_name in all_losses.keys():
                    all_losses[loss_name] = all_losses[loss_name] / self.grad_accum_steps
        batch.update(all_losses)

        self.train_step_count += 1
        if self.is_train:
            if self.scaler is not None:
                self.scaler.scale(batch["loss"]).backward()
                if batch_idx % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_grad_norm()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                batch["loss"].backward()
                if batch_idx % self.grad_accum_steps == 0:
                    self._clip_grad_norm()
                    self.optimizer.step()

            if self.lr_scheduler is not None and batch_idx % self.grad_accum_steps == 0:
                self.lr_scheduler.step()

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
            metrics.update(loss_name, batch[loss_name].item() * self.grad_accum_steps)

        for met in metric_funcs:
            metric_result = met(**batch)
            if isinstance(metric_result, dict):
                for key in metric_result.keys():
                    metrics.update(met.name + '_' + key, metric_result[key])
            else:
                metrics.update(met.name, metric_result)

        return batch

    def convert_image(self, img):
        img = img.detach().long().cpu().numpy()
        out = np.zeros((*img.shape, 3), dtype=np.uint8)

        out[img == 0] = [0, 0, 0]
        out[img == 1] = [255, 255, 255]
        out[img == 2] = [127, 127, 127]

        return out

    def _log_batch(self, batch_idx, batch, mode="train"):
        if not self.log_batch_to_writer:
            return

        if mode != "train":
            mask = batch['gt_mask'][0]
            indices = np.arange(mask.shape[0])
            logits = batch['logits'][0]
            prob = torch.softmax(logits, dim=0).max(dim=0)
            pred = logits.argmax(dim=0)
            mask = self.convert_image(mask)
            pred = self.convert_image(pred)
            slices = {
                "mask_axial_z": [mask[i, :, :] for i in indices],
                "pred_axial_z": [pred[i, :, :] for i in indices],
                "prob_axial_z": [prob[i, :, :] for i in indices],
                "mask_coronal_y": [mask[:, i, :] for i in indices],
                "pred_coronal_y": [pred[:, i, :] for i in indices],
                "prob_coronal_y": [prob[:, i, :] for i in indices],
                "mask_sagittal_x": [mask[:, :, i] for i in indices],
                "pred_sagittal_x": [pred[:, :, i] for i in indices],
                "prob_sagittal_x": [prob[:, :, i] for i in indices],
            }
            self.writer.add_slices(slices)
