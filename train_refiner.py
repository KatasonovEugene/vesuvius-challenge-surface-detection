import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import RefinerTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train_refiner")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if len(config.writer.run_name) == 0:
        config.writer.run_name = "plug"

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    assert hasattr(model, "get_inner_model")

    teacher_model = instantiate(config.teacher_model).to(device)
    assert hasattr(teacher_model, "get_inner_model")

    loss_function = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    epoch_len = config.trainer.get("epoch_len")

    if 'steps' in config.lr_scheduler:
        config.lr_scheduler['steps'] = config.trainer.n_epochs * epoch_len
    if 'max_epochs' in config.lr_scheduler:
        config.lr_scheduler['max_epochs'] = config.trainer.n_epochs

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    trainer = RefinerTrainer(
        teacher_model=teacher_model,
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        log_batch_plots=config.trainer.get("log_batch_plots", False),
        view_3d_online=config.trainer.get("view_3d_online", False),
    )

    trainer.train()


if __name__ == "__main__":
    main()
