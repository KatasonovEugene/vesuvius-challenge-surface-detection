python3 train.py \
    trainer.override=True \
    dataloader.batch_size=2 \
    dataloader.num_workers=0 \
    datasets.train.override=True \
    datasets.val.override=True \
    datasets.train.val_size=0.5 \
    datasets.val.val_size=0.5 \
    dataloader=debug \
    instance_transforms=debug \
    trainer.n_epochs=1 \
    trainer.epoch_len=3 \
    $@
