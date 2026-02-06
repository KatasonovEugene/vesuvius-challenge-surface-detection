python3 inference.py \
    instance_transforms=debug \
    tta_transforms=flips \
    'model.roi_size=[64, 64, 64]' \
    $@
