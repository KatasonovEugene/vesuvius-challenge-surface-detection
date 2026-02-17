python3 inference.py \
    model=nnunet \
    inferencer.from_pretrained="models/patch192_pseudotarget.pth" \
    'model.roi_size=[64, 64, 64]' \
    $@
