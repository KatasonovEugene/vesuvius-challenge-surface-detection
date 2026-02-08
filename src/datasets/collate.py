import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    result_batch['volume'] = torch.cat([elem['volume'] for elem in dataset_items])
    result_batch['image_id'] = [elem['image_id'] for elem in dataset_items]
    if 'gt_mask' in dataset_items[0]:
        result_batch['gt_mask'] = torch.cat([elem['gt_mask'] for elem in dataset_items])
        result_batch['gt_skel'] = torch.cat([elem['gt_skel'] for elem in dataset_items])
    if 'volume_struct' in dataset_items[0]:
        result_batch['volume_struct'] = torch.cat([elem['volume_struct'] for elem in dataset_items])
    if 'volume_sem1' in dataset_items[0]:
        result_batch['volume_sem1'] = torch.cat([elem['volume_sem1'] for elem in dataset_items])
        result_batch['volume_sem2'] = torch.cat([elem['volume_sem2'] for elem in dataset_items])
    return result_batch
