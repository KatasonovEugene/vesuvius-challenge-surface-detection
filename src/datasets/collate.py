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
        if dataset_items[0]['loss_weights'] is not None:
            result_batch['loss_weights'] = torch.cat([elem['loss_weights'] for elem in dataset_items])
    return result_batch
