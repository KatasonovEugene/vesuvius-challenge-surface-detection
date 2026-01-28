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
    result_batch['volume'] = torch.stack([elem['volume'] for elem in dataset_items])
    if 'target' in dataset_items[0]:
        result_batch['target'] = torch.stack([elem['volume'] for elem in dataset_items]).unsqueeze(-1)
    return result_batch
