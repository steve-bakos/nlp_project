import logging
from torch.utils.data import DataLoader
import torch


def get_next_or_restart(dataloader: DataLoader, iterator, name=None):
    """
    Get next element of an iterable or re-create the iterable and starts over
    """
    name = name or str(dataloader)
    try:
        batch = next(iterator)
        restarted = False
    except StopIteration:
        logging.warning(f"Reached end of Dataloader {name}. Starting over")
        iterator = iter(dataloader)
        batch = next(iterator)
        restarted = True
    return iterator, batch, restarted


def bring_batch_to_model(batch, model):
    """
    Move all tensors from a batch to the same device as the model
    """
    return {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def prefix_dictionary(dictionary, prefix=None):
    """
    prefix all the keys of a given dictionary with a given prefix, does nothing
    if the prefix is None
    """
    if prefix is None:
        return dictionary
    return {f"{prefix}_{k}": v for k, v in dictionary.items()}
