import torch
import logging
import numpy as np
import os

from multilingual_eval.datasets.realignment_task import (
    get_realignment_dataset_for_one_pair,
    get_multiparallel_realignment_dataset,
)
from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset
from multilingual_eval.datasets.collators import RealignmentCollator
from multilingual_eval.training.utils import bring_batch_to_model
from multilingual_eval.models.utils import remove_batch_dimension, sum_ranges_and_put_together
from multilingual_eval.retrieval import evaluate_alignment_with_cosim


def evaluate_alignment(
    tokenizer,
    model,
    alignment_dataset,
    nb_pairs=5000,
    batch_size=4,
    model_back_to_cpu=False,
    device_for_search="cpu:0",
    strong_alignment=False,
    layers=None,
    boostrap=False,
    boostrap_p=0.8,
    seed=None,
):
    n_dim = model.config.hidden_size
    layers = layers or [-1]

    if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
        alignment_dataset = TorchCompatibleIterableDataset(alignment_dataset)

    dataloader = torch.utils.data.DataLoader(
        alignment_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=RealignmentCollator(tokenizer),
    )

    left_embs = torch.zeros((len(layers), nb_pairs, n_dim))
    right_embs = torch.zeros((len(layers), nb_pairs, n_dim))

    alignment_found = 0

    for batch in dataloader:
        left_batch = {k.split("_", 1)[1]: v for k, v in batch.items() if k.startswith("left_")}
        right_batch = {k.split("_", 1)[1]: v for k, v in batch.items() if k.startswith("right_")}

        left_batch = bring_batch_to_model(left_batch, model)
        left_output = model(**left_batch, output_hidden_states=True).hidden_states
        right_batch = bring_batch_to_model(right_batch, model)
        right_output = model(**right_batch, output_hidden_states=True).hidden_states

        end = min(alignment_found + np.sum(batch["alignment_nb"].numpy()), left_embs.shape[1])

        for i, layer in enumerate(layers):
            left_reprs = left_output[layer]
            right_reprs = right_output[layer]

            aligned_left_repr = sum_ranges_and_put_together(
                left_reprs,
                batch["alignment_left_positions"],
                ids=batch["alignment_left_ids"],
            )
            aligned_right_repr = sum_ranges_and_put_together(
                right_reprs,
                batch["alignment_right_positions"],
                ids=batch["alignment_right_ids"],
            )

            aligned_left_repr = (
                remove_batch_dimension(aligned_left_repr, batch["alignment_nb"]).detach().cpu()
            )
            aligned_right_repr = (
                remove_batch_dimension(aligned_right_repr, batch["alignment_nb"]).detach().cpu()
            )

            left_embs[i, alignment_found:end] = aligned_left_repr[: end - alignment_found]
            right_embs[i, alignment_found:end] = aligned_right_repr[: end - alignment_found]

        alignment_found = end

        if alignment_found == left_embs.shape[1]:
            break

    if model_back_to_cpu:
        model = model.cpu()

    if not boostrap:
        ids = None
    else:
        if seed:
            np.random.seed(seed)
        ids = np.random.choice(
            list(range(alignment_found)),
            size=(int(boostrap_p * alignment_found),),
            replace=False,
        )
        ids = torch.from_numpy(ids)

    scores_fwd = []
    scores_bwd = []

    logging.info(f"Alignment found: {alignment_found}")

    for i in range(len(layers)):
        scores_fwd.append(
            evaluate_alignment_with_cosim(
                left_embs[i][:alignment_found],
                right_embs[i][:alignment_found],
                device=device_for_search,
                strong_alignment=strong_alignment,
            )
        )
        scores_bwd.append(
            evaluate_alignment_with_cosim(
                right_embs[i][:alignment_found],
                left_embs[i][:alignment_found],
                device=device_for_search,
                strong_alignment=strong_alignment,
            )
        )

    return scores_fwd, scores_bwd


def evaluate_alignment_for_pairs(
    tokenizer,
    model,
    translation_path,
    alignment_path,
    lang_pairs,
    max_length=None,
    nb_pairs=5000,
    batch_size=4,
    model_back_to_cpu=False,
    device_for_search="cpu:0",
    strong_alignment=False,
    seed=None,
    split="train",
    layers=None,
):
    scores_fwd = []
    scores_bwd = []
    device_before = model.device
    for left_lang, right_lang in lang_pairs:
        dataset = get_realignment_dataset_for_one_pair(
            tokenizer,
            os.path.join(translation_path, f"{left_lang}-{right_lang}.tokenized.{split}.txt"),
            os.path.join(alignment_path, f"{left_lang}-{right_lang}.{split}"),
            max_length=max_length,
            seed=seed,
            left_id=None,
            right_id=None,
        )

        fwd, bwd = evaluate_alignment(
            tokenizer,
            model,
            dataset,
            nb_pairs=nb_pairs,
            batch_size=batch_size,
            model_back_to_cpu=model_back_to_cpu,
            device_for_search=device_for_search,
            strong_alignment=strong_alignment,
            layers=layers,
        )

        scores_fwd.append(fwd)
        scores_bwd.append(bwd)

        if model_back_to_cpu:
            model = model.to(device_before)

    return scores_fwd, scores_bwd


def evaluate_multiparallel_alignment(
    tokenizer,
    model,
    translation_path,
    alignment_path,
    lang_pairs,
    max_length=None,
    nb_pairs=5000,
    batch_size=4,
    model_back_to_cpu=False,
    device_for_search="cpu:0",
    strong_alignment=False,
    seed=None,
    split="train",
    layers=None,
    boostrap=False,
    boostrap_p=0.8,
):
    assert (
        len(set(map(lambda x: x[0], lang_pairs))) == 1
    ), f"evaluate_multiparallel_alignment accepts for lang_pairs only pairs involving the same left language (pivot) got {len(len(set(map(lambda x: x[0], lang_pairs))))} differents left lang"

    left_lang = lang_pairs[0][0]
    right_langs = list(map(lambda x: x[1], lang_pairs))

    datasets = get_multiparallel_realignment_dataset(
        tokenizer,
        [
            os.path.join(translation_path, f"{left_lang}-{right_lang}.tokenized.{split}.txt")
            for right_lang in right_langs
        ],
        [
            os.path.join(alignment_path, f"{left_lang}-{right_lang}.{split}")
            for right_lang in right_langs
        ],
        max_length=max_length,
        seed=seed,
        left_id=None,
        right_id=None,
    )

    scores_fwd = []
    scores_bwd = []
    device_before = model.device
    for dataset in datasets:
        fwd, bwd = evaluate_alignment(
            tokenizer,
            model,
            dataset,
            nb_pairs=nb_pairs,
            batch_size=batch_size,
            model_back_to_cpu=model_back_to_cpu,
            device_for_search=device_for_search,
            strong_alignment=strong_alignment,
            layers=layers,
            seed=seed,
            boostrap=boostrap,
            boostrap_p=boostrap_p,
        )

        scores_fwd.append(fwd)
        scores_bwd.append(bwd)

        if model_back_to_cpu:
            model = model.to(device_before)

    return scores_fwd, scores_bwd
