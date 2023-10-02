import logging
import numpy as np

from multilingual_eval.contextualized_pairs import (
    compute_pair_representations,
    generate_pairs,
)
from multilingual_eval.retrieval import evaluate_alignment_with_cosim
from multilingual_eval.utils import get_nb_layers, get_tokenizer_type


def select_pairs_for_evaluation(
    tokenizer,
    translation_dataset,
    dico_path,
    left_lang,
    right_lang,
    nb_selected=5000,
    max_length=512,
    dict_tuple=None,
    split="all",
    ignore_not_enough=False,
):
    def remove_bad_samples(example):
        if example[left_lang] is None:
            return False
        if example[right_lang] is None:
            return False

        left_subwords = tokenizer.tokenize(example[left_lang])
        right_subwords = tokenizer.tokenize(example[right_lang])

        if len(left_subwords) > max_length or len(right_subwords) > max_length:
            return False
        if (
            len(list(filter(lambda x: x != "[UNK]", left_subwords))) == 0
            or len(list(filter(lambda x: x != "[UNK]", right_subwords))) == 0
        ):
            return False
        return True

    translation_dataset = translation_dataset.filter(remove_bad_samples)

    pairs = generate_pairs(
        translation_dataset,
        dico_path,
        left_lang=left_lang,
        right_lang=right_lang,
        avoid_repetition=False,
        max_pairs=nb_selected * 50,
        dict_tuple=dict_tuple,
        split=split,
    )

    if len(pairs) < nb_selected and not ignore_not_enough:
        raise Exception(f"Not enough pair with pair {left_lang}-{right_lang}")
    elif len(pairs) < nb_selected:
        logging.warning(
            f"Not enough pair with pair {left_lang}-{right_lang} ({len(pairs)} instead of {nb_selected})"
        )

    if nb_selected < len(pairs):
        selected_pairs = np.random.choice(pairs, size=(nb_selected,), replace=False)
    else:
        selected_pairs = pairs
    return selected_pairs


def evaluate_alignment_on_pairs(
    model,
    tokenizer,
    selected_pairs,
    left_lang=None,
    right_lang=None,
    left_lang_id=None,
    right_lang_id=None,
    batch_size=2,
    device="cpu:0",
    strong_alignment=False,
    move_model_back_to_cpu=True,
):
    left_embs, right_embs = compute_pair_representations(
        model,
        tokenizer,
        selected_pairs,
        batch_size=batch_size,
        dim_size=model.config.hidden_size,
        n_layers=get_nb_layers(model),
        device=device,
        left_lang=left_lang,
        right_lang=right_lang,
        left_lang_id=left_lang_id,
        right_lang_id=right_lang_id,
        split_type=get_tokenizer_type(tokenizer),
    )

    if move_model_back_to_cpu:
        model = model.cpu()

    res = []
    for layer in range(get_nb_layers(model)):
        score = evaluate_alignment_with_cosim(
            left_embs[layer],
            right_embs[layer],
            device=device,
            strong_alignment=strong_alignment,
        )
        res.append(score)
    return res
