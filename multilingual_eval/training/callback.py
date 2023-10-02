import logging
import torch
from multilingual_eval.datasets.translation_dataset import get_news_commentary, get_opus100
from multilingual_eval.evaluate_alignment import (
    evaluate_alignment_on_pairs,
    select_pairs_for_evaluation,
)


def get_alignment_eval_callback(
    tokenizer,
    dico_path,
    left_lang,
    right_lang,
    limit=500,
    dataset_name="news_commentary",
    cache_dir=None,
    max_length=512,
    split="eval",
    prefix=None,
    log_in_wandb=False,
    translation_kwargs=None,
    batch_size=2,
    left_lang_id=None,
    right_lang_id=None,
):
    """
    Callback for evaluating for a fine-tuning task
    on a given evaluation dataset
    """
    if log_in_wandb:
        import wandb
    translation_kwargs = translation_kwargs or {}

    prefix = prefix or split
    whole_prefix = f"{prefix}_{left_lang}_{right_lang}"

    if dataset_name == "news_commentary":
        dataset_getter = get_news_commentary
    elif dataset_name == "opus100":
        dataset_getter = get_opus100
    else:
        raise NotImplementedError(f"dataset_name `{dataset_name}` is not expected.")

    translation_dataset = dataset_getter(
        left_lang, right_lang, cache_dir=cache_dir, **translation_kwargs
    )

    pairs = select_pairs_for_evaluation(
        tokenizer,
        translation_dataset,
        dico_path,
        left_lang,
        right_lang,
        nb_selected=limit,
        max_length=max_length,
        split=split,
        ignore_not_enough=True,
    )

    def callback(model):
        score = evaluate_alignment_on_pairs(
            model,
            tokenizer,
            pairs,
            device=None,
            batch_size=batch_size,
            move_model_back_to_cpu=False,
            left_lang_id=left_lang_id,
            right_lang_id=right_lang_id,
        )
        res = {f"{whole_prefix}_nb_pairs": len(pairs), f"{whole_prefix}_bli_accuracy": score[-1]}
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        return res

    return callback


def get_averaged_alignment_callback(
    tokenizer,
    dico_path,
    left_lang,
    right_langs,
    limit=5000,
    dataset_name="news_commentary",
    cache_dir=None,
    max_length=512,
    split="eval",
    prefix=None,
    log_in_wandb=False,
    translation_kwargs=None,
    batch_size=2,
    lang_to_id=None,
):
    """
    Callback for performing multiple evaluation for different language and
    computing aggregate metrics
    """
    if log_in_wandb:
        import wandb
    prefix = prefix or split
    whole_prefix = f"{prefix}_{left_lang}_avg"
    callbacks_by_language = [
        get_alignment_eval_callback(
            tokenizer,
            dico_path,
            left_lang,
            elt,
            limit=limit,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            max_length=max_length,
            split=split,
            log_in_wandb=log_in_wandb,
            translation_kwargs=translation_kwargs,
            batch_size=batch_size,
            prefix=prefix,
            left_lang_id=lang_to_id[left_lang] if lang_to_id is not None else None,
            right_lang_id=lang_to_id[elt] if lang_to_id is not None else None,
        )
        for elt in right_langs
    ]

    def callback(model):
        results = [
            f(model)[f"{prefix}_{left_lang}_{lang}_bli_accuracy"]
            for f, lang in zip(callbacks_by_language, right_langs)
        ]
        avg = sum(results) / max(1, len(results))
        res = {f"{whole_prefix}_bli_accuracy": avg}
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        return res

    return callback


def orthogonalize_callback(model):
    """
    Callback for peforming the orthogonalizing step of the mapping
    of a model (throws an error if it has no mapping)
    """
    model.orthogonalize()

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0

    def __call__(self, current_val_loss):
        if self.best_val_loss - current_val_loss > self.min_delta:
            self.best_val_loss = current_val_loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        if self.no_improvement_epochs >= self.patience:
            return True  # Trigger early stopping
        return False
