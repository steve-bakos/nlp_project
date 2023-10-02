"""
Evaluate the alignment of representation before and after fine-tuning
and perform controlled experiment with sequential and joint realignment 
and different aligners
"""
import os
import sys
import torch
import logging
import datasets
from typing import List
from transformers import AutoTokenizer, set_seed

sys.path.append(os.curdir)


from multilingual_eval.seeds import seeds
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.training.wandb_utils import (
    wrap_train,
    imitate_wandb_sweep,
    store_dicts_in_csv,
)
from multilingual_eval.utils import get_nb_layers
from multilingual_eval.loggers import (
    WandbResultStore,
    DictResultStore,
    DefaultResultStore,
)
from multilingual_eval.datasets.dispatch_datasets import (
    get_dataset_fn,
    get_dataset_metric_fn,
    model_fn,
    model_fn_from_scratch,
    collator_fn,
)
from multilingual_eval.training.training_loops import realignment_training_loop
from multilingual_eval.training.batch_sizes import get_batch_size
from multilingual_eval.training.alignment_evaluation_loops import (
    evaluate_alignment_for_pairs,
    evaluate_multiparallel_alignment,
)

logging.basicConfig(level=logging.INFO)


def train(
    left_lang: str,
    right_langs: List[str],
    translation_dir: str,
    alignment_dir: str,
    config=None,
    sweep_config=None,
    zh_segmenter=None,
    debug=False,
    cache_dir=None,
    large_gpu=False,
    eval_layers=None,
    strong_alignment=False,
    multiparallel=False,
    n_epochs=5,
    tokenizers=None,
    from_scratch=False,
    result_store=None,
):
    model_name = config["model"]
    task_name = config["task"]
    seed = config["seed"]

    # result_store allows to gather information along the experiment
    # By default, only logs them in the console
    result_store = result_store or DefaultResultStore()

    # Compute batch size and gradient accumulation from real batch size (32)
    # and an empirical batch size based on the model name
    cumul_batch_size = 32
    batch_size = get_batch_size(model_name, cumul_batch_size, large_gpu=large_gpu)
    accumulation_steps = cumul_batch_size // batch_size

    assert cumul_batch_size % batch_size == 0

    # Compute caching directory for HuggingFace datasets and models
    data_cache_dir = (
        os.path.join(cache_dir, "datasets") if cache_dir is not None else cache_dir
    )
    model_cache_dir = (
        os.path.join(cache_dir, "transformers") if cache_dir is not None else cache_dir
    )

    # Instantiate tokenizer, model and set the seed
    if tokenizers is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    else:
        tokenizer_name = tokenizers[
            sweep_config["parameters"]["model"]["values"].index(model_name)
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=model_cache_dir
        )
    set_seed(seed)
    if from_scratch:
        model = model_fn_from_scratch(task_name, with_realignment=False)(
            model_name, cache_dir=model_cache_dir
        )
    else:
        model = model_fn(task_name, with_realignment=False)(
            model_name, cache_dir=model_cache_dir
        )

    n_layers = get_nb_layers(model)

    eval_layers = eval_layers or range(n_layers)

    # Load fine-tuning dataset
    training_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="train",
        limit=1000 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    # Load test dataset for target languages
    validation_datasets = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        right_langs,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
        interleave=False,
    )

    # Load test dataset for source language
    source_validation_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    # Load realignment datatset
    # With an option to use multiparallel datasets (dataset where the same source
    # sentence is translated in all the target languages)
    # multiparallel option was not used in the scope of the ACL Findings paper
    evaluation_fn = (
        evaluate_multiparallel_alignment
        if multiparallel
        else evaluate_alignment_for_pairs
    )

    # Evaluate alignment before
    model.to("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
    scores_before_fwd, score_before_bwd = evaluation_fn(
        tokenizer,
        model,
        translation_dir,
        alignment_dir,
        [(left_lang, right_lang) for right_lang in right_langs],
        batch_size=batch_size,
        device_for_search=f"cuda:{torch.cuda.device_count()-1}"
        if torch.cuda.device_count() > 0
        else "cpu",
        strong_alignment=strong_alignment,
        seed=42,
        max_length=128,
        layers=eval_layers,
        nb_pairs=200 if debug else 1000,
    )
    model.to("cpu")

    res = {}
    for right_lang, scores_fwd, scores_bwd in zip(
        right_langs, scores_before_fwd, score_before_bwd
    ):
        for layer, fwd, bwd in zip(eval_layers, scores_fwd, scores_bwd):
            res[f"alignment_before_fwd_{right_lang}_{layer}"] = fwd
            res[f"alignment_before_bwd_{right_lang}_{layer}"] = bwd
    result_store.log(res)

    # perform fine-tuning (realignment_dataset is set to None and strategy is baseline, so no realignment)
    realignment_training_loop(
        tokenizer,
        model,
        training_dataset,
        None,
        strategy="baseline",
        evaluation_datasets=validation_datasets if task_name not in ["xquad"] else None,
        same_language_evaluation_dataset=source_validation_dataset
        if task_name not in ["xquad"]
        else None,
        evaluation_prefixes=right_langs,
        seed=seed,
        task_batch_size=batch_size,
        n_epochs=n_epochs,
        accumulation_steps=accumulation_steps,
        result_store=result_store,
        metric_fn=get_dataset_metric_fn(task_name)(),
        data_collator=collator_fn(task_name)(tokenizer),
        learning_rate=7.5e-6 if "roberta" in model_name else 2e-5,
    )

    # Evaluate alignment after fine-tuning
    model.to("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
    scores_after_fwd, score_after_bwd = evaluation_fn(
        tokenizer,
        model,
        translation_dir,
        alignment_dir,
        [(left_lang, right_lang) for right_lang in right_langs],
        batch_size=batch_size,
        device_for_search="cuda:0" if torch.cuda.device_count() > 0 else "cpu",
        strong_alignment=strong_alignment,
        seed=42,
        layers=eval_layers,
        nb_pairs=200 if debug else 1000,
    )
    model.to("cpu")

    res = {}
    for right_lang, scores_fwd, scores_bwd in zip(
        right_langs, scores_after_fwd, score_after_bwd
    ):
        for layer, fwd, bwd in zip(eval_layers, scores_fwd, scores_bwd):
            res[f"alignment_after_fwd_{right_lang}_{layer}"] = fwd
            res[f"alignment_after_bwd_{right_lang}_{layer}"] = bwd
    result_store.log(res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "translation_dir",
        type=str,
        help="Directory where the parallel dataset can be found, expecting to contain files of the form {source_lang}-{target_lang}.tokenized.train.txt",
    )
    parser.add_argument(
        "alignment_dir",
        type=str,
        help="Directory where the alignment pairs can be found, expecting to contain files of the form {source_lang}-{target_lang}.train",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=[
            "distilbert-base-multilingual-cased",
            "bert-base-multilingual-cased",
            "xlm-roberta-base",
            "xlm-roberta-large",
        ],
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        type=str,
        default=None,
        help="List of tokenizer slugs that can be provided to override --models when instantiating with AutoTokenizer",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, default=["wikiann", "udpos", "xnli"]
    )
    parser.add_argument("--left_lang", type=str, default="en", help="Source language")
    parser.add_argument(
        "--right_langs",
        type=str,
        nargs="+",
        default=["ar", "es", "fr", "ru", "zh"],
        help="Target languages",
    )
    parser.add_argument(
        "--eval_layers",
        type=int,
        nargs="+",
        default=None,
        help="List of (indexed) layers of which the alignment must be evaluated (default: all layers)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory which will contain subdirectories 'transformers' and 'datasets' for caching HuggingFace models and datasets",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="If using wandb, useful to restart a sweep or launch several run in parallel for a same sweep",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="Use this to perform a quicker test run with less samples",
    )
    parser.add_argument(
        "--large_gpu",
        action="store_true",
        dest="large_gpu",
        help="Use this option for 45GB GPUs (less gradient accumulation needed)",
    )
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument(
        "--strong_alignment",
        action="store_true",
        dest="strong_alignment",
        help="Option to use strong alignment instead of weak for evaluation",
    )
    parser.add_argument(
        "--multiparallel",
        action="store_true",
        dest="multiparallel",
        help="Option to use when the translation dataset is multiparallel (not used in the paper)",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        dest="from_scratch",
        help="Option to use when we want to instantiate the models with random weights instead of pre-trained ones",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The path to the output CSV file containing results (used only if wandb is not use, which is the case by default)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        dest="use_wandb",
        help="Use this option to use wandb (but must be installed first)",
    )
    parser.set_defaults(
        debug=False,
        large_gpu=False,
        strong_alignemnt=False,
        multiparallel=False,
        from_scratch=False,
        use_wandb=False,
    )
    args = parser.parse_args()

    if not args.use_wandb and args.output_file is None:
        raise Exception(
            f"Either wandb must be used (--use_wandb) or an output csv file must be set (--output_file) to store results"
        )
    if not args.use_wandb:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": seeds[: args.n_seeds]},
            "model": {"values": args.models},
            "task": {"values": args.tasks},
        },
    }

    if args.debug:
        sweep_config["parameters"]["seed"]["values"] = sweep_config["parameters"][
            "seed"
        ]["values"][:1]

    with StanfordSegmenter() as zh_segmenter:  # Calls Stanford Segmenter in another process, hence the context manager
        if args.use_wandb:
            import wandb

            result_store = WandbResultStore()

            if args.sweep_id is None:
                sweep_id = wandb.sweep(sweep_config, project="finetuning_and_alignment")
            else:
                sweep_id = args.sweep_id

            final_train_fn = wrap_train(
                lambda cfg, sweep_cfg, zh_sgm: train(
                    args.left_lang,
                    args.right_langs,
                    args.translation_dir,
                    args.alignment_dir,
                    config=cfg,
                    sweep_config=sweep_cfg,
                    zh_segmenter=zh_sgm,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    eval_layers=args.eval_layers,
                    strong_alignment=args.strong_alignment,
                    multiparallel=args.multiparallel,
                    n_epochs=args.n_epochs,
                    tokenizers=args.tokenizers,
                    from_scratch=args.from_scratch,
                    result_store=result_store,
                ),
                sweep_config,
                sweep_id,
                zh_segmenter=zh_segmenter,
                # ts=ts,
            )

            wandb.agent(sweep_id, final_train_fn, project="finetuning_and_alignment")
        else:
            datasets.disable_progress_bar()
            results = []
            # Looping over all possible configuration of runs provided in sweep_config
            for run_config in imitate_wandb_sweep(sweep_config):
                result_store = DictResultStore()
                result_store.log(run_config)
                train(
                    args.left_lang,
                    args.right_langs,
                    args.translation_dir,
                    args.alignment_dir,
                    config=run_config,
                    sweep_config=sweep_config,
                    zh_segmenter=zh_segmenter,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    eval_layers=args.eval_layers,
                    strong_alignment=args.strong_alignment,
                    multiparallel=args.multiparallel,
                    n_epochs=args.n_epochs,
                    tokenizers=args.tokenizers,
                    from_scratch=args.from_scratch,
                    result_store=result_store,
                )
                results.append(result_store.get_results())

            store_dicts_in_csv(args.output_file, results)
