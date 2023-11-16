"""
Script to compare different realignment methods with simple fine-tuning
"""


import os
import sys
import logging
import datasets
from typing import List
from transformers import AutoTokenizer, set_seed

sys.path.append(os.curdir)

from multilingual_eval.seeds import seeds
from multilingual_eval.loggers import (
    WandbResultStore,
    DictResultStore,
    DefaultResultStore,
)
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.training.wandb_utils import (
    wrap_train,
    imitate_wandb_sweep,
    store_dicts_in_csv,
)
from multilingual_eval.training.training_loops import realignment_training_loop
from multilingual_eval.training.batch_sizes import get_batch_size
from multilingual_eval.datasets.dispatch_datasets import (
    get_dataset_fn,
    get_dataset_metric_fn,
    model_fn,
    collator_fn,
)
from multilingual_eval.datasets.realignment_task import (
    get_multilingual_realignment_dataset,
)


def train(
    left_lang: str,
    right_langs: List[str],
    translation_dir: str,
    fastalign_dir: str,
    dico_dir: str,
    awesome_dir: str,
    config=None,
    sweep_config=None,
    zh_segmenter=None,
    debug=False,
    cache_dir=None,
    large_gpu=False,
    n_epochs=5,
    layers=None,
    result_store=None,
    aligner=None
):

    print('Inside train...')

    layers = layers or [-1]
    model_name = config["model"]
    task_name = config["task"]
    seed = config["seed"]
    method = config["method"]
    # if method == "baseline":
    #     aligner = None
    # else:
    #     # method, aligner = method.split("_")
    #     aligner = method.split("_")[-1]
    #     method = "_".join(x for x in method.split("_")[:-1])

    print(f'METHOD : {method}')
    print(f'ALIGNER: {aligner}')

    # result_store allows to gather information along the experiment
    # By default, only logs them in the console
    result_store = result_store or DefaultResultStore()

    # Compute batch size and gradient accumulation from real batch size (32)
    # and an empirical batch size based on the model name
    cumul_batch_size = 32
    batch_size = get_batch_size(model_name, cumul_batch_size, large_gpu=large_gpu)
    accumulation_steps = cumul_batch_size // batch_size

    realignment_batch_size = 16

    assert cumul_batch_size % batch_size == 0

    # Compute caching directory for HuggingFace datasets and models
    data_cache_dir = (
        os.path.join(cache_dir, "datasets") if cache_dir is not None else cache_dir
    )
    model_cache_dir = (
        os.path.join(cache_dir, "transformers") if cache_dir is not None else cache_dir
    )

    # Instantiate tokenizer, model and set the seed
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    set_seed(seed)
    if method == "baseline":
        model = model_fn(task_name, with_realignment=False)(
            model_name, cache_dir=model_cache_dir
        )
    else:
        model = model_fn(task_name, with_realignment=True)(
            model_name,
            cache_dir=model_cache_dir,
            nb_pairs=len(right_langs),
            strong_alignment=True,
            realignment_loss="contrastive",
            with_mapping=False,
            regularization_to_init=False,
            realignment_layers=layers,
        )

    logging.debug(model)

    # Load fine-tuning dataset
    training_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="train",
        limit=1000 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    # print()
    # print('fine-tuning dataset')
    # print(training_dataset[:5])
    # print()

    # Load test dataset for target languages
    validation_datasets = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        right_langs,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
        interleave=False,
    )

    # print()
    # print('test dataset for target languages')
    # for dataset in validation_datasets:
    #     print(dataset[:5])
    # print()

    # Load test dataset for source language
    source_validation_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
        left_lang,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
    )

    # print()
    # print('test dataset for source language')
    # print(source_validation_dataset[:5])
    # print()

    # Load realignment datatset
    lang_pairs = [(left_lang, right_lang) for right_lang in right_langs]
    if aligner == "fastalign":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            fastalign_dir,
            lang_pairs,
            # lang_to_id={'en':0, 'ar':1, 'es':2, 'fr':3, 'ru':4, 'zh':5},
            max_length=96,
            seed=seed,
        )
    elif aligner == "dico":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, dico_dir, lang_pairs, max_length=96, seed=seed
        )
    elif aligner == "awesome":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            awesome_dir,
            lang_pairs,
            max_length=96,
            seed=seed,
        )
    elif aligner is None:
        alignment_dataset = None
    else:
        raise KeyError(aligner)

    # perform realignment and fine-tuning
    realignment_training_loop(
        tokenizer,
        model,
        training_dataset,
        alignment_dataset,
        strategy=method,
        evaluation_datasets=validation_datasets if task_name not in ["xquad"] else None,
        same_language_evaluation_dataset=source_validation_dataset
        if task_name not in ["xquad"]
        else None,
        evaluation_prefixes=right_langs,
        seed=seed,
        task_batch_size=batch_size,
        learning_rate=7.5e-6 if "roberta" in model_name else 2e-5,
        realignment_batch_size=realignment_batch_size,
        realignment_steps_by_finetuning=1,
        n_epochs=n_epochs,
        accumulation_steps=accumulation_steps,
        result_store=result_store,
        metric_fn=get_dataset_metric_fn(task_name)(),
        data_collator=collator_fn(task_name)(tokenizer),
        model_name=model_name
    )


def rl_main(config):
    logging.basicConfig(level=logging.INFO)

    # Ensure necessary directories are created
    if not config.get('use_wandb') and config.get('output_file'):
        os.makedirs(os.path.dirname(config['output_file']), exist_ok=True)

    # Prepare the sweep configuration
    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": seeds[:config.get('n_seeds', 5)]},
            "model": {"values": [config.get('models', [])]},
            "task": {"values": [config.get('tasks', [])]},
            "method": {"values": [config.get('strategies', [])]},
        },
    }

    print()
    print(f'SWEEP CONFIG: {sweep_config}')
    print()

    if config.get('debug'):
        sweep_config["parameters"]["seed"]["values"] = sweep_config["parameters"]["seed"]["values"][:1]

    with StanfordSegmenter() as zh_segmenter:
        if config.get('use_wandb'):
            import wandb
            result_store = WandbResultStore()

            # Determine the project and sweep ID
            project = config.get('project')
            sweep_id = config.get('sweep_id') or wandb.sweep(sweep_config, project=project)

            final_train_fn = wrap_train(
                lambda cfg, sweep_cfg, zh_sgm: train(
                    config['left_lang'],
                    config['right_langs'],
                    config['translation_dir'],
                    config['fastalign_dir'],
                    config['dico_dir'],
                    config['awesome_dir'],
                    layers=config.get('layers', [-1]),
                    config=cfg,
                    sweep_config=sweep_cfg,
                    zh_segmenter=zh_sgm,
                    debug=config.get('debug', False),
                    large_gpu=config.get('large_gpu', False),
                    cache_dir=config.get('cache_dir'),
                    n_epochs=config.get('n_epochs', 5),
                    result_store=result_store,
                    aligner=config['aligners']
                ),
                sweep_config,
                sweep_id,
                zh_segmenter=zh_segmenter,
            )

            wandb.agent(sweep_id, final_train_fn, project=project)
        else:
            datasets.disable_progress_bar()
            results = []

            print('Running without WandB...')

            for run_config in imitate_wandb_sweep(sweep_config):
                print()
                print(f'RUN CONFIG: {run_config}')
                print()
                result_store = DictResultStore()
                result_store.log(run_config)
                train(
                    config['left_lang'],
                    config['right_langs'],
                    config['translation_dir'],
                    config['fastalign_dir'],
                    config['dico_dir'],
                    config['awesome_dir'],
                    layers=config.get('layers', [-1]),
                    config=run_config,
                    sweep_config=sweep_config,
                    zh_segmenter=zh_segmenter,
                    debug=config.get('debug', False),
                    large_gpu=config.get('large_gpu', False),
                    cache_dir=config.get('cache_dir'),
                    n_epochs=config.get('n_epochs', 5),
                    result_store=result_store,
                    aligner=config['aligners'],
                )
                results.append(result_store.get_results())

            store_dicts_in_csv(config['output_file'], results)