import torch
import random
import logging
import numpy as np

from transformers import DataCollatorForTokenClassification, AdapterSetup
from transformers.optimization import get_scheduler, AdamW
from transformers.adapters.composition import Stack

from multilingual_eval.training.epoch_loop import epoch_loop
from multilingual_eval.training.evaluation_loops import evaluate_token_classification


def finetuning_loop_with_adapters(
    tokenizer,
    model,
    task_dataset,
    lang,
    eval_datasets=None,
    eval_languages=None,
    n_epochs=2,
    batch_size=4,
    accumulation_steps=8,
    learning_rate=1e-4,
    logging_steps=None,
    log_in_wandb=False,
    collator=None,
    metric_fn=None,
    seed=None,
    num_workers=0,
    baseline=False,
    no_adapter_for=None,
):
    eval_datasets = eval_datasets or []
    eval_languages = eval_languages or []

    assert len(eval_datasets) == len(eval_languages)

    collator = collator or DataCollatorForTokenClassification(tokenizer)

    if log_in_wandb:
        import wandb

    # Put model to GPU if available
    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # Fix random seed for Pytorch and numpy
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        g = None
        seed_worker = None

    # Create dataloader for the fine-tuning task
    task_dataloader = torch.utils.data.DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
    )

    logging_steps = logging_steps or max(1, len(task_dataloader) // 100)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = get_scheduler(
    #     "linear",
    #     optimizer,
    #     num_warmup_steps=int(0.1 * len(task_dataloader) * 5),
    #     num_training_steps=len(task_dataloader) * 5,
    # )
    scheduler = None

    training_state = None

    model.train_adapter(["task"])

    if baseline:
        model.set_active_adapters("task")

    for i_epoch in range(n_epochs):

        logging.info(f"Starting epoch {i_epoch + 1} / {n_epochs}")

        if not baseline:
            model.set_active_adapters(
                Stack(f"{lang}_adapter", "task") if lang != no_adapter_for else "task"
            )

        training_state = epoch_loop(
            model,
            optimizer,
            scheduler=scheduler,
            task_dataloader=task_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            training_state=training_state,
        )

        for eval_lang, eval_dataset in zip(eval_languages, eval_datasets):
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=collator,
            )

            if not baseline:
                model.set_active_adapters(
                    Stack(f"{eval_lang}_adapter", "task") if eval_lang != no_adapter_for else "task"
                )
            res = evaluate_token_classification(
                model, eval_dataloader, prefix=f"eval_{eval_lang}", metric_fn=metric_fn
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
