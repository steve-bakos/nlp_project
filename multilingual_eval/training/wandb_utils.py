import sys
import csv
import datasets
import traceback
import itertools
from typing import Callable, Any, Optional, List

from multilingual_eval.loggers import post_on_slack
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter


def wrap_train(
    train_fn: Callable[[dict, dict, Optional[StanfordSegmenter]], Any],
    sweep_config: dict,
    sweep_id: str,
    zh_segmenter: Optional[StanfordSegmenter] = None,
    ts=None,
):
    import wandb

    datasets.disable_progress_bar()

    def train(config=None):
        with wandb.init():
            config = wandb.config
            run = wandb.run

            partial_config = {
                k: v for k, v in config.items() if k in list(sweep_config["parameters"].keys())
            }

            if ts is not None:
                post_on_slack(
                    f"Starting new run {run.name} (id: {run.id})\n\n"
                    + f"config:\n\n```\n{partial_config}\n```",
                    thread_ts=ts,
                )

            try:
                train_fn(
                    config,
                    sweep_config,
                    zh_segmenter,
                )
            except Exception as e:
                print(traceback.print_exc(), file=sys.stderr)

                post_on_slack(
                    (f"Run from sweep: {sweep_id} failed.\n\n" if ts is None else "")
                    + f"Run: {run.name} (id: {run.id})\n\n"
                    + f"Trace:\n\n```\n{traceback.format_exc()}```",
                    thread_ts=ts,
                )

                raise e

    return train


def imitate_wandb_sweep(sweep_config):
    if sweep_config["method"] != "grid":
        raise NotImplementedError(
            f"function `imitate_wandb_sweep` was only implemented for method 'grid', got {sweep_config['method']}"
        )

    fixed_keys = list(sweep_config["parameters"].keys())

    for values in itertools.product(
        *map(lambda x: sweep_config["parameters"][x]["values"], fixed_keys)
    ):
        yield dict(zip(fixed_keys, values))


def store_dicts_in_csv(fname: str, infos: List[dict]):
    if len(infos) == 0:
        return
    column_names = set(infos[0].keys()).union(*map(lambda x: x.keys(), infos))

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(column_names)
        for info in infos:
            writer.writerow([str(info.get(c, "")) for c in column_names])
