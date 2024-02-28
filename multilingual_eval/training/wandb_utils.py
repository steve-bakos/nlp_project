import os
import sys
import csv
import datasets
import traceback
import itertools
import logging
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

class CSVRecorder:

    def __init__(self, fname: str, config_props = None):
        self.fname = fname
        self.config_props = config_props
        self.props = None
        self.file = None
        self.writer = None
        self.already_passed = set()
        self.logger = logging.getLogger('csv_recorder')

    def __enter__(self):
        possible_props = []
        config_props_ids = None
        if os.path.isfile(self.fname):
            self.logger.info(f"Result file {self.fname} already exists")
            with open(self.fname, "r") as f:
                reader = csv.reader(f, delimiter=',')
                line_iter = iter(reader)
                try:
                    possible_props = next(line_iter)
                    self.logger.debug(f"Header was parsed successfuly")
                except StopIteration:
                    pass
                if possible_props and self.config_props:
                    try:
                        config_props_ids = [possible_props.index(p) for p in self.config_props]
                    except ValueError:
                        config_props_ids = []
                if config_props_ids:
                    for line in line_iter:
                        self.already_passed.add(tuple(line[i] for i in config_props_ids))
            self.logger.info(f"Found {len(self.already_passed)} existing runs")
        if len(possible_props) > 0:
            self.file = open(self.fname, "a").__enter__()
            self.props = possible_props
        else:
            self.file = open(self.fname, "w").__enter__()
        self.writer = csv.writer(self.file, delimiter=",")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.__exit__(self, exc_type, exc_val, exc_tb)
        self.props = None
        self.file = None
        self.writer = None

    def is_already_passed(self, config: dict):
        if self.config_props:
            key = tuple(str(config[key]) for key in self.config_props)
            return key in self.already_passed
        return False

    def add(self, info: dict):
        if self.props is None:
            self.props = list(info.keys())
            self.writer.writerow(self.props)
        self.writer.writerow([str(info.get(c, "")) for c in self.props])
        self.file.flush()
        if self.config_props:
            key = tuple(info[key] for key in self.config_props)
            self.already_passed.add(key)