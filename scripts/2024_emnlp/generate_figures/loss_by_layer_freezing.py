import os
import sys
import json
import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE, get_statistics, plot_with_statistics


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_layers", type=int, default=13)
    args = parser.parse_args()
    result_file = args.results
    n_layers = args.n_layers
    output_dir = args.output_dir

    assert os.path.isfile(result_file)

    if not os.path.isdir(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    methods = ["during_partial_freeze", "freeze_realign_unfreeze"]
    aligners = ["awesome", "fastalign", "dico"]

    langs = ["ar", "es", "fr", "zh", "ru"]

    df = pd.read_csv(
        result_file, converters={
            k: json.loads for k in [
                "finetuning_steps", 
                "realignment_steps", 
                "distinct_realignment_samples", 
                "repeated_realignment_samples",
                "train_loss",
                "task_loss",
                "realignment_loss",
                *(f"eval_{subset}_accuracy" for subset in langs + ["avg", "same"])
            ]
        }
    )

    print(df.columns)

    for method, aligner in itertools.product(methods, aligners):
        c_idx = 0
        for i in range(args.n_layers):

            subdf = df[df.method == f"{method}_0_{i}_{aligner}"]
            if len(subdf) == 0:
                continue
            
            
            task_loss = list(subdf["task_loss"])
            train_loss = list(subdf["train_loss"])

            task_loss = np.array(task_loss)
            train_loss = np.array(train_loss)

            print(task_loss.shape, train_loss.shape)
            if len(task_loss.shape) < 2 or len(train_loss.shape) < 2 or task_loss.shape[0] != train_loss.shape[0] or train_loss.shape[1] < task_loss.shape[1]:
                continue
            train_loss = train_loss[:,-task_loss.shape[1]:]

            realignment_loss = train_loss - task_loss

            plot_with_statistics(
                list(range(realignment_loss.shape[1])),
                *get_statistics(task_loss.T),
                color=COLOR_PALETTE[c_idx],
                legend=f"task loss {i}",
                alpha=0.01
            )
            plot_with_statistics(
                list(range(realignment_loss.shape[1])),
                *get_statistics(realignment_loss.T),
                color=COLOR_PALETTE[c_idx],
                legend=f"align loss {i}",
                linestyle="dashed",
                alpha=0.01
            )
            c_idx += 1
        plt.legend()
        plt.ylim((0,0.5))
        plt.xlabel("log steps")
        plt.ylabel("loss")
        plt.savefig(os.path.join(output_dir, f"{method}_{aligner}.png"))
        plt.clf()

        for subset in langs + ["avg", "same"]:
            c_idx = 0
            for i in range(args.n_layers):

                subdf = df[df.method == f"{method}_0_{i}_{aligner}"]
                if len(subdf) == 0:
                    continue

                eval_subset_accuracy = np.array(list(subdf[f"eval_{subset}_accuracy"]))

                plot_with_statistics(
                    list(range(eval_subset_accuracy.shape[1])),
                    *get_statistics(eval_subset_accuracy.T),
                    color=COLOR_PALETTE[c_idx],
                    legend=f"layer {i}",
                    alpha=0.1
                )

                c_idx += 1

            plt.legend()
            plt.xlabel("log steps")
            plt.ylabel("accuracy")
            plt.savefig(os.path.join(output_dir, f"{method}_{aligner}_{subset}_accuracy.png"))
            plt.clf()


