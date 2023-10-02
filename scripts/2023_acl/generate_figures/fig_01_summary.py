"""
Script to generate Figure 1 of paper
It requires the csv ouput of scripts/2023_acl/controlled_realignment.py
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brokenaxes import brokenaxes

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files containing results of scripts/2023_acl/controlled_realignment.py",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="path of the output file for the figure",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["udpos", "xnli"],
        help="List of fine-tuning tasks to stack on the figure, (udpos, wikiann and/or xnli)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ar",
        help="Target language on which cross-lingual transfer is evaluated",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="before_dico",
        help="Realignment method compared with plain finetuning, before/during_fastalign/dico/awesome",
    )
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_files])

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI"}

    available_tasks = df["task"].unique()
    assert all(map(lambda x: x in available_tasks, args.tasks))

    tasks = args.tasks

    model_sizes = {
        "distilbert-base-multilingual-cased": 66,
        "bert-base-multilingual-cased": 110,
        "xlm-roberta-base": 125,
        "xlm-roberta-large": 345,
    }
    existing_models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R base",
        "xlm-roberta-large": "XLM-R large",
    }

    summary_lang = args.lang
    summary_method = args.method

    fig = plt.figure()
    print(len(tasks))
    print(available_tasks)

    gs = fig.add_gridspec(len(tasks), hspace=0)
    # _ = gs.subplots(sharex=True, sharey=True)
    for i, task in enumerate(tasks):
        ax = brokenaxes(
            xlims=((60 - 10, 125 + 10), (345 - 10, 345 + 8)), subplot_spec=gs[i]
        )

        ax.set(ylim=[0.0, 0.85])

        ax.text(
            90,
            0.4,
            task_to_name[task],
            fontsize=14,
            horizontalalignment="center",
        )

        for j, (model, size) in enumerate(model_sizes.items()):
            if i == 0:
                ax.text(
                    size - 4 if model == "xlm-roberta-large" else size,
                    0.75,
                    existing_models[model],
                    fontsize=11,
                    horizontalalignment="center",
                    verticalalignment="bottom"
                    if model == "xlm-roberta-base"
                    else "top",
                )

            values_without = df[
                (df.model == model) & (df.task == task) & (df.method == "baseline")
            ][f"final_eval_{summary_lang}_{task_to_metric[task]}"].dropna()
            values_with = df[
                (df.model == model) & (df.task == task) & (df.method == summary_method)
            ][f"final_eval_{summary_lang}_{task_to_metric[task]}"].dropna()

            ax.bar(
                [size],
                [np.mean(values_without)],
                width=-7,
                align="edge",
                yerr=[np.std(values_without)],
                color=COLOR_PALETTE[0],
                **({"label": "without"} if i == len(tasks) - 1 and j == 0 else {}),
            )
            ax.bar(
                [size],
                [np.mean(values_with)],
                width=7,
                align="edge",
                yerr=[np.std(values_with)],
                color=COLOR_PALETTE[1],
                **({"label": "with"} if i == len(tasks) - 1 and j == 0 else {}),
            )

        ax.label_outer()

        ax.set_ylabel(task_to_metric[task])

        if i == len(tasks) - 1:
            ax.legend(loc="lower left")
            ax.set_xlabel("model size (million parameters)")

    plt.savefig(args.output_file or "summary.pdf")
