"""
Script to plot cross-lingual generalization versus alignment across several languages, models and seeds
It requires the csv output of scripts/2023_acl/finetuning_and_alignment.py
"""


import os
import sys
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brokenaxes import brokenaxes
from matplotlib.markers import MarkerStyle
from transformers import AutoModel

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE
from multilingual_eval.utils import get_nb_layers

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files containing results of scripts/2023_acl/finetuning_and_alignment.py",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="path of the output file for the figure",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="xnli",
        help="Fine-tuning task to use (udpos, wikiann or xnli)",
    )
    parser.add_argument(
        "--layer",
        default=-1,
        help="Positive or negative index of layer (-1 indicates last)",
    )
    parser.add_argument(
        "--moment",
        choices=["before", "after"],
        default="after",
        help="Whether use alignment measured before or after fine-tuning",
    )
    parser.add_argument(
        "--direction",
        choices=["fwd", "bwd"],
        default="fwd",
        help="Whether to use source-target alignment (fwd) or target-source (bwd)",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=["ar", "es", "fr", "ru", "zh"],
        help="List of target languages",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models and tokenizers",
    )
    args = parser.parse_args()

    task = args.task
    layer = args.layer
    moment = args.moment
    direction = args.direction
    langs = args.langs

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_files])

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    model_to_nlayer = {}

    models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R Base",
        "xlm-roberta-large": "XLM-R Large",
    }

    model_in_df = set(df["model"].unique())

    models = {key: value for key, value in models.items() if key in model_in_df}

    for model_name in models:
        model = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir)
        n_layers = get_nb_layers(model)
        model_to_nlayer[model_name] = n_layers

    if layer < 0:
        available_models = list(
            filter(lambda x: model_to_nlayer[x] >= -layer, models.keys())
        )
    else:
        available_models = list(
            filter(lambda x: model_to_nlayer[x] > layer, models.keys())
        )

    line_for_models = []
    line_for_langs = []

    for j, model in enumerate(available_models):
        model_name = models[model]
        subdf = df[(df.model == model) & (df.task == task)]
        if layer < 0:
            model_layer = model_to_nlayer[model] + layer
        else:
            model_layer = layer

        line_for_models.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=MarkerStyle.filled_markers[j],
                linestyle="",
            )
        )

        for i, lang in enumerate(langs):
            alignment_scores = list(
                subdf[f"alignment_{moment}_{direction}_{lang}_{model_layer}"]
            )
            delta_scores = list(
                (
                    subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                    - subdf[f"final_eval_same_{task_to_metric[task]}"]
                )
                / subdf[f"final_eval_same_{task_to_metric[task]}"]
            )
            (line,) = plt.plot(
                alignment_scores,
                delta_scores,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                marker=MarkerStyle.filled_markers[j],
                linestyle=""
                # label=f"{model_name} {lang}",
            )
            if j == 0:
                line_for_langs.append(line)

    legend1 = plt.legend(
        line_for_langs,
        langs,
        loc="lower right",
        bbox_to_anchor=(0.80, 0.05, 0.15, 0.30),
    )
    plt.legend(
        line_for_models,
        list(map(models.__getitem__, available_models)),
        loc="lower right",
        bbox_to_anchor=(0.50, 0.05, 0.25, 0.30),
    )
    plt.gca().add_artist(legend1)
    plt.xlabel("alignment score")
    plt.ylabel("cross-lingual generalization")
    plt.savefig(args.output_file or f"{layer}_{moment}_fwd.pdf")
