"""
Script to produce bar plot of alignment measured at each layer before and after fine-tuning for a specific task
It requires the csv output of scripts/2023_acl/finetuning_and_alignment.py
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        default="udpos",
        help="Fine-tuning task to use (udpos, wikiann or xnli)",
    )
    parser.add_argument(
        "--direction",
        choices=["fwd", "bwd"],
        default="fwd",
        help="Whether to use source-target alignment (fwd) or target-source (bwd)",
    )
    parser.add_argument(
        "--lang", type=str, default="ar", help="Target language for alignment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Path of the model (HuggingFace)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models and tokenizers",
    )
    args = parser.parse_args()

    task = args.task
    model_slug = args.model
    direction = args.direction
    lang = args.lang

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_files])

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R Base",
        "xlm-roberta-large": "XLM-R Large",
    }

    model_in_df = set(df["model"].unique())

    models = {key: value for key, value in models.items() if key in model_in_df}

    model_to_nlayer = {}
    for model_name in models:
        model = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir)
        n_layers = get_nb_layers(model)
        model_to_nlayer[model_name] = n_layers

    model_name = models.get(model_slug, model_slug)

    bar_chart_df = df[(df.model == model_slug) & (df.task == task)]

    property_names_before = [
        f"alignment_before_{direction}_{lang}_{i}"
        for i in range(model_to_nlayer[model_slug])
    ]
    property_names_after = [
        f"alignment_after_{direction}_{lang}_{i}"
        for i in range(model_to_nlayer[model_slug])
    ]
    layers = list(range(model_to_nlayer[model_slug]))

    property_names_after.sort(key=lambda x: int(x.split("_")[-1]) + 1)
    property_names_before.sort(key=lambda x: int(x.split("_")[-1]) + 1)
    layers.sort()

    mean_after = [np.mean(bar_chart_df[c].dropna()) for c in property_names_after]
    mean_before = [np.mean(bar_chart_df[c].dropna()) for c in property_names_before]
    std_after = [np.std(bar_chart_df[c].dropna()) for c in property_names_after]
    std_before = [np.std(bar_chart_df[c].dropna()) for c in property_names_before]

    plt.bar(
        layers,
        mean_before,
        width=-0.4,
        align="edge",
        yerr=std_before,
        color=COLOR_PALETTE[0],
        label="before",
    )
    plt.bar(
        layers,
        mean_after,
        width=0.4,
        align="edge",
        yerr=std_after,
        color=COLOR_PALETTE[1],
        label="after",
    )

    plt.legend(loc="upper right")
    plt.xlabel("layer")
    plt.ylabel("retrieval accuracy")
    plt.savefig(args.output_file)
