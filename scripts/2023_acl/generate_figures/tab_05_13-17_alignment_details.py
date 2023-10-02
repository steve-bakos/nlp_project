"""
Script plot accuracy brought by various realignment methods (or the baseline) on various fine-tuning tasks detailed by language
It requires the csv ouput of scripts/2023_acl/controlled_realignment.py
"""

import pandas as pd
import sys, os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import get_statistics, COLOR_PALETTE
from multilingual_eval.seeds import seeds

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files containing results of scripts/2023_acl/controlled_realignment.py",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="udpos",
        help="Fine-tuning task to show on the table, (udpos, wikiann or xnli)",
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
        "--models",
        type=str,
        nargs="+",
        default=[
            "distilbert-base-multilingual-cased",
            "bert-base-multilingual-cased",
            "xlm-roberta-base",
            "xlm-roberta-large",
        ],
        help="Paths of the models (HuggingFace or local directory) to evaluate",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=[
            "baseline",
            "before_fastalign",
            "before_awesome",
            "before_dico",
            "during_fastalign",
            "during_awesome",
            "during_dico",
        ],
        help="Realignment strategies to show on the table",
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(fname) for fname in args.csv_files]

    df = pd.concat(dfs)

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI"}

    def get_strategy_name(model_name, strategy):
        if strategy == "baseline":
            return model_name
        parts = strategy.split("_")
        if len(parts) != 2:
            raise NotImplementedError(f"Unrecognized strategy: {strategy}")
        method, aligner = parts
        if method == "before":
            return f"+ before {aligner}"
        if method == "during":
            return f"+ joint {aligner}"
        raise NotImplementedError(f"Unrecognized strategy: {strategy}")

    strategies = args.strategies
    assert strategies[0] == "baseline"
    strategy_to_rank = {s: i for i, s in enumerate(strategies)}

    existing_models = {
        "distilbert-base-multilingual-cased": "distilmBERT",
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R base",
        "xlm-roberta-large": "XLM-R large",
    }
    model_to_rank = {
        "distilbert-base-multilingual-cased": 0,
        "bert-base-multilingual-cased": 1,
        "xlm-roberta-base": 2,
        "xlm-roberta-large": 3,
    }

    models = args.models

    taskdf = df[df.task == args.task]

    res = f"{task_to_name[args.task]} & "

    langs = ["same", *args.right_langs]

    res += " & ".join([args.left_lang, *args.right_langs]) + "\\\\\n"
    res += "\\hline\n"

    for model in sorted(models, key=model_to_rank.__getitem__):

        subdf = taskdf[taskdf.model == model]

        model_name = existing_models.get(model, model)

        strategies = subdf["method"].unique()

        means = np.zeros((len(langs), len(strategies)))
        stds = np.zeros((len(langs), len(strategies)))

        for j, strategy in enumerate(
            sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))
        ):
            if "method" in df.columns:
                df_by_strategy = subdf[(subdf.method == strategy)]
            else:
                df_by_strategy = subdf[(subdf.realignment_strategy == strategy)]
            for i, lang in enumerate(langs):
                scores = list(
                    df_by_strategy[f"final_eval_{lang}_{task_to_metric[args.task]}"]
                )
                means[i, j] = np.mean(scores)
                stds[i, j] = np.std(scores)

        for j, strategy in enumerate(
            sorted(strategies, key=lambda x: strategy_to_rank.get(x, 10))
        ):
            res += get_strategy_name(model_name, strategy)
            for i, lang in enumerate(langs):
                mean_argmax = np.argmax([0 if np.isnan(m) else m for m in means[i]])
                is_max = j == mean_argmax
                is_significant = means[i, j] - means[i, 0] > stds[i, 0]
                is_significantly_lower = (
                    means[i, 0] - means[i, j] > stds[i, 0] and j != 0
                )
                is_not_significant = (
                    not (is_significant or is_significantly_lower) and j != 0
                )
                res += (
                    " & "
                    + (
                        "\cellcolor{gray!30}"
                        if is_not_significant
                        else ("\cellcolor{gray!80}" if is_significantly_lower else "")
                    )
                    + ("\\textit{" if j == 0 else "")
                    + ("\\textbf{" if is_max else "")
                    + f"{means[i,j]*100:.1f}"
                    + ("}" if is_max else "")
                    + ("}" if j == 0 else "")
                    + f"$_{{\pm {stds[i,j]*100:.1f}}}$"
                )
            res += " \\\\\n"

        res += "\\hline\n"

    print(res)
