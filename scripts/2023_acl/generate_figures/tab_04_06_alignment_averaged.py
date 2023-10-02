"""
Script plot accuracy brought by various realignment methods (or the baseline) on various fine-tuning tasks averaged over languages
It requires the csv ouput of scripts/2023_acl/controlled_realignment.py
"""

import pandas as pd
import numpy as np
from collections import defaultdict

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files containing results of scripts/2023_acl/controlled_realignment.py",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of fine-tuning tasks to show on the table, (udpos, wikiann and/or xnli)",
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

    available_tasks = df["task"].unique()
    if args.tasks is None:
        tasks = list(available_tasks)
    else:
        tasks = sorted(
            list(set(available_tasks).intersection(args.tasks)), key=args.tasks.index
        )

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

    res = "& "

    res += " & ".join(map(task_to_name.__getitem__, tasks)) + "\\\\\n"
    res += "\\hline\n"

    for model in sorted(models, key=model_to_rank.__getitem__):

        subdf = df[df.model == model]

        model_name = existing_models.get(model, model)

        relevant_strategies = list(
            set(strategies).intersection(subdf["method"].unique())
        )

        means = np.zeros((len(relevant_strategies), len(tasks)))
        stds = np.zeros((len(relevant_strategies), len(tasks)))

        for i, strategy in enumerate(
            sorted(relevant_strategies, key=strategy_to_rank.__getitem__)
        ):

            subsubdf = subdf[subdf.method == strategy]

            if len(subsubdf) == 0:
                continue

            for j, task in enumerate(tasks):
                subsubsubdf = subsubdf[subsubdf.task == task]

                scores = list(subsubsubdf[f"final_eval_avg_{task_to_metric[task]}"])

                mean = np.mean(scores)
                std = np.std(scores)

                means[i, j] = mean
                stds[i, j] = std

        for i, strategy in enumerate(
            sorted(relevant_strategies, key=strategy_to_rank.__getitem__)
        ):

            res += f"{get_strategy_name(model_name, strategy)} "

            for j, task in enumerate(tasks):
                if i == 0:
                    res += f"& \\textit{{ {means[i,j]*100:.1f} }} "
                else:
                    res += "& "
                    if means[0, j] - means[i, j] > stds[0, j]:
                        res += "\\cellcolor{gray!80} "
                    elif not means[i, j] - means[0, j] > stds[0, j]:
                        res += "\\cellcolor{gray!30} "

                    if f"{means[i,j]*100:.1f}" == f"{max(means[:,j])*100:.1f}":
                        res += f"\\textbf{{ {means[i,j]*100:.1f} }} "
                    else:
                        res += f"{means[i,j]*100:.1f} "

            res += "\\\\\n"

        res += "\\hline\n"

    print(res)
