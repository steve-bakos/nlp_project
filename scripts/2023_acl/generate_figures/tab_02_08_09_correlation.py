"""
Script to print LaTeX-like table of correlation between alignment (before and after fine-tuning) and cross-lingual transfer
It requires the csv output of scripts/2023_acl/finetuning_and_alignment.py
"""

import os
import sys
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoModel

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import spearman_with_bootstrap_ci
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
        "--tasks",
        type=str,
        nargs="+",
        default=["udpos", "wikiann", "xnli"],
        help="List of fine-tuning tasks to stack on the table, (udpos, wikiann and/or xnli)",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=["ar", "es", "fr", "ru", "zh"],
        help="List of target languages",
    )
    parser.add_argument(
        "--with_ci",
        action="store_true",
        dest="with_ci",
        help="Option to include confidence interval in table",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models and tokenizers",
    )
    parser.set_defaults(with_ci=False)
    args = parser.parse_args()

    langs = args.langs

    df = pd.concat([pd.read_csv(fname) for fname in args.csv_files])

    task_to_metric = defaultdict(lambda: "accuracy")
    task_to_metric["wikiann"] = "f1"

    task_to_name = {"wikiann": "NER", "udpos": "POS", "xnli": "NLI"}

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

    res = "task & layer & \multicolumn{2}{|c}{en-X} & \multicolumn{2}{|c}{X-en}\\\\\n"
    res += "& & before & after & before & after \\\\\n\\hline\n"

    models_for_layers = ["bert-base-multilingual-cased", "xlm-roberta-base"]

    for task in args.tasks:
        task_name = task_to_name[task]
        res += f"\\multirow{{2}}{{*}}{{{task_name}}}"

        alignment_scores_fwd_before = []
        alignment_scores_fwd_after = []
        alignment_scores_bwd_before = []
        alignment_scores_bwd_after = []
        delta_scores = []

        for model, n_layer in model_to_nlayer.items():
            if model not in df["model"].unique():
                continue

            layer = n_layer - 1
            subdf = df[((df.model == model) & (df.task == task))]

            for lang in langs:
                alignment_scores_fwd_before += list(
                    subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_fwd_after += list(
                    subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_bwd_before += list(
                    subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_bwd_after += list(
                    subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                )
                delta_scores += list(
                    (
                        subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                        - subdf[f"final_eval_same_{task_to_metric[task]}"]
                    )
                    / subdf[f"final_eval_same_{task_to_metric[task]}"]
                )

        rho_fwd_before, p_fwd_before = spearmanr(
            alignment_scores_fwd_before, delta_scores
        )
        rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
        rho_bwd_before, p_bwd_before = spearmanr(
            alignment_scores_bwd_before, delta_scores
        )
        rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

        if args.with_ci:
            boostrap_fwd_before = spearman_with_bootstrap_ci(
                alignment_scores_fwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_fwd_after = spearman_with_bootstrap_ci(
                alignment_scores_fwd_after, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_before = spearman_with_bootstrap_ci(
                alignment_scores_bwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_after = spearman_with_bootstrap_ci(
                alignment_scores_bwd_after, delta_scores, n_resamples=2_000
            )

            res += (
                " & penult. "
                + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
            )
        else:
            res += (
                " & last & "
                + ("\cellcolor{gray!30}" if p_fwd_before > 0.05 else "")
                + f"{rho_fwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_fwd_after > 0.05 else "")
                + f"{rho_fwd_after:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_before > 0.05 else "")
                + f"{rho_bwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_after > 0.05 else "")
                + f"{rho_bwd_after:.2f} \\\\\n"
            )

        alignment_scores_fwd_before = []
        alignment_scores_fwd_after = []
        alignment_scores_bwd_before = []
        alignment_scores_bwd_after = []
        delta_scores = []

        for model, n_layer in model_to_nlayer.items():
            if model not in df["model"].unique():
                continue
            layer = n_layer - 2
            subdf = df[((df.model == model) & (df.task == task))]

            for lang in langs:
                alignment_scores_fwd_before += list(
                    subdf[f"alignment_before_fwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_fwd_after += list(
                    subdf[f"alignment_after_fwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_bwd_before += list(
                    subdf[f"alignment_before_bwd_{lang}_{layer}"].dropna()
                )
                alignment_scores_bwd_after += list(
                    subdf[f"alignment_after_bwd_{lang}_{layer}"].dropna()
                )
                delta_scores += list(
                    (
                        subdf[f"final_eval_{lang}_{task_to_metric[task]}"]
                        - subdf[f"final_eval_same_{task_to_metric[task]}"]
                    )
                    / subdf[f"final_eval_same_{task_to_metric[task]}"]
                )

        rho_fwd_before, p_fwd_before = spearmanr(
            alignment_scores_fwd_before, delta_scores
        )
        rho_fwd_after, p_fwd_after = spearmanr(alignment_scores_fwd_after, delta_scores)
        rho_bwd_before, p_bwd_before = spearmanr(
            alignment_scores_bwd_before, delta_scores
        )
        rho_bwd_after, p_bwd_after = spearmanr(alignment_scores_bwd_after, delta_scores)

        if args.with_ci:
            boostrap_fwd_before = spearman_with_bootstrap_ci(
                alignment_scores_fwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_fwd_after = spearman_with_bootstrap_ci(
                alignment_scores_fwd_after, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_before = spearman_with_bootstrap_ci(
                alignment_scores_bwd_before, delta_scores, n_resamples=2_000
            )
            boostrap_bwd_after = spearman_with_bootstrap_ci(
                alignment_scores_bwd_after, delta_scores, n_resamples=2_000
            )

            res += (
                " & penult. "
                + f"& {rho_fwd_before:.2f} ({boostrap_fwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_fwd_after:.2f} ({boostrap_fwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_fwd_after.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_before:.2f} ({boostrap_bwd_before.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_before.confidence_interval.high:.2f}) "
                + f"& {rho_bwd_after:.2f} ({boostrap_bwd_after.confidence_interval.low:.2f} - "
                + f"{boostrap_bwd_after.confidence_interval.high:.2f}) \\\\\n"
            )
        else:
            res += (
                " & penult. & "
                + ("\cellcolor{gray!30}" if p_fwd_before > 0.05 else "")
                + f"{rho_fwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_fwd_after > 0.05 else "")
                + f"{rho_fwd_after:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_before > 0.05 else "")
                + f"{rho_bwd_before:.2f} & "
                + ("\cellcolor{gray!30}" if p_bwd_after > 0.05 else "")
                + f"{rho_bwd_after:.2f} \\\\\n"
            )
            res += "\\hline\n"

    print(res)
