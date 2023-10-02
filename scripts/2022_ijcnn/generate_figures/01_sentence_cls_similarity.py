"""
Script to plot the similarity between representation of the CLS tokens of translated sentences and random pairs
IJCNN 2022: Fig. 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import get_statistics, plot_with_statistics, COLOR_PALETTE

if __name__ == "__main__":

    pair_for_sim = ("ru", "en")
    model = "bert-base-multilingual-cased"
    dataset = "wmt19"
    data_dir = os.getenv("DATA_DIR")

    assert data_dir is not None, "env var DATA_DIR should be defined"
    assert os.path.isdir(data_dir), f"value provided for DATA_DIR {data_dir} is not a directory"

    base_dir = os.path.join(data_dir, "figures")

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    cls_sim_file = os.path.join(data_dir, "raw_results", "01_cls_similarities.csv")

    cls_df = pd.read_csv(cls_sim_file)

    cls_subset = cls_df[
        (cls_df.model == model)
        & (cls_df.dataset == dataset)
        & (cls_df.left_lang == pair_for_sim[0])
        & (cls_df.right_lang == pair_for_sim[1])
    ]

    n_layers = max(list(cls_subset["layer"])) + 1

    similar_pairs = [
        list(cls_subset[(cls_subset.type == "sim") & (cls_subset.layer == layer)]["score"])
        for layer in range(n_layers)
    ]
    random_pairs = [
        list(cls_subset[(cls_subset.type == "random") & (cls_subset.layer == layer)]["score"])
        for layer in range(n_layers)
    ]

    plot_with_statistics(
        range(n_layers),
        *get_statistics(np.array(similar_pairs)),
        COLOR_PALETTE[0],
        f"[CLS] tokens of translated pairs",
    )
    plot_with_statistics(
        range(n_layers),
        *get_statistics(np.array(random_pairs)),
        COLOR_PALETTE[1],
        f"[CLS] tokens of random pairs",
    )
    plt.legend(loc="lower left")
    plt.ylim((0.0, 1.0))
    plt.xlabel("layer")
    plt.ylabel("cos")
    plt.savefig(
        os.path.join(
            base_dir, f"01_cls_similarities_{model}_{pair_for_sim[0]}_{pair_for_sim[1]}.pdf"
        )
    )
    plt.clf()
