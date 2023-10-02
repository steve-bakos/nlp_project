import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.curdir)

from multilingual_eval.plotting_utils import COLOR_PALETTE, get_statistics, plot_with_statistics

if __name__ == "__main__":

    model_dir = "bert-base-multilingual-cased"
    model_name = "mBERT"

    dataset = "wmt19"

    lang_pairs = [
        ("de", "en", COLOR_PALETTE[0]),
        ("ru", "en", COLOR_PALETTE[1]),
        ("zh", "en", COLOR_PALETTE[4]),
    ]

    data_dir = os.getenv("DATA_DIR")

    assert data_dir is not None, "env var DATA_DIR should be defined"
    assert os.path.isdir(data_dir), f"value provided for DATA_DIR {data_dir} is not a directory"

    base_dir = os.path.join(data_dir, "figures")

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    word_nn_file = os.path.join(data_dir, "raw_results", "02_word_nn.csv")

    df = pd.read_csv(word_nn_file)
    fasttext_df = df[(df.dataset == dataset) & (df.model == "fasttext")]
    subset = df[(df.dataset == dataset) & (df.model == model_dir)]

    n_layers = max(list(subset["layer"])) + 1

    # Plot weak alignment
    for i, (left, right, color) in enumerate(lang_pairs):

        model_scores = [
            list(
                subset[
                    (subset.left_lang == left)
                    & (subset.right_lang == right)
                    & (subset.layer == layer)
                    & (subset.strong == False)
                ]["score"]
            )
            for layer in range(n_layers)
        ]
        print(f"number of runs: {len(model_scores[0])}")
        fasttext_scores = [
            list(
                fasttext_df[
                    (fasttext_df.left_lang == left)
                    & (fasttext_df.right_lang == right)
                    & (fasttext_df.strong == False)
                ]["score"]
            )
            for layer in range(n_layers)
        ]
        print(f"number of runs: {len(fasttext_scores[0])}")

        plot_with_statistics(
            range(n_layers),
            *get_statistics(np.array(model_scores)),
            color,
            f"{model_name} {left}-{right}",
        )
        plot_with_statistics(
            range(n_layers),
            *get_statistics(np.array(fasttext_scores)),
            color,
            f"FastText {left}-{right}",
            linestyle="dashed",
        )

    plt.legend(loc="upper left")
    plt.ylim((0.0, 1.0))
    plt.xlabel("layer")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(base_dir, "03_weak_alignment.pdf"))
    plt.clf()

    # Plot strong alignment
    for i, (left, right, color) in enumerate(lang_pairs):

        model_scores = [
            list(
                subset[
                    (subset.left_lang == left)
                    & (subset.right_lang == right)
                    & (subset.layer == layer)
                    & (subset.strong == True)
                ]["score"]
            )
            for layer in range(n_layers)
        ]
        fasttext_scores = [
            list(
                fasttext_df[
                    (fasttext_df.left_lang == left)
                    & (fasttext_df.right_lang == right)
                    & (fasttext_df.strong == True)
                ]["score"]
            )
            for layer in range(n_layers)
        ]

        plot_with_statistics(
            range(n_layers),
            *get_statistics(np.array(model_scores)),
            color,
            f"{model_name} {left}-{right}",
        )
        plot_with_statistics(
            range(n_layers),
            *get_statistics(np.array(fasttext_scores)),
            color,
            f"FastText {left}-{right}",
            linestyle="dashed",
        )

    plt.legend(loc="upper left")
    plt.ylim((0.0, 1.0))
    plt.xlabel("layer")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(base_dir, "04_strong_alignment.pdf"))

    models = {
        "& mBERT": "bert-base-multilingual-cased",
        "& XLM-100": "xlm-mlm-100-1280",
        "& XLM-R Base": "xlm-roberta-base",
        "& XLM-R Large": "xlm-roberta-large",
        "& XLM-15$^\mathrm{a,b}$": "xlm-mlm-tlm-xnli15-1024",
        "& AWESOME$^\mathrm{a}$": "awesome-align-with-co",
        "& mBART$^\mathrm{b,c}$": "mbart-large-50",
    }

    for strong_alignment in [True, False]:
        lines = []

        # Add header
        lines.append("\hline")
        header = "layer & model"
        for left, right, _ in lang_pairs:
            header += f" & {left}-{right}"
        header += "\\\\"
        lines.append(header)
        lines.append("\hline")

        best_lines = []
        last_lines = []


        fasttext_line = "- & aligned FastText"
        for left, right, _ in lang_pairs:
            scores = list(
                df[
                    (df.model == "fasttext")
                    & (df.left_lang == left)
                    & (df.right_lang == right)
                    & (df.strong == strong_alignment)
                ]["score"]
            )
            fasttext_line += f" & {np.mean(scores)*100:.1f} ({np.std(scores)*100:.2f})"
        fasttext_line += "\\\\"
        lines.append(fasttext_line)
        lines.append("\hline")

        for model, model_name in models.items():
            if len(df[df.model == model_name]) == 0:
                continue

            best_line = f"{model}"
            last_line = f"{model}"

            n_layers = max(list(df[df.model == model_name]["layer"])) + 1

            if model_name == "mbart-large-50":
                n_layers = 13

            for left, right, _ in lang_pairs:
                scores = [
                    list(
                        df[
                            (df.model == model_name)
                            & (df.left_lang == left)
                            & (df.right_lang == right)
                            & (df.layer == layer)
                            & (df.strong == strong_alignment)
                        ]["score"]
                    )
                    for layer in range(n_layers)
                ]

                mean_scores = list(map(np.mean, scores))

                best_layer = np.argmax(mean_scores)

                best_line += (
                    f" & {np.mean(scores[best_layer])*100:.1f} ({np.std(scores[best_layer])*100:.2f})"
                )
                last_line += f" & {np.mean(scores[-1])*100:.1f} ({np.std(scores[-1])*100:.2f})"

            best_line += "\\\\"
            last_line += "\\\\"
            best_lines.append(best_line)
            last_lines.append(last_line)

        best_lines[0] = "\multirow{7}{*}{best} " + best_lines[0]
        last_lines[0] = "\multirow{7}{*}{last} " + last_lines[0]

        lines += best_lines + ["\hline"] + last_lines + ["\hline"]

        with open(os.path.join(base_dir, "05_weak_alignment_table.txt" if not strong_alignment else "06_strong_alignment_table.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")
