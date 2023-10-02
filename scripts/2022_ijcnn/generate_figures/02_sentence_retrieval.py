import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    data_dir = os.getenv("DATA_DIR")

    assert data_dir is not None, "env var DATA_DIR should be defined"
    assert os.path.isdir(data_dir), f"value provided for DATA_DIR {data_dir} is not a directory"

    base_dir = os.path.join(data_dir, "figures")

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    sentence_nn_file = os.path.join(data_dir, "raw_results", "02_sentence_nn.csv")
    table_file = os.path.join(base_dir, "02_sentence_retrieval_table.txt")

    df = pd.read_csv(sentence_nn_file)

    model = "bert-base-multilingual-cased"

    header = "representations"

    fasttext_line = "FastText avg"

    cls_lines = [
        "CLS first",
        "CLS best",
        "CLS last",
    ]

    avg_lines = [
        "avg first",
        "avg best",
        "avg last",
    ]

    lang_pairs = [("de", "en"), ("ru", "en"), ("zh", "en")]

    n_layers = max(list(df[df.model == model]["layer"])) + 1

    for left_lang, right_lang in lang_pairs:
        header += f" & {left_lang}-{right_lang}"

        fasttext_scores = list(
            df[
                (df.model == "fasttext")
                & (df.left_lang == left_lang)
                & (df.right_lang == right_lang)
                & (df.centered == False)
            ]["score"]
        )
        fasttext_line += (
            f" & {np.mean(fasttext_scores)*100:.1f} ({np.std(fasttext_scores)*100:.1f})"
        )

        cls_scores = [
            list(
                df[
                    (df.model == model)
                    & (df.left_lang == left_lang)
                    & (df.right_lang == right_lang)
                    & (df.centered == False)
                    & (df.layer == layer)
                    & (df.type == "cls")
                ]["score"]
            )
            for layer in range(n_layers)
        ]

        avg_scores = [
            list(
                df[
                    (df.model == model)
                    & (df.left_lang == left_lang)
                    & (df.right_lang == right_lang)
                    & (df.centered == False)
                    & (df.layer == layer)
                    & (df.type == "avg")
                ]["score"]
            )
            for layer in range(n_layers)
        ]

        # Remove absurd values
        mean_cls = list(map(np.mean, cls_scores))
        mean_avg = list(map(np.mean, avg_scores))
        cls_scores = [
            list(filter(lambda x: x > mean_score * 0.01, elt))
            for elt, mean_score in zip(cls_scores, mean_cls)
        ]
        avg_scores = [
            list(filter(lambda x: x > mean_score * 0.01, elt))
            for elt, mean_score in zip(avg_scores, mean_avg)
        ]

        mean_cls_scores = list(map(np.mean, cls_scores))
        std_cls_scores = list(map(np.std, cls_scores))
        mean_avg_scores = list(map(np.mean, avg_scores))
        std_avg_scores = list(map(np.std, avg_scores))

        cls_best_idx = np.argmax(mean_cls_scores)
        avg_best_idx = np.argmax(mean_avg_scores)

        cls_lines[0] += f" & {mean_cls_scores[0]*100:.1f} ({std_cls_scores[0]*100:.1f})"
        cls_lines[
            1
        ] += f" & {mean_cls_scores[cls_best_idx]*100:.1f} ({std_cls_scores[cls_best_idx]*100:.1f})"
        cls_lines[2] += f" & {mean_cls_scores[-1]*100:.1f} ({std_cls_scores[-1]*100:.1f})"

        avg_lines[0] += f" & {mean_avg_scores[0]*100:.1f} ({std_avg_scores[0]*100:.1f})"
        avg_lines[
            1
        ] += f" & {mean_avg_scores[avg_best_idx]*100:.1f} ({std_avg_scores[avg_best_idx]*100:.1f})"
        avg_lines[2] += f" & {mean_avg_scores[-1]*100:.1f} ({std_avg_scores[-1]*100:.1f})"

    with open(table_file, "w") as f:
        f.write("\hline\n")
        f.write(header + "\\\\\n")
        f.write("\hline\n")
        f.write(fasttext_line + "\\\\\n")
        f.write("\hline\n")
        for i in range(3):
            f.write(cls_lines[i] + "\\\\\n")
            f.write(avg_lines[i] + "\\\\\n")
            f.write("\hline\n")
