import os, shutil
import argparse
from pathlib import Path
from comet import download_model, load_from_checkpoint
import itertools
from contextlib import ExitStack


def prepare_directories(data_dir: str, dataset: str, subdirs, thresholds):
    for subdir, thresh in itertools.product(subdirs, thresholds):
        new_dir = os.path.join(data_dir, subdir, f"{dataset}_filtered_{thresh}")
        if os.path.exists(new_dir):
            print(f"Target directory {new_dir} already exist, will erase it first")
            shutil.rmtree(new_dir)
        Path(new_dir).mkdir()


def get_languages(langs, translation_dir, left_lang):
    if langs:
        return langs
    return list(
        map(
            lambda x: x.split(".")[0].split("-")[1],
            filter(
                lambda x: x.startswith(f"{left_lang}-")
                and x.endswith(".tokenized.train.txt"),
                os.listdir(translation_dir),
            ),
        )
    )


if __name__ == "__main__":
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--translation_dir", type=str, default="translation")
    parser.add_argument(
        "--alignment_dirs",
        type=str,
        default=["awesome-align", "dico-align", "fastalign"],
    )
    parser.add_argument("--left_lang", type=str, default="en")
    parser.add_argument("--right_langs", type=str, nargs="+", default=None)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    prepare_directories(
        args.data_dir,
        args.dataset,
        [args.translation_dir, *args.alignment_dirs],
        args.thresholds,
    )

    translation_dir = os.path.join(args.data_dir, args.translation_dir, args.dataset)
    alignment_dirs = [os.path.join(args.data_dir, args.alignment_dirs, args.dataset)]

    languages = get_languages(args.right_langs, translation_dir, args.left_lang)

    thresholds = sorted(args.thresholds)

    for lang in languages:
        with open(
            os.path.join(
                translation_dir, f"{args.left_lang}-{lang}.tokenized.train.txt"
            ),
            "r",
        ) as file:
            # Loop through each line in the file
            data = []
            for line in file:
                data.append(
                    {
                        "src": line.strip().split("|||")[0],
                        "mt": line.strip().split("|||")[1],
                    }
                )
        alignment_data = []
        for alignment_dir in alignment_dirs:
            this_data = []
            with open(
                os.path.join(
                    alignment_dir,
                    f"{args.left_lang}-{lang}.tokenized.train.txt",
                )
            ) as f:
                for line in file:
                    this_data.append(line.strip())
            alignment_data.append(this_data)

        model_output = model.predict(data, batch_size=args.batch_size, gpus=1)

        line_counters = [0] * len(thresholds)
        with ExitStack() as stack:
            translation_writers = [
                stack.enter_context(
                    open(
                        os.path.join(
                            args.data_dir,
                            args.translation_dir,
                            f"{args.dataset}_filtered_{t}",
                            f"{args.left_lang}-{lang}.train",
                        ),
                        "w",
                    )
                )
                for t in thresholds
            ]
            alignment_writers = [
                [
                    stack.enter_context(
                        open(
                            os.path.join(
                                args.data_dir,
                                subdir,
                                f"{args.dataset}_filtered_{t}",
                                f"{args.left_lang}-{lang}.train",
                            ),
                            "w",
                        )
                    )
                    for subdir in args.alignment_dirs
                ]
                for t in thresholds
            ]

            for i_data in range(len(data)):
                line = f"{data[i_data]['src']} ||| {data[i_data]['mt']}\n"
                for i_threshold, t in enumerate(thresholds):
                    if model_output[0][i_data] < t:
                        break
                    line_counters[i_threshold] += 1
                    translation_writers[i_threshold].write(line)
                    for i_alignment, writer in enumerate(
                        alignment_writers[i_threshold]
                    ):
                        writer.write(f"{alignment_data[i_alignment][i_data]}\n")

            for i_t, t in enumerate(thresholds):
                print(
                    f"lang {lang}: filtered out {len(data) - line_counters[i_t]} pairs out of {len(data)}"
                )
