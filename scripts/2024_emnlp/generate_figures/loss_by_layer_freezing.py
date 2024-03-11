import os
import json
import pandas as pd
import itertools
from pathlib import Path

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_layers", type=int, default=13)
    args = parser.parse_args()
    result_file = args.results
    n_layers = args.n_layers
    output_dir = args.output_dir

    assert os.path.isfile(result_file)

    if not os.path.isdir(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    methods = ["during_partial_freeze"]
    aligners = ["awesome", "fastalign", "dico"]

    langs = ["ar", "es", "fr", "zh", "ru"]

    df = pd.read_csv(
        result_file, converters={
            k: json.loads for k in [
                "finetuning_steps", 
                "realignment_steps", 
                "distinct_realignment_samples", 
                "repeated_realignment_samples",
                "train_loss",
                "task_loss",
                "realignment_loss",
                *(f"eval_{subset}_accuracy" for subset in langs + ["avg", "same"])
            ]
        }
    )

    print(df.columns)

    for method, aligner in itertools.product(methods, aligners):
        for i in range(args.n_layers):
            subdf = df[df.method == f"{method}_0_{i}_{aligner}"]
            if len(subdf) == 0:
                continue

            



