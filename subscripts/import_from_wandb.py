import pandas as pd
import wandb

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wandb_path", type=str, help="Path of the sweep id (user_or_organisation/project/sweep_id)"
    )
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    wandb_path = args.wandb_path
    output_file = args.output_file

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    sweep = api.sweep(wandb_path)
    runs = sweep.runs
    summary_list, config_list, name_list = [], [], []
    df_keys = set()
    run_dicts = []
    for run in runs:

        if run.state != "finished":
            continue

        new_run_dict = {
            "name": run.name,
            **{k: v for k, v in run.config.items()},
            **run.summary._json_dict,
        }

        df_keys = df_keys.union(new_run_dict.keys())
        run_dicts.append(new_run_dict)

    runs_df = pd.DataFrame({key: [elt.get(key, None) for elt in run_dicts] for key in df_keys})

    runs_df.to_csv(output_file)
