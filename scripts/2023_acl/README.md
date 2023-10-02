# Exploring the Relationship between Alignment and Cross-lingual Transfer in Multilingual Transformers

Code for the paper: Exploring the Relationship between Alignment and Cross-lingual Transfer in Multilingual Transformers. Félix Gaschi, Patricio Cerda, Parisa Rastin, Yannick Toussaint. Findings of ACL 2023 (link coming soon)

## Requirements

This was tested in Python 3.9, with requirements provided in `requirements.txt`, where `wandb` is actually optional and `brokenaxes` is only useful for generating figures.

## How to use

All the scripts are meant to be run from the run of the repository (e.g. `scripts/2023_acl/reproduce_all_opus100.sh` instead of `cd scripts/2023_acl && ./reproduce_all_opus100.sh`).

### Recommended usage

`reproduce_all_opus100.sh` allows to perform all experiments present in the paper, except those concerning the Multi-UN dataset (in Appendix). But it would take a very long time to run entirely.

Instead it is recommended to download the preprocessed realignment data [here](https://drive.google.com/file/d/12mrv3tUHNvGLCYtjiv6VackjmNtW_PRn/view?usp=sharing).

Then you can run any of the two main scripts on it:

- `finetuning_and_alignment.py` is the main script for the experiment where multilingual alignment is measured before and after fine-tuning (then we can look at correlation, drop in alignment etc...)
- `controlled_realignment.py` is the main script for comparing different realignment methods (before or during fine-tuning and with different aligners)

Options of those scripts are specified below.

By default, this scripts produces CSV files with the raw data produced by the experiments. Figures from the papers can then be reproduced from those CSV files using the scripts in the `generate_figures` subdirectory.

### Finetuning and alignment

`finetuning_and_alignment.py` measure multilingual alignment before and after fine-tuning (then we can look at correlation, drop in alignment etc...).

```
usage: finetuning_and_alignment.py [-h] [--models MODELS [MODELS ...]] [--tokenizers TOKENIZERS [TOKENIZERS ...]] [--tasks TASKS [TASKS ...]] [--left_lang LEFT_LANG] [--right_langs RIGHT_LANGS [RIGHT_LANGS ...]] [--eval_layers EVAL_LAYERS [EVAL_LAYERS ...]] [--cache_dir CACHE_DIR] [--sweep_id SWEEP_ID]
                                   [--debug] [--large_gpu] [--n_seeds N_SEEDS] [--n_epochs N_EPOCHS] [--strong_alignment] [--multiparallel] [--from_scratch] [--output_file OUTPUT_FILE] [--use_wandb]
                                   translation_dir alignment_dir

positional arguments:
  translation_dir       Directory where the parallel dataset can be found, expecting to contain files of the form {source_lang}-{target_lang}.tokenized.train.txt
  alignment_dir         Directory where the alignment pairs can be found, expecting to contain files of the form {source_lang}-{target_lang}.train

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
  --tokenizers TOKENIZERS [TOKENIZERS ...]
                        List of tokenizer slugs that can be provided to override --models when instantiating with AutoTokenizer
  --tasks TASKS [TASKS ...]
  --left_lang LEFT_LANG
                        Source language
  --right_langs RIGHT_LANGS [RIGHT_LANGS ...]
                        Target languages
  --eval_layers EVAL_LAYERS [EVAL_LAYERS ...]
                        List of (indexed) layers of which the alignment must be evaluated (default: all layers)
  --cache_dir CACHE_DIR
                        Cache directory which will contain subdirectories 'transformers' and 'datasets' for caching HuggingFace models and datasets
  --sweep_id SWEEP_ID   If using wandb, useful to restart a sweep or launch several run in parallel for a same sweep
  --debug               Use this to perform a quicker test run with less samples
  --large_gpu           Use this option for 45GB GPUs (less gradient accumulation needed)
  --n_seeds N_SEEDS
  --n_epochs N_EPOCHS
  --strong_alignment    Option to use strong alignment instead of weak for evaluation
  --multiparallel       Option to use when the translation dataset is multiparallel (not used in the paper)
  --from_scratch        Option to use when we want to instantiate the models with random weights instead of pre-trained ones
  --output_file OUTPUT_FILE
                        The path to the output CSV file containing results (used only if wandb is not use, which is the case by default)
  --use_wandb           Use this option to use wandb (but must be installed first)
```

### Controlled realignment 

`controlled_realignment.py` compares different realignment methods (before or during fine-tuning and with different aligners)

```
usage: controlled_realignment.py [-h] [--translation_dir TRANSLATION_DIR] [--fastalign_dir FASTALIGN_DIR] [--dico_dir DICO_DIR] [--awesome_dir AWESOME_DIR] [--models MODELS [MODELS ...]] [--tasks TASKS [TASKS ...]] [--strategies STRATEGIES [STRATEGIES ...]] [--left_lang LEFT_LANG]
                                 [--right_langs RIGHT_LANGS [RIGHT_LANGS ...]] [--cache_dir CACHE_DIR] [--sweep_id SWEEP_ID] [--debug] [--large_gpu] [--n_epochs N_EPOCHS] [--n_seeds N_SEEDS] [--layers LAYERS [LAYERS ...]] [--output_file OUTPUT_FILE] [--use_wandb]

optional arguments:
  -h, --help            show this help message and exit
  --translation_dir TRANSLATION_DIR
                        Directory where the parallel dataset can be found, must be set if other strategy than baseline is used.
  --fastalign_dir FASTALIGN_DIR
                        Directory where fastalign alignments can be found, must be set if strategy ending in _fastalign is used
  --dico_dir DICO_DIR   Directory where bilingual dictionary alignments can be found, must be set if strategy ending in _dico is used
  --awesome_dir AWESOME_DIR
                        Directory where awesome alignments can be found, must be set if strategy ending in awesome is used
  --models MODELS [MODELS ...]
  --tasks TASKS [TASKS ...]
  --strategies STRATEGIES [STRATEGIES ...]
                        Realignment strategies to use, of the form strategy_aligner, with strategy being either before or after and aligner being either dico, fastalign or awesome
  --left_lang LEFT_LANG
                        Source language for cross-lingual transfer
  --right_langs RIGHT_LANGS [RIGHT_LANGS ...]
                        Target languages for cross-lingual transfer
  --cache_dir CACHE_DIR
                        Cache directory which will contain subdirectories 'transformers' and 'datasets' for caching HuggingFace models and datasets
  --sweep_id SWEEP_ID   If using wandb, useful to restart a sweep or launch several run in parallel for a same sweep
  --debug               Use this to perform a quicker test run with less samples
  --large_gpu           Use this option for 45GB GPUs (less gradient accumulation needed)
  --n_epochs N_EPOCHS
  --n_seeds N_SEEDS
  --layers LAYERS [LAYERS ...]
                        The layer (or list of layers) on which we want to perform realignment (default -1 for the last one)
  --output_file OUTPUT_FILE
                        The path to the output CSV file containing results (used only if wandb is not use, which is the case by default)
  --use_wandb           Use this option to use wandb (but must be installed first)
```

### Reproducing all the results

To retrieve all the data and run all the experiments at once, there are the two following scripts:

- `reproduce_all_opus100.sh` allows to perform all experiments present in the paper, except those concerning the Multi-UN dataset (in Appendix)
- `realignment_with_multiun.sh` allows to perform those additional experiments with the multi-UN dataset

Given a directory DATA_DIR where everything will be stored (preprocessed datasets and all), either of the two scripts can be used as follows:

```
bash scripts/2023_acl/reproduce_all_opus100.sh DATA_DIR
```

Please note that adding `--debug` to this command will allow you to do a relatively quick test run of all the experiments.

