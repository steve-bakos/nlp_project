# Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation

Code for the paper: [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076v1), Félix Gaschi, François Plesse, Parisa Rastin, Yannick Toussaint. IJCNN 2022

## Requirements

This was tested in Python 3.7, with requirements provided in `requirements.txt`

## How to use

All scripts are meant to be run from the root of the repository (e.g. `./scripts/2022_ijcnn/reproduce_all.sh` instead of `cd scripts/2022_ijcnn && ./reproduce_all.sh`).

### Reproduce all results

If you can reproduce all the figure from the paper, you should first create a `parameters.sh` file following the template in `sample_parameters.sh` in the root of the directory. Most importantly, the `DATA_DIR` variable defines the directory where all data and results will be stored. Be aware than it is in the orders of several hundreds of GB (mainly for downloading WMT19). 

Then, all the figures can be reproduced by simply running the following (after having installed the required dependencies): 

```
./scripts/2022_ijcnn/reproduce_all.sh
```

This scripts also takes care of downloading the necessary resources: FastText embeddings and MUSE dictionaries.

### Run a specific scripts

There are two scripts for performing the experiments:

- `scripts/2022_ijcnn/compare_sentence_representations.py`: compares multilingual sentence representations and writes two files: `01_cls_similarities.csv` for storing average similarity of CLS token for translated and random pairs of sentences, and `02_sentence_nn.csv` for multilingual nearest-neighbor retrieval score for various sentence representations.
- `scripts/2022_ijcnn/word_level_alignment.py`: computes the nearest-neighbor retrieval score for various word-level representations.

Both scripts rely on the following environment variables:

- `DATA_DIR`: the directory where all the data will be stored
- `ALIGNED_FASTTEXT_DIR`: the directory where aligned FastText embeddings can be found (can be downloaded with `bash download_resources/aligned_fasttext.sh $ALIGNED_FASTTEXT_DIR "en fr zh"`)
- `DICO_PATH`: the directory where MUSE bilingual dictionary are stored (can be downloaded with `bash download_resources/muse_dictionaries.sh $DICO_PATH "fr zh"`), usefull only for the second script on word-level alignment
- `DATASETS_CACHE_DIR`: optional but highly recommended to set. If set, it is the directory that will be used for caching in `datasets.load_dataset`, be aware that the scripts load WMT19 which needs several hundreds of GB of space on disk
- `TRANSFORMERS_CACHE_DIR`: optional, the directory used for caching HuggingFace models.

The two scripts generate raw results in the form of csv files in `$DATA_DIR/raw_results`. To generate figures and tables from that, you must use the scripts in `scripts/2022_ijcnn/generate_figures`
