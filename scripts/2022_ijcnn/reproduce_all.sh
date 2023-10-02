#!/bin/bash

set -e

source parameters.sh

mkdir -p $DATA_DIR

if [ -z ${ALIGNED_FASTTEXT_DIR+x} ]; then
    ALIGNED_FASTTEXT_DIR=$DATA_DIR/fasttext_aligned_embeddings
fi

if [ -z ${TRANSFORMERS_CACHE_DIR+x} ]; then
    TRANSFORMERS_CACHE_DIR=$DATA_DIR/cache/transformers
fi

if [ -z ${DATASETS_CACHE_DIR+x} ]; then
    DATASETS_CACHE_DIR=$DATA_DIR/cache/datasets
fi

if [ -z ${DICO_PATH+x} ]; then
    DICO_PATH=$DATA_DIR/muse_dictionaries
fi

mkdir -p $ALIGNED_FASTTEXT_DIR
mkdir -p $TRANSFORMERS_CACHE_DIR
mkdir -p $DATASETS_CACHE_DIR
mkdir -p $DICO_PATH

export DATA_DIR=$DATA_DIR
export ALIGNED_FASTTEXT_DIR=$ALIGNED_FASTTEXT_DIR
export TRANSFORMERS_CACHE_DIR=$TRANSFORMERS_CACHE_DIR
export DATASETS_CACHE_DIR=$DATASETS_CACHE_DIR
export DICO_PATH=$DICO_PATH

# download aligned fasttext embeddings
source download_resources/aligned_fasttext.sh $ALIGNED_FASTTEXT_DIR "en de ru zh"

# download bilingual dictionaries
source download_resources/muse_dictionaries.sh $DICO_PATH "de ru zh"

if [ ! -f $DATA_DIR/raw_results/01_cls_similarities.csv ] || [ ! -f $DATA_DIR/raw_results/02_sentence_nn.csv ] ; then
    python scripts/2022_ijcnn/compare_sentence_representations.py
fi

if [ ! -f $DATA_DIR/figures/01_cls_similarities_bert-base-multilingual-cased_ru_en.pdf ]; then
    python scripts/2022_ijcnn/generate_figures/01_sentence_cls_similarity.py
fi

if [ ! -f $DATA_DIR/figures/02_sentence_retrieval_table.txt ]; then
    python scripts/2022_ijcnn/generate_figures/02_sentence_retrieval.py
fi

if [ ! -f $DATA_DIR/raw_results/02_word_nn.csv ]; then
    python scripts/2022_ijcnn/word_level_alignment.py
fi

if [ ! -f $DATA_DIR/figures/03_weak_alignment.pdf ] || [ ! -f $DATA_DIR/figures/04_strong_alignment.pdf ] || [ ! -f $DATA_DIR/figures/05_weak_alignment_table.txt ]  || [ ! -f $DATA_DIR/figures/06_strong_alignment_table.txt ] ; then
    python scripts/2022_ijcnn/generate_figures/03_evaluate_strong_and_weak_alignment.py
fi