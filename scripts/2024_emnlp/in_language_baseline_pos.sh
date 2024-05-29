#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
MODEL=$3
ADD_ARGS=$4

# Removed because no train dataset: fi sv th  tr
langs="bg cs de es lv af ar ca da el fa fr he hi hu it ja ko lt no pl pt ro ru sk sl ta uk vi zh"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/raw_results

mkdir -p $CACHE_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR

export DATA_DIR=$DATA_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR


################
# BASELINES
################

for lang in $langs; do
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/$DATASET \
        --fastalign_dir $FASTALIGN_DIR/$DATASET \
        --dico_dir $DICOALIGN_DIR/$DATASET \
        --awesome_dir $AWESOME_DIR/$DATASET \
        --strategies baseline \
        --models $MODEL \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --left_lang $lang \
        --right_langs $lang en \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__in_language_${lang}.csv $ADD_ARGS
done