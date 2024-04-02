#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
ADD_ARGS=$3

#langs="ar es fr ru zh af fa hi"
langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"

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

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies before_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies before_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies before_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - FRONT FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - BACK FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_last_6_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_last_6_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies freeze_realign_unfreeze_last_6_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT DURING - FRONT FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_front_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_front_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_front_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


##########################################
# PARTIAL-REALIGNMENT DURING - BACK FROZEN
##########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_back_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_back_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies during_partial_freeze_back_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --project_prefix "34langs_" \
    --output_file $RESULT_DIR/$DATASET.csv $ADD_ARGS
