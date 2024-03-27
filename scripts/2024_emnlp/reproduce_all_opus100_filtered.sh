#!/bin/bash

set -e

DATA_DIR=$1
ADD_ARGS=$2

langs="ar es fr ru zh af fa hi"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
OPUS_DIR=$DATA_DIR/opus100
FILTERED_OPUS_DIR=$DATA_DIR/opus100_filtered
MUSE_DIR=$DATA_DIR/muse_dictionaries
AWESOME_MODEL_DIR=$DATA_DIR/awesome-models
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/raw_results

mkdir -p $CACHE_DIR
mkdir -p $OPUS_DIR
mkdir -p $FILTERED_OPUS_DIR
mkdir -p $MUSE_DIR
mkdir -p $AWESOME_MODEL_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR

export DATA_DIR=$DATA_DIR
export OPUS_DIR=$OPUS_DIR
export MUSE_DIR=$MUSE_DIR
export AWESOME_MODEL_DIR=$AWESOME_MODEL_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR

# download muse dictionaries
bash download_resources/muse_dictionaries.sh $MUSE_DIR "$langs"

# install fastalign
bash download_resources/fastalign.sh

# install Stanford segmenter for Chinese
bash download_resources/stanford_tokenizer.sh

# installe awesome-align and download model without-co
bash download_resources/awesome_align.sh  $AWESOME_MODEL_DIR

# Create virtualenv for awesome-align
if [ ! -d $DATA_DIR/venvs/awesome-align ]; then
    python -m venv $DATA_DIR/venvs/awesome-align

    cd tools/awesome-align
    source $DATA_DIR/venvs/awesome-align/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python setup.py install
    deactivate
    cd ../..
fi


for lang in $langs; do
    echo "parsing lang $lang for opus-100"

    mkdir -p $TRANSLATION_DIR/opus100_filtered
    mkdir -p $DICOALIGN_DIR/opus100_filtered
    mkdir -p $AWESOME_DIR/opus100_filtered

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Align with FastAlign
    if [ ! -f $FASTALIGN_DIR/opus100_filtered/en-$lang.train ]; then
        bash scripts/word_align_with_fastalign.sh $TRANSLATION_DIR/opus100_filtered/en-$lang.tokenized.train.txt $FASTALIGN_DIR/opus100_filtered/en-$lang.train
    fi

    # Align with bilingual dictionaries
    if [ ! -f $DICOALIGN_DIR/opus100_filtered/en-$lang.train ]; then
        python scripts/word_align_with_dictionary.py $TRANSLATION_DIR/opus100_filtered/en-$lang.tokenized.train.txt $MUSE_DIR en $lang $DICOALIGN_DIR/opus100_filtered/en-$lang.train
    fi

    # Align with AWESOME-align
    if [ ! -f $AWESOME_DIR/opus100_filtered/en-$lang.train ]; then
        source $DATA_DIR/venvs/awesome-align/bin/activate
        bash scripts/word_align_with_awesome.sh $TRANSLATION_DIR/opus100_filtered/en-$lang.tokenized.train.txt $AWESOME_DIR/opus100_filtered/en-$lang.train $AWESOME_MODEL_DIR/model_without_co
        deactivate
    fi

done

##################
# FINE-TUNING ONLY
##################

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies baseline \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

################
# BASELINES
################

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies before_fastalign \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies before_awesome \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies before_dico \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies during_fastalign \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies during_awesome \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100_filtered \
        --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
        --dico_dir $DICOALIGN_DIR/opus100_filtered \
        --awesome_dir $AWESOME_DIR/opus100_filtered \
        --strategies during_dico \
        --models xlm-roberta-base \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
        --use_wandb --project_prefix "filtered_0.7_"
fi 

###########################################
# PARTIAL-REALIGNMENT BEFORE - FRONT FROZEN
###########################################

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_fastalign \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_awesome \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_dico \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# ###########################################
# # PARTIAL-REALIGNMENT BEFORE - BACK FROZEN
# ###########################################

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_last_6_fastalign \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_last_6_awesome \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies freeze_realign_unfreeze_last_6_dico \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# ###########################################
# # PARTIAL-REALIGNMENT DURING - FRONT FROZEN
# ###########################################

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_front_fastalign \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_front_awesome \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_front_dico \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# ##########################################
# # PARTIAL-REALIGNMENT DURING - BACK FROZEN
# ##########################################

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_back_fastalign \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_back_awesome \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_filtered.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100_filtered \
#         --fastalign_dir $FASTALIGN_DIR/opus100_filtered \
#         --dico_dir $DICOALIGN_DIR/opus100_filtered \
#         --awesome_dir $AWESOME_DIR/opus100_filtered \
#         --strategies during_partial_freeze_back_dico \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_filtered_0.7_tagging_large_staged_filtered.csv $ADD_ARGS \
#         --use_wandb --project_prefix "filtered_0.7_"
# fi 