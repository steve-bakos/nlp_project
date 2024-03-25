#!/bin/bash

set -e

DATA_DIR=$1
ADD_ARGS=$2

#langs="ar es fr ru zh af fa hi"
langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
OPUS_DIR=$DATA_DIR/opus100
MUSE_DIR=$DATA_DIR/muse_dictionaries
AWESOME_MODEL_DIR=$DATA_DIR/awesome-models
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/raw_results

mkdir -p $CACHE_DIR
mkdir -p $OPUS_DIR
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

# download OPUS 100
bash download_resources/opus100.sh $OPUS_DIR "$langs"

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

    mkdir -p $TRANSLATION_DIR/opus100
    mkdir -p $DICOALIGN_DIR/opus100
    mkdir -p $AWESOME_DIR/opus100

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    python subscripts/prepare_pharaoh_dataset.py \
        $OPUS_DIR/$pair/opus.$pair-train.en \
        $OPUS_DIR/$pair/opus.$pair-train.$lang \
        $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang


    # Align with FastAlign
    if [ ! -f $FASTALIGN_DIR/opus100/en-$lang.train ]; then
        bash scripts/word_align_with_fastalign.sh $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $FASTALIGN_DIR/opus100/en-$lang.train
    fi

    # Align with bilingual dictionaries
    if [ ! -f $DICOALIGN_DIR/opus100/en-$lang.train ]; then
        python scripts/word_align_with_dictionary.py $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $MUSE_DIR en $lang $DICOALIGN_DIR/opus100/en-$lang.train
    fi

    # Align with AWESOME-align
    if [ ! -f $AWESOME_DIR/opus100/en-$lang.train ]; then
        source $DATA_DIR/venvs/awesome-align/bin/activate
        bash scripts/word_align_with_awesome.sh $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt $AWESOME_DIR/opus100/en-$lang.train $AWESOME_MODEL_DIR/model_without_co
        deactivate
    fi

done

python subscripts/pharaoh_postprocessing.py $TRANSLATION_DIR/opus100

##################
# FINE-TUNING ONLY
##################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies baseline \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


################
# BASELINES
################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies before_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies before_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies before_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - FRONT FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT BEFORE - BACK FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_last_6_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_last_6_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies freeze_realign_unfreeze_last_6_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


###########################################
# PARTIAL-REALIGNMENT DURING - FRONT FROZEN
###########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_front_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_front_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_front_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


##########################################
# PARTIAL-REALIGNMENT DURING - BACK FROZEN
##########################################

echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_back_fastalign \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_back_awesome \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS


echo ""
echo "Testing controlled_realignment.py staged-realignment..."
echo ""
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/opus100 \
    --fastalign_dir $FASTALIGN_DIR/opus100 \
    --dico_dir $DICOALIGN_DIR/opus100 \
    --awesome_dir $AWESOME_DIR/opus100 \
    --strategies during_partial_freeze_back_dico \
    --models xlm-roberta-base \
    --tasks udpos \
    --cache_dir $CACHE_DIR \
    --n_epochs 5 \
    --right_langs $langs \
    --use_wandb --project_prefix "34langs_" $ADD_ARGS
