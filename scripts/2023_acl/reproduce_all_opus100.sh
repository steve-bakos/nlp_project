#!/bin/bash

set -e

DATA_DIR=$1
ADD_ARGS=$2

langs="ar es fr ru zh"

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
    source ../../$DATA_DIR/venvs/awesome-align/bin/activate
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


# if [ ! -f $RESULT_DIR/finetuning_and_alignment_weak.csv ]; then
#     echo ""
#     echo "Running finetuning_and_alignment.py for weak alignment..."
#     echo ""
#     python scripts/2023_acl/finetuning_and_alignment.py \
#         $TRANSLATION_DIR/opus100 \
#         $DICOALIGN_DIR/opus100 \
#         --cache_dir $CACHE_DIR \
#         --output_file $RESULT_DIR/finetuning_and_alignment_weak.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi


# if [ ! -f $RESULT_DIR/finetuning_and_alignment_strong.csv ]; then
#     echo ""
#     echo "Running finetuning_and_alignment.py for strong alignment..."
#     echo ""
#     python scripts/2023_acl/finetuning_and_alignment.py \
#         $TRANSLATION_DIR/opus100 \
#         $DICOALIGN_DIR/opus100 \
#         --cache_dir $CACHE_DIR \
#         --strong_alignment \
#         --output_file $RESULT_DIR/finetuning_and_alignment_strong.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_tagging.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --strategies baseline \
#         --models xlm-roberta-base \
#         --tasks wikiann \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_tagging_large_staged.csv $ADD_ARGS \
#         --use_wandb
# fi 

if [ ! -f $RESULT_DIR/controlled_realignment_opus100_tagging.csv ]; then
    echo ""
    echo "Testing controlled_realignment.py staged-realignment..."
    echo ""
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/opus100 \
        --fastalign_dir $FASTALIGN_DIR/opus100 \
        --dico_dir $DICOALIGN_DIR/opus100 \
        --awesome_dir $AWESOME_DIR/opus100 \
        --strategies freeze_realign_finetune_0to2_9to11_fastalign \
        --models xlm-roberta-base \
        --tasks wikiann \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --output_file $RESULT_DIR/controlled_realignment_opus100_tagging_large_staged.csv $ADD_ARGS \
        --
fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_tagging.csv ]; then
#     echo ""
#     echo "Testing controlled_realignment.py staged-realignment..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --strategies baseline \
#         --models xlm-roberta-base \
#         --tasks udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_tagging_large_staged.csv $ADD_ARGS \
#         --use_wandb
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_tagging.csv ]; then
#     echo ""
#     echo "Running controlled_realignment.py experiment 1..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --tasks wikiann udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_tagging.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_xnli.csv ]; then
#     echo ""
#     echo "Running controlled_realignment.py experiment 2..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --strategies baseline during_dico before_dico \
#         --tasks xnli \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 2 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_xnli.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_tagging_large_baseline.csv ]; then
#     echo ""
#     echo "Running controlled_realignment.py experiment 3..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --strategies baseline \
#         --models xlm-roberta-large \
#         --tasks wikiann udpos \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 5 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_tagging_large_baseline.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi 

# if [ ! -f $RESULT_DIR/controlled_realignment_opus100_xnli_large_baseline.csv ]; then
#     echo ""
#     echo "Running controlled_realignment.py experiment 4..."
#     echo ""
#     python scripts/2023_acl/controlled_realignment.py \
#         --translation_dir $TRANSLATION_DIR/opus100 \
#         --fastalign_dir $FASTALIGN_DIR/opus100 \
#         --dico_dir $DICOALIGN_DIR/opus100 \
#         --awesome_dir $AWESOME_DIR/opus100 \
#         --strategies baseline \
#         --models xlm-roberta-large \
#         --tasks xnli \
#         --cache_dir $CACHE_DIR \
#         --n_epochs 2 \
#         --output_file $RESULT_DIR/controlled_realignment_opus100_xnli_large_baseline.csv $ADD_ARGS \
#         --debug \
#         --use_wandb
# fi 