
set -e

DATA_DIR=$1

TRANSLATION_DIR=$DATA_DIR/translation

mkdir -p $TRANSLATION_DIR

export DATA_DIR=$DATA_DIR

langs="ar es fr ru zh af fa hi"

for lang in $langs; do
    echo "parsing lang $lang for opus-100"

    mkdir -p $TRANSLATION_DIR/opus100_filtered

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")


    if [ ! -f $TRANSLATION_DIR/opus100_filtered/en-$lang.tokenized.train.txt ]; then
        python scripts/2024_emnlp/qualityFilter.py \
            $TRANSLATION_DIR/opus100/en-$lang.tokenized.train.txt \
            $TRANSLATION_DIR/opus100_filtered/en-$lang.tokenized.train.txt
    fi
done