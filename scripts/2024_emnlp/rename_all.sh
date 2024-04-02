set -e

DATA_DIR=$1

thresholds="0.4 0.5 0.6 0.7 0.8"
langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"

for thresh in $thresholds; do
    for lang in $langs; do
        mv $DATA_DIR/translation/opus100_filtered_$thresh/en-$lang.train $DATA_DIR/translation/opus100_filtered_$thresh/en-$lang.tokenized.train.txt
    done
done 