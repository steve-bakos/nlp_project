#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

for lang in $langs; do

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    if [ ! -d $OUTPUT_DIR/$pair ]; then

        wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-$pair-v1.0.tar.gz -O $OUTPUT_DIR/$pair.tar.gz

        tar -xvzf $OUTPUT_DIR/$pair.tar.gz -C $OUTPUT_DIR

        mv $OUTPUT_DIR/opus-100-corpus/v1.0/supervised/$pair $OUTPUT_DIR/$pair   

        rm $OUTPUT_DIR/$pair.tar.gz
    fi
done

if [ -d $OUTPUT_DIR/opus-100-corpus ]; then
    rm -rf $OUTPUT_DIR/opus-100-corpus
fi