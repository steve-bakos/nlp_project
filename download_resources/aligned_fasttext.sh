#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

for lang in $langs; do
    if [ ! -f $OUTPUT_DIR/wiki.$lang.align.vec ]; then
        wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.$lang.align.vec -O $OUTPUT_DIR/wiki.$lang.align.vec
    fi
done