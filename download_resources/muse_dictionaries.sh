#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

for lang in $langs; do
    if [ ! -f $OUTPUT_DIR/en-$lang.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.txt -O $OUTPUT_DIR/en-$lang.txt
    fi
    if [ ! -f $OUTPUT_DIR/$lang-en.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.txt -O $OUTPUT_DIR/$lang-en.txt
    fi
    if [ ! -f $OUTPUT_DIR/en-$lang.0-5000.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.0-5000.txt -O $OUTPUT_DIR/en-$lang.0-5000.txt
    fi
    if [ ! -f $OUTPUT_DIR/$lang-en.0-5000.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.0-5000.txt -O $OUTPUT_DIR/$lang-en.0-5000.txt
    fi
    if [ ! -f $OUTPUT_DIR/en-$lang.5000-6500.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.5000-6500.txt -O $OUTPUT_DIR/en-$lang.5000-6500.txt
    fi
    if [ ! -f $OUTPUT_DIR/$lang-en.5000-6500.txt ]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$lang.5000-6500.txt -O $OUTPUT_DIR/$lang-en.5000-6500.txt
    fi
done