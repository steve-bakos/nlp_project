#!/bin/bash

set -e

OUTPUT_DIR=$1

download_gcs () {
    local id=$1
    local output=$2
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${id}" -O $output && rm -rf /tmp/cookies.txt
}

if [ ! -f $OUTPUT_DIR/en-fr/UNv1.0.en-fr.fr ]; then
    download_gcs 126paJ81dFHu1wSiXrM47bkFuHDMMHpN_ $OUTPUT_DIR/en-fr.tar.gz.00
    download_gcs 1bs617LoEkn84O_lwOl1NSl4tpF0PvwvE $OUTPUT_DIR/en-fr.tar.gz.01
    download_gcs 11LzF_iQo3-8pwov6Iu7ZeMBl3nOOJECL $OUTPUT_DIR/en-fr.tar.gz.02
    cat $OUTPUT_DIR/en-fr.tar.gz.* > $OUTPUT_DIR/en-fr.tar.gz

    rm -f $OUTPUT_DIR/en-fr.tar.gz.*

    tar -xzf $OUTPUT_DIR/en-fr.tar.gz -C $OUTPUT_DIR

    rm $OUTPUT_DIR/en-fr.tar.gz
fi

if [ ! -f $OUTPUT_DIR/en-es/UNv1.0.en-es.es ]; then
    download_gcs 1WFjXnj0i2yhubmFJFAhZ4-Ap4YHh-1p5 $OUTPUT_DIR/en-es.tar.gz.00
    download_gcs 1I-uE0On0l33kbi23qxSMKl5nPzqVE4Wj $OUTPUT_DIR/en-es.tar.gz.01
    cat $OUTPUT_DIR/en-es.tar.gz.* > $OUTPUT_DIR/en-es.tar.gz

    rm -f $OUTPUT_DIR/en-es.tar.gz.*

    tar -xzf $OUTPUT_DIR/en-es.tar.gz -C $OUTPUT_DIR

    rm $OUTPUT_DIR/en-es.tar.gz
fi

if [ ! -f $OUTPUT_DIR/en-ru/UNv1.0.en-ru.ru ]; then
    download_gcs 1kM4FV2d7tBXmjWx0_Nc-klDZNRoKCRHY $OUTPUT_DIR/en-ru.tar.gz.00
    download_gcs 1T76T6SsB3PL0OUjfGxegbFZaLhz7rUjG $OUTPUT_DIR/en-ru.tar.gz.01
    download_gcs 17xvY_z-tGgqM-QiC9mwQus5WkxRYQRSI $OUTPUT_DIR/en-ru.tar.gz.02
    cat $OUTPUT_DIR/en-ru.tar.gz.* > $OUTPUT_DIR/en-ru.tar.gz

    rm -f $OUTPUT_DIR/en-ru.tar.gz.*

    tar -xzf $OUTPUT_DIR/en-ru.tar.gz -C $OUTPUT_DIR

    rm $OUTPUT_DIR/en-ru.tar.gz
fi

if [ ! -f $OUTPUT_DIR/en-zh/UNv1.0.en-zh.zh ]; then
    download_gcs 1rv2Yh5j-5da5RZO3DEaYvYRZKxE841hT $OUTPUT_DIR/en-zh.tar.gz.00
    download_gcs 1cfUezEOv5UPzF-d1uIm9-dkIUjtyZ9ys $OUTPUT_DIR/en-zh.tar.gz.01
    cat $OUTPUT_DIR/en-zh.tar.gz.* > $OUTPUT_DIR/en-zh.tar.gz

    rm -f $OUTPUT_DIR/en-zh.tar.gz.*

    tar -xzf $OUTPUT_DIR/en-zh.tar.gz -C $OUTPUT_DIR

    rm $OUTPUT_DIR/en-zh.tar.gz
fi

if [ ! -f $OUTPUT_DIR/en-ar/UNv1.0.en-ar.ar ]; then
    download_gcs 1EIf7Gh8bPO-gPm69r_F_kI076Fx75zr8 $OUTPUT_DIR/en-ar.tar.gz.00
    download_gcs 1jv9UOCzspiEgTR5VI13OKd2uGOTfA-gx $OUTPUT_DIR/en-ar.tar.gz.01
    cat $OUTPUT_DIR/en-ar.tar.gz.* > $OUTPUT_DIR/en-ar.tar.gz

    rm -f $OUTPUT_DIR/en-ar.tar.gz.*

    tar -xzf $OUTPUT_DIR/en-ar.tar.gz -C $OUTPUT_DIR

    rm $OUTPUT_DIR/en-ar.tar.gz
fi