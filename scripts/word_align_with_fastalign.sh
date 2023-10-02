
set -e

INPUT_FILE=$1
OUTPUT_FILE=$2

mkdir -p $(dirname $OUTPUT_FILE)

./tools/fast_align/build/fast_align -i $INPUT_FILE -d -o -v > $OUTPUT_FILE.forward
./tools/fast_align/build/fast_align -i $INPUT_FILE -d -o -v -r > $OUTPUT_FILE.backward
./tools/fast_align/build/atools -i $OUTPUT_FILE.forward -j $OUTPUT_FILE.backward -c grow-diag-final-and > $OUTPUT_FILE
