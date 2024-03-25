"""
Tokenize OPUS100 dataset and produce a FastAlign-compatile output 
"""

import os
import sys
from typing import Optional
from contextlib import ExitStack
import logging
from tqdm import tqdm  # <-- Import tqdm

sys.path.append(os.curdir)

from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.utils import LanguageSpecificTokenizer
from multilingual_eval.utils import count_lines

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("left_file")
    parser.add_argument("right_file")
    parser.add_argument("output_file")
    parser.add_argument("--left_lang", type=str, default=None)
    parser.add_argument("--right_lang", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", dest="overwrite", help="Option to restart from where it stopped")
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    left_lang = args.left_lang
    right_lang = args.right_lang

    left_lang_file = args.left_file
    right_lang_file = args.right_file

    # Compute number of lines already parsed if not overriding
    if not args.overwrite and os.path.isfile(args.output_file):
        lines_to_parse = count_lines(left_lang_file)
        lines_parsed = count_lines(args.output_file)

        start_line = lines_to_parse if lines_parsed >= lines_to_parse else lines_parsed
    else:
        start_line = 0

    with ExitStack() as stack:
        if "zh" in [left_lang, right_lang]:
            zh_segmenter = stack.enter_context(StanfordSegmenter())
        else:
            zh_segmenter = None
        left_reader = stack.enter_context(open(left_lang_file, "r"))
        right_reader = stack.enter_context(open(right_lang_file, "r"))
        writer = stack.enter_context(open(args.output_file, "w" if args.overwrite else "a"))

        logging.info(f"Skipping {start_line} lines")
        for _ in range(start_line):
            _ = next(left_reader)
            _ = next(right_reader)

        left_tokenizer = LanguageSpecificTokenizer(lang=left_lang, zh_segmenter=zh_segmenter)
        right_tokenizer = LanguageSpecificTokenizer(lang=right_lang, zh_segmenter=zh_segmenter)

        # Wrap the zip iterator with tqdm for progress bar
        for left_line, right_line in tqdm(zip(left_reader, right_reader), total=count_lines(left_lang_file) - start_line, desc="Tokenizing"):
            left_tokens = left_tokenizer.tokenize(left_line)
            right_tokens = right_tokenizer.tokenize(right_line)

            writer.write(" ".join(left_tokens).strip() + " ||| " + " ".join(right_tokens).strip() + "\n")
