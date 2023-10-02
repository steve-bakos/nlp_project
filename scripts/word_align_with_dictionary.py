import sys
import os
import logging

sys.path.append(os.curdir)

from multilingual_eval.datasets.realignment_dataset import DatasetMapperForRealignment


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("dico_path", type=str)
    parser.add_argument("left_lang", type=str)
    parser.add_argument("right_lang", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    mapper = DatasetMapperForRealignment(args.left_lang, args.right_lang, args.dico_path)

    with open(args.input_file) as reader, open(args.output_file, "w") as writer:

        for line in reader:
            parts = line.split("|||")
            if len(parts) != 2:
                logging.error(f"Found {len(parts)} splits instead of two in the following line:\n{line}\nCan't find alignment")
                writer.write("\n")
            left_sent, right_sent = parts
            left_tokens = list(filter(lambda x: len(x)> 0, left_sent.split()))
            right_tokens = list(filter(lambda x: len(x)> 0, right_sent.split()))

            left_ids, right_ids = mapper.find_aligned_words(left_tokens, right_tokens)

            writer.write(" ".join(map(lambda x: "-".join(map(str,x)), zip(left_ids, right_ids))) + "\n")