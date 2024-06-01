import os
import re

def average_token_ratio(filename):
    ratio_accumulation = 0
    nb_samples = 0
    with open(filename) as f:
        for line in f:
            parts = line.strip().split(" ||| ")
            if len(parts) != 2:
                    continue
            left, right = parts
            left = left.strip()
            right = right.strip()
            nb_left = len(list(filter(len, left.split())))
            nb_right = len(list(filter(len, right.split())))
            if nb_left == 0 or nb_right == 0:
                    continue
            ratio_accumulation += nb_left / nb_right
            nb_samples += 1
    return ratio_accumulation / nb_samples

if __name__ == "__main__":
       
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    for fname in os.listdir(args.dir):
        if not re.match(r"en-[a-z]+\.tokenized\.train\.txt", fname):
            continue
        ratio = average_token_ratio(os.path.join(args.dir, fname))
        print(f"{fname}: {ratio:.2f} (inverse={1/ratio:.2f})")