import os
import re

def remove_specific_spaces(filename):
    new_lines = []
    found_different = False
    with open(filename) as f:
        for line in f:
            parts = line.strip().split(" ||| ")
            if len(parts) != 2:
                    continue
            left, right = parts
            new_left = " ".join(left.split())
            new_right = " ".join(right.split())

            if left != new_left or right != new_right:
                found_different = True
            
            new_lines.append(f"{new_left} ||| {new_right}\n")

    if found_different:
        print(f"Found specific spaces in file: {filename}")
        with open(filename, "w") as f:
            for line in new_lines:
                f.write(line)

if __name__ == "__main__":
       
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    for fname in os.listdir(args.dir):
        if not re.match(r"en-[a-z]+\.tokenized\.train\.txt", fname):
            continue
        remove_specific_spaces(os.path.join(args.dir, fname))