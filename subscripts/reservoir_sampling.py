"""
Performs reservoir sampling of the lines in a list of parallel files
(meaning that the same lines from each input files must be sampled together)
"""

import random
from contextlib import ExitStack


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", type=str, nargs="+")
    parser.add_argument("--output_files", type=str, nargs="+")
    parser.add_argument("--num_samples", type=int, default=1_000_000)
    args = parser.parse_args()

    line_positions = []

    input_files = args.input_files
    output_files = args.output_files

    if not isinstance(input_files, list):
        input_files = [input_files]
    if not isinstance(output_files, list):
        output_files = [output_files]

    assert len(input_files) == len(output_files)

    with ExitStack() as stack:
        file_readers = [stack.enter_context(open(fname, "rb")) for fname in input_files]
        positions = [f.tell() for f in file_readers]

        for i, lines in enumerate(zip(*file_readers)):
            if i < args.num_samples:
                line_positions.append(positions)
            else:
                j = random.randrange(i + 1)
                if j < args.num_samples:
                    line_positions[j] = positions
            positions = [f.tell() for f in file_readers]

        writers = [stack.enter_context(open(fname, "wb")) for fname in output_files]
        for pos_list in line_positions:
            for f, pos in zip(file_readers, pos_list):
                f.seek(pos)
            for writer, reader in zip(writers, file_readers):
                writer.write(reader.readline())
