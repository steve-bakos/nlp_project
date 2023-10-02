from datasets.iterable_dataset import IterableDataset, ExamplesIterable
import os


class PositionAlignmentMapper:
    """
    Take parsed output of FastAlign and build a dataset for huggingface models
    """

    def __init__(self, tokenizer, max_length=None, first_subword_only=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.first_subword_only = first_subword_only

    def __call__(self, examples):
        """
        examples is a dictionary with keys:
        - left_tokens: list of str for the tokens of left sentence
        - right_tokens: list of str for the tokens of right sentence (translation)
        - left_positions: list of int for the position of each left element of a pair of aligned word
        - right_positions: list of int for the corresponding right element of each alignment pair
        """
        tokenized_left = self.tokenizer(
            examples["left_tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )
        tokenized_right = self.tokenizer(
            examples["right_tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )

        alignment_left_ids = []
        alignment_right_ids = []
        alignment_left_positions = []
        alignment_right_positions = []
        alignment_nb = []
        alignment_left_length = []
        alignment_right_length = []
        for i, (left_pos, right_pos) in enumerate(
            zip(examples["left_positions"], examples["right_positions"])
        ):
            left_word_ids = tokenized_left.word_ids(batch_index=i)
            right_word_ids = tokenized_right.word_ids(batch_index=i)

            this_alignment_left_ids = []
            this_alignment_right_ids = []
            this_alignment_left_positions = []
            this_alignment_right_positions = []

            previous_word_idx = None
            previous_position = None
            left_word_id_to_index = {}
            for j, word_idx in enumerate(left_word_ids):
                # record position if necessary
                if previous_word_idx is not None and word_idx != previous_word_idx:
                    this_alignment_left_positions.append(previous_position)
                    left_word_id_to_index[previous_word_idx] = (
                        len(this_alignment_left_positions) - 1
                    )

                if word_idx is None:
                    previous_position = None
                elif word_idx != previous_word_idx:
                    previous_position = [j, j + 1]
                elif not self.first_subword_only:
                    previous_position[1] = j + 1

                previous_word_idx = word_idx

            previous_word_idx = None
            previous_position = None
            right_word_id_to_index = {}
            for j, word_idx in enumerate(right_word_ids):
                # record position if necessary
                if previous_word_idx is not None and word_idx != previous_word_idx:
                    this_alignment_right_positions.append(previous_position)
                    right_word_id_to_index[previous_word_idx] = (
                        len(this_alignment_right_positions) - 1
                    )

                if word_idx is None:
                    previous_position = None
                elif word_idx != previous_word_idx:
                    previous_position = [j, j + 1]
                elif not self.first_subword_only:
                    previous_position[1] = j + 1

                previous_word_idx = word_idx

            for left_word_idx, right_word_idx in zip(left_pos, right_pos):
                if (
                    left_word_idx not in left_word_id_to_index
                    or right_word_idx not in right_word_id_to_index
                ):
                    continue
                this_alignment_left_ids.append(left_word_id_to_index[left_word_idx])
                this_alignment_right_ids.append(right_word_id_to_index[right_word_idx])

            alignment_left_ids.append(this_alignment_left_ids)
            alignment_right_ids.append(this_alignment_right_ids)
            alignment_left_positions.append(this_alignment_left_positions)
            alignment_right_positions.append(this_alignment_right_positions)

            alignment_nb.append(len(this_alignment_left_ids))
            alignment_left_length.append(len(this_alignment_left_positions))
            alignment_right_length.append(len(this_alignment_right_positions))

        return {
            **{f"left_{k}": v for k, v in tokenized_left.items()},
            **{f"right_{k}": v for k, v in tokenized_right.items()},
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_nb": alignment_nb,
            "alignment_left_length": alignment_left_length,
            "alignment_right_length": alignment_right_length,
        }


def get_raw_fastalign_realignment_dataset(
    fastalign_input_file: str,
    fastalign_output_file: str,
    keep_only_one_to_one=True,
    ignore_identical=True,
):
    """
    Parses FastAlign output and corresponding aligned dataset such as to yield dictionaries with keys:
    - left_tokens: list of str for the tokens of left sentence
    - right_tokens: list of str for the tokens of right sentence (translation)
    - left_positions: list of int for the position of each left element of a pair of aligned word
    - right_positions: list of int for the corresponding right element of each alignment pair
    """

    def generator():
        with open(fastalign_input_file, "r") as input_f, open(
            fastalign_output_file, "r"
        ) as output_f:
            for i, (input_line, output_line) in enumerate(zip(input_f, output_f)):
                parts = input_line.split("|||")
                if len(parts) != 2:
                    continue
                left_sent, right_sent = parts
                left_tokens = left_sent.strip().split(" ")
                right_tokens = right_sent.strip().split(" ")

                pairs = output_line.strip().split(" ")
                pairs = list(map(lambda x: tuple(map(int, x.split("-"))), pairs))

                left_positions = list(map(lambda x: x[0], pairs))
                right_positions = list(map(lambda x: x[1], pairs))

                if keep_only_one_to_one:
                    new_left_positions = []
                    new_right_positions = []
                    for a, b in zip(left_positions, right_positions):
                        if left_positions.count(a) > 1 or right_positions.count(b) > 1:
                            continue
                        new_left_positions.append(a)
                        new_right_positions.append(b)
                    left_positions = new_left_positions
                    right_positions = new_right_positions

                if ignore_identical:
                    new_left_positions = []
                    new_right_positions = []
                    for a, b in zip(left_positions, right_positions):
                        if left_tokens[a] == right_tokens[b]:
                            continue
                        new_left_positions.append(a)
                        new_right_positions.append(b)
                    left_positions = new_left_positions
                    right_positions = new_right_positions

                yield i, {
                    "left_tokens": left_tokens,
                    "right_tokens": right_tokens,
                    "left_positions": left_positions,
                    "right_positions": right_positions,
                }

    return IterableDataset(ExamplesIterable(generator, {}))


def get_fastalign_realignment_dataset(
    tokenizer,
    fastalign_input_file: str,
    fastalign_output_file: str,
    max_length=None,
    first_subword_only=True,
    ignore_identical=True,
):
    """
    Get realignment dataset from a parallel dataset in FastAlign format and FastAlign output
    """
    raw_dataset = get_raw_fastalign_realignment_dataset(
        fastalign_input_file, fastalign_output_file, ignore_identical=ignore_identical
    )
    mapper = PositionAlignmentMapper(
        tokenizer, max_length=max_length, first_subword_only=first_subword_only
    )
    return raw_dataset.map(mapper, batched=True).remove_columns(
        ["left_positions", "right_positions", "left_tokens", "right_tokens"]
    )


def get_fastalign_realignment_dataset_from_path(
    tokenizer,
    fastalign_path: str,
    left_lang: str,
    right_lang: str,
    left_lang_id=0,
    right_lang_id=0,
    dataset_name="news_commentary",
    max_length=None,
    first_subword_only=True,
    ignore_identical=True,
    seed=None,
):
    """
    Get interleaved realignment dataset for different languages based on FastAlign,
    simply from the directory where inputs and outputs of FastAlign are kept
    """

    fastalign_input_file = os.path.join(
        fastalign_path, f"{dataset_name}_{left_lang}_{right_lang}.txt"
    )
    fastalign_ouput_file = os.path.join(
        fastalign_path, f"{dataset_name}_{left_lang}_{right_lang}", "symmetrized.align"
    )
    dataset = get_fastalign_realignment_dataset(
        tokenizer,
        fastalign_input_file,
        fastalign_ouput_file,
        max_length=max_length,
        first_subword_only=first_subword_only,
        ignore_identical=ignore_identical,
    )
    dataset = (
        dataset.map(
            lambda x: {**x, "left_lang_id": [left_lang_id], "right_lang_id": [right_lang_id]}
        )
        .filter(lambda x: len(x["alignment_left_ids"]) > 0)
        .shuffle(seed=seed)
        .with_format("torch")
    )
    return dataset


if __name__ == "__main__":

    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("translation_file", type=str)
    parser.add_argument("alignment_file", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = get_fastalign_realignment_dataset(
        tokenizer, args.translation_file, args.alignment_file
    )

    for i, sample in enumerate(dataset):
        if i >= 10:
            break
        print(sample)
