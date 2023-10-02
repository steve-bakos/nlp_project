"""
Source code for language-specific preprocessing
"""
import logging

from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter


class StanfordSegmenterWithLabelAlignmentMapper:
    """
    Class for a Callable for mapping datasets (as argument of Dataset.map) that use the Stanford Segmenter
    to re-segment a chinese token classification dataset that has been tokenized by character (such as wikiann)
    """

    def __init__(self, zh_segmenter: StanfordSegmenter, label_name="labels"):
        """
        - label_name, str: name of the label in the original dataset
        """
        self.label_name = label_name
        self.zh_segmenter = zh_segmenter

    def relabel(self, labels, new_tokens):
        offset = 0

        new_labels = []

        new_segments = []

        for new_token in new_tokens:
            segments_to_add = []
            labels_to_add = []
            current_segment = ""
            current_label = None
            for char, label in zip(new_token, labels[offset : offset + len(new_token)]):
                # if current segment was previously empty, create it with current label
                if current_label is None:
                    current_segment += char
                    current_label = label
                # if the label is of type B-X, create a new segment
                elif label % 2 == 1:
                    segments_to_add.append(current_segment)
                    labels_to_add.append(current_label)
                    current_segment = char
                    current_label = label
                # if the label is equal to previous or previous is B-X and new is I-X, keep the segment together
                elif current_label == label or (
                    current_label % 2 == 1 and label == current_label + 1
                ):
                    current_segment += char
                # otherwise, resegment
                else:
                    segments_to_add.append(current_segment)
                    labels_to_add.append(current_label)
                    current_segment = char
                    current_label = label

            if len(current_segment) > 0:
                segments_to_add.append(current_segment)
                labels_to_add.append(current_label)

            new_segments += segments_to_add
            new_labels += labels_to_add

            offset += len(new_token)

        new_tokens = new_segments

        return new_tokens, new_labels

    def __call__(self, example):
        tokens = example["tokens"]
        labels = example[self.label_name]

        if "None" in tokens:
            logging.warning(f"Found 'None' token in sentence: {tokens}")
            tokens = list(map(lambda x: " " if x == "None" else x, tokens))

        if any(map(lambda x: len(x) != 1, tokens)):
            raise Exception(
                f"StanfordSegmenterWithLabelAlignmentMapper expects character-tokenized text. Got: {tokens}"
            )

        sent = "".join(tokens)

        new_tokens = self.zh_segmenter(sent)

        new_tokens, new_labels = self.relabel(labels, new_tokens)

        return {"tokens": new_tokens, self.label_name: new_labels}
