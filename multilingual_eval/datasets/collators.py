import torch
from transformers import DataCollatorWithPadding


class RealignmentCollator:
    """
    Data collator for building and padding batch for the realignment task
    """

    def __init__(self, tokenizer, **kwargs):
        self.usual_collator = DataCollatorWithPadding(tokenizer, **kwargs)

    def __call__(self, examples):
        left_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("left_")}
            for sample in examples
        ]
        right_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("right_")}
            for sample in examples
        ]
        batch_left = {f"left_{k}": v for k, v in self.usual_collator(left_inputs).items()}
        batch_right = {f"right_{k}": v for k, v in self.usual_collator(right_inputs).items()}

        max_nb = max(map(lambda x: x["alignment_nb"], examples))
        max_left_length = max(map(lambda x: x["alignment_left_length"], examples))
        max_right_length = max(map(lambda x: x["alignment_right_length"], examples))

        alignment_left_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)
        alignment_right_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)
        alignment_left_positions = torch.zeros(
            (len(examples), max_left_length, 2), dtype=torch.long
        )
        alignment_right_positions = torch.zeros(
            (len(examples), max_right_length, 2), dtype=torch.long
        )

        for i, ex in enumerate(examples):
            alignment_left_ids[i, : ex["alignment_nb"]] = torch.LongTensor(ex["alignment_left_ids"])
            alignment_right_ids[i, : ex["alignment_nb"]] = torch.LongTensor(
                ex["alignment_right_ids"]
            )
            alignment_left_positions[i, : ex["alignment_left_length"]] = torch.LongTensor(
                ex["alignment_left_positions"]
            )
            alignment_right_positions[i, : ex["alignment_right_length"]] = torch.LongTensor(
                ex["alignment_right_positions"]
            )

        return {
            **batch_left,
            **batch_right,
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_nb": torch.LongTensor([ex["alignment_nb"] for ex in examples]),
            "alignment_left_length": torch.LongTensor(
                [ex["alignment_left_length"] for ex in examples]
            ),
            "alignment_right_length": torch.LongTensor(
                [ex["alignment_right_length"] for ex in examples]
            ),
        }
