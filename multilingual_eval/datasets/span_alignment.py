import logging


class SpanAligner:

    """
    Class for a callable that can be used as argument of datasets.Dataset.map()
    It will perform pre-processing for span extraction (question answering)
    """

    def __init__(self, tokenizer, max_length=128, stride=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def __call__(self, examples):
        """
        This code is drawn from here:

        https://github.com/huggingface/transformers/blob/588faad1062198e45cf3aebed21dc1fc1e1ed0d7/examples/pytorch/question-answering/run_qa.py#L370
        """

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        try:
            tokenized_examples = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.max_length else False,
            )
        except:
            exclude_ids = []
            for i, (question, context) in enumerate(zip(examples["question"], examples["context"])):
                try:
                    self.tokenizer(
                        [question],
                        [context],
                        truncation="only_second",
                        max_length=self.max_length,
                        stride=self.stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length" if self.max_length else False,
                    )
                except:
                    logging.error(
                        f"Error with example, will ignore it. question: {question} / context: {context}"
                    )
                    exclude_ids.append(i)
            examples = {
                k: [v[i] for i in range(len(v)) if i not in exclude_ids]
                for k, v in examples.items()
            }
            tokenized_examples = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.max_length else False,
            )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
