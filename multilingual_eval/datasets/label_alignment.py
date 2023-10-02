from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast


class LabelAlignmentMapper:
    """
    Class for a callable that can be used as argument of datasets.Dataset.map()
    It will perform label alignment for token classification tasks
    """

    def __init__(
        self,
        tokenizer,
        label_name="labels",
        first_subword_only=False,
        max_length=None,
        return_overflowing_tokens=False,
        stride=0,
        remove_underscore_if_roberta=True,
    ):
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.first_subword_only = first_subword_only
        self.max_length = max_length
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride
        self.remove_underscore = remove_underscore_if_roberta and isinstance(
            self.tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)
        )
        if self.remove_underscore:
            self.underscore_id = self.tokenizer.convert_tokens_to_ids(["▁"])[0]

    def __call__(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_overflowing_tokens=self.return_overflowing_tokens,
            stride=self.stride,
        )

        n = len(tokenized_inputs["input_ids"])

        if self.remove_underscore:
            result = {key: [] for key in tokenized_inputs if key != "overflow_to_sample_mapping"}

        labels = []
        for i in range(n):
            previous_batch_id = tokenized_inputs.get("overflow_to_sample_mapping", list(range(n)))[
                i
            ]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            if self.remove_underscore:
                result_sample = {key: [] for key in tokenized_inputs if key in result}
            label = examples[self.label_name][previous_batch_id]
            for j, word_idx in enumerate(word_ids):

                if self.remove_underscore:
                    if tokenized_inputs["input_ids"][i][j] == self.underscore_id:
                        continue
                    for key in result_sample:
                        result_sample[key].append(tokenized_inputs[key][i][j])

                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the first_subword_only flag.
                elif self.first_subword_only:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

            if self.remove_underscore:
                for key, value in result_sample.items():
                    result[key].append(value)

        if self.remove_underscore:
            tokenized_inputs = {
                **result,
                **(
                    {
                        "overflow_to_sample_mapping": tokenized_inputs.get(
                            "overflow_to_sample_mapping"
                        )
                    }
                    if "overflow_to_sample_mapping" in tokenized_inputs
                    else {}
                ),
            }

        tokenized_inputs["labels"] = labels

        return tokenized_inputs
