model_to_batch_size_small = {
    "xlm-roberta-base": 2,
    "bert-base-multilingual-cased": 4,
    "distilbert-base-multilingual-cased": 8,
    "xlm-roberta-large": 1,
}

model_to_batch_size_big = {
    "xlm-roberta-base": 32,
    "bert-base-multilingual-cased": 32,
    "distilbert-base-multilingual-cased": 32,
    "xlm-roberta-large": 8,
}


def get_batch_size(model_name, real_batch_size=32, large_gpu=False):
    """
    Empirical heuristics to get batch size according to model name
    in order to avoid OOM. This is very device-specific, and should not
    be used as is
    """
    if large_gpu:
        return min(model_to_batch_size_big.get(model_name, real_batch_size), real_batch_size)
    return min(model_to_batch_size_small.get(model_name, 4), real_batch_size)
