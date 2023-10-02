from datasets import load_dataset
import functools
from multilingual_eval.datasets.question_answering import get_question_answering_getter

get_xquad_inner = get_question_answering_getter(
    lambda lang, cache_dir=None: load_dataset(
        "squad" if lang == "en" else "xquad",
        name=f"xquad.{lang}" if lang != "en" else None,
        cache_dir=cache_dir,
    )
)


@functools.wraps(get_xquad_inner)
def get_xquad(lang, *args, split=None, **kwargs):
    if split == "test":
        split = "validation"
    return get_xquad_inner(lang, *args, split=split, **kwargs)
