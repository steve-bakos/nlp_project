from transformers import AutoConfig, AutoModel
from transformers.models.auto.auto_factory import _get_model_class


def get_class_from_model_path(model_path: str, cache_dir=None):

    config = AutoConfig.from_pretrained(
        model_path, return_unused_kwargs=False, trust_remote_code=False, cache_dir=cache_dir
    )

    model_class = _get_model_class(config, AutoModel._model_mapping)

    return model_class
