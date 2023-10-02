from typing import List, Union
import pycountry

from datasets import load_dataset, get_dataset_config_names

from multilingual_eval.datasets.token_classification import get_token_classification_getter


get_xtreme_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "xtreme",
        f"udpos.{pycountry.languages.get(alpha_2=lang).name}",
        cache_dir=cache_dir,
    ),
    "pos_tags",
)

wuetal_subsets = {
    "en": "en_ewt",
    "ar": "ar_padt",
    "bg": "bg_btb",
    "de": "de_gsd",
    "el": "el_gdt",
    "es": "es_gsd",
    "fr": "fr_gsd",
    "hi": "hi_pud",
    "ru": "ru_gsd",
    "th": "th_pud",
    "tr": "tr_pud",
    "vi": "vi_vtb",
    "zh": "zh_gsd",
}

get_wuetal_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "universal_dependencies", wuetal_subsets[lang], cache_dir=cache_dir
    ),
    "upos",
)


def get_xtreme_udpos_langs():
    """
    Get the list of availabel languages in xtreme.udpos
    """
    return list(
        map(
            lambda x: "el"
            if x.split(".")[1] == "Greek"
            else pycountry.languages.get(name=x.split(".")[1]).alpha_2,
            filter(lambda x: x.startswith("udpos."), get_dataset_config_names("xtreme")),
        )
    )
