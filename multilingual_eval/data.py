import os
import sys
import re
import random
from typing import Dict, Set, Tuple
from collections import defaultdict
from datasets import load_dataset, get_dataset_infos
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm

sys.path.append(os.curdir)

from multilingual_eval.utils import UniversalTokenizer


class LazyStopwords:
    """
    Class for reusing already loaded stopwords
    """

    code_to_lang = {"en": "english", "de": "german", "ru": "russian"}
    code_to_stops = {}

    def stopwords(self, code):
        assert code in self.code_to_lang
        if code not in self.code_to_stops:
            self.code_to_stops[code] = set(stopwords.words(self.code_to_lang[code]))
        return self.code_to_stops[code]


STOPWORDS = LazyStopwords()


def load_europarl(data_dir: str = "/data2/europarl", lang="de", seed=None, shuffle=True):
    """
    Load parallel sentences from the Europarl dataset (en-lang)
    """
    with open(os.path.join(data_dir, f"europarl-v7.{lang}-en.{lang}"), "r") as lang_file, open(
        os.path.join(data_dir, f"europarl-v7.{lang}-en.en"), "r"
    ) as en_file:
        lines = list(zip(lang_file.readlines(), en_file.readlines()))
        if shuffle:
            random.shuffle(lines)
    return map(lambda x: {"translation": {lang: x[0], "en": x[1]}}, lines)


def load_from_fastalign_input(fname: str, left_lang="de", right_lang="en", shuffle=True):
    res = []
    with open(fname, "r") as f:
        for line in f:
            parts = line.strip("\n").split(" ||| ")
            if len(parts) != 2:
                continue
            left_sent, right_sent = parts
            res.append({left_lang: left_sent, right_lang: right_sent})
    if shuffle:
        random.shuffle(res)
    return res


def load_biomedical_wmt19(
    data_dir: str = "/data2/datasets/wmt19biomedical", lang="de", shuffle=True
):
    """
    Load parallel sentences from the biomedical WMT10 dataset (en-lang)
    """

    base_dir = os.path.join(data_dir, "test_sets", "Medline")
    lang_lang2en_file = os.path.join(base_dir, f"medline_{lang}2en_{lang}.txt")
    en_lang2en_file = os.path.join(base_dir, "gold_standard", f"medline_{lang}2en_en.txt")
    lang_en2lang_file = os.path.join(base_dir, "gold_standard", f"medline_en2{lang}_{lang}.txt")
    en_en2lang_file = os.path.join(base_dir, f"medline_en2{lang}_en.txt")
    align_file = os.path.join(base_dir, "gold_standard", f"align_validation_{lang}_en.txt")
    mapdocs_file = os.path.join(base_dir, "gold_standard", f"mapdocs_{lang}_en.txt")

    pid_to_doc = {}
    with open(mapdocs_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            pid_to_doc[parts[0].strip()] = parts[1].strip()

    lang_sentences: Dict[str, Dict[int, str]] = defaultdict(lambda: {})
    with open(lang_lang2en_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            doc, s_id, sent = parts
            s_id = int(s_id)
            lang_sentences[doc][s_id] = sent
    with open(lang_en2lang_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            doc, s_id, sent = parts
            s_id = int(s_id)
            lang_sentences[doc][s_id] = sent

    en_sentences: Dict[str, Dict[int, str]] = defaultdict(lambda: {})
    with open(en_lang2en_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            doc, s_id, sent = parts
            s_id = int(s_id)
            en_sentences[doc][s_id] = sent
    with open(en_en2lang_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            doc, s_id, sent = parts
            s_id = int(s_id)
            en_sentences[doc][s_id] = sent

    res = []
    with open(align_file, "r") as f:
        for line in f:
            pid, rule, status = line.strip().split("\t")
            if status != "OK":
                continue
            left, right = rule.split(" <=> ")
            if not re.match(r"^[0-9]+$", left) or not re.match(r"^[0-9]+$", right):
                continue
            left = int(left)
            right = int(right)
            doc_id = pid_to_doc[pid]
            lang_sentence = lang_sentences[doc_id][left]
            en_sentence = en_sentences[doc_id][right]

            if len(lang_sentence.split()) <= 2:
                continue
            res.append({"translation": {lang: lang_sentence, "en": en_sentence}})
    if shuffle:
        random.shuffle(res)
    return res


def sample_sentences(
    dataset_slug: str,
    left_lang="en",
    right_lang="de",
    nb=5000,
    shuffle=True,
    seed=None,
    cache_dir=None,
    tokenizer=None,
    max_length=400,
):
    """
    Sample `nb` sentence pairs at random for the en-lang pair
    """
    tokenizer = tokenizer or UniversalTokenizer()
    if dataset_slug == "wmt19":
        subsets = set(get_dataset_infos("wmt19").keys())
        subset_name = (
            f"{left_lang}-{right_lang}"
            if f"{left_lang}-{right_lang}" in subsets
            else f"{right_lang}-{left_lang}"
        )
        dataset = load_dataset(
            "wmt19", subset_name, **({"cache_dir": cache_dir} if cache_dir is not None else {})
        )["train"]
        if shuffle:
            dataset = dataset.shuffle(seed=np.random.randint(0, 1024) if seed is None else seed)
    elif dataset_slug == "wmt16":
        assert left_lang == "en" or right_lang == "en"
        lang = left_lang if right_lang == "en" else right_lang
        dataset = load_dataset("wmt16", f"{lang}-en")["train"]
        if shuffle:
            dataset = dataset.shuffle(seed=np.random.randint(0, 1024) if seed is None else seed)
    elif dataset_slug == "europarl":
        assert left_lang == "en" or right_lang == "en"
        lang = left_lang if right_lang == "en" else right_lang
        dataset = load_europarl(lang=lang, shuffle=shuffle)
    elif dataset_slug == "biomedicalwmt19":
        assert left_lang == "en" or right_lang == "en"
        lang = left_lang if right_lang == "en" else right_lang
        dataset = load_biomedical_wmt19(lang=lang, shuffle=shuffle)
    elif dataset_slug.split(":")[0] == "fastalign":
        dataset = load_from_fastalign_input(
            dataset_slug.split(":")[1], left_lang=left_lang, right_lang=right_lang, shuffle=shuffle
        )
    else:
        raise NotImplementedError(f"Unrecognized dataset_slug: `{dataset_slug}`")

    pbar = tqdm(total=nb)
    count = 0
    for elt in dataset:
        if nb is not None and count >= nb:
            break
        if "translation" in elt:
            res = (elt["translation"][left_lang], elt["translation"][right_lang])
        else:
            res = (elt[left_lang], elt[right_lang])

        if res[0] is not None and res[1] is not None:
            left_subwords = tokenizer.base_tokenizer.tokenize(res[0])
            right_subwords = tokenizer.base_tokenizer.tokenize(res[1])
            if len(left_subwords) > max_length or len(right_subwords) > max_length:
                continue
            if (
                len(list(filter(lambda x: x != "[UNK]", left_subwords))) == 0
                or len(list(filter(lambda x: x != "[UNK]", right_subwords))) == 0
            ):
                continue
            yield res
            count += 1
            pbar.update()


def get_dicos(
    left_lang,
    right_lang,
    dico_path,
    ignore_identical=False,
    right_voc=None,
    left_voc=None,
    tokenizer=None,
    tokenize=True,
    split="all",
):
    """
    Retrieve forward and backward crosslingual dictionaries under the
    form of two Python dictionaries. `only_atomic` indicates that we
    keep only words that are not splitted by the tokenizer. `ignore_identical`
    indicates that we ignore pairs of words that are identically spelled across
    languages.
    """
    if split == "all":
        dico_suffix = ""
    elif split == "train":
        dico_suffix = ".0-5000"
    elif split == "eval":
        dico_suffix = ".5000-6500"
    else:
        raise NotImplementedError(f"Unsuported split type {split} expected all, train or eval")

    forward_path = os.path.join(dico_path, f"{left_lang}-{right_lang}{dico_suffix}.txt")
    backward_path = os.path.join(dico_path, f"{right_lang}-{left_lang}{dico_suffix}.txt")

    if tokenize:
        tokenizer = tokenizer or UniversalTokenizer()

    forward_dico: Dict[Tuple[str], Set[Tuple[str]]] = {}
    backward_dico: Dict[Tuple[str], Set[Tuple[str]]] = {}

    with open(forward_path, "r") as f:
        for line in f:
            words = line.strip("\n").split()
            if len(words) != 2:
                continue
            w1, w2 = words
            if tokenize:
                new_w1 = tuple(tokenizer.tokenize(w1))
                new_w2 = tuple(tokenizer.tokenize(w2))

                w1 = "".join(new_w1)
                w2 = "".join(new_w2)
            else:
                new_w1 = w1
                new_w2 = w2

            if ignore_identical and new_w1 == new_w2:
                continue

            if right_voc is not None and w2 not in right_voc:
                continue
            if left_voc is not None and w1 not in left_voc:
                continue
            if new_w1 not in forward_dico:
                forward_dico[new_w1] = set()
            if new_w2 not in backward_dico:
                backward_dico[new_w2] = set()
            forward_dico[new_w1].add(new_w2)
            backward_dico[new_w2].add(new_w1)

    with open(backward_path, "r") as f:
        for line in f:
            words = line.strip("\n").split()
            if len(words) != 2:
                continue
            w1, w2 = words
            if tokenize:
                new_w1 = tuple(tokenizer.tokenize(w1))
                new_w2 = tuple(tokenizer.tokenize(w2))

                w1 = "".join(new_w1)
                w2 = "".join(new_w2)
            else:
                new_w1 = w1
                new_w2 = w2

            if ignore_identical and new_w1 == new_w2:
                continue

            if right_voc is not None and w1 not in right_voc:
                continue
            if left_voc is not None and w2 not in left_voc:
                continue
            if new_w1 not in backward_dico:
                backward_dico[new_w1] = set()
            if new_w2 not in forward_dico:
                forward_dico[new_w2] = set()
            backward_dico[new_w1].add(new_w2)
            forward_dico[new_w2].add(new_w1)

    return forward_dico, backward_dico


def load_alignemnt_evaluation(alignment_script_path: str, left_lang="en", right_lang="de"):
    dir_path = os.path.join(alignment_script_path, "test")
    assert (left_lang, right_lang) in [
        ("de", "en"),
        ("en", "fr"),
        ("ro", "en"),
    ], f"{(left_lang, right_lang)} not available, try one of: de-en, en-fr or ro-en"

    left_file = os.path.join(dir_path, f"{left_lang}{right_lang}.src")
    right_file = os.path.join(dir_path, f"{left_lang}{right_lang}.tgt")
    align_file = os.path.join(dir_path, f"{left_lang}{right_lang}.talp")

    with open(align_file, "r") as align_f, open(left_file, "r") as left_f, open(
        right_file, "r"
    ) as right_f:
        for alignment, left_sent, right_sent in zip(align_f, left_f, right_f):
            alignments = list(
                map(
                    lambda x: tuple(map(int, x.split("-"))),
                    alignment.strip().replace("p", "-").split(" "),
                )
            )
            left_sent = left_sent.strip()
            right_sent = right_sent.strip()
            yield left_sent, right_sent, alignments
