from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple
import torch
import numpy as np

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


sys.path.append(os.curdir)

from multilingual_eval.utils import (
    UniversalTokenizer,
    compute_model_hidden_reprs,
    find_lang_key_for_mbart_like,
    load_embedding,
    subwordlist_to_wordlist,
)
from multilingual_eval.data import get_dicos, sample_sentences


@dataclass
class ContextualizedPair:
    left_sentence: str
    right_sentence: str
    left_word: str
    right_word: str
    left_offset: Tuple[int, int]
    right_offset: Tuple[int, int]
    left_sent_id: int
    right_sent_id: int


def generate_pairs(
    sentence_pair_generator,
    dico_path: str,
    left_lang="en",
    right_lang="de",
    max_length=512,
    already_tokenized=False,
    left_voc=None,
    right_voc=None,
    avoid_repetition=False,
    tokenizer=None,
    dict_tuple=None,
    selected_source=None,
    selected_target=None,
    max_pairs=None,
    split="all",
):

    tokenizer = tokenizer or UniversalTokenizer()

    if dict_tuple is not None:
        forward_dict, backward_dict = dict_tuple
    else:
        forward_dict, backward_dict = get_dicos(
            left_lang, right_lang, dico_path, left_voc=left_voc, right_voc=right_voc, split=split
        )

    max_length_left = max(map(len, forward_dict))
    max_length_right = max(map(len, backward_dict))

    res: List[ContextualizedPair] = []

    selected_source = selected_source or set()
    selected_target = selected_target or set()
    for sent_id, pair in enumerate(sentence_pair_generator):
        if isinstance(pair, dict):
            left_sent = pair[left_lang]
            right_sent = pair[right_lang]
        else:
            left_sent, right_sent = pair

        if left_sent is None or right_sent is None:
            continue

        left_tokens, left_offsets = tokenizer.tokenize(
            left_sent, return_offsets=True, already_tokenized_by_space=already_tokenized
        )
        right_tokens, right_offsets = tokenizer.tokenize(
            right_sent, return_offsets=True, already_tokenized_by_space=already_tokenized
        )
        # We don't embarass ourselves with sequence that are longer than the model max length
        if len(left_tokens) + 2 > max_length or len(right_tokens) > max_length:
            continue

        # Generate potential multi-word expression (essentially for languages like mandarin chinese
        # which are tokenized by character and have words spanning several tokens
        left_words = []
        left_word_offsets = []
        for length in range(1, max_length_left + 1):
            for i in range(0, len(left_tokens) - length + 1):
                left_words.append(tuple(left_tokens[i : i + length]))
                left_word_offsets.append((left_offsets[i][0], left_offsets[i + length - 1][1]))

        right_words = []
        right_word_offsets = []
        for length in range(1, max_length_right + 1):
            for i in range(0, len(right_tokens) - length + 1):
                right_words.append(tuple(right_tokens[i : i + length]))
                right_word_offsets.append((right_offsets[i][0], right_offsets[i + length - 1][1]))

        for word_pos, word in zip(left_word_offsets, left_words):
            candidates = forward_dict.get(word, set())
            candidates = candidates.intersection(right_words)

            # Verify that there is one and only one candidate in set of words
            if len(candidates) != 1:
                continue

            # Verify that there is only one occurence of this word and extract it
            i = -1
            for i, (tgt_pos, tgt_word) in enumerate(
                filter(lambda x: x[1] in candidates, zip(right_word_offsets, right_words))
            ):
                if i == 1:
                    break
            if i != 0:
                continue

            # Verify that the target word is not the translation of another word in the source
            backward_candidates = backward_dict.get(tgt_word, set())
            counter = 0
            for w in left_words:
                if w in backward_candidates:
                    counter += 1
            if counter != 1:
                continue

            if word not in selected_source and tgt_word not in selected_target:
                res.append(
                    ContextualizedPair(
                        left_sentence=left_sent,
                        right_sentence=right_sent,
                        left_word="".join(word),
                        right_word="".join(tgt_word),
                        left_offset=word_pos,
                        right_offset=tgt_pos,
                        left_sent_id=sent_id,
                        right_sent_id=sent_id,
                    )
                )
                if avoid_repetition:
                    selected_source.add(word)
                    selected_target.add(tgt_word)
            if max_pairs and len(res) >= max_pairs:
                return res

    return res


def get_fastalign_pairs(
    sentence_pairs: List[Tuple[str, str]],
    alignment_file: str,
    max_length=512,
    tokenizer=None,
):
    tokenizer = tokenizer or UniversalTokenizer()

    alignment_pairs: List[List[Tuple[int, int]]] = []
    with open(alignment_file, "r") as f:
        for line in f:
            alignment_pairs.append(
                list(map(lambda x: tuple(map(int, x.split("-"))), line.strip().split()))
            )

    assert len(sentence_pairs) == len(alignment_pairs)

    res: List[ContextualizedPair] = []

    for i, ((left_sent, right_sent), alignment) in enumerate(zip(sentence_pairs, alignment_pairs)):
        left_tokens, left_offsets = tokenizer.tokenize(
            left_sent, already_tokenized_by_space=True, return_offsets=True
        )
        right_tokens, right_offsets = tokenizer.tokenize(
            right_sent, already_tokenized_by_space=True, return_offsets=True
        )

        if len(left_tokens) > max_length - 3 or len(right_tokens) > max_length - 3:
            continue

        for left_i, right_i in alignment:
            res.append(
                ContextualizedPair(
                    left_sentence=left_sent,
                    right_sentence=right_sent,
                    left_word=left_tokens[left_i],
                    right_word=right_tokens[right_i],
                    left_offset=left_offsets[left_i],
                    right_offset=right_offsets[right_i],
                    left_sent_id=i,
                    right_sent_id=i,
                )
            )

    return res


def get_random_intra_sentence_pairs(
    sentence_pairs: List[Tuple[str, int]],
    max_length=512,
    tokenizer=None,
    already_tokenized=True,
    nb=10_000,
):
    tokenizer = tokenizer or UniversalTokenizer()

    res = []

    ids = np.random.randint(0, len(sentence_pairs), size=(nb,))

    for idx in ids:
        left_sent, right_sent = sentence_pairs[idx]

        left_tokens, left_offsets = tokenizer.tokenize(
            left_sent, already_tokenized_by_space=already_tokenized, return_offsets=True
        )
        right_tokens, right_offsets = tokenizer.tokenize(
            right_sent, already_tokenized_by_space=already_tokenized, return_offsets=True
        )

        left_idx = np.random.randint(0, min(max_length, len(left_tokens)))
        right_idx = np.random.randint(0, min(max_length, len(right_tokens)))

        res.append(
            ContextualizedPair(
                left_sentence=left_sent,
                right_sentence=right_sent,
                left_word=left_tokens[left_idx],
                right_word=right_tokens[right_idx],
                left_offset=left_offsets[left_idx],
                right_offset=right_offsets[right_idx],
                left_sent_id=idx,
                right_sent_id=idx,
            )
        )

    return res


def get_random_inter_sentence_pairs(
    sentence_pairs: List[Tuple[str, int]],
    max_length=512,
    tokenizer=None,
    already_tokenized=True,
    nb=10_000,
):
    tokenizer = tokenizer or UniversalTokenizer()

    res = []

    left_ids = list(np.random.randint(0, len(sentence_pairs), size=(nb,)))
    right_ids = list(np.random.randint(1, len(sentence_pairs), size=(nb,)))
    right_ids += left_ids
    right_ids %= len(sentence_pairs)

    for left_sent_id, right_sent_id in zip(left_ids, right_ids):
        left_sent = sentence_pairs[left_sent_id][0]
        right_sent = sentence_pairs[right_sent_id][1]

        left_tokens, left_offsets = tokenizer.tokenize(
            left_sent, already_tokenized_by_space=already_tokenized, return_offsets=True
        )
        right_tokens, right_offsets = tokenizer.tokenize(
            right_sent, already_tokenized_by_space=already_tokenized, return_offsets=True
        )

        left_idx = np.random.randint(0, min(max_length, len(left_tokens)))
        right_idx = np.random.randint(0, min(max_length, len(right_tokens)))

        res.append(
            ContextualizedPair(
                left_sentence=left_sent,
                right_sentence=right_sent,
                left_word=left_tokens[left_idx],
                right_word=right_tokens[right_idx],
                left_offset=left_offsets[left_idx],
                right_offset=right_offsets[right_idx],
                left_sent_id=left_sent_id,
                right_sent_id=right_sent_id,
            )
        )

    return res


def compute_pair_representations(
    model,
    tokenizer,
    pairs: List[ContextualizedPair],
    batch_size=2,
    dim_size=768,
    n_layers=13,
    device="cpu",
    left_lang=None,
    right_lang=None,
    left_lang_id=None,
    right_lang_id=None,
    split_type="wordpiece",
    universal_tokenizer=None,
    mask_word=False,
):
    model.eval()
    if device is not None:
        model.to(device)

    left_embeddings = torch.zeros((n_layers, len(pairs), dim_size), device="cpu")
    right_embeddings = torch.zeros((n_layers, len(pairs), dim_size), device="cpu")

    if left_lang is not None:
        left_key = find_lang_key_for_mbart_like(tokenizer, left_lang)
    else:
        left_key = None
    if right_lang is not None:
        right_key = find_lang_key_for_mbart_like(tokenizer, right_lang)
    else:
        right_key = None

    for i in range(0, len(pairs), batch_size):
        j = min(i + batch_size, len(pairs))

        if mask_word:
            left_sentences = list(map(lambda x: x.left_sentence, pairs[i:j]))
            batch = tokenizer(
                left_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                **(
                    {"return_offsets_mapping": True}
                    if isinstance(tokenizer, PreTrainedTokenizerFast)
                    else {}
                ),
            )
            for k, p in enumerate(pairs[i:j]):
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    offset_mappings = list(
                        map(lambda x: list(map(int, x)), batch["offset_mapping"][k])
                    )
                    start_offset_candidates = list(
                        filter(
                            lambda x: x[1][1] > 0 and x[1][0] <= p.left_offset[0] < x[1][1],
                            enumerate(offset_mappings),
                        )
                    )
                    end_offset_candidates = list(
                        filter(
                            lambda x: x[1][1] > 0 and x[1][0] < p.left_offset[1] <= x[1][1],
                            enumerate(offset_mappings),
                        )
                    )
                    if len(start_offset_candidates) != 1 and len(end_offset_candidates) != 1:
                        logging.error(
                            f"Found too many candidates for position {p.left_offset} in {offset_mappings}"
                        )
                        start_offset, end_offset = -1, -1
                    else:
                        try:
                            start_offset = start_offset_candidates[0][0]
                            end_offset = end_offset_candidates[0][0] + 1
                            p.left_sentence = (
                                p.left_sentence[: offset_mappings[start_offset][0]]
                                + f"{tokenizer.special_tokens_map['mask_token']} "
                                + p.left_sentence[offset_mappings[end_offset][0] :]
                            )
                            p.left_offset = (
                                p.left_offset[0],
                                p.left_offset[0] + len(tokenizer.special_tokens_map["mask_token"]),
                            )
                        except IndexError:
                            logging.error(
                                f"IndexError: \n`start_offset = start_offset_candidates[0][0]` -> start_offset_candidates={start_offset_candidates} \n`end_offset = end_offset_candidates[0][0]` -> end_offset_candidates={end_offset_candidates}"
                            )
                            start_offset, end_offset = -1, -1
                else:
                    universal_tokenizer = universal_tokenizer or UniversalTokenizer()
                    subwords = tokenizer.tokenize(p.left_sentence)
                    words, word_pos = subwordlist_to_wordlist(subwords, split_type=split_type)
                    normalized_words = list(
                        map(lambda x: "".join(universal_tokenizer.tokenize(x)), words)
                    )
                    try:
                        start_offset, end_offset = word_pos[normalized_words.index(p.left_word)]
                        p.left_sentence = (
                            p.left_sentence[: offset_mappings[start_offset][0]]
                            + f"{tokenizer.special_tokens_map['mask_token']} "
                            + p.left_sentence[offset_mappings[end_offset][0] :]
                        )
                        p.left_offset = (
                            p.left_offset[0],
                            p.left_offset[0] + len(tokenizer.special_tokens_map["mask_token"]),
                        )
                    except ValueError:
                        start_offset, end_offset = -1, -1

        left_sentences = list(map(lambda x: x.left_sentence, pairs[i:j]))
        batch = tokenizer(
            left_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            **(
                {"return_offsets_mapping": True}
                if isinstance(tokenizer, PreTrainedTokenizerFast)
                else {}
            ),
        )
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "offset_mapping"}
        hidden_repr = compute_model_hidden_reprs(
            inputs,
            model,
            tokenizer,
            left_lang,
            device=device,
            lang_key=left_key,
            lang_id=left_lang_id,
        )

        for k, p in enumerate(pairs[i:j]):
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                offset_mappings = list(map(lambda x: list(map(int, x)), batch["offset_mapping"][k]))
                start_offset_candidates = list(
                    filter(
                        lambda x: x[1][1] > 0 and x[1][0] <= p.left_offset[0] < x[1][1],
                        enumerate(offset_mappings),
                    )
                )
                end_offset_candidates = list(
                    filter(
                        lambda x: x[1][1] > 0 and x[1][0] < p.left_offset[1] <= x[1][1],
                        enumerate(offset_mappings),
                    )
                )
                if len(start_offset_candidates) == 0 or len(end_offset_candidates) == 0:
                    logging.error(
                        f"Found {len(start_offset_candidates)} and {len(end_offset_candidates)} candidates for position {p.left_offset} in {offset_mappings}"
                    )
                    start_offset, end_offset = -1, -1
                else:
                    try:
                        start_offset = min(start_offset_candidates, key=lambda x: x[0])[0]
                        end_offset = max(start_offset_candidates, key=lambda x: x[0])[0] + 1
                    except IndexError:
                        logging.error(
                            f"IndexError: \n`start_offset = start_offset_candidates[0][0]` -> start_offset_candidates={start_offset_candidates} \n`end_offset = end_offset_candidates[0][0]` -> end_offset_candidates={end_offset_candidates}"
                        )
                        start_offset, end_offset = -1, -1
            else:
                universal_tokenizer = universal_tokenizer or UniversalTokenizer()
                subwords = tokenizer.tokenize(p.left_sentence)
                words, word_pos = subwordlist_to_wordlist(subwords, split_type=split_type)
                normalized_words = list(
                    map(lambda x: "".join(universal_tokenizer.tokenize(x)), words)
                )
                try:
                    start_offset, end_offset = word_pos[normalized_words.index(p.left_word)]
                except ValueError:
                    start_offset, end_offset = -1, -1

            for layer in range(n_layers):
                if start_offset == -1 or end_offset == -1:
                    left_embeddings[layer, i + k] = torch.rand((dim_size,))
                else:
                    left_embeddings[layer, i + k, :] = (
                        (
                            torch.sum(hidden_repr[layer][k, start_offset:end_offset], axis=0)
                            / (end_offset - start_offset)
                        )
                        .detach()
                        .cpu()
                    )

        if mask_word:
            right_sentences = list(map(lambda x: x.right_sentence, pairs[i:j]))
            batch = tokenizer(
                right_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                **(
                    {"return_offsets_mapping": True}
                    if isinstance(tokenizer, PreTrainedTokenizerFast)
                    else {}
                ),
            )
            for k, p in enumerate(pairs[i:j]):
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    offset_mappings = list(
                        map(lambda x: list(map(int, x)), batch["offset_mapping"][k])
                    )
                    start_offset_candidates = list(
                        filter(
                            lambda x: x[1][1] > 0 and x[1][0] <= p.right_offset[0] < x[1][1],
                            enumerate(offset_mappings),
                        )
                    )
                    end_offset_candidates = list(
                        filter(
                            lambda x: x[1][1] > 0 and x[1][0] < p.right_offset[1] <= x[1][1],
                            enumerate(offset_mappings),
                        )
                    )
                    if len(start_offset_candidates) != 1 and len(end_offset_candidates) != 1:
                        logging.error(
                            f"Found too many candidates for position {p.right_offset} in {offset_mappings}"
                        )
                        start_offset, end_offset = -1, -1
                    else:
                        try:
                            start_offset = start_offset_candidates[0][0]
                            end_offset = end_offset_candidates[0][0] + 1
                            p.right_sentence = (
                                p.right_sentence[: offset_mappings[start_offset][0]]
                                + f"{tokenizer.special_tokens_map['mask_token']} "
                                + p.right_sentence[offset_mappings[end_offset][0] :]
                            )
                            p.right_offset = (
                                p.right_offset[0],
                                p.right_offset[0] + len(tokenizer.special_tokens_map["mask_token"]),
                            )
                        except IndexError:
                            logging.error(
                                f"IndexError: \n`start_offset = start_offset_candidates[0][0]` -> start_offset_candidates={start_offset_candidates} \n`end_offset = end_offset_candidates[0][0]` -> end_offset_candidates={end_offset_candidates}"
                            )
                            start_offset, end_offset = -1, -1
                else:
                    universal_tokenizer = universal_tokenizer or UniversalTokenizer()
                    subwords = tokenizer.tokenize(p.right_sentence)
                    words, word_pos = subwordlist_to_wordlist(subwords, split_type=split_type)
                    normalized_words = list(
                        map(lambda x: "".join(universal_tokenizer.tokenize(x)), words)
                    )
                    try:
                        start_offset, end_offset = word_pos[normalized_words.index(p.right_word)]
                        p.right_sentence = (
                            p.right_sentence[: offset_mappings[start_offset][0]]
                            + f"{tokenizer.special_tokens_map['mask_token']} "
                            + p.right_sentence[offset_mappings[end_offset][0] :]
                        )
                        p.right_offset = (
                            p.right_offset[0],
                            p.right_offset[0] + len(tokenizer.special_tokens_map["mask_token"]),
                        )
                    except ValueError:
                        start_offset, end_offset = -1, -1
        right_sentences = list(map(lambda x: x.right_sentence, pairs[i:j]))
        batch = tokenizer(
            right_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            **(
                {"return_offsets_mapping": True}
                if isinstance(tokenizer, PreTrainedTokenizerFast)
                else {}
            ),
        )
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "offset_mapping"}
        hidden_repr = compute_model_hidden_reprs(
            inputs,
            model,
            tokenizer,
            right_lang,
            device=device,
            lang_key=right_key,
            lang_id=right_lang_id,
        )

        for k, p in enumerate(pairs[i:j]):
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                offset_mappings = list(map(lambda x: list(map(int, x)), batch["offset_mapping"][k]))
                start_offset_candidates = list(
                    filter(
                        lambda x: x[1][1] > 0 and x[1][0] <= p.right_offset[0] < x[1][1],
                        enumerate(offset_mappings),
                    )
                )
                end_offset_candidates = list(
                    filter(
                        lambda x: x[1][1] > 0 and x[1][0] < p.right_offset[1] <= x[1][1],
                        enumerate(offset_mappings),
                    )
                )
                if len(start_offset_candidates) == 0 or len(end_offset_candidates) == 0:
                    logging.error(
                        f"Found {len(start_offset_candidates)} and {len(end_offset_candidates)} candidates for position {p.right_offset} in {offset_mappings}"
                    )
                    start_offset, end_offset = -1, -1
                else:
                    try:
                        start_offset = min(start_offset_candidates, key=lambda x: x[0])[0]
                        end_offset = max(start_offset_candidates, key=lambda x: x[0])[0] + 1
                    except IndexError:
                        logging.error(
                            f"IndexError: \n`start_offset = start_offset_candidates[0][0]` -> start_offset_candidates={start_offset_candidates} \n`end_offset = end_offset_candidates[0][0]` -> end_offset_candidates={end_offset_candidates}"
                        )
                        start_offset, end_offset = -1, -1
            else:
                universal_tokenizer = universal_tokenizer or UniversalTokenizer()
                subwords = tokenizer.tokenize(p.right_sentence)
                words, word_pos = subwordlist_to_wordlist(subwords, split_type=split_type)
                normalized_words = list(
                    map(lambda x: "".join(universal_tokenizer.tokenize(x)), words)
                )
                try:
                    start_offset, end_offset = word_pos[normalized_words.index(p.right_word)]
                except ValueError:
                    start_offset, end_offset = -1, -1

            for layer in range(n_layers):
                if start_offset == -1 or end_offset == -1:
                    right_embeddings[layer, i + k] = torch.rand((dim_size,))
                else:
                    right_embeddings[layer, i + k, :] = (
                        (
                            torch.sum(hidden_repr[layer][k, start_offset:end_offset], axis=0)
                            / (end_offset - start_offset)
                        )
                        .detach()
                        .cpu()
                    )

    return left_embeddings, right_embeddings


def compute_pair_representations_with_fasttext(
    pairs: List[ContextualizedPair],
    left_lang: str,
    right_lang: str,
    fasttext_dir: str,
    tokenizer=None,
    left_words=None,
    left_embedding=None,
    right_words=None,
    right_embedding=None,
):
    tokenizer = tokenizer or UniversalTokenizer()

    if left_words is None or left_embedding is None:
        left_words, left_embedding = load_embedding(
            os.path.join(fasttext_dir, f"wiki.{left_lang}.align.vec")
        )
    if right_words is None or right_embedding is None:
        right_words, right_embedding = load_embedding(
            os.path.join(fasttext_dir, f"wiki.{right_lang}.align.vec")
        )
    left_w2i = {w: i for i, w in enumerate(left_words)}
    right_w2i = {w: i for i, w in enumerate(right_words)}

    size = left_embedding.shape[1]

    left_pair_embeddings = torch.zeros((len(pairs), size), device="cpu")
    right_pair_embeddings = torch.zeros((len(pairs), size), device="cpu")

    left_unk = np.mean(left_embedding, axis=0)
    right_unk = np.mean(right_embedding, axis=0)

    for i, p in enumerate(pairs):
        if p.left_word not in left_w2i:
            left_pair_embeddings[i] = torch.from_numpy(left_unk)
        else:
            left_pair_embeddings[i] = torch.from_numpy(left_embedding[left_w2i[p.left_word]])
        if p.right_word not in right_w2i:
            right_pair_embeddings[i] = torch.from_numpy(right_unk)
        else:
            right_pair_embeddings[i] = torch.from_numpy(right_embedding[right_w2i[p.right_word]])

    return left_pair_embeddings, right_pair_embeddings
