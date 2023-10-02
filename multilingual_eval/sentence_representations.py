from typing import List, Tuple
from tqdm import tqdm
import torch
import os
import sys
import re
import numpy as np

sys.path.append(os.curdir)

from multilingual_eval.data import STOPWORDS
from multilingual_eval.utils import (
    UniversalTokenizer,
    compute_model_hidden_reprs,
    find_lang_key_for_mbart_like,
    load_embedding,
    subwordlist_to_wordlist,
)


def weight_by_words(sentences, inputs, tokenizer, split_type, device="cpu"):
    """
    Compute weights for computing the average over words (and not subwords)
    For a given batch (`inputs`) typically returned by a tokenizer, as well as the input sentences
    """
    weights = torch.zeros(inputs["input_ids"].size(), dtype=float).to(device)
    for k in range(len(sentences)):
        _, word_ranges = subwordlist_to_wordlist(
            tokenizer.tokenize(sentences[k]), split_type=split_type
        )
        for start, end in word_ranges:
            length = end - start
            weights[k, 1 + start : 1 + end] = 1.0 / length
    return weights


def mask_special_token(inputs, tokenizer, device="cpu"):
    """
    Compute a mask for special tokens like [PAD] but also [CLS], [SEP] and [UNK]
    """
    mask = torch.ones(inputs["input_ids"].size(), dtype=float).to(device)
    for special_tok in tokenizer.special_tokens_map.values():
        if not isinstance(special_tok, list):
            special_tok = [special_tok]
        for st in special_tok:
            tok_id = tokenizer.encode(st, add_special_tokens=False)[0]
            mask *= 1 - (inputs["input_ids"] == tok_id).float()
    return mask


def mask_stopwords(sentences, inputs, tokenizer, split_type, stopwords, mask=None, device="cpu"):
    """
    Mask stopwords (as well as the special tokens)
    """
    mask = mask or mask_special_token(inputs, tokenizer, device=device)
    special_mask = torch.ones(inputs["input_ids"].size(), dtype=float).to(device)
    special_mask *= mask
    for k in range(len(sentences)):
        words, word_ranges = subwordlist_to_wordlist(
            tokenizer.tokenize(sentences[k]), split_type=split_type
        )
        for word, (start, end) in zip(words, word_ranges):
            if word.lower() in stopwords or not re.match(r"\w", word):
                special_mask[k, 1 + start : 1 + end] = 0
    return special_mask


def build_sentence_representations_transformer(
    model,
    tokenizer,
    sentence_pairs: List[Tuple[str, str]],
    batch_size=2,
    dim_size=768,
    n_layers=13,
    device="cpu",
    pooling=["avg", "cls"],
    left_lang="en",
    right_lang="de",
    split_type="wordpiece",
):
    """
    Compute and stores sentence representations in a CPU tensor
    """
    model.eval()
    model.to(device)

    if "avg" in pooling:
        avg_left_embeddings = torch.zeros((n_layers, len(sentence_pairs), dim_size), device="cpu")
        avg_right_embeddings = torch.zeros((n_layers, len(sentence_pairs), dim_size), device="cpu")
    if "cls" in pooling:
        cls_left_embeddings = torch.zeros((n_layers, len(sentence_pairs), dim_size), device="cpu")
        cls_right_embeddings = torch.zeros((n_layers, len(sentence_pairs), dim_size), device="cpu")
    if "reweighted" in pooling:
        reweighted_left_embeddings = torch.zeros(
            (n_layers, len(sentence_pairs), dim_size), device="cpu"
        )
        reweighted_right_embeddings = torch.zeros(
            (n_layers, len(sentence_pairs), dim_size), device="cpu"
        )
    if "stopwords" in pooling:
        stopwords_left_embeddings = torch.zeros(
            (n_layers, len(sentence_pairs), dim_size), device="cpu"
        )
        stopwords_right_embeddings = torch.zeros(
            (n_layers, len(sentence_pairs), dim_size), device="cpu"
        )
        left_stopwords = STOPWORDS.stopwords(left_lang)
        right_stopwords = STOPWORDS.stopwords(right_lang)
    if "word_average" in pooling:
        sum_left_embeddings = torch.zeros((n_layers, dim_size), device="cpu")
        sum_right_embeddings = torch.zeros((n_layers, dim_size), device="cpu")
        count_left = 0
        count_right = 0

    left_key = find_lang_key_for_mbart_like(tokenizer, left_lang)
    right_key = find_lang_key_for_mbart_like(tokenizer, right_lang)

    for i in tqdm(range(0, len(sentence_pairs), batch_size)):
        j = min(i + batch_size, len(sentence_pairs))

        left_sentences = list(map(lambda x: x[0], sentence_pairs[i:j]))
        batch = tokenizer(
            left_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in batch.items()}
        hidden_repr = compute_model_hidden_reprs(
            inputs,
            model,
            tokenizer,
            left_lang,
            device=device,
            lang_key=left_key,
        )

        # Compute masks
        mask = None
        if "avg" in pooling or "word_average" in pooling:
            mask = mask_special_token(inputs, tokenizer, device=device)
        if "reweighted" in pooling:
            weights = weight_by_words(left_sentences, inputs, tokenizer, split_type, device=device)
        if "stopwords" in pooling:
            stopword_mask = mask_stopwords(
                left_sentences,
                inputs,
                tokenizer,
                split_type,
                left_stopwords,
                mask=mask,
                device=device,
            )

        for k in range(n_layers):
            if "cls" in pooling:
                cls_left_embeddings[k, i:j] = hidden_repr[k][:, 0].detach().cpu()
            if "avg" in pooling:
                avg_left_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * mask.unsqueeze(-1), axis=1)
                        / torch.sum(mask, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "reweighted" in pooling:
                reweighted_left_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * weights.unsqueeze(-1), axis=1)
                        / torch.sum(weights, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "stopwords" in pooling:
                stopwords_left_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * stopword_mask.unsqueeze(-1), axis=1)
                        / torch.sum(stopword_mask, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "word_average" in pooling:
                sum_left_embeddings[k] = (
                    (torch.sum(hidden_repr[k] * mask.unsqueeze(-1), axis=(0, 1))).detach().cpu()
                )
                count_left += int(torch.sum(mask).detach().cpu())

        right_sentences = list(map(lambda x: x[1], sentence_pairs[i:j]))
        batch = tokenizer(
            right_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in batch.items()}
        hidden_repr = compute_model_hidden_reprs(
            inputs, model, tokenizer, right_lang, device=device, lang_key=right_key
        )

        # Compute masks
        mask = None
        if "avg" in pooling or "word_average" in pooling:
            mask = mask_special_token(inputs, tokenizer, device=device)
        if "reweighted" in pooling:
            weights = weight_by_words(right_sentences, inputs, tokenizer, split_type, device=device)
        if "stopwords" in pooling:
            stopword_mask = mask_stopwords(
                right_sentences,
                inputs,
                tokenizer,
                split_type,
                right_stopwords,
                mask=mask,
                device=device,
            )

        for k in range(n_layers):
            if "cls" in pooling:
                cls_right_embeddings[k, i:j] = hidden_repr[k][:, 0].detach().cpu()
            if "avg" in pooling:
                avg_right_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * mask.unsqueeze(-1), axis=1)
                        / torch.sum(mask, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "reweighted" in pooling:
                reweighted_right_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * weights.unsqueeze(-1), axis=1)
                        / torch.sum(weights, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "stopwords" in pooling:
                stopwords_right_embeddings[k, i:j] = (
                    (
                        torch.sum(hidden_repr[k] * stopword_mask.unsqueeze(-1), axis=1)
                        / torch.sum(stopword_mask, axis=1, keepdim=True)
                    )
                    .detach()
                    .cpu()
                )
            if "word_average" in pooling:
                sum_right_embeddings[k] = (
                    (torch.sum(hidden_repr[k] * mask.unsqueeze(-1), axis=(0, 1))).detach().cpu()
                )
                count_right += int(torch.sum(mask).detach().cpu())

    res = tuple()
    for p in pooling:
        if p == "avg":
            res += (avg_left_embeddings, avg_right_embeddings)
        if p == "cls":
            res += (cls_left_embeddings, cls_right_embeddings)
        if p == "reweighted":
            res += (reweighted_left_embeddings, reweighted_right_embeddings)
        if p == "stopwords":
            res += (stopwords_left_embeddings, stopwords_right_embeddings)
        if p == "word_average":
            res += (sum_left_embeddings / count_left, sum_right_embeddings / count_right)
    return res


def compute_sentence_representations_with_fasttext(
    sentence_pairs: List[Tuple[str, str]],
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

    left_pair_embeddings = torch.zeros((len(sentence_pairs), size), device="cpu")
    right_pair_embeddings = torch.zeros((len(sentence_pairs), size), device="cpu")

    for i, (left_sent, right_sent) in enumerate(sentence_pairs):
        left_available = list(filter(lambda x: x in left_w2i, tokenizer.tokenize(left_sent)))
        right_available = list(filter(lambda x: x in right_w2i, tokenizer.tokenize(right_sent)))
        if len(left_available) == 0:
            left_pair_embeddings[i] = torch.rand((size,))
        else:
            left_ids = np.array([left_w2i[w] for w in left_available])
            left_pair_embeddings[i] = torch.from_numpy(np.mean(left_embedding[left_ids], axis=0))
        if len(right_available) == 0:
            right_pair_embeddings[i] = torch.rand((size,))
        else:
            right_ids = np.array([right_w2i[w] for w in right_available])
            right_pair_embeddings[i] = torch.from_numpy(np.mean(right_embedding[right_ids], axis=0))

    return left_pair_embeddings, right_pair_embeddings
