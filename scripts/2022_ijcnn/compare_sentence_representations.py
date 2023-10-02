import sys
import os

from transformers import AutoModel, AutoTokenizer


sys.path.append(os.curdir)

from multilingual_eval.data import sample_sentences
from multilingual_eval.sentence_representations import (
    build_sentence_representations_transformer,
    compute_sentence_representations_with_fasttext,
)
from multilingual_eval.utils import get_tokenizer_type, UniversalTokenizer, load_embedding
from multilingual_eval.retrieval import compute_average_similarity, evaluate_alignment_with_cosim

if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR")
    fasttext_dir = os.getenv("ALIGNED_FASTTEXT_DIR")
    transformers_cache_dir = os.getenv("TRANSFORMERS_CACHE_DIR")
    data_cache_dir = os.getenv("DATASETS_CACHE_DIR")

    assert data_dir is not None, "env var DATA_DIR should be defined"
    assert os.path.isdir(data_dir), f"value provided for DATA_DIR {data_dir} is not a directory"
    assert os.path.isdir(
        fasttext_dir
    ), f"value provided for fasstext_dir {fasttext_dir} is not a directory (create the directory first?)"
    
    base_dir = os.path.join(data_dir, "raw_results")

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    cls_sim_file = os.path.join(base_dir, "01_cls_similarities.csv")
    sentence_nn_file = os.path.join(base_dir, "02_sentence_nn.csv")

    models = [
        "bert-base-multilingual-cased",
    ]

    dataset_tuples = [
        ("wmt19", "de", "en", True),
        ("wmt19", "ru", "en", True),
        ("wmt19", "zh", "en", True),
    ]

    n_runs = 10

    if not os.path.isfile(cls_sim_file):
        with open(cls_sim_file, "w") as f:
            f.write("model,dataset,left_lang,right_lang,type,layer,score\n")
    if not os.path.isfile(sentence_nn_file):
        with open(sentence_nn_file, "w") as f:
            f.write("model,dataset,left_lang,right_lang,centered,type,layer,score\n")

    # Retrieve embeddings in advance
    voc_emb_tuples = {}
    for _, left_lang, right_lang, fasttext_available in dataset_tuples:
        if not fasttext_available:
            continue
        if left_lang not in voc_emb_tuples:
            voc_emb_tuples[left_lang] = load_embedding(
                os.path.join(fasttext_dir, f"wiki.{left_lang}.align.vec")
            )
        if right_lang not in voc_emb_tuples:
            voc_emb_tuples[right_lang] = load_embedding(
                os.path.join(fasttext_dir, f"wiki.{right_lang}.align.vec")
            )

    universal_tokenizer = UniversalTokenizer()

    for _ in range(n_runs):
        for dataset, left_lang, right_lang, fasttext_available in dataset_tuples:
            sentence_pairs = list(
                sample_sentences(
                    dataset,
                    left_lang=left_lang,
                    right_lang=right_lang,
                    nb=5000,
                    cache_dir=data_cache_dir,
                )
            )

            for model_dir in models:
                model = AutoModel.from_pretrained(model_dir, cache_dir=transformers_cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir, cache_dir=transformers_cache_dir
                )
                model_name = model_dir.split("/")[-1]
                n_layers = model.config.num_hidden_layers + 1
                size = model.config.hidden_size
                split_type = get_tokenizer_type(tokenizer)

                (
                    cls_left,
                    cls_right,
                    avg_left,
                    avg_right,
                ) = build_sentence_representations_transformer(
                    model,
                    tokenizer,
                    sentence_pairs,
                    batch_size=2,
                    dim_size=size,
                    n_layers=n_layers,
                    device="cpu" if size > 800 else "cuda:1",
                    pooling=["cls", "avg"],
                    left_lang=left_lang,
                    right_lang=right_lang,
                    split_type=split_type,
                )

                for layer in range(n_layers):

                    pair_sim, random_sim = compute_average_similarity(
                        cls_left[layer], cls_right[layer], device="cuda:1"
                    )

                    for score in list(pair_sim):
                        with open(cls_sim_file, "a") as f:
                            f.write(
                                f"{model_name},{dataset},{left_lang},{right_lang},sim,{layer},{score}\n"
                            )

                    for score in list(random_sim):
                        with open(cls_sim_file, "a") as f:
                            f.write(
                                f"{model_name},{dataset},{left_lang},{right_lang},random,{layer},{score}\n"
                            )

                    # compute nearest-neighbor search for model
                    cls_score = evaluate_alignment_with_cosim(
                        cls_left[layer], cls_right[layer], device="cuda:1", csls_k=10
                    )
                    cls_score_centered = evaluate_alignment_with_cosim(
                        cls_left[layer],
                        cls_right[layer],
                        device="cuda:1",
                        csls_k=10,
                        mean_centered=True,
                    )
                    avg_score = evaluate_alignment_with_cosim(
                        avg_left[layer], avg_right[layer], device="cuda:1", csls_k=10
                    )
                    avg_score_centered = evaluate_alignment_with_cosim(
                        avg_left[layer],
                        avg_right[layer],
                        device="cuda:1",
                        csls_k=10,
                        mean_centered=True,
                    )

                    with open(sentence_nn_file, "a") as f:
                        f.write(
                            f"{model_name},{dataset},{left_lang},{right_lang},True,cls,{layer},{cls_score_centered}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{left_lang},{right_lang},False,cls,{layer},{cls_score}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{left_lang},{right_lang},True,avg,{layer},{avg_score_centered}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{left_lang},{right_lang},False,avg,{layer},{avg_score}\n"
                        )

                    # do the same thing but permute right_lang and left_lang
                    cls_score = evaluate_alignment_with_cosim(
                        cls_right[layer], cls_left[layer], device="cuda:1", csls_k=10
                    )
                    cls_score_centered = evaluate_alignment_with_cosim(
                        cls_right[layer],
                        cls_left[layer],
                        device="cuda:1",
                        csls_k=10,
                        mean_centered=True,
                    )
                    avg_score = evaluate_alignment_with_cosim(
                        avg_right[layer], avg_left[layer], device="cuda:1", csls_k=10
                    )
                    avg_score_centered = evaluate_alignment_with_cosim(
                        avg_right[layer],
                        avg_left[layer],
                        device="cuda:1",
                        csls_k=10,
                        mean_centered=True,
                    )

                    with open(sentence_nn_file, "a") as f:
                        f.write(
                            f"{model_name},{dataset},{right_lang},{left_lang},True,cls,{layer},{cls_score_centered}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{right_lang},{left_lang},False,cls,{layer},{cls_score}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{right_lang},{left_lang},True,avg,{layer},{avg_score_centered}\n"
                        )
                        f.write(
                            f"{model_name},{dataset},{right_lang},{left_lang},False,avg,{layer},{avg_score}\n"
                        )

            # compute nearest-neighbor search for fasttext
            if fasttext_available:
                left_emb, right_emb = compute_sentence_representations_with_fasttext(
                    sentence_pairs,
                    left_lang,
                    right_lang,
                    fasttext_dir,
                    tokenizer=universal_tokenizer,
                    left_words=voc_emb_tuples[left_lang][0],
                    left_embedding=voc_emb_tuples[left_lang][1],
                    right_words=voc_emb_tuples[right_lang][0],
                    right_embedding=voc_emb_tuples[right_lang][1],
                )

                fasttext_score = evaluate_alignment_with_cosim(
                    left_emb, right_emb, device="cuda:1", csls_k=10
                )
                fasttext_score_centered = evaluate_alignment_with_cosim(
                    left_emb, right_emb, device="cuda:1", csls_k=10, mean_centered=True
                )

                with open(sentence_nn_file, "a") as f:
                    f.write(
                        f"fasttext,{dataset},{left_lang},{right_lang},False,avg,0,{fasttext_score}\n"
                    )
                    f.write(
                        f"fasttext,{dataset},{left_lang},{right_lang},True,avg,0,{fasttext_score_centered}\n"
                    )

                # Do the same thing but permute right_lang and left_lang
                fasttext_score = evaluate_alignment_with_cosim(
                    right_emb, left_emb, device="cuda:1", csls_k=10
                )
                fasttext_score_centered = evaluate_alignment_with_cosim(
                    right_emb, left_emb, device="cuda:1", csls_k=10, mean_centered=True
                )

                with open(sentence_nn_file, "a") as f:
                    f.write(
                        f"fasttext,{dataset},{right_lang},{left_lang},False,avg,0,{fasttext_score}\n"
                    )
                    f.write(
                        f"fasttext,{dataset},{right_lang},{left_lang},True,avg,0,{fasttext_score_centered}\n"
                    )
