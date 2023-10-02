import os
import sys
import numpy as np

from transformers import AutoModel, AutoTokenizer

sys.path.append(os.curdir)

from multilingual_eval.utils import (
    UniversalTokenizer,
    get_tokenizer_type,
    load_embedding,
    get_nb_layers,
)
from multilingual_eval.data import sample_sentences
from multilingual_eval.retrieval import evaluate_alignment_with_cosim
from multilingual_eval.contextualized_pairs import (
    compute_pair_representations,
    compute_pair_representations_with_fasttext,
    generate_pairs,
)


if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR")
    dico_path = os.getenv("DICO_PATH")
    fasttext_dir = os.getenv("ALIGNED_FASTTEXT_DIR")
    transformers_cache_dir = os.getenv("TRANSFORMERS_CACHE_DIR")
    data_cache_dir = os.getenv("DATASETS_CACHE_DIR")
    assert data_dir is not None, "env var DATA_DIR should be defined"
    assert os.path.isdir(data_dir), f"value provided for DATA_DIR {data_dir} is not a directory"
    assert os.path.isdir(
        dico_path
    ), f"value provided for dico_path {dico_path} is not a directory (create the directory first?)"

    assert os.path.isdir(
        fasttext_dir
    ), f"value provided for fasstext_dir {fasttext_dir} is not a directory (create the directory first?)"

    model_dirs = [
        "bert-base-multilingual-cased",
        "aneuraz/awesome-align-with-co",
        "xlm-mlm-tlm-xnli15-1024",
        "xlm-roberta-base",
        "facebook/mbart-large-50",
        "xlm-mlm-100-1280",
        "xlm-roberta-large",
    ]

    dataset_tuples = [("wmt19", "de", "en"), ("wmt19", "zh", "en"), ("wmt19", "ru", "en")]

    n_runs = 10



    base_dir = os.path.join(data_dir, "raw_results")

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    word_nn_file = os.path.join(base_dir, "02_word_nn.csv")

    if not os.path.isfile(word_nn_file):
        with open(word_nn_file, "w") as f:
            f.write("model,dataset,left_lang,right_lang,layer,strong,score\n")

    models = [
        AutoModel.from_pretrained(model_dir, cache_dir=transformers_cache_dir)
        for model_dir in model_dirs
    ]
    tokenizers = [
        AutoTokenizer.from_pretrained(model_dir, cache_dir=transformers_cache_dir)
        for model_dir in model_dirs
    ]
    model_names = list(map(lambda x: x.split("/")[-1], model_dirs))
    n_layers_list = list(map(lambda x: get_nb_layers(x), models))
    size_list = list(map(lambda x: x.config.hidden_size, models))
    split_types = list(map(get_tokenizer_type, tokenizers))

    universal_tokenizer = UniversalTokenizer()

    # Retrieve embeddings in advance
    voc_emb_tuples = {}
    for _, left_lang, right_lang in dataset_tuples:
        if left_lang not in voc_emb_tuples:
            voc_emb_tuples[left_lang] = load_embedding(
                os.path.join(fasttext_dir, f"wiki.{left_lang}.align.vec")
            )
        if right_lang not in voc_emb_tuples:
            voc_emb_tuples[right_lang] = load_embedding(
                os.path.join(fasttext_dir, f"wiki.{right_lang}.align.vec")
            )

    for i_run in range(n_runs):
        print(f"RUN {i_run+1}/{n_runs}")

        for dataset, left_lang, right_lang in dataset_tuples:
            print(f"Evaluating {dataset} on {left_lang}-{right_lang}")
            sentence_generator = list(
                sample_sentences(
                    dataset,
                    left_lang=left_lang,
                    right_lang=right_lang,
                    nb=100_000 if left_lang != "zh" and right_lang != "zh" else 1_000_000,
                    cache_dir=data_cache_dir,
                )
            )

            pairs = generate_pairs(
                sentence_generator,
                dico_path,
                left_lang=left_lang,
                right_lang=right_lang,
                avoid_repetition=True,
                tokenizer=universal_tokenizer,
            )

            if len(pairs) < 5000:
                raise Exception(
                    f"Not enough pair with dataset {dataset} and pair {left_lang}-{right_lang}"
                )

            selected_pairs = np.random.choice(pairs, size=(5000,), replace=False)

            for i, (model, tokenizer, name, n_layers, size, split_type) in enumerate(
                zip(models, tokenizers, model_names, n_layers_list, size_list, split_types)
            ):
                print(f"    Evaluating model {name}")
                left_embs, right_embs = compute_pair_representations(
                    model,
                    tokenizer,
                    selected_pairs,
                    batch_size=1 if size > 800 else 2,
                    dim_size=size,
                    n_layers=n_layers,
                    device="cuda:0",
                    left_lang=left_lang,
                    right_lang=right_lang,
                    split_type=split_type,
                )

                # Remove references to GPU
                model = model.cpu()
                models[i] = model

                for layer in range(n_layers):
                    # evaluate
                    score = evaluate_alignment_with_cosim(
                        left_embs[layer], right_embs[layer], device="cuda:0", csls_k=10
                    )
                    score_strong = evaluate_alignment_with_cosim(
                        left_embs[layer],
                        right_embs[layer],
                        device="cuda:0",
                        csls_k=10,
                        strong_alignment=1.0,
                    )

                    # write in file
                    with open(word_nn_file, "a") as f:
                        f.write(
                            f"{name},{dataset},{left_lang},{right_lang},{layer},False,{score}\n"
                        )
                    with open(word_nn_file, "a") as f:
                        f.write(
                            f"{name},{dataset},{left_lang},{right_lang},{layer},True,{score_strong}\n"
                        )

            print("   Evaluating fasttext")
            # compute fasttext representations
            left_embs, right_embs = compute_pair_representations_with_fasttext(
                selected_pairs,
                left_lang,
                right_lang,
                fasttext_dir,
                tokenizer=universal_tokenizer,
                left_words=voc_emb_tuples[left_lang][0],
                left_embedding=voc_emb_tuples[left_lang][1],
                right_words=voc_emb_tuples[right_lang][0],
                right_embedding=voc_emb_tuples[right_lang][1],
            )

            # evaluate
            score = evaluate_alignment_with_cosim(left_embs, right_embs, device="cuda:0", csls_k=10)
            score_strong = evaluate_alignment_with_cosim(
                left_embs,
                right_embs,
                device="cuda:0",
                csls_k=10,
                strong_alignment=1.0,
            )

            # write in file
            with open(word_nn_file, "a") as f:
                f.write(f"fasttext,{dataset},{left_lang},{right_lang},{layer},False,{score}\n")
            with open(word_nn_file, "a") as f:
                f.write(
                    f"fasttext,{dataset},{left_lang},{right_lang},{layer},True,{score_strong}\n"
                )
