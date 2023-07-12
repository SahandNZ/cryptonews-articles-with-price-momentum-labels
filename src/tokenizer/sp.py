import io
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import sentencepiece as spm

from definitions import DATASET_DIR


def load_corpus(textual_source: str, labeling_method: str) -> List[str]:
    dataset_path = os.path.join(DATASET_DIR, textual_source, labeling_method, 'dataset.csv')
    sdf = pd.read_csv(dataset_path)
    corpus = sdf.text.to_list()

    return corpus


def cross_validation(corpus: List[str], n_split: int) -> Tuple[List[str], List[str]]:
    split_size = len(corpus) // n_split
    for split_index in range(n_split):
        test_split_start_index = split_index * split_size
        test_split_stop_index = test_split_start_index + split_size
        test_split = corpus[test_split_start_index: test_split_stop_index]
        train_split = corpus[:test_split_start_index] + corpus[test_split_stop_index:]

        yield train_split, test_split


def save_spp(model: io.BytesIO, path: str):
    with open(path, 'wb') as outfile:
        outfile.write(model.getvalue())


def load_spp(path: str) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=path)


def train_spp(corpus: list[str], vocab_size: int) -> io.BytesIO:
    model = io.BytesIO()
    spm.SentencePieceTrainer.Train(sentence_iterator=iter(corpus), model_writer=model, vocab_size=vocab_size,
                                   pad_id=0, unk_id=1, bos_id=2, eos_id=3)

    return model


def calculate_unk_rate(corpus: List[str], sp: spm.SentencePieceProcessor):
    encoded_corpus = sp.Encode(corpus)

    unk_count = 0
    for encoded_text in encoded_corpus:
        unk_count += (1 == np.array(encoded_text)).sum()

    return round(unk_count / len(encoded_corpus) * 100, 2)

