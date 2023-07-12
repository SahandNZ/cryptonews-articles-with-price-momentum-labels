import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from definitions import DATASET_DIR
from src.feature.feed_forward import FeedForward


def load_dataset(textual_source: str, labeling_method: str) -> pd.DataFrame:
    dataset_path = os.path.join(DATASET_DIR, textual_source, labeling_method, 'dataset.csv')
    sdf = pd.read_csv(dataset_path)

    return sdf


def create_dataset_with_sentence_as_feature(sdf: pd.DataFrame) -> Tuple:
    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        x.append(len(text))
        y.append(label)

    return x, y


def create_dataset_with_word_length_as_feature(sdf: pd.DataFrame, sequence_length: int) -> Tuple:
    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        feature = [0] * sequence_length
        for i, word in enumerate(text.split()):
            feature[i] = len(word)

        x.append(feature)
        y.append(label)

    return x, y


def create_dataset_with_words_as_feature(sdf: pd.DataFrame, sequence_length: int) -> Tuple:
    word2index = {}
    for text in sdf.text:
        for word in text.split():
            if word not in word2index:
                word2index[word] = len(word2index)

    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        words = text.split()
        feature = [0] * sequence_length
        for i in range(min(len(words), sequence_length)):
            feature[i] = word2index[words[i]]

        x.append(feature)
        y.append(label)

    return x, y


def create_dataset_with_words_bigram_as_feature(sdf: pd.DataFrame, sequence_length: int) -> Tuple:
    word2index = {}
    for text in sdf.text:
        for word in text.split():
            if word not in word2index:
                word2index[word] = len(word2index)

    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        words = text.split()
        feature = [0] * sequence_length
        for i in range(1, sequence_length + 1):
            if i < len(words):
                feature[i - 1] = [word2index[words[i]], word2index[words[i + 1]]]
            else:
                feature[i - 1] = [0, 0]

        x.append(feature)
        y.append(label)

    return x, y


def create_dataset_with_word2vec_as_feature(sdf: pd.DataFrame, model_path: str, sequence_length: int) -> Tuple:
    word2index = {}
    for text in sdf.text:
        for word in text.split():
            if word not in word2index:
                word2index[word] = len(word2index)

    word2vec = torch.jit.load(model_path)
    word2vec.eval()

    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        words = text.split()
        feature = [0] * sequence_length
        for i in i in range(1, min(len(words), sequence_length)):
            feature[i] = word2vec[word2index[words[i]]]

        x.append(feature)
        y.append(label)

    return x, y


def create_dataset_with_word2vec_bigram_as_feature(sdf: pd.DataFrame, model_path: str, sequence_length: int) -> Tuple:
    word2index = {}
    for text in sdf.text:
        for word in text.split():
            if word not in word2index:
                word2index[word] = len(word2index)

    word2vec = torch.jit.load(model_path)
    word2vec.eval()

    x, y = [], []
    bar = tqdm(range(len(sdf)))
    bar.set_description_str("Creating dataset")
    for index in bar:
        text = sdf.text.iloc[index]
        label = sdf.label.iloc[index]

        words = text.split()
        feature = [0] * sequence_length
        for i in range(1, sequence_length + 1):
            if i < len(words):
                feature[i - 1] = [word2vec[word2index[words[i - 1]]], word2vec[word2index[words[i]]]]
            else:
                feature[i - 1] = [0, 0]

        x.append(feature)
        y.append(label)

    return x, y


def create_dataset_with_bert_as_feature(sdf: pd.DataFrame, sequence_length: int) -> Tuple:
    pass


def to_tensor(x, y) -> Tuple:
    return torch.tensor(x), torch.tensor(y)


def create_splits(x, y, train_percentage: float = 0.8, val_percentage: float = 0.1) -> Tuple[Tuple, Tuple, Tuple]:
    samples = len(x)
    train_start_index = 0
    train_stop_index = round(samples * train_percentage)
    validation_start_index = train_stop_index
    validation_stop_index = round(samples * (train_percentage + val_percentage))
    test_start_index = validation_stop_index
    test_stop_index = samples

    train_split = x[train_start_index:train_stop_index], y[train_start_index:train_stop_index]
    val_split = x[validation_start_index:validation_stop_index], y[validation_start_index:validation_stop_index]
    test_split = x[test_start_index:test_start_index], y[test_start_index:test_stop_index]

    return train_split, val_split, test_split


def run_experiment(feature_transform):
    method = feature_transform["method"]
    params = feature_transform["params"]
    x, y = method(**params)
    x_tensor, y_tensor = to_tensor(x, y)
    train_set, val_set, test_set = create_splits(x_tensor, y_tensor)

    layers = [np.array(x.shape())[1:].prod(), 100, 100]
    dropout = 0
    activation_fn = torch.nn.Tanh()
    prediction_fn = lambda output: (0.5 <= output).int()
    model = FeedForward(layers=layers, dropout=dropout, activation_fn=activation_fn, prediction_fn=prediction_fn)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_experiments(textual_source: str, labeling_method: str, word2vec_model_path: str, sequence_length: int):
    sdf = load_dataset(textual_source=textual_source, labeling_method=labeling_method)
    transforms = [
        {
            "method": create_dataset_with_sentence_as_feature,
            "params": {"sdf": sdf}
        },
        {
            "method": create_dataset_with_word_length_as_feature,
            "params": {"sdf": sdf, "sequence_length": sequence_length}
        },
        {
            "method": create_dataset_with_words_as_feature,
            "params": {"sdf": sdf, "sequence_length": sequence_length}
        },
        {
            "method": create_dataset_with_words_bigram_as_feature,
            "params": {"sdf": sdf, "sequence_length": sequence_length}
        },
        {
            "method": create_dataset_with_word2vec_as_feature,
            "params": {"sdf": sdf, "model_path": word2vec_model_path, "sequence_length": sequence_length}
        },
        {
            "method": create_dataset_with_word2vec_bigram_as_feature,
            "params": {"sdf": sdf, "model_path": word2vec_model_path, "sequence_length": sequence_length}
        },
        {
            "method": create_dataset_with_bert_as_feature,
            "params": {"sdf": sdf, "sequence_length": sequence_length}
        }

    ]

    for transform in transforms:
        run_experiment(transform)
