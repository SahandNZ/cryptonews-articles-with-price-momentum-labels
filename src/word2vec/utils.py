import os
import random
from collections import Counter
from typing import List

import pandas as pd
import torch
from tqdm import tqdm

from definitions import DATASET_DIR


def load_words(textual_source: str, labeling_method: str, label: int) -> List[str]:
    dataset_path = os.path.join(DATASET_DIR, textual_source, labeling_method, 'dataset.csv')
    sdf = pd.read_csv(dataset_path)
    if label is not None:
        sdf = sdf[label == sdf.label]

    words = []
    for text in sdf.text:
        words.extend(text.split())

    return words


def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)  # descending freq order
    int2vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab2int = {word: ii for ii, word in int2vocab.items()}

    return vocab2int, int2vocab


def get_targets(words, idx, window_size=5):
    R = random.randint(1, window_size)
    start = max(0, idx - R)
    end = min(idx + R, len(words) - 1)
    targets = words[start:idx] + words[idx + 1:end + 1]

    return targets


def get_batches(words, batch_size):
    for i in range(0, len(words), batch_size):
        curr = words[i:i + batch_size]
        batch_x, batch_y = [], []

        for ii in range(len(curr)):
            x = [curr[ii]]
            y = get_targets(curr, ii)
            batch_x.extend(x * len(y))
            batch_y.extend(y)

        yield batch_x, batch_y


def train_skip_gram(tokens, model, criterion, optimizer, n_negative_samples: int, batch_size: int, n_epochs: int):
    device = next(model.parameters()).device

    bar = tqdm(range(n_epochs))
    bar.set_description_str("Training Word2Vec model")
    for _ in bar:
        for inputs, targets in get_batches(tokens, batch_size=batch_size):
            inputs = torch.LongTensor(inputs).to(device)
            targets = torch.LongTensor(targets).to(device)

            embedded_input_words = model.forward_input(inputs)
            embedded_target_words = model.forward_target(targets)
            embedded_noise_words = model.forward_noise(batch_size=inputs.shape[0], n_samples=n_negative_samples)

            loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
