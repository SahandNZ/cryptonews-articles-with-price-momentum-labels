import math
import os.path
from typing import List, Set, Dict, Tuple

import pandas as pd
from matplotlib import pyplot as plt

from definitions import STATS_DIR
from src.analyser.analyser import Analyser
from src.utils.string import has_digit


class DatasetAnalyser(Analyser):
    def __init__(self, dataframe: pd.DataFrame):
        self.df: pd.DataFrame = dataframe

        self.__labels: List[int] = None
        self.__words: List[str] = None
        self.__unique_words: Set[str] = None
        self.__word2label: Dict[str, int] = None
        self.__word2count: Dict[str, int] = None
        self.__word2rnf: Dict[str, int] = None
        self.__label2unique_words: Dict[int, Set[str]] = None
        self.__overlapping_unique_words: Set[str] = None
        self.__label2word2count: Dict[int, Dict[str, int]] = None
        self.__label2non_overlapping_word2count: Dict[int, Dict[str, int]] = None
        self.__label2sum_word_count: Dict[int, int] = None
        self.__word2rnf: Dict[str, float] = None
        self.__label2word2rnf: Dict[int, Dict[str, float]] = None
        self.__label2word2tfidf: Dict[int, Dict[str, float]] = None

        self.__ten_most_repetitive_words: List[Tuple[str, int]] = None
        self.__label2ten_most_non_overlapping_repetitive_words_count: Dict[int, List[Tuple[str, int]]] = None
        self.__ten_most_rnf_words: List[Tuple[str, int]] = None
        self.__label2ten_most_tfidf_words: Dict[int, List[Tuple[str, int]]] = None

    @property
    def samples_count(self) -> int:
        return len(self.df)

    @property
    def sentences_count(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> List[int]:
        if self.__labels is None:
            self.__labels = self.df.label.unique().tolist()

        return self.__labels

    @property
    def words(self) -> List[str]:
        if self.__words is None:
            self.__words = [word for sentence in self.df.text for word in sentence.split() if not has_digit(word)]

        return self.__words

    @property
    def unique_words(self) -> Set[str]:
        if self.__unique_words is None:
            self.__unique_words = set(self.words)

        return self.__unique_words

    @property
    def word2label(self) -> Dict[str, int]:
        if self.__word2label is None:
            self.__word2label = {}
            for index in range(len(self.df)):
                sentence, label = self.df.iloc[index]
                for word in sentence.split():
                    if not has_digit(word):
                        value = self.__word2label[word] | label if word in self.__word2label else label
                        self.__word2label[word] = value

        return self.__word2label

    @property
    def word2count(self) -> Dict[str, int]:
        if self.__word2count is None:
            self.__word2count = {}
            for word in self.words:
                self.__word2count[word] = 1 if word not in self.__word2count else self.__word2count[word] + 1
            self.__word2count = {w: c for w, c in sorted(self.__word2count.items(), key=lambda x: x[1], reverse=True)}

        return self.__word2count

    @property
    def label2unique_words(self) -> Dict[int, Set[str]]:
        if self.__label2unique_words is None:
            self.__label2unique_words = {label: set() for label in self.labels}
            for index in range(len(self.df)):
                sentence, label = self.df.iloc[index]
                for word in sentence.split():
                    if not has_digit(word):
                        self.__label2unique_words[label].add(word)

        return self.__label2unique_words

    @property
    def overlapping_unique_words(self) -> Set[str]:
        if self.__overlapping_unique_words is None:
            self.__overlapping_unique_words = self.label2unique_words[1].intersection(self.label2unique_words[2])

        return self.__overlapping_unique_words

    @property
    def label2word2count(self) -> Dict[int, Dict[str, int]]:
        if self.__label2word2count is None:
            self.__label2word2count = {label: {} for label in self.labels}
            for index in range(len(self.df)):
                sentence, label = self.df.iloc[index]
                for word in sentence.split():
                    if not has_digit(word):
                        if word not in self.__label2word2count[label]:
                            self.__label2word2count[label][word] = 1
                        else:
                            self.__label2word2count[label][word] += 1

            for label in self.labels:
                w2c = self.__label2word2count[label]
                sorted_w2c = {w: c for w, c in sorted(w2c.items(), key=lambda x: x[1], reverse=True)}
                self.__label2word2count[label] = sorted_w2c

        return self.__label2word2count

    @property
    def label2non_overlapping_word2count(self) -> Dict[int, Dict[str, int]]:
        if self.__label2non_overlapping_word2count is None:
            self.__label2non_overlapping_word2count = {}
            for label in self.labels:
                other = 1 if 2 == label else 2
                w2c = self.label2word2count[label]
                l2uw = self.label2unique_words
                uw2c = {w: c for w, c in w2c.items() if w in l2uw[label] and w not in l2uw[other]}
                self.__label2non_overlapping_word2count[label] = uw2c

        return self.__label2non_overlapping_word2count

    @property
    def label2sum_word_count(self) -> Dict[int, int]:
        if self.__label2sum_word_count is None:
            self.__label2sum_word_count = {}
            for label in self.labels:
                self.__label2sum_word_count[label] = sum(self.label2word2count[label].values())

        return self.__label2sum_word_count

    @property
    def word2rnf(self) -> Dict[str, float]:
        if self.__word2rnf is None:
            self.__word2rnf = {}
            l2w2c = self.label2word2count
            l2swc = self.label2sum_word_count
            for word in self.overlapping_unique_words:
                if word in l2w2c[1] and word in l2w2c[2]:
                    self.__word2rnf[word] = (l2w2c[1][word] / l2swc[1]) / (l2w2c[2][word] / l2swc[2])

            w2rnf = self.__word2rnf
            self.__word2rnf = {w: rnf for w, rnf in sorted(w2rnf.items(), key=lambda x: x[1], reverse=True)}

        return self.__word2rnf

    @property
    def label2word2rnf(self) -> Dict[int, Dict[str, float]]:
        if self.__label2word2rnf is None:
            self.__label2word2rnf = {label: {} for label in self.labels}
            for label in self.labels:
                for word in self.label2unique_words[label]:
                    if word in self.word2rnf:
                        self.__label2word2rnf[label][word] = self.word2rnf[word]

                w2rnf = self.__label2word2rnf[label]
                sorted_w2rnf = {w: rnf for w, rnf in sorted(w2rnf.items(), key=lambda x: x[1], reverse=True)}
                self.__label2word2rnf[label] = sorted_w2rnf

        return self.__label2word2rnf

    @property
    def label2word2tfidf(self) -> Dict[str, float]:
        if self.__label2word2tfidf is None:
            self.__label2word2tfidf = {label: {} for label in self.labels}
            for label in self.labels:
                for word in self.label2unique_words[label]:
                    if word in self.label2word2count[label]:
                        tf = self.label2word2count[label][word] / self.label2sum_word_count[label]
                        nt = 2 if 3 == self.word2label[word] else 1
                        idf = -math.log(nt / 2)
                        self.__label2word2tfidf[label][word] = tf * idf

                w2tfidf = self.__label2word2tfidf[label]
                sorted_w2tfidf = {w: tfidf for w, tfidf in sorted(w2tfidf.items(), key=lambda x: x[1], reverse=True)}
                self.__label2word2tfidf[label] = sorted_w2tfidf

        return self.__label2word2tfidf

    @property
    def ten_most_repetitive_words(self) -> List[Tuple[str, int]]:
        if self.__ten_most_repetitive_words is None:
            self.__ten_most_repetitive_words = list(self.word2count.items())[:10]

        return self.__ten_most_repetitive_words

    @property
    def label2ten_most_non_overlapping_repetitive_words(self) -> Dict[int, List[Tuple[str, int]]]:
        if self.__label2ten_most_non_overlapping_repetitive_words_count is None:
            l2now2c = self.label2non_overlapping_word2count
            l2ttnow = {label: list(l2now2c[label].items())[:10] for label in self.label2unique_words}
            self.__label2ten_most_non_overlapping_repetitive_words_count = l2ttnow

        return self.__label2ten_most_non_overlapping_repetitive_words_count

    @property
    def ten_most_rnf_words(self) -> List[Tuple[str, int]]:
        if self.__ten_most_rnf_words is None:
            self.__ten_most_rnf_words = list(self.word2rnf.items())[:10]

        return self.__ten_most_rnf_words

    @property
    def label2ten_most_tfidf_words(self) -> Dict[int, List[Tuple[str, float]]]:
        if self.__label2ten_most_tfidf_words is None:
            l2w2tfidf = self.label2word2tfidf
            self.__label2ten_most_tfidf_words = {label: list(l2w2tfidf[label].items())[:10] for label in self.labels}

        return self.__label2ten_most_tfidf_words

    def show_words_hist(self):
        plt.figure(figsize=(10, 10))
        plt.hist(self.words, bins=100)
        plt.xlabel("Words")
        plt.xticks([])
        plt.grid()
        plt.show()

    def save_unique_words_count(self):
        columns = ["label", "unique words count"]
        values = [["green", len(self.label2unique_words[2])],
                  ["red", len(self.label2unique_words[1])],
                  ["overlapping", len(self.overlapping_unique_words)]]
        df = pd.DataFrame(values, columns=columns)
        csv_path = os.path.join(STATS_DIR, 'unique words count.csv')
        df.to_csv(csv_path, index=False)

    def save_ten_most_repetitive_words(self):
        columns = ["word", "count"]
        values = [list(item) for item in self.ten_most_repetitive_words]
        df = pd.DataFrame(values, columns=columns)
        csv_path = os.path.join(STATS_DIR, '10 most repetitive words count.csv')
        df.to_csv(csv_path, index=False)

    def save_ten_most_non_overlapping_repetitive_words(self):
        for label, label_name in [(1, "red"), (2, "green")]:
            columns = ["word", "count"]
            values = [list(item) for item in self.label2ten_most_non_overlapping_repetitive_words[label]]
            df = pd.DataFrame(values, columns=columns)
            file_name = '10 most non-overlapping repetitive words count of {} label.csv'.format(label_name)
            csv_path = os.path.join(STATS_DIR, file_name)
            df.to_csv(csv_path, index=False)

    def save_ten_most_rnf_words(self):
        columns = ["word", "rnf"]
        values = [list(item) for item in self.ten_most_rnf_words]
        df = pd.DataFrame(values, columns=columns)
        csv_path = os.path.join(STATS_DIR, '10 most RNF words.csv')
        df.to_csv(csv_path, index=False)

    def save_ten_most_tfidf_words(self):
        for label, label_name in [(1, "red"), (2, "green")]:
            columns = ["word", "tf-idf"]
            values = [list(item) for item in self.label2ten_most_tfidf_words[label]]
            df = pd.DataFrame(values, columns=columns)
            file_name = '10 most TF-IDF of {} label.csv'.format(label_name)
            csv_path = os.path.join(STATS_DIR, file_name)
            df.to_csv(csv_path, index=False)

    def save_words_hist(self):
        figure_path = os.path.join(STATS_DIR, 'words hist figure.png')
        plt.figure(figsize=(10, 10))
        plt.hist(self.words, bins=100)
        plt.xlabel("Words")
        plt.xticks([])
        plt.grid()
        plt.savefig(figure_path)

    def save(self):
        self.save_unique_words_count()
        self.save_ten_most_repetitive_words()
        self.save_ten_most_non_overlapping_repetitive_words()
        self.save_ten_most_rnf_words()
        self.save_ten_most_tfidf_words()
        self.save_words_hist()

    def __str__(self):
        tmrw = self.ten_most_repetitive_words
        l2tmnorw = self.label2ten_most_non_overlapping_repetitive_words
        tmrnfw = self.ten_most_rnf_words
        return "Statistics" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40}" \
               "\n\t\t- {:<40} {}" \
               "\n\t\t- {:<40} {}" \
               "\n\t\t- {:<40} {}" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40}" \
               "\n\t\t- {:<40} {}" \
               "\n\t\t- {:<40} {}" \
               "\n\t- {:<40} {}" \
               "\n\t- {:<40}" \
               "\n\t\t- {:<40} {}" \
               "\n\t\t- {:<40} {}" \
            .format("Samples count", self.samples_count,
                    "Sentences count", self.sentences_count,
                    "Words count", len(self.words),
                    "Unique words count", len(self.unique_words),
                    "Unique words count of each label",
                    "Green", len(self.label2unique_words[2]),
                    "Red", len(self.label2unique_words[1]),
                    "Overlapping", len(self.overlapping_unique_words),
                    "10 most overlapping repetitive words", ''.join([f"\n\t\t- {w:<20}{c}" for w, c in tmrw]),
                    "10 most non-overlapping repetitive words of each label",
                    "Green", ''.join([f"\n\t\t\t- {w:<20}{c}" for w, c in l2tmnorw[2]]),
                    "Red", ''.join([f"\n\t\t\t- {w:<20}{c}" for w, c in l2tmnorw[1]]),
                    "10 most overlapping RNF words", ''.join([f"\n\t\t- {w:<20}{rnf:.4f}" for w, rnf in tmrnfw]),
                    "10 most TF-IDF words of each label",
                    "Green", ''.join([f"\n\t\t\t- {w:<20}{v:.4f}" for w, v in self.label2ten_most_tfidf_words[2]]),
                    "Red", ''.join([f"\n\t\t\t- {w:<20}{v:.4f}" for w, v in self.label2ten_most_tfidf_words[1]]))
