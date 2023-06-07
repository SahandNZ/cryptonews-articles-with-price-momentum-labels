import os.path
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from definitions import NUMERICAL_CLEAN_DATA_DIR, TEXTUAL_CLEAN_DATA_DIR, DATASET_DIR
from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame
from src.model.candle import Candle
from src.model.news import News
from src.utils.directory import create_directory_recursively


class Labeler(ABC):
    def __init__(self, numerical_source: str, symbol: Symbol, time_frame: TimeFrame, look_ahead: int,
                 textual_source: str, name: str, train_percent: float = 0.8, validation_percent: float = 0.1):
        self.numerical_source: str = numerical_source
        self.symbol: Symbol = symbol
        self.time_frame: TimeFrame = time_frame
        self.look_ahead: int = look_ahead
        self.textual_source: str = textual_source

        self.name: str = name
        self.train_percent: float = train_percent
        self.validation_percent: float = validation_percent

    def pre_add_label(self) -> pd.DataFrame:
        time_frame_str = TimeFrame.to_str(self.time_frame)
        numerical_root = os.path.join(NUMERICAL_CLEAN_DATA_DIR, self.numerical_source, self.symbol)
        numerical_csv_path = os.path.join(numerical_root, '{}.csv'.format(time_frame_str))
        numerical_data: List[Candle] = Candle.from_csv(numerical_csv_path)
        ndf = Candle.to_dataframe(numerical_data)
        ndf['datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in ndf.timestamp]
        ndf['fclose'] = ndf.close.shift(self.look_ahead)
        ndf = ndf.set_index('datetime')
        ndf = self.add_label(ndf)
        ndf = ndf.dropna()

        textual_json_path = os.path.join(TEXTUAL_CLEAN_DATA_DIR, '{}.json'.format(self.textual_source))
        textual_data: List[News] = News.from_json(textual_json_path)
        values = []
        for news in textual_data:
            news_date = news.date.split()[0]
            if 5 < len(news.title.split()):
                values.append([news_date, news.title, news.url])
            for paragraph in news.paragraphs:
                if 5 < len(paragraph.split()):
                    values.append([news_date, paragraph, news.url])
        tdf = pd.DataFrame(values, columns=['datetime', 'text', 'url'])
        tdf = tdf.set_index('datetime')

        df = tdf.copy()
        df['label'] = ndf.label
        df = df.reset_index()
        df = df.sort_values(by='datetime')
        df = df.reset_index(drop=True)

        return df

    @abstractmethod
    def add_label(self, ndf: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        indices = (np.array([self.train_percent, self.validation_percent]).cumsum() * len(df)).astype(int)
        train_df = df[:indices[0]]
        validation_df = df[indices[0]: indices[1]]
        test_df = df[indices[1]:]

        return train_df, validation_df, test_df

    def to_csv(self):
        dataset_df = self.pre_add_label()
        train_df, validation_df, test_df = self.create_splits(dataset_df)

        dataset_root = os.path.join(DATASET_DIR, self.textual_source, self.name)
        dataset_csv_path = os.path.join(dataset_root, 'dataset.csv')
        train_csv_path = os.path.join(dataset_root, 'train.csv')
        validation_csv_path = os.path.join(dataset_root, 'validation.csv')
        test_csv_path = os.path.join(dataset_root, 'test.csv')
        create_directory_recursively(dataset_root)

        dataset_df.to_csv(dataset_csv_path, index=False)
        train_df.to_csv(train_csv_path, index=False)
        validation_df.to_csv(validation_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
