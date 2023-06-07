import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from nltk.corpus import stopwords


class Cleaner(ABC):
    def __init__(self, raw_data: List):
        self.raw_data: List = raw_data

    @staticmethod
    def clean_datetime(raw_datetime: str) -> datetime:
        clean_datetime_str = raw_datetime.split('+')[0].replace('T', ' ')
        return clean_datetime_str

    @staticmethod
    def clean_string(raw_string: str):
        # Normalize string
        text = raw_string.lower()

        # Remove unicode characters
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

        # Remove stop words
        words = text.split()
        useless_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in useless_words]

        # join words
        clean_string = ' '.join(filtered_words)

        return clean_string

    @abstractmethod
    def clean(self, raw_data: List) -> List:
        raise NotImplementedError()

    @abstractmethod
    def to_json(self, path: str):
        raise NotImplementedError()
