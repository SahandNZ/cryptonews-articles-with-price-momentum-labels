from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    @staticmethod
    @abstractmethod
    def from_binance(values: List):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_list(values: List):
        raise NotImplementedError()

    @abstractmethod
    def to_list(self) -> List:
        raise NotImplementedError()
