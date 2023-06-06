from abc import ABC, abstractmethod
from enum import Enum


class Constant(Enum):
    @staticmethod
    @abstractmethod
    def from_binance(value):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def to_binance(value):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def to_str(value):
        raise NotImplementedError()
