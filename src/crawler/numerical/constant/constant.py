from abc import ABC, abstractmethod


class Constant(ABC):
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
