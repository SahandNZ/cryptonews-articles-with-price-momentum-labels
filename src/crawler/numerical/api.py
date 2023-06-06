from abc import abstractmethod, ABC
from typing import List

from src.crawler.numerical.constant.symbol import Symbol
from src.crawler.numerical.constant.time_frame import TimeFrame
from src.crawler.numerical.model.candle import Candle


class API(ABC):
    def __init__(self, name: str, base_url: str, maximum_requestable_candles: int):
        self.name: str = name
        self.base_url: str = base_url
        self.maximum_requestable_candles: int = maximum_requestable_candles

    @abstractmethod
    def get_candles(self, symbol: Symbol, time_frame: TimeFrame, from_timestamp: int, to_timestamp: int) \
            -> List[Candle]:
        raise NotImplementedError()
