from datetime import datetime
from typing import List

import pandas as pd


class Candle:
    def __init__(self):
        self.timestamp: int = None
        self.datetime: datetime = None
        self.open: float = None
        self.high: float = None
        self.low: float = None
        self.close: float = None
        self.volume: float = None
        self.trade: float = None

    @staticmethod
    def from_binance(values: List):
        instance = Candle()

        instance.timestamp = int(int(values[0]) / 1000)
        instance.datetime = datetime.fromtimestamp(instance.timestamp)
        instance.open = float(values[1])
        instance.high = float(values[2])
        instance.low = float(values[3])
        instance.close = float(values[4])
        instance.volume = float(values[5])
        instance.trade = int(values[8])

        return instance

    @staticmethod
    def from_list(values: List):
        instance = Candle()

        instance.timestamp = int(values[0])
        instance.datetime = datetime.fromtimestamp(instance.timestamp)
        instance.open = float(values[1])
        instance.high = float(values[2])
        instance.low = float(values[3])
        instance.close = float(values[4])
        instance.volume = float(values[5])
        instance.trade = int(values[6])

        return instance

    def to_list(self) -> List:
        return [self.timestamp, self.open, self.high, self.low, self.close, self.volume, self.trade]

    @staticmethod
    def to_dataframe(candles: List):
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade']
        values = [candle.to_list() for candle in candles]
        df = pd.DataFrame(values, columns=columns)
        return df
