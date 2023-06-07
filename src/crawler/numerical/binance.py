import json
from typing import List
from urllib import parse

import requests

from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame
from src.crawler.numerical.api import API
from src.model.candle import Candle


class BinanceDataAPI(API):
    def __init__(self):
        super().__init__(name='binance', base_url="https://fapi.binance.com", maximum_requestable_candles=1500)

    def get_candles(self, symbol: str, time_frame: TimeFrame, from_timestamp: int, to_timestamp: int) -> List[Candle]:
        endpoint = '/fapi/v1/klines'
        params = {
            'symbol': Symbol.to_binance(symbol),
            'interval': TimeFrame.to_binance(time_frame),
            'startTime': from_timestamp * 1000,
            'endTime': to_timestamp * 1000,
            'limit': int((to_timestamp - from_timestamp) / time_frame)
        }

        url = parse.urljoin(self.base_url, endpoint)
        response_json = requests.get(url=url, params=params)
        response_dict = json.loads(response_json.text)
        candles = [Candle.from_binance(item) for item in response_dict]
        return candles
