import argparse
import math
import os.path
from datetime import datetime

from definition import NUMERICAL_DATA_DIR
from src.crawler.numerical.binance import BinanceDataApi
from src.crawler.numerical.constant.time_frame import TimeFrame
from src.crawler.numerical.model.candle import Candle
from src.utils.directory import create_directory_recursively

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', action='store', type=str, required=False, default='BTC-USDT')
    parser.add_argument('--timeframe', action='store', type=int, required=False, default=60 * 60 * 24 * 7)
    args = parser.parse_args()

    symbol = args.symbol
    time_frame = args.timeframe
    count = 200
    to_timestamp = math.floor(datetime.now().timestamp() / time_frame) * time_frame
    from_timestamp = to_timestamp - time_frame * count

    api = BinanceDataApi()
    candles = api.get_candles(symbol=symbol, time_frame=time_frame, from_timestamp=from_timestamp,
                              to_timestamp=to_timestamp)

    df = Candle.to_dataframe(candles)
    data_root = os.path.join(NUMERICAL_DATA_DIR, symbol)
    create_directory_recursively(data_root)
    data_path = os.path.join(data_root, '{}.csv'.format(TimeFrame.to_str(time_frame)))
    df.to_csv(data_path, index=False)
