import argparse

from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame
from src.labeler.color import ColorLabeler
from src.labeler.roc import RateOfChangeLabeler


def add_label(method: str, numerical_source: str, symbol: Symbol, time_frame: TimeFrame, look_ahead: int,
              textual_source: str):
    if 'color' == method:
        labeler = ColorLabeler(numerical_source, symbol, time_frame, look_ahead, textual_source)
    elif 'roc' == method:
        labeler = RateOfChangeLabeler(numerical_source, symbol, time_frame, look_ahead, textual_source)
    labeler.to_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--numerical-source', action='store', type=str, required=False, default='binance')
    parser.add_argument('--symbol', action='store', type=str, required=False, default=Symbol.BTCUSDT)
    parser.add_argument('--time-frame', action='store', type=int, required=False, default=TimeFrame.DAY1)
    parser.add_argument('--look-ahead', action='store', type=int, required=False, default=7)
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--method', action='store', type=str, required=False, default='roc')
    args = parser.parse_args()

    add_label(args.method, args.numerical_source, args.symbol, args.time_frame, args.look_ahead, args.textual_source)
