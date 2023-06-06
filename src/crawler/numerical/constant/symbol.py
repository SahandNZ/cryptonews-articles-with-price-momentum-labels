from src.crawler.numerical.constant.constant import Constant


class Symbol(Constant):
    BTCUSDT = 'BTC-USDT'

    @staticmethod
    def to_binance(value):
        return ''.join(value.split('-'))

    @staticmethod
    def from_binance(value):
        return value.replace('USDT', '') + '-' + 'USDT'

    @staticmethod
    def to_str(value):
        return value
