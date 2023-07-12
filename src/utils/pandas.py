import pandas as pd

from src.constant.time_frame import TimeFrame


def resample_based_on_time_frame(tohlcv: pd.DataFrame, source_time_frame: TimeFrame,
                                 destination_time_frame: TimeFrame) -> pd.DataFrame:
    df = tohlcv.copy()
    step = destination_time_frame // source_time_frame

    df['is_first'] = 0 == (df.timestamp % destination_time_frame)
    df['is_last'] = df.is_first.shift(step - 1).fillna(False)

    df['open'] = df.open[df.is_first]
    df['high'] = df.high.rolling(step).max().shift(-step + 1)
    df['low'] = df.low.rolling(step).min().shift(-step + 1)
    df['close'] = df.close[df.is_last]
    df['close'] = df.close.bfill()
    df['volume'] = df.volume.rolling(step).sum().shift(-step + 1)

    df = df.dropna()
    df = df.drop(['is_first', 'is_last'], axis=1)

    return df
