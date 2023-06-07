import numpy as np
import pandas as pd

from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame
from src.labeler.labler import Labeler


class RateOfChangeLabeler(Labeler):
    def __init__(self, numerical_source: str, symbol: Symbol, time_frame: TimeFrame, look_ahead: int,
                 textual_source: str):
        super().__init__(numerical_source, symbol, time_frame, look_ahead, textual_source, 'roc')

    def add_label(self, ndf: pd.DataFrame):
        ndf['label'] = np.round(ndf.fclose / ndf.close - 1, 4)

        return ndf
