from src.crawler.numerical.constant.constant import Constant


class TimeFrame(Constant):
    MIN1: int = 60
    MIN3: int = 3 * MIN1
    MIN5: int = 5 * MIN1
    MIN15: int = 15 * MIN1
    MIN30: int = 30 * MIN1
    HOUR1: int = 60 * MIN1
    HOUR2: int = 2 * HOUR1
    HOUR4: int = 4 * HOUR1
    HOUR6: int = 6 * HOUR1
    HOUR8: int = 8 * HOUR1
    HOUR12: int = 12 * HOUR1
    DAY1: int = 24 * HOUR1
    DAY3: int = 3 * DAY1
    WEEK1: int = 7 * DAY1
    MON1: int = 30 * DAY1

    @staticmethod
    def from_binance(value):
        raise NotImplementedError()

    @staticmethod
    def to_binance(value):
        minute = int(value / 60)
        hour = int(minute / 60)
        day = int(hour / 24)
        week = int(day / 7)

        if minute < 60:
            return "{}m".format(minute)
        elif hour < 24:
            return "{}h".format(hour)
        elif day < 7:
            return "{}d".format(day)
        elif week < 4:
            return "{}w".format(week)
        return "1M"

    @staticmethod
    def to_str(time_frame: int) -> str:
        minute = time_frame // 60
        hour = minute // 60
        day = hour // 24
        week = day // 7

        if minute < 60:
            return "{} Minute{}".format(minute, 's' if 1 != minute else '')
        elif hour < 24:
            return "{} Hour{}".format(hour, 's' if 1 != hou else '')
        elif day < 7:
            return "{} Day{}".format(day, 's' if 1 != day else '')
        elif 1 == week:
            return "{} Week".format(week)
        else:
            return "1 Month"
